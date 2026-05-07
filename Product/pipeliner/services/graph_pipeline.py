from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from typing import Any, AsyncGenerator

import neo4j

from core.config import PipelineConfig
from core.pipeline_history import PipelineHistory
from services.llm_service import get_llm, LLMInterface
from services.chunking_service import get_chunker
from services.extraction_service import EntityRelationshipExtractor
from services.resolution_service import EntityResolver
from services.neo4j_writer import GraphRAGNeo4jWriter
from services.community_service import CommunityDetector, CommunitySummarizer
from services.embedding_utils import hf_embed

logger = logging.getLogger(__name__)


class GraphRAGPipeline:
    def __init__(self, config: PipelineConfig, chat_id: str = "", document_id: str = "") -> None:
        self.chat_id = chat_id
        self.document_id = document_id
        self.config = config

        # Override embedding model with the chat's selected model to ensure consistency
        if self.chat_id:
            from core.db import connection
            try:
                with connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT embedding_model FROM chats WHERE chat_id = %s", (self.chat_id,))
                        row = cur.fetchone()
                        if row and row.get("embedding_model"):
                            self.config.embedding.model = row["embedding_model"]
                            logger.info("Override config embedding model to match chat: %s", self.config.embedding.model)
            except Exception as e:
                logger.error("Failed to fetch embedding model for chat %s: %s", self.chat_id, e)
        self._run_id: str | None = None

        try:
            self._history = PipelineHistory()
        except Exception as e:
            logger.warning("Pipeline history disabled: %s", e)
            self._history = None

        self.llm: LLMInterface = get_llm(
            provider=config.llm.provider.value,
            model=config.llm.model,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
        )
        self.chunker = get_chunker(
            strategy=config.chunking.strategy,
            chunk_size=config.chunking.chunk_size,
            chunk_overlap=config.chunking.chunk_overlap,
            llm=self.llm,
        )
        self._driver = neo4j.GraphDatabase.driver(
            config.neo4j.uri,
            auth=(config.neo4j.user, config.neo4j.password),
        )
        self.writer = GraphRAGNeo4jWriter(
            driver=self._driver,
            database=config.neo4j.database,
        )
        self.er_extractor = EntityRelationshipExtractor(
            llm=self.llm,
            max_concurrency=config.extraction.max_concurrency,
        )
        self.step_times: dict[str, float] = {}

    async def run(self, document_text: str, doc_title: str = "", doc_source: str = "") -> AsyncGenerator[str, None]:
        """Run the full pipeline, yielding status messages for real-time streaming."""
        total_start = time.time()

        yield "Initializing pipeline..."
        self.writer.create_indexes()

        if self.chat_id:
            self.writer.write_chat(self.chat_id)

        doc_id = hashlib.md5(f"{self.chat_id}:{doc_title}".encode()).hexdigest()
        doc_id = self.writer.write_document(
            doc_id=doc_id,
            chat_id=self.chat_id,
            title=doc_title,
            source=doc_source,
        )
        yield f"Document registered: {doc_id}"

        if self._history and self.chat_id and self.document_id:
            try:
                self._run_id = self._history.create_run(
                    chat_id=self.chat_id,
                    document_id=self.document_id,
                    config_snapshot=self.config.model_dump(mode="json"),
                )
            except Exception as e:
                logger.warning("Failed to create pipeline_run record: %s", e)

        try:
            # Step 1: Chunking
            self._update_run_status("CHUNKING")
            async for msg in self._chunk_step(doc_id, document_text):
                yield msg

            # Step 2: Extraction
            self._update_run_status("EXTRACTING")
            async for msg in self._extraction_step(doc_id):
                yield msg

            # Step 3: Entity Resolution
            if self.config.entity_resolution.enabled:
                self._update_run_status("RESOLVING")
                async for msg in self._resolution_step(doc_id):
                    yield msg

            # Step 4: Embedding
            self._update_run_status("EMBEDDING")
            async for msg in self._embedding_step(doc_id):
                yield msg

            # Step 5: Community Detection
            self._update_run_status("COMMUNITY_DETECTION")
            async for msg in self._community_step(doc_id):
                yield msg

            # Step 6: Community Summarization
            self._update_run_status("SUMMARIZING")
            async for msg in self._summarization_step(doc_id):
                yield msg

            total_time = time.time() - total_start
            summary = self._build_summary(total_time)
            yield summary

            self._complete_run()

            yield "Pipeline completed."

        except Exception as e:
            if self._history and self._run_id:
                try:
                    self._history.fail_run(self._run_id, str(e))
                except Exception:
                    pass
            raise

    async def _chunk_step(self, doc_id: str, document_text: str) -> AsyncGenerator[str, None]:
        status = self.writer.get_document_status(doc_id)
        chunks_count = self.writer.get_chunks_count(doc_id)

        if status in ("CHUNKED", "EXTRACTED", "RESOLVED", "COMPLETED") or chunks_count > 0:
            yield "Step 1/6: Chunking — skipped (already completed)."
            return

        yield f"Step 1/6: Chunking ({self.config.chunking.strategy} strategy)..."
        t0 = time.time()
        yield f"detail:chunk:Splitting document ({len(document_text):,} chars) into chunks..."
        chunks = self.chunker.chunk(document_text)
        yield f"detail:chunk:Created {len(chunks)} chunks (avg {len(document_text) // max(len(chunks), 1)} chars each)"
        yield f"detail:chunk:Writing chunks to Neo4j..."
        self.writer.write_chunks(doc_id, chunks)
        yield f"detail:chunk:All {len(chunks)} chunks persisted"
        self.step_times["1. Chunking"] = time.time() - t0
        yield f"Step 1/6: Chunking complete — {len(chunks)} chunks produced."

    async def _extraction_step(self, doc_id: str) -> AsyncGenerator[str, None]:
        unextracted = self.writer.get_unextracted_chunks(doc_id)
        if not unextracted:
            yield "Step 2/6: Extraction — skipped (all chunks already extracted)."
            return

        yield f"Step 2/6: Extracting entities & relations from {len(unextracted)} chunks..."
        t0 = time.time()

        chunk_ids = [u[0] for u in unextracted]
        chunk_texts = [u[1] for u in unextracted]
        total = len(chunk_texts)

        yield f"detail:extract:Starting LLM extraction (concurrency={self.er_extractor.max_concurrency})..."

        # Use a queue so progress messages stream in real-time
        progress_queue: asyncio.Queue = asyncio.Queue()

        def on_chunk_done(completed: int, total: int, result):
            e_count = len(result.entities)
            r_count = len(result.relations)
            pct = int(completed / total * 100)
            progress_queue.put_nowait(
                f"detail:extract:Chunk {result.chunk_index + 1}/{total} → "
                f"{e_count} entities, {r_count} relations"
            )
            progress_queue.put_nowait(f"progress:extract:{completed}/{total}")

        task = asyncio.create_task(
            self.er_extractor.extract_from_chunks_with_progress(
                chunk_texts, progress_callback=on_chunk_done,
            )
        )

        # Yield progress messages as they arrive
        completed = 0
        while completed < total:
            try:
                msg = await asyncio.wait_for(progress_queue.get(), timeout=0.5)
                yield msg
                if msg.startswith("progress:"):
                    completed = int(msg.rsplit("/", 1)[0].rsplit(":", 1)[1])
            except asyncio.TimeoutError:
                if task.done():
                    break

        # Drain any remaining messages
        while not progress_queue.empty():
            msg = progress_queue.get_nowait()
            yield msg

        extraction_results = await task

        yield f"detail:extract:Writing results to Neo4j..."
        self.writer.write_entities_and_relations(chunk_ids, extraction_results)

        total_entities = sum(len(r.entities) for r in extraction_results)
        total_relations = sum(len(r.relations) for r in extraction_results)
        self.step_times["2. Extraction"] = time.time() - t0
        self.writer.update_document_status(doc_id, "EXTRACTED")
        yield f"Step 2/6: Extraction complete — {total_entities} entities, {total_relations} relations."

    async def _resolution_step(self, doc_id: str) -> AsyncGenerator[str, None]:
        status = self.writer.get_document_status(doc_id)
        if status == "RESOLVED":
            yield "Step 3/6: Entity Resolution — skipped (already completed)."
            return

        yield "Step 3/6: Entity Resolution (KNN → WCC → Word Distance → LLM)..."
        t0 = time.time()
        er_config = self.config.entity_resolution
        embedding_model = self.config.embedding.model

        async def embed_fn(texts: list[str]) -> list[list[float]]:
            return await hf_embed(texts, model_name=embedding_model, show_progress=True)

        resolver = EntityResolver(
            driver=self._driver,
            llm=self.llm,
            embedding_fn=embed_fn,
            k_neighbors=er_config.k_neighbors,
            similarity_threshold=er_config.similarity_threshold,
            word_distance_threshold=er_config.word_distance_threshold,
            max_concurrency=er_config.max_concurrency,
            database=self.config.neo4j.database,
            use_llm=er_config.use_llm,
        )
        yield f"detail:resolve:Computing entity embeddings..."
        yield f"detail:resolve:LLM verification {'enabled' if er_config.use_llm else 'disabled'}"
        decisions = await resolver.resolve()
        for d in decisions:
            yield f"detail:resolve:Merged [{', '.join(d.group)}] → '{d.canonical_name}'"
        self.step_times["3. Entity Resolution"] = time.time() - t0
        self.writer.update_document_status(doc_id, "RESOLVED")
        yield f"Step 3/6: Entity Resolution complete — {len(decisions)} merge groups resolved."

    async def _embedding_step(self, doc_id: str) -> AsyncGenerator[str, None]:
        yield "Step 4/6: Embedding entities and chunks..."
        t0 = time.time()

        # 1. Embed Entities
        entity_rows: list[tuple[str, str]] = []
        with self._driver.session(database=self.config.neo4j.database) as session:
            result = session.run(
                "MATCH (e:Entity) WHERE e.text_embedding IS NULL "
                "RETURN e.name AS name, e.description AS description"
            )
            entity_rows = [(r["name"], r["description"] or "") for r in result if r["name"]]

        if entity_rows:
            yield f"detail:embed:Embedding {len(entity_rows)} entities using {self.config.embedding.model}..."
            embed_texts = [
                f"{name}: {desc}".strip(": ") if desc else name
                for name, desc in entity_rows
            ]
            entity_embeddings = await hf_embed(
                embed_texts,
                model_name=self.config.embedding.model,
                show_progress=True,
            )
            name_to_emb = {name: emb for (name, _), emb in zip(entity_rows, entity_embeddings)}
            self.writer.write_entity_embeddings(name_to_emb)
            yield f"detail:embed:✔ {len(entity_rows)} entity embeddings written to Neo4j"
            yield f"  - Embedded {len(entity_rows)} entities."

        # 2. Embed Chunks
        chunk_rows: list[tuple[str, str]] = []
        with self._driver.session(database=self.config.neo4j.database) as session:
            result = session.run(
                "MATCH (c:Chunk) WHERE c.text_embedding IS NULL "
                "RETURN c.id AS id, c.text AS text"
            )
            chunk_rows = [(r["id"], r["text"]) for r in result if r["text"]]

        if chunk_rows:
            yield f"detail:embed:Embedding {len(chunk_rows)} chunks using {self.config.embedding.model}..."
            chunk_texts = [text for _, text in chunk_rows]
            chunk_embeddings = await hf_embed(
                chunk_texts,
                model_name=self.config.embedding.model,
                show_progress=True,
            )
            id_to_emb = {cid: emb for (cid, _), emb in zip(chunk_rows, chunk_embeddings)}
            self.writer.write_chunk_embeddings(id_to_emb)
            yield f"detail:embed:✔ {len(chunk_rows)} chunk embeddings written to Neo4j"
            yield f"  - Embedded {len(chunk_rows)} chunks."

        self.step_times["4. Embedding"] = time.time() - t0
        
        if not entity_rows and not chunk_rows:
            yield "Step 4/6: Embedding — skipped (all entities and chunks already embedded)."
        else:
            yield f"Step 4/6: Embedding complete."

    async def _community_step(self, doc_id: str) -> AsyncGenerator[str, None]:
        yield "Step 5/6: Community detection..."
        t0 = time.time()
        yield f"detail:community:Running {self.config.community_detection.algorithm} algorithm (resolution={self.config.community_detection.resolution})..."
        detector = CommunityDetector(
            driver=self._driver,
            database=self.config.neo4j.database,
            algorithm=self.config.community_detection.algorithm,
            resolution=self.config.community_detection.resolution,
            level=self.config.community_detection.level,
        )
        communities = detector.detect()
        if communities:
            sizes = [len(members) for members in communities.values()]
            yield f"detail:community:Found {len(communities)} communities (avg size: {sum(sizes)/len(sizes):.1f}, range: {min(sizes)}-{max(sizes)})"
        yield f"detail:community:Writing communities to Neo4j..."
        self.writer.write_communities(communities)
        self.step_times["5. Community Detection"] = time.time() - t0
        yield f"Step 5/6: Community detection complete — {len(communities)} communities detected."

    async def _summarization_step(self, doc_id: str) -> AsyncGenerator[str, None]:
        yield "Step 6/6: Summarizing communities..."
        t0 = time.time()
        summarizer = CommunitySummarizer(
            driver=self._driver,
            llm=self.llm,
            max_concurrency=self.config.extraction.max_concurrency,
            database=self.config.neo4j.database,
        )
        communities_to_summarize = summarizer.get_unsummarized_communities()
        if not communities_to_summarize:
            yield "Step 6/6: Summarization — skipped (all communities already summarized)."
            return

        total = len(communities_to_summarize)
        yield f"detail:summarize:Summarizing {total} communities (concurrency={self.config.extraction.max_concurrency})..."

        progress_queue: asyncio.Queue = asyncio.Queue()

        def on_community_done(completed: int, total: int, comm_id: int):
            progress_queue.put_nowait(
                f"detail:summarize:Community {completed}/{total} summarized"
            )
            progress_queue.put_nowait(f"progress:summarize:{completed}/{total}")

        task = asyncio.create_task(
            summarizer.summarize(communities_to_summarize, progress_callback=on_community_done)
        )

        completed = 0
        while completed < total:
            try:
                msg = await asyncio.wait_for(progress_queue.get(), timeout=0.5)
                yield msg
                if msg.startswith("progress:"):
                    completed = int(msg.rsplit("/", 1)[0].rsplit(":", 1)[1])
            except asyncio.TimeoutError:
                if task.done():
                    break

        # Drain any remaining messages
        while not progress_queue.empty():
            msg = progress_queue.get_nowait()
            yield msg

        summaries = await task
        self.writer.write_community_summaries(summaries)
        self.step_times["6. Summarize Communities"] = time.time() - t0
        self.writer.update_document_status(doc_id, "COMPLETED")
        yield f"Step 6/6: Summarization complete — {len(summaries)} community summaries generated."

    def _build_summary(self, total_time: float) -> str:
        lines = ["--- Pipeline Summary ---"]
        for step, t in self.step_times.items():
            lines.append(f"  {step}: {t:.2f}s")
        lines.append(f"  TOTAL: {total_time:.2f}s")

        prompt_tokens = getattr(self.llm, "total_prompt_tokens", 0)
        completion_tokens = getattr(self.llm, "total_completion_tokens", 0)
        total_requests = getattr(self.llm, "total_requests", 0)
        lines.append(f"  LLM: {total_requests} requests, {prompt_tokens + completion_tokens} tokens")

        with self._driver.session(database=self.config.neo4j.database) as session:
            doc_count = session.run("MATCH (d:Document) RETURN count(d) as c").single()["c"]
            chunk_count = session.run("MATCH (c:Chunk) RETURN count(c) as c").single()["c"]
            entity_count = session.run("MATCH (e:Entity) RETURN count(e) as c").single()["c"]
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as c").single()["c"]
            community_count = session.run("MATCH (c:Community) RETURN count(c) as c").single()["c"]

        lines.append(f"  Neo4j: {doc_count} docs, {chunk_count} chunks, {entity_count} entities, {rel_count} rels, {community_count} communities")
        return "\n".join(lines)

    def _update_run_status(self, status: str) -> None:
        if self._history and self._run_id:
            try:
                self._history.update_status(self._run_id, status)
            except Exception as e:
                logger.warning("Failed to update pipeline_run status: %s", e)

    def _complete_run(self) -> None:
        if not self._history or not self._run_id:
            return
        try:
            prompt_tokens = getattr(self.llm, "total_prompt_tokens", 0)
            completion_tokens = getattr(self.llm, "total_completion_tokens", 0)
            total_requests = getattr(self.llm, "total_requests", 0)

            neo4j_stats = {}
            with self._driver.session(database=self.config.neo4j.database) as session:
                neo4j_stats["documents"] = session.run("MATCH (d:Document) RETURN count(d) as c").single()["c"]
                neo4j_stats["chunks"] = session.run("MATCH (c:Chunk) RETURN count(c) as c").single()["c"]
                neo4j_stats["entities"] = session.run("MATCH (e:Entity) RETURN count(e) as c").single()["c"]
                neo4j_stats["relationships"] = session.run("MATCH ()-[r]->() RETURN count(r) as c").single()["c"]
                neo4j_stats["communities"] = session.run("MATCH (c:Community) RETURN count(c) as c").single()["c"]

            self._history.complete_run(
                run_id=self._run_id,
                step_times=self.step_times,
                llm_usage={
                    "total_requests": total_requests,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                neo4j_stats=neo4j_stats,
            )
        except Exception as e:
            logger.warning("Failed to complete pipeline_run record: %s", e)

    def close(self) -> None:
        self._driver.close()
