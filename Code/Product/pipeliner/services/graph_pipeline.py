from __future__ import annotations

import hashlib
import logging
import time
from typing import Any, AsyncGenerator

import neo4j

from core.config import PipelineConfig, SupabaseConfig
from core.supabase_client import SupabasePipelineHistory
from services.llm_service import get_llm, LLMInterface
from services.chunking_service import get_chunker
from services.extraction_service import EntityRelationshipExtractor
from services.resolution_service import EntityResolver
from services.neo4j_writer import GraphRAGNeo4jWriter
from services.community_service import CommunityDetector, CommunitySummarizer
from services.embedding_utils import hf_embed

logger = logging.getLogger(__name__)


class GraphRAGPipeline:
    def __init__(self, config: PipelineConfig, user_id: str = "", document_id: str = "") -> None:
        self.user_id = user_id
        self.document_id = document_id
        self._run_id: str | None = None

        # Initialize Supabase history (optional — won't fail if creds missing)
        try:
            self._history = SupabasePipelineHistory(config.supabase)
        except (ValueError, Exception) as e:
            logger.warning("Supabase history disabled: %s", e)
            self._history = None
        self.config = config
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

        doc_id = hashlib.md5(doc_title.encode()).hexdigest()
        doc_id = self.writer.write_document(doc_id=doc_id, title=doc_title, source=doc_source)
        yield f"Document registered: {doc_id}"

        # Create a pipeline_run record in Supabase
        if self._history and self.user_id and self.document_id:
            try:
                self._run_id = self._history.create_run(
                    user_id=self.user_id,
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

            # Persist final stats to Supabase
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
        chunks = self.chunker.chunk(document_text)
        self.writer.write_chunks(doc_id, chunks)
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

        extraction_results = await self.er_extractor.extract_from_chunks(chunk_texts)
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
        decisions = await resolver.resolve()
        self.step_times["3. Entity Resolution"] = time.time() - t0
        self.writer.update_document_status(doc_id, "RESOLVED")
        yield f"Step 3/6: Entity Resolution complete — {len(decisions)} merge groups resolved."

    async def _embedding_step(self, doc_id: str) -> AsyncGenerator[str, None]:
        yield "Step 4/6: Embedding entities..."
        t0 = time.time()

        entity_rows: list[tuple[str, str]] = []
        with self._driver.session(database=self.config.neo4j.database) as session:
            result = session.run(
                "MATCH (e:Entity) WHERE e.text_embedding IS NULL "
                "RETURN e.name AS name, e.description AS description"
            )
            entity_rows = [(r["name"], r["description"] or "") for r in result if r["name"]]

        if not entity_rows:
            self.step_times["4. Embedding"] = time.time() - t0
            yield "Step 4/6: Embedding — skipped (all entities already embedded)."
            return

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

        self.step_times["4. Embedding"] = time.time() - t0
        yield f"Step 4/6: Embedding complete — {len(entity_rows)} entities embedded."

    async def _community_step(self, doc_id: str) -> AsyncGenerator[str, None]:
        yield "Step 5/6: Community detection..."
        t0 = time.time()
        detector = CommunityDetector(
            driver=self._driver,
            database=self.config.neo4j.database,
            algorithm=self.config.community_detection.algorithm,
            resolution=self.config.community_detection.resolution,
            level=self.config.community_detection.level,
        )
        communities = detector.detect()
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

        summaries = await summarizer.summarize(communities_to_summarize)
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
        lines.append(
            f"  LLM: {total_requests} requests | "
            f"Input: {prompt_tokens} tokens | Output: {completion_tokens} tokens | "
            f"Total: {prompt_tokens + completion_tokens} tokens"
        )

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
