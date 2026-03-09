from __future__ import annotations
import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

import neo4j

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graphrag_pipeline.config import PipelineConfig, load_config
from graphrag_pipeline.llm import get_llm, LLMInterface
from graphrag_pipeline.chunkers import get_chunker
from graphrag_pipeline.extractors import EntityRelationshipExtractor
from graphrag_pipeline.resolvers import EntityResolver
from graphrag_pipeline.writers import GraphRAGNeo4jWriter
from graphrag_pipeline.analyzers import CommunityDetector, CommunitySummarizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

from graphrag_pipeline.utils import hf_embed
class GraphRAGPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.llm: LLMInterface = get_llm(
            provider=config.llm.provider,
            model=config.llm.model,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
        )
        logger.info("LLM: %s", self.llm)

        self.chunker = get_chunker(
            strategy=config.chunking.strategy,
            chunk_size=config.chunking.chunk_size,
            chunk_overlap=config.chunking.chunk_overlap,
            llm=self.llm
        )
        logger.info("Chunker: %s", self.chunker)

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

    async def run(self, document_text: str, doc_title: str = "", doc_source: str = "") -> None:
        import time
        logger.info("=" * 60)
        logger.info("Starting GraphRAG Pipeline")
        logger.info("=" * 60)
        
        total_start_time = time.time()

        self.writer.create_indexes()

        doc_id = self.writer.write_document(title=doc_title, source=doc_source)

        logger.info("Step 1/8: Chunking (%s strategy)...", self.config.chunking.strategy)
        t0 = time.time()
        chunks = self.chunker.chunk(document_text)
        self.step_times["1. Chunking"] = time.time() - t0
        logger.info("  → %d chunks produced.", len(chunks))

        logger.info("Step 2/8: Skipping proposition generation (strategy != propositional).")
        self.step_times["2. Propositions"] = 0.0

        t0 = time.time()
        chunk_ids = self.writer.write_chunks(doc_id, chunks)
        self.step_times["2.5 Write Chunks"] = time.time() - t0

        logger.info("Step 3/8: Extracting entities & relations (parallel, concurrency=%d)...",
                     self.config.extraction.max_concurrency)
        t0 = time.time()
        extraction_results = await self.er_extractor.extract_from_chunks(chunks)
        self.step_times["3. Extraction"] = time.time() - t0
        total_entities = sum(len(r.entities) for r in extraction_results)
        total_relations = sum(len(r.relations) for r in extraction_results)
        logger.info("  → %d entities, %d relations extracted.", total_entities, total_relations)

        logger.info("Step 4/8: Writing entities & relations to Neo4j...")
        t0 = time.time()
        self.writer.write_entities_and_relations(chunk_ids, extraction_results)
        self.step_times["4. Write Ents/Rels"] = time.time() - t0

        if self.config.entity_resolution.enabled:
            logger.info("Step 5/8: Entity Resolution (KNN → WCC → Word Distance → LLM)...")
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
            self.step_times["5. Entity Resolution"] = time.time() - t0
            logger.info("  → %d merge groups resolved.", len(decisions))
        else:
            logger.info("Step 5/9: Entity resolution DISABLED — skipping.")
            self.step_times["5. Entity Resolution"] = 0.0

        logger.info("Step 5b/9: Embedding entities for local search...")
        t0 = time.time()
        entity_rows: list[tuple[str, str]] = []
        with self._driver.session(database=self.config.neo4j.database) as session:
            result = session.run("MATCH (e:Entity) RETURN e.name AS name, e.description AS description")
            entity_rows = [(r["name"], r["description"] or "") for r in result if r["name"]]
        logger.info("  → Embedding %d entities...", len(entity_rows))
        if entity_rows:
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
        self.step_times["5b. Embedding"] = time.time() - t0

        algo = self.config.community_detection.algorithm
        logger.info("Step 6/9: Community detection (%s via Neo4j GDS)...", algo)
        t0 = time.time()
        detector = CommunityDetector(
            driver=self._driver,
            database=self.config.neo4j.database,
            algorithm=algo,
            resolution=self.config.community_detection.resolution,
            level=self.config.community_detection.level,
        )
        communities = detector.detect()
        self.step_times["6. Community Detection"] = time.time() - t0
        logger.info("  → %d communities detected.", len(communities))

        logger.info("Step 7/9: Writing communities to Neo4j...")
        t0 = time.time()
        self.writer.write_communities(communities)
        self.step_times["7. Write Communities"] = time.time() - t0

        logger.info("Step 8/9: Summarizing communities (parallel)...")
        t0 = time.time()
        summarizer = CommunitySummarizer(
            driver=self._driver,
            llm=self.llm,
            max_concurrency=self.config.extraction.max_concurrency,
            database=self.config.neo4j.database,
        )
        summaries = await summarizer.summarize(communities)
        self.writer.write_community_summaries(summaries)
        self.step_times["8. Summarize Communities"] = time.time() - t0
        
        total_time = time.time() - total_start_time

        logger.info("=" * 60)
        logger.info("GraphRAG Pipeline COMPLETE")
        logger.info("=" * 60)
        
        self._print_pipeline_summary(total_time)
        
    def _print_pipeline_summary(self, total_time: float) -> None:
        """Query Neo4j for final counts and print a summary table of time and graph data."""
        with self._driver.session(database=self.config.neo4j.database) as session:
            doc_count = session.run("MATCH (d:Document) RETURN count(d) as c").single()["c"]
            chunk_count = session.run("MATCH (c:Chunk) RETURN count(c) as c").single()["c"]
            entity_count = session.run("MATCH (e:Entity) RETURN count(e) as c").single()["c"]
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as c").single()["c"]
            community_count = session.run("MATCH (c:Community) RETURN count(c) as c").single()["c"]
            
        print("\n" + "=" * 60)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 60)
        
        print("\n--- Time Taken Per Step ---")
        for step, t in self.step_times.items():
            print(f"{step:.<30} {t:8.2f} seconds")
        print("-" * 45)
        print(f"{'TOTAL TIME':.<30} {total_time:8.2f} seconds")
        
        print("\n--- LLM Usage ---")
        prompt_tokens = getattr(self.llm, "total_prompt_tokens", 0)
        completion_tokens = getattr(self.llm, "total_completion_tokens", 0)
        total_tokens = prompt_tokens + completion_tokens
        total_requests = getattr(self.llm, "total_requests", 0)
        total_429_errors = getattr(self.llm, "total_429_errors", 0)
        
        print(f"Total Requests:    {total_requests:,}")
        print(f"429 Rate Limits:   {total_429_errors:,}")
        print(f"Prompt Tokens:     {prompt_tokens:,}")
        print(f"Completion Tokens: {completion_tokens:,}")
        print(f"Total Tokens:      {total_tokens:,}")
        
        print("\n--- Neo4j Database Contents ---")
        print(f"Documents:     {doc_count:,}")
        print(f"Chunks:        {chunk_count:,}")
        print(f"Entities:      {entity_count:,}")
        print(f"Relationships: {rel_count:,}")
        print(f"Communities:   {community_count:,}")
        print("=" * 60 + "\n")


    def close(self) -> None:
        """Clean up resources."""
        self._driver.close()



def _read_document(path: str) -> str:
    """Read a document file. Supports .txt and .pdf."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    if p.suffix.lower() == ".pdf":
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(str(p))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            raise ImportError("Install PyPDF2 to read PDF files: pip install PyPDF2")
    else:
        return p.read_text(encoding="utf-8")


async def async_main( document_paths: list[str]) -> None:
    
    config = load_config()
    pipeline = GraphRAGPipeline(config)

    try:
        for i, doc_path in enumerate(document_paths, 1):
            logger.info("=" * 60)
            logger.info("Processing document %d / %d: %s", i, len(document_paths), doc_path)
            logger.info("=" * 60)
            document_text = _read_document(doc_path)
            await pipeline.run(
                document_text=document_text,
                doc_title=Path(doc_path).stem,
                doc_source=doc_path,
            )
    finally:
        pipeline.close()


def _collect_documents(path: str) -> list[str]:
    """Collect document paths from a file or directory."""
    p = Path(path)
    if p.is_file():
        return [str(p)]
    elif p.is_dir():
        extensions = {".txt", ".pdf"}
        files = sorted(f for f in p.rglob("*") if f.suffix.lower() in extensions)
        if not files:
            raise FileNotFoundError(f"No .txt or .pdf files found in: {path}")
        logger.info("Found %d documents in %s", len(files), path)
        return [str(f) for f in files]
    else:
        raise FileNotFoundError(f"Path not found: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Microsoft GraphRAG Pipeline (Neo4j Edition)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--document", type=str, help="Path to a single document (.txt or .pdf)")
    group.add_argument("--directory", type=str, help="Path to a directory of documents (processes all .txt and .pdf files)")

    args = parser.parse_args()

    path = args.document or args.directory
    document_paths = _collect_documents(path)
    asyncio.run(async_main(document_paths))


if __name__ == "__main__":
    main()

