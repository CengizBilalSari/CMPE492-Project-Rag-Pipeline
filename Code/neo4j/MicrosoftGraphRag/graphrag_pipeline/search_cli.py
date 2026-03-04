"""
CLI for running Global Search queries on a GraphRAG Neo4j database.
"""

import argparse
import asyncio
import logging
import sys
from typing import Optional

import neo4j

from graphrag_pipeline.config import load_config
from graphrag_pipeline.llm import get_llm
from graphrag_pipeline.retrievers import GlobalRetriever, LocalRetriever
from graphrag_pipeline.utils import hf_embed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("graphrag_search")

async def run_search(query: str, config_path: Optional[str] = None, mode: str = "global"):
    logger.info("[1/5] Loading config from: %s", config_path or "default location")
    config = load_config(config_path)
    logger.info("      Config loaded. LLM provider=%s  model=%s  temperature=%s  max_tokens=%s",
                config.llm.provider, config.llm.model,
                config.llm.temperature, config.llm.max_tokens)
    if mode == "global":
        logger.info("      Global search: top_k=%s  max_concurrency=%s  token_limit=%s",
                    config.global_search.top_k,
                    config.global_search.max_concurrency,
                    config.global_search.max_token_per_batch)
    else:
        logger.info("      Local search: top_k_entities=%s  hop_depth=%s  max_chunks=%s",
                    config.local_search.top_k_entities,
                    config.local_search.hop_depth,
                    config.local_search.max_chunks)

    logger.info("[2/5] Initialising LLM (%s / %s)...", config.llm.provider, config.llm.model)
    llm = get_llm(
        provider=config.llm.provider,
        model=config.llm.model,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
    )
    logger.info("      LLM ready: %s", type(llm).__name__)

    if not config.neo4j.uri or not config.neo4j.password:
        logger.error("Neo4j credentials missing. Ensure NEO4J_URI and NEO4J_PASSWORD are set in .env")
        return

    logger.info("[3/5] Connecting to Neo4j at %s  (database=%s)...",
                config.neo4j.uri, config.neo4j.database)
    driver = neo4j.GraphDatabase.driver(
        config.neo4j.uri,
        auth=(config.neo4j.user, config.neo4j.password),
    )
    logger.info("      Neo4j driver created successfully.")

    try:
        if mode == "global":
            retriever = GlobalRetriever(
                driver=driver,
                llm=llm,
                token_limit=config.global_search.max_token_per_batch,
                max_concurrency=config.global_search.max_concurrency,
                top_k=config.global_search.top_k,
                database=config.neo4j.database,
            )
        else:
            async def embed_fn(texts: list[str]) -> list[list[float]]:
                return await hf_embed(texts, model_name=config.embedding.model, show_progress=False)
                
            retriever = LocalRetriever(
                driver=driver,
                llm=llm,
                embedding_fn=embed_fn,
                top_k_entities=config.local_search.top_k_entities,
                hop_depth=config.local_search.hop_depth,
                max_chunks=config.local_search.max_chunks,
                database=config.neo4j.database,
            )

        print(f"\n{'='*60}")
        print(f"MODE: {mode.upper()}")
        print(f"QUERY: {query}")
        print(f"{'='*60}\n")

        logger.info("[4/5] Running %s.search() ...", type(retriever).__name__)
        result = await retriever.search(query)
        logger.info("[4/5] Retrieval complete. Answer length: %d chars", len(result))

        logger.info("[5/5] Printing final answer.")
        print(f"RESULT:\n\n{result}")
        print(f"\n{'='*60}")

    finally:
        logger.info("      Closing Neo4j driver.")
        driver.close()

def main():
    print("[0/5] Imports done, starting GraphRAG Search CLI...")

    parser = argparse.ArgumentParser(description="GraphRAG Search CLI")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--mode", type=str, choices=["global", "local"], default="global", 
                        help="Search mode: global (thematic) or local (entity-focused)")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")

    args = parser.parse_args()

    asyncio.run(run_search(args.query, args.config, args.mode))

if __name__ == "__main__":
    main()
