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
from graphrag_pipeline.retrievers import GlobalRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("graphrag_search")

async def run_search(query: str, config_path: Optional[str] = None):
    config = load_config(config_path)

    llm = get_llm(
        provider=config.llm.provider,
        model=config.llm.model,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
    )

    if not config.neo4j.uri or not config.neo4j.password:
        logger.error("Neo4j credentials missing. Ensure NEO4J_URI and NEO4J_PASSWORD are set in .env")
        return

    driver = neo4j.GraphDatabase.driver(
        config.neo4j.uri,
        auth=(config.neo4j.user, config.neo4j.password),
    )

    try:
        retriever = GlobalRetriever(
            driver=driver,
            llm=llm,
            token_limit=config.global_search.max_token_per_batch,
            max_concurrency=config.global_search.max_concurrency,
            top_k=config.global_search.top_k,
            database=config.neo4j.database,
        )

        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print(f"{'='*60}\n")
        
        result = await retriever.search(query)
        
        print(f"RESULT:\n\n{result}")
        print(f"\n{'='*60}")

    finally:
        driver.close()

def main():
    parser = argparse.ArgumentParser(description="GraphRAG Global Search CLI")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    
    args = parser.parse_args()
    
    asyncio.run(run_search(args.query, args.config))

if __name__ == "__main__":
    main()
