from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Awaitable

from ..llm.base import LLMInterface

logger = logging.getLogger(__name__)

PPR_SEARCH_PROMPT = """--- ROLE ---
You are a knowledgeable expert answering a question based on a reranked set of knowledge graph text passages.

--- CONTEXT ---
Source Text Passages (ranked by graph importance relative to query):
{chunks}

--- QUERY ---
{query}

--- INSTRUCTIONS ---
Answer the query using ONLY the context above. Be specific and cite relevant facts.
If the context is insufficient or irrelevant to the query, state clearly that the provided context does not contain the answer.
"""

class PPRRerankerRetriever:
    def __init__(
        self,
        driver: Any,
        llm: LLMInterface,
        embedding_fn: Callable[[list[str]], Awaitable[list[list[float]]]],
        top_k_entities: int = 10,
        max_chunks: int = 15,
        database: str = "neo4j",
    ) -> None:
        self.driver = driver
        self.llm = llm
        self.embedding_fn = embedding_fn
        self.top_k_entities = top_k_entities
        self.max_chunks = max_chunks
        self.database = database
        
        try:
            from graphdatascience import GraphDataScience
        except ImportError:
            raise ImportError("Install the GDS Python client: pip install graphdatascience")
            
        self._gds = GraphDataScience.from_neo4j_driver(driver=driver, database=database)

    async def search(self, query: str) -> str:
        logger.info("[PPRRetriever] Starting PPR-based search for query: '%s'", query)
        
        # 1. Find seed entities based on semantic similarity
        logger.info("[PPRRetriever] Embedding query and finding seed entities...")
        qemb = (await self.embedding_fn([query]))[0]
        seed_entity_ids = self._find_seed_entities(qemb)
        
        if not seed_entity_ids:
            logger.warning("[PPRRetriever] No relevant seed entities found in the graph.")
            return "I couldn't find any relevant entities in the knowledge graph to answer your query."
        
        # 2. Run Personalized PageRank using seed entities as source nodes
        logger.info("[PPRRetriever] Running Personalized PageRank from %d seeds...", len(seed_entity_ids))
        entity_scores = self._run_ppr(seed_entity_ids)
        
        # 3. Rank chunks based on the PPR scores of entities they mention
        logger.info("[PPRRetriever] Ranking chunks based on graph scores...")
        chunks = self._fetch_reranked_chunks(entity_scores)
        
        if not chunks:
            logger.warning("[PPRRetriever] No source chunks found for the relevant entities.")
            return "The graph analysis found relevant entities, but no source text passages were associated with them."

        # 4. Generate final answer with LLM
        logger.info("[PPRRetriever] Generating final answer using %d ranked chunks...", len(chunks))
        prompt = PPR_SEARCH_PROMPT.format(
            chunks="\n\n".join(chunks),
            query=query
        )
        return await self.llm.ainvoke(prompt)

    def _find_seed_entities(self, query_embedding: list[float]) -> list[int]:
        """Find Neo4j node IDs of entities most similar to the query."""
        # Using internal IDs for GDS
        records, _, _ = self.driver.execute_query(
            """
            MATCH (e:Entity)
            WHERE e.text_embedding IS NOT NULL
            WITH e, gds.similarity.cosine(e.text_embedding, $qemb) AS score
            WHERE score > 0.3
            RETURN id(e) AS id
            ORDER BY score DESC
            LIMIT $top_k
            """,
            qemb=query_embedding,
            top_k=self.top_k_entities,
            database_=self.database
        )
        return [r["id"] for r in records]

    def _run_ppr(self, seed_entity_ids: list[int]) -> dict[int, float]:
        """Execute Personalized PageRank on a projected graph."""
        graph_name = "__ppr_search_projection__"
        
        # Ensure clean state
        if self._gds.graph.exists(graph_name).get("exists", False):
            self._gds.graph.drop(self._gds.graph.get(graph_name))
            
        # Project undirected graph for PageRank
        G, _ = self._gds.graph.project(
            graph_name,
            "Entity",
            {"RELATED_TO": {"orientation": "UNDIRECTED"}}
        )
        
        try:
            # PPR: sourceNodes tells PageRank to "start" or "jump back" to these nodes
            results = self._gds.pageRank.stream(
                G,
                sourceNodes=seed_entity_ids,
                maxIterations=20,
                dampingFactor=0.85
            )
            scores = {int(row["nodeId"]): float(row["score"]) for _, row in results.iterrows()}
        finally:
            self._gds.graph.drop(G)
            
        return scores

    def _fetch_reranked_chunks(self, entity_scores: dict[int, float]) -> list[str]:
        """Fetch chunks and rank them by the importance of mentioned entities."""
        entity_ids = list(entity_scores.keys())
        if not entity_ids:
            return []
            
        # Map chunks to entities they mention
        records, _, _ = self.driver.execute_query(
            """
            MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
            WHERE id(e) IN $eids
            RETURN c.text AS text, collect(id(e)) AS mentions
            """,
            eids=entity_ids,
            database_=self.database
        )
        
        chunk_ranked = []
        for r in records:
            # Score = sum of PPR scores of entities mentioned in this chunk
            score = sum(entity_scores.get(eid, 0) for eid in r["mentions"])
            chunk_ranked.append((r["text"], score))
            
        # Sort by total graph-importance score
        chunk_ranked.sort(key=lambda x: x[1], reverse=True)
        
        # Return top K unique chunk texts
        seen = set()
        top_chunks = []
        for text, score in chunk_ranked:
            if text not in seen:
                top_chunks.append(text)
                seen.add(text)
                if len(top_chunks) >= self.max_chunks:
                    break
                    
        return top_chunks
