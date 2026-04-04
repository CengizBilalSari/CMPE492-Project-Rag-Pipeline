from __future__ import annotations

import logging
from typing import Any, Callable, Awaitable

from ..llm_base import LLMInterface

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

        from graphdatascience import GraphDataScience
        self._gds = GraphDataScience.from_neo4j_driver(driver=driver, database=database)

    async def search(self, query: str) -> str:
        qemb = (await self.embedding_fn([query]))[0]
        seed_entity_ids = self._find_seed_entities(qemb)

        if not seed_entity_ids:
            return "I couldn't find any relevant entities in the knowledge graph to answer your query."

        entity_scores = self._run_ppr(seed_entity_ids)
        chunks = self._fetch_reranked_chunks(entity_scores)

        if not chunks:
            return "The graph analysis found relevant entities, but no source text passages were associated with them."

        prompt = PPR_SEARCH_PROMPT.format(chunks="\n\n".join(chunks), query=query)
        return await self.llm.ainvoke(prompt)

    def _find_seed_entities(self, query_embedding: list[float]) -> list[int]:
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
            qemb=query_embedding, top_k=self.top_k_entities,
            database_=self.database,
        )
        return [r["id"] for r in records]

    def _run_ppr(self, seed_entity_ids: list[int]) -> dict[int, float]:
        graph_name = "__ppr_search_projection__"

        if self._gds.graph.exists(graph_name).get("exists", False):
            self._gds.graph.drop(self._gds.graph.get(graph_name))

        G, _ = self._gds.graph.project(
            graph_name, "Entity",
            {"RELATED_TO": {"orientation": "UNDIRECTED"}},
        )

        try:
            results = self._gds.pageRank.stream(
                G, sourceNodes=seed_entity_ids, maxIterations=20, dampingFactor=0.85,
            )
            scores = {int(row["nodeId"]): float(row["score"]) for _, row in results.iterrows()}
        finally:
            self._gds.graph.drop(G)

        return scores

    def _fetch_reranked_chunks(self, entity_scores: dict[int, float]) -> list[str]:
        entity_ids = list(entity_scores.keys())
        if not entity_ids:
            return []

        records, _, _ = self.driver.execute_query(
            """
            MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
            WHERE id(e) IN $eids
            RETURN c.text AS text, collect(id(e)) AS mentions
            """,
            eids=entity_ids, database_=self.database,
        )

        chunk_ranked = []
        for r in records:
            score = sum(entity_scores.get(eid, 0) for eid in r["mentions"])
            chunk_ranked.append((r["text"], score))

        chunk_ranked.sort(key=lambda x: x[1], reverse=True)

        seen, top_chunks = set(), []
        for text, _ in chunk_ranked:
            if text not in seen:
                top_chunks.append(text)
                seen.add(text)
                if len(top_chunks) >= self.max_chunks:
                    break
        return top_chunks
