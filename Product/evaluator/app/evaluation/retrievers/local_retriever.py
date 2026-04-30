from __future__ import annotations

import logging
from typing import Any, Callable, Awaitable

from ..llm_base import LLMInterface

logger = logging.getLogger(__name__)

LOCAL_SEARCH_PROMPT = """--- ROLE ---
You are a knowledgeable expert answering a focused question based on a provided knowledge graph context.

--- CONTEXT ---
Seed Entities:
{entities}

Relationships:
{relationships}

Source Text Passages:
{chunks}

--- QUERY ---
{query}

--- INSTRUCTIONS ---
Answer the query using ONLY the context above. Be specific and cite entity names where appropriate.
If the context is insufficient or irrelevant to the query, state clearly that the provided knowledge graph context does not contain the answer.
"""


class LocalRetriever:
    def __init__(
        self,
        driver: Any,
        llm: LLMInterface,
        embedding_fn: Callable[[list[str]], Awaitable[list[list[float]]]],
        top_k_entities: int = 5,
        hop_depth: int = 1,
        max_chunks: int = 10,
        database: str = "neo4j",
    ) -> None:
        self.driver = driver
        self.llm = llm
        self.embedding_fn = embedding_fn
        self.top_k_entities = top_k_entities
        self.hop_depth = hop_depth
        self.max_chunks = max_chunks
        self.database = database

    async def search(self, query: str) -> tuple[str, list[str], bool]:
        qemb = (await self.embedding_fn([query]))[0]
        seed_entity_scores = self._find_similar_entities(qemb)
        if not seed_entity_scores:
            return "I couldn't find any relevant entities in the knowledge graph to answer your query.", [], True

        seed_entities = [name for name, _ in seed_entity_scores]
        entity_details = self._fetch_entity_details(seed_entities)
        relationships = self._fetch_subgraph(seed_entities, self.hop_depth)
        chunks = self._fetch_source_chunks(seed_entities)

        contexts: list[str] = []
        if entity_details:
            contexts.append(f"Seed Entities:\n{entity_details}")
        if relationships:
            contexts.append(f"Relationships:\n{relationships}")
        if chunks:
            contexts.append(f"Source Text Passages:\n{chunks}")

        prompt = LOCAL_SEARCH_PROMPT.format(
            entities=entity_details or "None",
            relationships=relationships or "None",
            chunks=chunks or "None",
            query=query,
        )
        answer = await self.llm.ainvoke(prompt)
        return answer, contexts, not contexts

    def _find_similar_entities(self, query_embedding: list[float]) -> list[tuple[str, float]]:
        records, _, _ = self.driver.execute_query(
            """
            MATCH (e:Entity)
            WHERE e.text_embedding IS NOT NULL
            WITH e, gds.similarity.cosine(e.text_embedding, $qemb) AS score
            WHERE score > 0
            RETURN e.name AS name, score
            ORDER BY score DESC
            LIMIT $top_k
            """,
            qemb=query_embedding, top_k=self.top_k_entities,
            database_=self.database,
        )
        return [(r["name"], r["score"]) for r in records]

    def _fetch_entity_details(self, entity_names: list[str]) -> str:
        records, _, _ = self.driver.execute_query(
            """
            MATCH (e:Entity)
            WHERE e.name IN $names
            RETURN e.name AS name, e.type AS type, e.description AS description
            """,
            names=entity_names, database_=self.database,
        )
        lines = []
        for r in records:
            desc = r["description"] or ""
            entry = f"- {r['name']} ({r['type']})"
            if desc:
                entry += f": {desc}"
            lines.append(entry)
        return "\n".join(lines)

    def _fetch_subgraph(self, seed_entities: list[str], depth: int) -> str:
        records, _, _ = self.driver.execute_query(
            """
            MATCH (seed:Entity)-[r:RELATED_TO*1..%d]-(neighbor:Entity)
            WHERE seed.name IN $seeds
            UNWIND r AS rel
            RETURN startNode(rel).name AS source, rel.predicate AS predicate, endNode(rel).name AS target
            """ % depth,
            seeds=seed_entities, database_=self.database,
        )
        edges = set()
        for r in records:
            edges.add(f"{r['source']} -[{r['predicate']}]-> {r['target']}")
        return "\n".join(sorted(edges))

    def _fetch_source_chunks(self, seed_entities: list[str]) -> str:
        records, _, _ = self.driver.execute_query(
            """
            MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
            WHERE e.name IN $seeds
            RETURN DISTINCT c.text AS text
            LIMIT $max_chunks
            """,
            seeds=seed_entities, max_chunks=self.max_chunks,
            database_=self.database,
        )
        chunks = []
        for i, r in enumerate(records, 1):
            text = (r["text"] or "").replace("\x00", "")
            chunks.append(f"--- Chunk {i} ---\n{text}")
        return "\n\n".join(chunks)
