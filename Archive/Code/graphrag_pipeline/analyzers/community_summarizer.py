
from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..llm.base import LLMInterface

logger = logging.getLogger(__name__)

COMMUNITY_SUMMARY_PROMPT = """You are a knowledge-graph analyst.

Below is a list of entities that belong to the same community in a knowledge graph, along with their relationships.

Entities: {entities}
Relationships: {relationships}

Write a concise summary (2-4 sentences) describing what this community is about,
what the key entities are, and how they relate to each other.

Return ONLY the summary text, no JSON, no markdown."""


class CommunitySummarizer:


    def __init__(
        self,
        driver: Any,
        llm: LLMInterface,
        max_concurrency: int = 5,
        database: str = "neo4j",
    ) -> None:
        self.driver = driver
        self.llm = llm
        self.max_concurrency = max_concurrency
        self.database = database
        self.completed_summaries: int = 0
        self.total_communities = 0

    def _get_unsummarized_communities(self) -> dict[int, list[str]]:
        """Return a dict of {comm_id: [entity_names]} for communities that lack a summary."""
        def _run_query():
            records, _, _ = self.driver.execute_query(
                """
                MATCH (com:Community)
                WHERE NOT (com)-[:HAS_SUMMARY]->(:CommunitySummary)
                RETURN com.id AS id
                """,
                database_=self.database,
            )
            return [r["id"] for r in records]

        unsummarized_ids = _run_query()
        if not unsummarized_ids:
            return {}

        def _fetch_entities(comm_ids):
            records, _, _ = self.driver.execute_query(
                """
                MATCH (e:Entity)-[:IN_COMMUNITY]->(com:Community)
                WHERE com.id IN $ids
                RETURN com.id AS id, collect(e.name) AS names
                """,
                ids=comm_ids,
                database_=self.database,
            )
            return {int(r["id"]): r["names"] for r in records}

        return _fetch_entities(unsummarized_ids)

    async def summarize(self, communities: dict[int, list[str]]) -> dict[int, str]:
        self.total_communities = len(communities)
        self.completed_summaries = 0
        sem = asyncio.Semaphore(self.max_concurrency)
        tasks = {
            comm_id: self._summarize_community(sem, comm_id, entity_names)
            for comm_id, entity_names in communities.items()
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        summaries: dict[int, str] = {}
        for (comm_id, _), result in zip(tasks.items(), results):
            if isinstance(result, Exception):
                logger.error("Summarization failed for community %d: %s", comm_id, result)
                summaries[comm_id] = ""
            else:
                summaries[comm_id] = result

        logger.info("Generated summaries for %d communities.", len(summaries))
        return summaries

    async def _summarize_community(
        self,
        sem: asyncio.Semaphore,
        comm_id: int,
        entity_names: list[str],
    ) -> str:
        async with sem:
            entity_details = await self._fetch_entity_details(entity_names)
            relationships = await self._fetch_community_relationships(entity_names)

            prompt = COMMUNITY_SUMMARY_PROMPT.format(
                entities=entity_details,
                relationships=relationships,
            )
            
            result = await self.llm.ainvoke(prompt)
            self.completed_summaries += 1
            logger.info("Summarized community %d  [%d / %d completed]", 
                        comm_id, self.completed_summaries, self.total_communities)
            return result

    async def _fetch_entity_details(self, entity_names: list[str]) -> str:
        def _run_query():
            records, _, _ = self.driver.execute_query(
                """
                MATCH (e:Entity)
                WHERE e.name IN $names
                RETURN e.name AS name, e.type AS type, e.description AS description
                """,
                names=entity_names,
                database_=self.database,
            )
            return records
            
        records = await asyncio.to_thread(_run_query)
        if not records:
            return ", ".join(entity_names)
        lines = []
        for r in records:
            desc = r["description"] or ""
            entry = f"- {r['name']} (type: {r['type']})"
            if desc:
                entry += f": {desc}"
            lines.append(entry)
        return "\n".join(lines)

    async def _fetch_community_relationships(self, entity_names: list[str]) -> str:
        def _run_query():
            records, _, _ = self.driver.execute_query(
                """
                MATCH (s:Entity)-[r:RELATED_TO]->(o:Entity)
                WHERE s.name IN $names AND o.name IN $names
                RETURN s.name AS source, r.predicate AS predicate, o.name AS target
                """,
                names=entity_names,
                database_=self.database,
            )
            return records
            
        records = await asyncio.to_thread(_run_query)
        if not records:
            return "No direct relationships found."
        lines = [f"{r['source']} --[{r['predicate']}]--> {r['target']}" for r in records]
        return "; ".join(lines)
