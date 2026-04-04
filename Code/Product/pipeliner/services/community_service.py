from __future__ import annotations

import asyncio
import logging
from typing import Any

from .llm_service import LLMInterface

logger = logging.getLogger(__name__)


COMMUNITY_SUMMARY_PROMPT = """You are a knowledge-graph analyst.

Below is a list of entities that belong to the same community in a knowledge graph, along with their relationships.

Entities: {entities}
Relationships: {relationships}

Write a concise summary (2-4 sentences) describing what this community is about,
what the key entities are, and how they relate to each other.

Return ONLY the summary text, no JSON, no markdown."""


class CommunityDetector:
    def __init__(
        self,
        driver: Any,
        database: str = "neo4j",
        algorithm: str = "louvain",
        resolution: float = 1.0,
        level: int = -1,
    ) -> None:
        self.driver = driver
        self.database = database
        self.algorithm = algorithm.lower()
        self.resolution = resolution
        self.level = level

        from graphdatascience import GraphDataScience
        self._gds = GraphDataScience.from_neo4j_driver(driver=driver, database=database)
        logger.info("GDS client initialized (server version: %s)", self._gds.version())

    def detect(self) -> dict[int, list[str]]:
        graph_name = "__graphrag_community_graph__"

        if self._gds.graph.exists(graph_name).get("exists", False):
            self._gds.graph.drop(self._gds.graph.get(graph_name))

        G, project_result = self._gds.graph.project(
            graph_name,
            node_spec="Entity",
            relationship_spec="RELATED_TO",
        )
        node_count = project_result.get("nodeCount", 0)
        rel_count = project_result.get("relationshipCount", 0)
        logger.info("GDS projection '%s': %d nodes, %d relationships.", graph_name, node_count, rel_count)

        if node_count == 0:
            logger.warning("No Entity nodes in the graph — skipping community detection.")
            self._gds.graph.drop(G)
            return {}

        result_df = self._run_algorithm(G)
        all_levels = self._extract_all_levels(result_df)
        num_levels = len(all_levels)
        logger.info("[%s] Hierarchy has %d level(s).", self.algorithm.upper(), num_levels)

        for lvl_idx, node_to_community in enumerate(all_levels):
            comm_count = len(set(node_to_community.values()))
            sizes: dict[int, int] = {}
            for comm_id in node_to_community.values():
                sizes[comm_id] = sizes.get(comm_id, 0) + 1
            size_vals = list(sizes.values())
            avg = sum(size_vals) / len(size_vals) if size_vals else 0
            logger.info(
                "  Level %d | communities=%d | avg_size=%.1f | min=%d | max=%d",
                lvl_idx, comm_count, avg,
                min(size_vals) if size_vals else 0,
                max(size_vals) if size_vals else 0,
            )

        level_idx = self.level if self.level >= 0 else num_levels + self.level
        level_idx = max(0, min(level_idx, num_levels - 1))
        selected = all_levels[level_idx]

        node_ids = list(selected.keys())
        id_to_name = self._batch_get_entity_names(node_ids)

        communities: dict[int, list[str]] = {}
        for node_id, comm_id in selected.items():
            name = id_to_name.get(node_id)
            if name:
                communities.setdefault(comm_id, []).append(name)

        logger.info(
            "[%s] Level %d → %d communities across %d entities.",
            self.algorithm.capitalize(), level_idx, len(communities),
            sum(len(v) for v in communities.values()),
        )

        self._gds.graph.drop(G)
        return communities

    def _run_algorithm(self, G: Any):
        if self.algorithm == "leiden":
            logger.info("Running GDS Leiden (resolution=%.2f, all levels)...", self.resolution)
            try:
                return self._gds.leiden.stream(G, includeIntermediateCommunities=True)
            except Exception:
                logger.warning("Leiden does not support includeIntermediateCommunities; falling back.")
                return self._gds.leiden.stream(G)
        else:
            logger.info("Running GDS Louvain (resolution=%.2f, all levels)...", self.resolution)
            return self._gds.louvain.stream(G, includeIntermediateCommunities=True)

    def _extract_all_levels(self, result_df) -> list[dict[int, int]]:
        levels: list[dict[int, int]] = []
        has_intermediate = "intermediateCommunityIds" in result_df.columns

        if has_intermediate:
            sample = result_df["intermediateCommunityIds"].iloc[0]
            num_intermediate = len(sample) if sample is not None else 0
            for i in range(num_intermediate):
                level_map: dict[int, int] = {}
                for _, row in result_df.iterrows():
                    level_map[int(row["nodeId"])] = int(row["intermediateCommunityIds"][i])
                levels.append(level_map)

        final_map: dict[int, int] = {
            int(row["nodeId"]): int(row["communityId"])
            for _, row in result_df.iterrows()
        }
        levels.append(final_map)
        return levels

    def _batch_get_entity_names(self, node_ids: list[int]) -> dict[int, str]:
        if not node_ids:
            return {}
        records, _, _ = self.driver.execute_query(
            "MATCH (e:Entity) WHERE id(e) IN $nids RETURN id(e) AS nid, e.name AS name",
            nids=node_ids, database_=self.database,
        )
        return {r["nid"]: r["name"] for r in records if r["name"]}


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

    def get_unsummarized_communities(self) -> dict[int, list[str]]:
        records, _, _ = self.driver.execute_query(
            """
            MATCH (com:Community)
            WHERE NOT (com)-[:HAS_SUMMARY]->(:CommunitySummary)
            RETURN com.id AS id
            """,
            database_=self.database,
        )
        unsummarized_ids = [r["id"] for r in records]
        if not unsummarized_ids:
            return {}

        records, _, _ = self.driver.execute_query(
            """
            MATCH (e:Entity)-[:IN_COMMUNITY]->(com:Community)
            WHERE com.id IN $ids
            RETURN com.id AS id, collect(e.name) AS names
            """,
            ids=unsummarized_ids, database_=self.database,
        )
        return {int(r["id"]): r["names"] for r in records}

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
        self, sem: asyncio.Semaphore, comm_id: int, entity_names: list[str],
    ) -> str:
        async with sem:
            entity_details = await self._fetch_entity_details(entity_names)
            relationships = await self._fetch_community_relationships(entity_names)

            prompt = COMMUNITY_SUMMARY_PROMPT.format(
                entities=entity_details, relationships=relationships,
            )
            result = await self.llm.ainvoke(prompt)
            self.completed_summaries += 1
            logger.info(
                "Summarized community %d  [%d / %d completed]",
                comm_id, self.completed_summaries, self.total_communities,
            )
            return result

    async def _fetch_entity_details(self, entity_names: list[str]) -> str:
        def _run():
            records, _, _ = self.driver.execute_query(
                """
                MATCH (e:Entity)
                WHERE e.name IN $names
                RETURN e.name AS name, e.type AS type, e.description AS description
                """,
                names=entity_names, database_=self.database,
            )
            return records

        records = await asyncio.to_thread(_run)
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
        def _run():
            records, _, _ = self.driver.execute_query(
                """
                MATCH (s:Entity)-[r:RELATED_TO]->(o:Entity)
                WHERE s.name IN $names AND o.name IN $names
                RETURN s.name AS source, r.predicate AS predicate, o.name AS target
                """,
                names=entity_names, database_=self.database,
            )
            return records

        records = await asyncio.to_thread(_run)
        if not records:
            return "No direct relationships found."
        lines = [f"{r['source']} --[{r['predicate']}]--> {r['target']}" for r in records]
        return "; ".join(lines)
