from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import Levenshtein
import networkx as nx

from .llm_service import LLMInterface

logger = logging.getLogger(__name__)


@dataclass
class MergeDecision:
    group: list[str]
    should_merge: bool
    canonical_name: str
    reason: str = ""


MERGE_EVALUATION_PROMPT = """You are an entity resolution expert.

Given a group of entity names that were identified as potentially referring to the same real-world entity, decide whether they should be merged.

Entity group: {entities}

Rules:
- Merge entities that refer to the same real-world concept (e.g., "Silicon Valley Bank" and "Silicon_Valley_Bank").
- Do NOT merge entities that are clearly different (e.g., "September 16, 2023" and "September 2, 2023").
- If they should be merged, pick the most canonical/clean name.

Return ONLY valid JSON:
{{
  "should_merge": true/false,
  "canonical_name": "the best name to keep",
  "reason": "brief explanation"
}}"""


class EntityResolver:
    def __init__(
        self,
        driver: Any,
        llm: LLMInterface,
        embedding_fn: Any,
        k_neighbors: int = 10,
        similarity_threshold: float = 0.85,
        word_distance_threshold: int = 3,
        max_concurrency: int = 5,
        database: str = "neo4j",
        use_llm: bool = True,
    ) -> None:
        self.driver = driver
        self.llm = llm
        self.embedding_fn = embedding_fn
        self.k_neighbors = k_neighbors
        self.similarity_threshold = similarity_threshold
        self.word_distance_threshold = word_distance_threshold
        self.max_concurrency = max_concurrency
        self.database = database
        self.use_llm = use_llm

        from graphdatascience import GraphDataScience
        self._gds = GraphDataScience.from_neo4j_driver(driver=driver, database=database)

    async def resolve(self) -> list[MergeDecision]:
        entity_names = self._fetch_entity_names()
        if len(entity_names) < 2:
            logger.info("Fewer than 2 entities — skipping resolution.")
            return []

        logger.info("Entity resolution: %d entities found.", len(entity_names))

        await self._store_embeddings(entity_names)

        knn_graph_name = "__er_knn_graph__"
        self._project_knn_graph(knn_graph_name)

        components = self._run_wcc(knn_graph_name)
        candidate_groups = [group for group in components.values() if len(group) > 1]
        logger.info("Found %d candidate merge groups from WCC.", len(candidate_groups))

        self._drop_graph(knn_graph_name)

        filtered_groups = self._word_distance_filter(candidate_groups)
        logger.info("After word-distance filtering: %d groups remain.", len(filtered_groups))

        if not filtered_groups:
            self._cleanup_embeddings()
            return []

        if self.use_llm:
            decisions = await self._llm_evaluate(filtered_groups)
            approved = [d for d in decisions if d.should_merge]
            logger.info("LLM approved %d / %d merge groups.", len(approved), len(decisions))
        else:
            logger.info("LLM evaluation DISABLED — auto-approving all %d word-distance groups.", len(filtered_groups))
            approved = [
                MergeDecision(group=g, should_merge=True, canonical_name=min(g, key=len))
                for g in filtered_groups
            ]

        for decision in approved:
            self._apply_merge(decision)

        self._cleanup_embeddings()
        return approved

    def _fetch_entity_names(self) -> list[str]:
        query = "MATCH (e:Entity) RETURN e.name AS name"
        records, _, _ = self.driver.execute_query(query, database_=self.database)
        return [r["name"] for r in records if r["name"]]

    async def _store_embeddings(self, names: list[str]) -> None:
        logger.info("Computing embeddings for %d entities...", len(names))
        embeddings = await self.embedding_fn(names)

        with self.driver.session(database=self.database) as session:
            for name, emb in zip(names, embeddings):
                session.run(
                    "MATCH (e:Entity {name: $name}) SET e.embedding = $emb",
                    name=name, emb=emb,
                )
        logger.info("Embeddings stored on Entity nodes.")

    def _cleanup_embeddings(self) -> None:
        self.driver.execute_query(
            "MATCH (e:Entity) REMOVE e.embedding",
            database_=self.database,
        )

    def _project_knn_graph(self, graph_name: str) -> None:
        if self._gds.graph.exists(graph_name).get("exists", False):
            self._gds.graph.drop(self._gds.graph.get(graph_name))

        source_graph_name = "__er_source_graph__"
        if self._gds.graph.exists(source_graph_name).get("exists", False):
            self._gds.graph.drop(self._gds.graph.get(source_graph_name))

        G_source, _ = self._gds.graph.project(
            source_graph_name,
            node_spec={"Entity": {"properties": ["embedding"]}},
            relationship_spec="*",
        )

        k = min(self.k_neighbors, max(1, G_source.node_count() - 1))
        logger.info("Running GDS KNN (k=%d, threshold=%.2f)...", k, self.similarity_threshold)

        self._gds.knn.mutate(
            G_source,
            nodeProperties=["embedding"],
            topK=k,
            similarityCutoff=self.similarity_threshold,
            mutateRelationshipType="SIMILAR",
            mutateProperty="score",
        )

        self._gds.graph.relationship.write(G_source, "SIMILAR", "score")
        self._gds.graph.drop(G_source)

        G_knn, project_result = self._gds.graph.project(
            graph_name,
            node_spec="Entity",
            relationship_spec={"SIMILAR": {"orientation": "UNDIRECTED"}},
        )
        logger.info(
            "KNN graph projected: %d nodes, %d relationships.",
            project_result.get("nodeCount", 0),
            project_result.get("relationshipCount", 0),
        )

    def _run_wcc(self, graph_name: str) -> dict[int, list[str]]:
        G = self._gds.graph.get(graph_name)
        result = self._gds.wcc.stream(G)

        components: dict[int, list[str]] = {}
        for _, row in result.iterrows():
            comp_id = int(row["componentId"])
            node_id = int(row["nodeId"])
            name = self._resolve_node_name(node_id)
            if name:
                components.setdefault(comp_id, []).append(name)

        logger.info("WCC found %d components.", len(components))
        return components

    def _resolve_node_name(self, node_id: int) -> str | None:
        records, _, _ = self.driver.execute_query(
            "MATCH (e:Entity) WHERE id(e) = $nid RETURN e.name AS name",
            nid=node_id, database_=self.database,
        )
        return records[0]["name"] if records else None

    def _drop_graph(self, graph_name: str) -> None:
        if self._gds.graph.exists(graph_name).get("exists", False):
            self._gds.graph.drop(self._gds.graph.get(graph_name))
        self.driver.execute_query(
            "MATCH ()-[r:SIMILAR]->() DELETE r",
            database_=self.database,
        )

    def _word_distance_filter(self, groups: list[list[str]]) -> list[list[str]]:
        def _numbers_in(text: str) -> set[str]:
            return set(re.findall(r'\d+(?:[.,]\d+)?', text))

        def _safe_to_merge(a: str, b: str) -> bool:
            nums_a, nums_b = _numbers_in(a), _numbers_in(b)
            if nums_a and nums_b and nums_a != nums_b:
                return False
            return True

        filtered: list[list[str]] = []
        for group in groups:
            sub = nx.Graph()
            sub.add_nodes_from(group)
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    if not _safe_to_merge(group[i], group[j]):
                        continue
                    dist = Levenshtein.distance(group[i].lower(), group[j].lower())
                    if dist <= self.word_distance_threshold:
                        sub.add_edge(group[i], group[j])
            for comp in nx.connected_components(sub):
                if len(comp) > 1:
                    filtered.append(list(comp))
        return filtered

    async def _llm_evaluate(self, groups: list[list[str]]) -> list[MergeDecision]:
        sem = asyncio.Semaphore(self.max_concurrency)
        tasks = [self._evaluate_group(sem, group) for group in groups]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        decisions: list[MergeDecision] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("LLM evaluation failed for group %d: %s", i, result)
                decisions.append(MergeDecision(group=groups[i], should_merge=False, canonical_name=""))
            else:
                decisions.append(result)
        return decisions

    async def _evaluate_group(self, sem: asyncio.Semaphore, group: list[str]) -> MergeDecision:
        async with sem:
            prompt = MERGE_EVALUATION_PROMPT.format(entities=json.dumps(group))
            response = await self.llm.ainvoke(prompt)
            return self._parse_merge_response(group, response)

    @staticmethod
    def _parse_merge_response(group: list[str], response: str) -> MergeDecision:
        try:
            clean = response.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(clean)
            return MergeDecision(
                group=group,
                should_merge=bool(data.get("should_merge", False)),
                canonical_name=str(data.get("canonical_name", group[0])),
                reason=str(data.get("reason", "")),
            )
        except (json.JSONDecodeError, KeyError):
            logger.warning("Failed to parse LLM merge response for group: %s", group)
            return MergeDecision(group=group, should_merge=False, canonical_name="")

    def _apply_merge(self, decision: MergeDecision) -> None:
        canonical = decision.canonical_name
        duplicates = [name for name in decision.group if name != canonical]
        if not duplicates:
            return

        logger.info("Merging entities %s → '%s'", duplicates, canonical)

        with self.driver.session(database=self.database) as session:
            for dup in duplicates:
                session.run(
                    """
                    MATCH (dup:Entity {name: $dup})<-[r:MENTIONS]-(c:Chunk)
                    MATCH (canon:Entity {name: $canonical})
                    MERGE (c)-[:MENTIONS]->(canon)
                    DELETE r
                    """,
                    dup=dup, canonical=canonical,
                )
                session.run(
                    """
                    MATCH (dup:Entity {name: $dup})-[r:RELATED_TO]->(target)
                    MATCH (canon:Entity {name: $canonical})
                    WHERE target <> canon
                    MERGE (canon)-[:RELATED_TO {predicate: r.predicate}]->(target)
                    DELETE r
                    """,
                    dup=dup, canonical=canonical,
                )
                session.run(
                    """
                    MATCH (source)-[r:RELATED_TO]->(dup:Entity {name: $dup})
                    MATCH (canon:Entity {name: $canonical})
                    WHERE source <> canon
                    MERGE (source)-[:RELATED_TO {predicate: r.predicate}]->(canon)
                    DELETE r
                    """,
                    dup=dup, canonical=canonical,
                )
                session.run(
                    "MATCH (dup:Entity {name: $dup}) DETACH DELETE dup",
                    dup=dup,
                )
