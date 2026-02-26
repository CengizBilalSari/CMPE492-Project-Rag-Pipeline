"""
Entity Resolution pipeline using Neo4j GDS:
  1. Embed entity names → store as node properties in Neo4j
  2. GDS KNN graph projection over embeddings
  3. GDS Weakly Connected Components → candidate merge groups
  4. Word-distance filtering within each WCC (Levenshtein)
  5. LLM evaluation for final merge / reject decision
  6. Apply merges in Neo4j
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

import Levenshtein

from ..llm.base import LLMInterface

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class MergeDecision:
    """Result of an LLM merge evaluation for a group of candidate entities."""
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


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------

class EntityResolver:
    """Performs entity resolution entirely within Neo4j using GDS.

    Pipeline:
      1. Compute embeddings for all entity names and store on nodes
      2. Build a KNN graph via GDS knn algorithm
      3. Find weakly connected components via GDS wcc
      4. Filter by Levenshtein word distance within each component
      5. Use LLM to evaluate each candidate group for merge/reject
      6. Apply merges in Neo4j

    Args:
        driver: Neo4j driver.
        llm: Language model for merge decisions.
        embedding_fn: Async callable: list[str] → list[list[float]].
        k_neighbors: K for the KNN algorithm.
        similarity_threshold: Minimum similarity for KNN edges.
        word_distance_threshold: Maximum Levenshtein distance within a WCC pair.
        max_concurrency: Parallel LLM calls for merge evaluation.
        database: Neo4j database name.
    """

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
    ) -> None:
        self.driver = driver
        self.llm = llm
        self.embedding_fn = embedding_fn
        self.k_neighbors = k_neighbors
        self.similarity_threshold = similarity_threshold
        self.word_distance_threshold = word_distance_threshold
        self.max_concurrency = max_concurrency
        self.database = database

        try:
            from graphdatascience import GraphDataScience
        except ImportError as exc:
            raise ImportError(
                "Install the GDS Python client: pip install graphdatascience"
            ) from exc

        self._gds = GraphDataScience.from_neo4j_driver(driver=driver, database=database)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def resolve(self) -> list[MergeDecision]:
        """Run the full entity resolution pipeline.

        Returns:
            A list of approved ``MergeDecision`` objects.
        """
        # Step 1: Gather entity names
        entity_names = self._fetch_entity_names()
        if len(entity_names) < 2:
            logger.info("Fewer than 2 entities — skipping resolution.")
            return []

        logger.info("Entity resolution: %d entities found.", len(entity_names))

        # Step 2: Compute and store embeddings on Entity nodes
        await self._store_embeddings(entity_names)

        # Step 3: GDS KNN graph
        knn_graph_name = "__er_knn_graph__"
        self._project_knn_graph(knn_graph_name)

        # Step 4: GDS WCC on the KNN graph
        components = self._run_wcc(knn_graph_name)
        candidate_groups = [group for group in components.values() if len(group) > 1]
        logger.info("Found %d candidate merge groups from WCC.", len(candidate_groups))

        # Clean up KNN projection
        self._drop_graph(knn_graph_name)

        # Step 5: Word distance filtering
        filtered_groups = self._word_distance_filter(candidate_groups)
        logger.info("After word-distance filtering: %d groups remain.", len(filtered_groups))

        if not filtered_groups:
            self._cleanup_embeddings()
            return []

        # Step 6: LLM evaluation
        decisions = await self._llm_evaluate(filtered_groups)
        approved = [d for d in decisions if d.should_merge]
        logger.info("LLM approved %d / %d merge groups.", len(approved), len(decisions))

        # Step 7: Apply merges in Neo4j
        for decision in approved:
            self._apply_merge(decision)

        # Cleanup: remove embedding property
        self._cleanup_embeddings()

        return approved

    # ------------------------------------------------------------------ #
    # Step 1-2: Embeddings
    # ------------------------------------------------------------------ #

    def _fetch_entity_names(self) -> list[str]:
        """Fetch all Entity node names from Neo4j."""
        query = "MATCH (e:Entity) RETURN e.name AS name"
        records, _, _ = self.driver.execute_query(query, database_=self.database)
        return [r["name"] for r in records if r["name"]]

    async def _store_embeddings(self, names: list[str]) -> None:
        """Compute embeddings and store them as an `embedding` property on Entity nodes."""
        logger.info("Computing embeddings for %d entities...", len(names))
        embeddings = await self.embedding_fn(names)

        with self.driver.session(database=self.database) as session:
            for name, emb in zip(names, embeddings):
                session.run(
                    "MATCH (e:Entity {name: $name}) SET e.embedding = $emb",
                    name=name,
                    emb=emb,
                )
        logger.info("Embeddings stored on Entity nodes.")

    def _cleanup_embeddings(self) -> None:
        """Remove temporary embedding property from Entity nodes."""
        self.driver.execute_query(
            "MATCH (e:Entity) REMOVE e.embedding",
            database_=self.database,
        )

    # ------------------------------------------------------------------ #
    # Step 3: GDS KNN
    # ------------------------------------------------------------------ #

    def _project_knn_graph(self, graph_name: str) -> None:
        """Use GDS KNN to build a similarity graph based on entity embeddings."""
        # Drop if exists
        if self._gds.graph.exists(graph_name).get("exists", False):
            self._gds.graph.drop(self._gds.graph.get(graph_name))

        # First, project Entity nodes with the embedding property
        source_graph_name = "__er_source_graph__"
        if self._gds.graph.exists(source_graph_name).get("exists", False):
            self._gds.graph.drop(self._gds.graph.get(source_graph_name))

        G_source, _ = self._gds.graph.project(
            source_graph_name,
            node_spec={"Entity": {"properties": ["embedding"]}},
            relationship_spec="*",
        )

        # Run KNN → mutate to create SIMILAR relationships in the projected graph
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

        # Write the KNN relationships back to Neo4j so we can project them for WCC
        self._gds.graph.relationship.write(G_source, "SIMILAR", "score")
        self._gds.graph.drop(G_source)

        # Now project using the new SIMILAR relationships for WCC
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

    # ------------------------------------------------------------------ #
    # Step 4: GDS WCC
    # ------------------------------------------------------------------ #

    def _run_wcc(self, graph_name: str) -> dict[int, list[str]]:
        """Run Weakly Connected Components and return component_id → entity names."""
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
        """Resolve a GDS internal node ID to an Entity name."""
        records, _, _ = self.driver.execute_query(
            "MATCH (e:Entity) WHERE id(e) = $nid RETURN e.name AS name",
            nid=node_id,
            database_=self.database,
        )
        return records[0]["name"] if records else None

    def _drop_graph(self, graph_name: str) -> None:
        """Drop a GDS graph projection and clean up SIMILAR relationships."""
        if self._gds.graph.exists(graph_name).get("exists", False):
            self._gds.graph.drop(self._gds.graph.get(graph_name))
        # Remove temporary SIMILAR relationships from the database
        self.driver.execute_query(
            "MATCH ()-[r:SIMILAR]->() DELETE r",
            database_=self.database,
        )

    # ------------------------------------------------------------------ #
    # Step 5: Word distance filtering
    # ------------------------------------------------------------------ #

    def _word_distance_filter(self, groups: list[list[str]]) -> list[list[str]]:
        """Filter each WCC group so only pairs within Levenshtein threshold remain.

        After filtering, re-derive connected sub-groups (in-memory, small scale).
        """
        import networkx as nx  # lightweight use: only for small sub-group filtering

        filtered: list[list[str]] = []
        for group in groups:
            sub = nx.Graph()
            sub.add_nodes_from(group)
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    dist = Levenshtein.distance(group[i].lower(), group[j].lower())
                    if dist <= self.word_distance_threshold:
                        sub.add_edge(group[i], group[j])
            for comp in nx.connected_components(sub):
                if len(comp) > 1:
                    filtered.append(list(comp))
        return filtered

    # ------------------------------------------------------------------ #
    # Step 6: LLM evaluation
    # ------------------------------------------------------------------ #

    async def _llm_evaluate(self, groups: list[list[str]]) -> list[MergeDecision]:
        """Use the LLM to decide whether each group should be merged."""
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

    # ------------------------------------------------------------------ #
    # Step 7: Apply merges
    # ------------------------------------------------------------------ #

    def _apply_merge(self, decision: MergeDecision) -> None:
        """Merge entities in Neo4j: keep canonical, redirect relationships, delete duplicates."""
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
