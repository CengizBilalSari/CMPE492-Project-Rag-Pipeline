from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class CommunityDetector:

    def __init__(
        self,
        driver: Any,
        database: str = "neo4j",
        algorithm: str = "louvain",
        resolution: float = 1.0,
        level: int = -1,
    ) -> None:
        """
        Args:
            level: Which hierarchy level to use.
                   0 = finest (many small communities),
                  -1 = coarsest (few large communities, default).
                  Any positive integer selects that level index.
                  If the requested level exceeds the available levels it is clamped to the last one.
        """
        self.driver = driver
        self.database = database
        self.algorithm = algorithm.lower()
        self.resolution = resolution
        self.level = level

        try:
            from graphdatascience import GraphDataScience
        except ImportError as exc:
            raise ImportError(
                "Install the GDS Python client: pip install graphdatascience"
            ) from exc

        self._gds = GraphDataScience.from_neo4j_driver(driver=driver, database=database)
        logger.info("GDS client initialized (server version: %s)", self._gds.version())

    def detect(self) -> dict[int, list[str]]:
        graph_name = "__graphrag_community_graph__"

        if self._gds.graph.exists(graph_name).get("exists", False):
            self._gds.graph.drop(self._gds.graph.get(graph_name))
            logger.info("Dropped existing GDS projection '%s'.", graph_name)

        G, project_result = self._gds.graph.project(
            graph_name,
            node_spec="Entity",
            relationship_spec="RELATED_TO",
        )
        node_count = project_result.get("nodeCount", 0)
        rel_count = project_result.get("relationshipCount", 0)
        logger.info(
            "GDS projection '%s': %d nodes, %d relationships.",
            graph_name, node_count, rel_count,
        )

        if node_count == 0:
            logger.warning("No Entity nodes in the graph — skipping community detection.")
            self._gds.graph.drop(G)
            return {}

        result_df = self._run_algorithm(G)

        logger.info(
            "[%s] Result columns: %s",
            self.algorithm.upper(), list(result_df.columns),
        )
        logger.info(
            "[%s] Result shape: %d rows  |  Sample (first 3 rows):\n%s",
            self.algorithm.upper(),
            len(result_df),
            result_df.head(3).to_string(index=False),
        )

        all_levels = self._extract_all_levels(result_df)
        num_levels = len(all_levels)
        logger.info("[%s] Hierarchy has %d level(s).", self.algorithm.upper(), num_levels)

        # Log every level's stats so the user can inspect and choose
        for lvl_idx, node_to_community in enumerate(all_levels):
            comm_count = len(set(node_to_community.values()))
            sizes = {}
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

        # Resolve negative index (Python-style: -1 = last)
        level_idx = self.level if self.level >= 0 else num_levels + self.level
        level_idx = max(0, min(level_idx, num_levels - 1))  # clamp
        logger.info(
            "Using level %d of %d (config level=%d).",
            level_idx, num_levels, self.level,
        )
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
            self.algorithm.capitalize(),
            level_idx,
            len(communities),
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
                logger.warning("Leiden does not support includeIntermediateCommunities; falling back to single level.")
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
            nids=node_ids,
            database_=self.database,
        )
        return {r["nid"]: r["name"] for r in records if r["name"]}
