from __future__ import annotations

import logging
from typing import Any

import networkx as nx
import networkx.algorithms.community as nx_comm

logger = logging.getLogger(__name__)


class CommunityDetector:

    def __init__(
        self,
        driver: Any,
        database: str = "neo4j",
        algorithm: str = "louvain",
        resolution: float = 1.0,
    ) -> None:
        self.driver = driver
        self.database = database
        self.algorithm = algorithm.lower()
        self.resolution = resolution

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
        logger.info("[%s] Hierarchy has %d level(s).", self.algorithm.upper(), len(all_levels))

        nx_graph = self._build_networkx_graph()

        best_level_idx = self._select_best_level(all_levels, nx_graph)
        selected = all_levels[best_level_idx]

        node_ids = list(selected.keys())
        id_to_name = self._batch_get_entity_names(node_ids)

        communities: dict[int, list[str]] = {}
        for node_id, comm_id in selected.items():
            name = id_to_name.get(node_id)
            if name:
                communities.setdefault(comm_id, []).append(name)

        logger.info(
            "[%s] Selected level %d → %d communities across %d entities.",
            self.algorithm.capitalize(),
            best_level_idx,
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

 
    def _select_best_level(
        self,
        all_levels: list[dict[int, int]],
        nx_graph: nx.Graph,
    ) -> int:
        best_idx = 0
        best_modularity = -1.0

        for lvl_idx, node_to_community in enumerate(all_levels):
            partition: dict[int, set[int]] = {}
            for node_id, comm_id in node_to_community.items():
                partition.setdefault(comm_id, set()).add(node_id)

            num_communities = len(partition)
            sizes = [len(s) for s in partition.values()]
            avg_size = sum(sizes) / len(sizes) if sizes else 0.0

            try:
                communities_list = [
                    frozenset(members & set(nx_graph.nodes))
                    for members in partition.values()
                ]
                communities_list = [c for c in communities_list if c]
                modularity = nx_comm.modularity(nx_graph, communities_list)
            except Exception as exc:
                modularity = 0.0
                logger.warning("Modularity computation failed for level %d: %s", lvl_idx, exc)

            logger.info(
                "  Level %d | communities=%d | avg_size=%.1f | min=%d | max=%d | modularity=%.4f",
                lvl_idx, num_communities, avg_size,
                min(sizes) if sizes else 0,
                max(sizes) if sizes else 0,
                modularity,
            )

            if modularity > best_modularity:
                best_modularity = modularity
                best_idx = lvl_idx

        logger.info(
            "Best level: %d  (modularity=%.4f)", best_idx, best_modularity,
        )
        return best_idx

    def _build_networkx_graph(self) -> nx.Graph:
        records, _, _ = self.driver.execute_query(
            "MATCH (s:Entity)-[:RELATED_TO]->(o:Entity) RETURN id(s) AS src, id(o) AS dst",
            database_=self.database,
        )
        G = nx.Graph()
        for r in records:
            G.add_edge(r["src"], r["dst"])
        logger.info(
            "NetworkX graph built: %d nodes, %d edges.", G.number_of_nodes(), G.number_of_edges(),
        )
        return G

    def _batch_get_entity_names(self, node_ids: list[int]) -> dict[int, str]:
        if not node_ids:
            return {}
        records, _, _ = self.driver.execute_query(
            "MATCH (e:Entity) WHERE id(e) IN $nids RETURN id(e) AS nid, e.name AS name",
            nids=node_ids,
            database_=self.database,
        )
        return {r["nid"]: r["name"] for r in records if r["name"]}

    def _get_entity_name(self, node_id: int) -> str | None:
        records, _, _ = self.driver.execute_query(
            "MATCH (e:Entity) WHERE id(e) = $nid RETURN e.name AS name",
            nid=node_id,
            database_=self.database,
        )
        return records[0]["name"] if records else None
