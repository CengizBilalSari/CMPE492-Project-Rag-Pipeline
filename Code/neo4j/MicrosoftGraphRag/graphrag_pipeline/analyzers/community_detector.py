"""
Community detection using Neo4j Graph Data Science (GDS).

Uses GDS graph projections and native Louvain/Leiden algorithms
instead of NetworkX, running everything server-side in Neo4j.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class CommunityDetector:
    """Detects communities in the entity graph using Neo4j GDS algorithms.

    Requires the Neo4j GDS plugin installed on the server and the
    ``graphdatascience`` Python client.

    Args:
        driver: Neo4j driver.
        database: Neo4j database name.
        algorithm: ``"louvain"`` or ``"leiden"``.
        resolution: Resolution parameter (higher → more communities).
    """

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

        # Initialize GDS client
        try:
            from graphdatascience import GraphDataScience
        except ImportError as exc:
            raise ImportError(
                "Install the GDS Python client: pip install graphdatascience"
            ) from exc

        self._gds = GraphDataScience.from_neo4j_driver(driver=driver, database=database)
        logger.info("GDS client initialized (server version: %s)", self._gds.version())

    def detect(self) -> dict[int, list[str]]:
        """Run community detection via GDS and return community_id → entity names.

        Returns:
            Dict mapping integer community IDs to lists of entity name strings.
        """
        graph_name = "__graphrag_community_graph__"

        # Drop existing projection if it exists
        if self._gds.graph.exists(graph_name).get("exists", False):
            self._gds.graph.drop(self._gds.graph.get(graph_name))
            logger.info("Dropped existing GDS projection '%s'.", graph_name)

        # Project the Entity→RELATED_TO→Entity subgraph into GDS
        G, project_result = self._gds.graph.project(
            graph_name,
            node_spec="Entity",
            relationship_spec="RELATED_TO",
        )
        logger.info(
            "GDS projection '%s': %d nodes, %d relationships.",
            graph_name,
            project_result.get("nodeCount", 0),
            project_result.get("relationshipCount", 0),
        )

        if project_result.get("nodeCount", 0) == 0:
            logger.warning("No Entity nodes in the graph — skipping community detection.")
            self._gds.graph.drop(G)
            return {}

        # Run the selected algorithm
        if self.algorithm == "leiden":
            result = self._run_leiden(G)
        else:
            result = self._run_louvain(G)

        # Parse results into community_id → [entity_names]
        communities: dict[int, list[str]] = {}
        for _, row in result.iterrows():
            comm_id = int(row["communityId"])
            node_id = int(row["nodeId"])
            # Map internal GDS nodeId back to entity name
            name = self._get_entity_name(node_id)
            if name:
                communities.setdefault(comm_id, []).append(name)

        logger.info(
            "%s detected %d communities across %d entities.",
            self.algorithm.capitalize(),
            len(communities),
            sum(len(v) for v in communities.values()),
        )

        # Clean up projection
        self._gds.graph.drop(G)
        return communities

    def _run_louvain(self, G: Any) -> Any:
        """Run GDS Louvain and return the result DataFrame."""
        logger.info("Running GDS Louvain (resolution=%.2f)...", self.resolution)
        return self._gds.louvain.stream(G)

    def _run_leiden(self, G: Any) -> Any:
        """Run GDS Leiden and return the result DataFrame."""
        logger.info("Running GDS Leiden (resolution=%.2f)...", self.resolution)
        return self._gds.leiden.stream(G)

    def _get_entity_name(self, node_id: int) -> str | None:
        """Resolve a GDS internal node ID to the Entity name."""
        records, _, _ = self.driver.execute_query(
            "MATCH (e:Entity) WHERE id(e) = $nid RETURN e.name AS name",
            nid=node_id,
            database_=self.database,
        )
        if records:
            return records[0]["name"]
        return None
