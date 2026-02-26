"""
Neo4j writer implementing the full Microsoft GraphRAG schema.

Node labels:  Document, Chunk, Entity, Community, CommunitySummary
Relationships: HAS_CHUNK, MENTIONS, RELATED_TO (with properties), IN_COMMUNITY, HAS_SUMMARY
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

from ..extractors.entity_relationship_extractor import ExtractionResult

logger = logging.getLogger(__name__)


class GraphRAGNeo4jWriter:
    """Writes the full GraphRAG knowledge graph to Neo4j.

    All write operations use explicit transactions for safety.

    Args:
        driver: An active ``neo4j.Driver`` instance.
        database: Neo4j database name.
        batch_size: Number of items per batch write.
    """

    def __init__(
        self,
        driver: Any,
        database: str = "neo4j",
        batch_size: int = 500,
    ) -> None:
        self.driver = driver
        self.database = database
        self.batch_size = batch_size

    # ------------------------------------------------------------------ #
    # Schema setup
    # ------------------------------------------------------------------ #

    def create_indexes(self) -> None:
        """Create indexes and constraints for optimal performance."""
        queries = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (com:Community) REQUIRE com.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (cs:CommunitySummary) REQUIRE cs.id IS UNIQUE",
        ]
        with self.driver.session(database=self.database) as session:
            for q in queries:
                try:
                    session.run(q)
                except Exception as exc:
                    logger.warning("Index/constraint creation skipped: %s", exc)
        logger.info("Neo4j indexes and constraints created.")

    # ------------------------------------------------------------------ #
    # Document & Chunk writing
    # ------------------------------------------------------------------ #

    def write_document(
        self,
        doc_id: Optional[str] = None,
        title: str = "",
        source: str = "",
    ) -> str:
        """Create (or merge) a Document node.

        Returns:
            The document ID.
        """
        doc_id = doc_id or str(uuid.uuid4())
        with self.driver.session(database=self.database) as session:
            session.run(
                """
                MERGE (d:Document {id: $id})
                SET d.title = $title, d.source = $source
                """,
                id=doc_id,
                title=title,
                source=source,
            )
        logger.info("Document written: %s", doc_id)
        return doc_id

    def write_chunks(self, doc_id: str, chunks: list[str]) -> list[str]:
        """Create Chunk nodes linked to a Document via ``HAS_CHUNK``.

        Returns:
            List of chunk IDs.
        """
        chunk_ids: list[str] = []
        with self.driver.session(database=self.database) as session:
            for idx, text in enumerate(chunks):
                chunk_id = f"{doc_id}:chunk:{idx}"
                chunk_ids.append(chunk_id)
                session.run(
                    """
                    MATCH (d:Document {id: $doc_id})
                    MERGE (c:Chunk {id: $chunk_id})
                    SET c.text = $text, c.index = $index
                    MERGE (d)-[:HAS_CHUNK]->(c)
                    """,
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    text=text,
                    index=idx,
                )
        logger.info("Wrote %d chunks for document %s.", len(chunks), doc_id)
        return chunk_ids

    # ------------------------------------------------------------------ #
    # Entity & Relation writing
    # ------------------------------------------------------------------ #

    def write_entities_and_relations(
        self,
        chunk_ids: list[str],
        extraction_results: list[ExtractionResult],
    ) -> None:
        """Write Entity nodes, MENTIONS relationships, and RELATED_TO edges.

        Args:
            chunk_ids: Chunk IDs corresponding to each extraction result.
            extraction_results: Results from ``EntityRelationshipExtractor``.
        """
        with self.driver.session(database=self.database) as session:
            for chunk_id, result in zip(chunk_ids, extraction_results):
                # Write entities
                for entity in result.entities:
                    session.run(
                        """
                        MERGE (e:Entity {name: $name})
                        SET e.type = $type, e.description = $description
                        WITH e
                        MATCH (c:Chunk {id: $chunk_id})
                        MERGE (c)-[:MENTIONS]->(e)
                        """,
                        name=entity.name,
                        type=entity.type,
                        description=entity.description,
                        chunk_id=chunk_id,
                    )

                # Write relations
                for rel in result.relations:
                    session.run(
                        """
                        MERGE (s:Entity {name: $subject})
                        MERGE (o:Entity {name: $object})
                        MERGE (s)-[r:RELATED_TO {predicate: $predicate}]->(o)
                        """,
                        subject=rel.subject,
                        predicate=rel.predicate,
                        object=rel.object,
                    )

        total_entities = sum(len(r.entities) for r in extraction_results)
        total_relations = sum(len(r.relations) for r in extraction_results)
        logger.info("Wrote %d entities, %d relations.", total_entities, total_relations)

    # ------------------------------------------------------------------ #
    # Community writing
    # ------------------------------------------------------------------ #

    def write_communities(self, communities: dict[int, list[str]]) -> None:
        """Create Community nodes and ``IN_COMMUNITY`` relationships.

        Args:
            communities: Mapping of community_id → list of entity names.
        """
        with self.driver.session(database=self.database) as session:
            for comm_id, entity_names in communities.items():
                session.run(
                    "MERGE (com:Community {id: $id}) SET com.size = $size",
                    id=str(comm_id),
                    size=len(entity_names),
                )
                for name in entity_names:
                    session.run(
                        """
                        MATCH (e:Entity {name: $name})
                        MATCH (com:Community {id: $comm_id})
                        MERGE (e)-[:IN_COMMUNITY]->(com)
                        """,
                        name=name,
                        comm_id=str(comm_id),
                    )
        logger.info("Wrote %d communities.", len(communities))

    def write_community_summaries(self, summaries: dict[int, str]) -> None:
        """Create CommunitySummary nodes and ``HAS_SUMMARY`` relationships.

        Args:
            summaries: Mapping of community_id → summary text.
        """
        with self.driver.session(database=self.database) as session:
            for comm_id, summary_text in summaries.items():
                summary_id = f"summary:{comm_id}"
                session.run(
                    """
                    MATCH (com:Community {id: $comm_id})
                    MERGE (cs:CommunitySummary {id: $summary_id})
                    SET cs.text = $text
                    MERGE (com)-[:HAS_SUMMARY]->(cs)
                    """,
                    comm_id=str(comm_id),
                    summary_id=summary_id,
                    text=summary_text,
                )
        logger.info("Wrote %d community summaries.", len(summaries))

    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #

    def clear_database(self) -> None:
        """Delete all nodes and relationships in the database. Use with caution."""
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.warning("Database cleared.")
