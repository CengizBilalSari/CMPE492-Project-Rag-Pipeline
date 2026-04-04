from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

from .extraction_service import ExtractionResult

logger = logging.getLogger(__name__)


class GraphRAGNeo4jWriter:
    def __init__(self, driver: Any, database: str = "neo4j", batch_size: int = 500) -> None:
        self.driver = driver
        self.database = database
        self.batch_size = batch_size

    def create_indexes(self) -> None:
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

    def write_document(self, doc_id: Optional[str] = None, title: str = "", source: str = "") -> str:
        doc_id = doc_id or str(uuid.uuid4())
        with self.driver.session(database=self.database) as session:
            session.run(
                """
                MERGE (d:Document {id: $id})
                SET d.title = $title, d.source = $source
                SET d.status = CASE WHEN d.status IS NULL THEN 'CREATED' ELSE d.status END
                """,
                id=doc_id, title=title, source=source,
            )
        logger.info("Document written: %s", doc_id)
        return doc_id

    def write_chunks(self, doc_id: str, chunks: list[str]) -> list[str]:
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
                    doc_id=doc_id, chunk_id=chunk_id, text=text, index=idx,
                )
        logger.info("Wrote %d chunks for document %s.", len(chunks), doc_id)
        self.update_document_status(doc_id, "CHUNKED")
        return chunk_ids

    def get_document_status(self, doc_id: str) -> Optional[str]:
        with self.driver.session(database=self.database) as session:
            result = session.run(
                "MATCH (d:Document {id: $id}) RETURN d.status AS status", id=doc_id,
            ).single()
            return result["status"] if result else None

    def update_document_status(self, doc_id: str, status: str) -> None:
        with self.driver.session(database=self.database) as session:
            session.run(
                "MATCH (d:Document {id: $id}) SET d.status = $status",
                id=doc_id, status=status,
            )
        logger.info("Document %s status updated to: %s", doc_id, status)

    def get_unextracted_chunks(self, doc_id: str) -> list[tuple[str, str, int]]:
        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)
                WHERE NOT (c)-[:MENTIONS]->(:Entity)
                RETURN c.id AS id, c.text AS text, c.index AS index
                ORDER BY c.index
                """,
                doc_id=doc_id,
            )
            return [(r["id"], r["text"], r["index"]) for r in result]

    def get_chunks_count(self, doc_id: str) -> int:
        with self.driver.session(database=self.database) as session:
            result = session.run(
                "MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk) RETURN count(c) as count",
                doc_id=doc_id,
            ).single()
            return result["count"] if result else 0

    def write_entities_and_relations(self, chunk_ids: list[str], extraction_results: list[ExtractionResult]) -> None:
        with self.driver.session(database=self.database) as session:
            for chunk_id, result in zip(chunk_ids, extraction_results):
                for entity in result.entities:
                    session.run(
                        """
                        MERGE (e:Entity {name: $name})
                        SET e.type = $type, e.description = $description
                        WITH e
                        MATCH (c:Chunk {id: $chunk_id})
                        MERGE (c)-[:MENTIONS]->(e)
                        """,
                        name=entity.name, type=entity.type,
                        description=entity.description, chunk_id=chunk_id,
                    )
                for rel in result.relations:
                    session.run(
                        """
                        MERGE (s:Entity {name: $subject})
                        MERGE (o:Entity {name: $object})
                        MERGE (s)-[r:RELATED_TO {predicate: $predicate}]->(o)
                        """,
                        subject=rel.subject, predicate=rel.predicate, object=rel.object,
                    )

        total_entities = sum(len(r.entities) for r in extraction_results)
        total_relations = sum(len(r.relations) for r in extraction_results)
        logger.info("Wrote %d entities, %d relations.", total_entities, total_relations)

    def write_communities(self, communities: dict[int, list[str]]) -> None:
        with self.driver.session(database=self.database) as session:
            for comm_id, entity_names in communities.items():
                session.run(
                    "MERGE (com:Community {id: $id}) SET com.size = $size",
                    id=str(comm_id), size=len(entity_names),
                )
                for name in entity_names:
                    session.run(
                        """
                        MATCH (e:Entity {name: $name})
                        MATCH (com:Community {id: $comm_id})
                        MERGE (e)-[:IN_COMMUNITY]->(com)
                        """,
                        name=name, comm_id=str(comm_id),
                    )
        logger.info("Wrote %d communities.", len(communities))

    def write_community_summaries(self, summaries: dict[int, str]) -> None:
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
                    comm_id=str(comm_id), summary_id=summary_id, text=summary_text,
                )
        logger.info("Wrote %d community summaries.", len(summaries))

    def write_entity_embeddings(self, name_to_embedding: dict[str, list[float]]) -> None:
        with self.driver.session(database=self.database) as session:
            for name, emb in name_to_embedding.items():
                session.run(
                    "MATCH (e:Entity {name: $name}) SET e.text_embedding = $emb",
                    name=name, emb=emb,
                )
        logger.info("Wrote text_embedding for %d entities.", len(name_to_embedding))

    def clear_database(self) -> None:
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.warning("Database cleared.")
