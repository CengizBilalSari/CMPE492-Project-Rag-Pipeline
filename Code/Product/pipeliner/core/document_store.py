from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

from .db import connection
from .storage import LocalDocumentStore
from .config import StorageConfig

logger = logging.getLogger(__name__)


class DocumentStore:
    """Persists documents to local storage and records metadata in Postgres."""

    def __init__(self, storage_config: StorageConfig | None = None) -> None:
        self._storage = LocalDocumentStore(storage_config or StorageConfig())

    def upload(
        self,
        file_bytes: bytes,
        filename: str,
        chat_id: str,
        content_type: str = "application/octet-stream",
    ) -> dict[str, Any]:
        doc_id = str(uuid.uuid4())
        stored_path = self._storage.save(chat_id, filename, file_bytes)

        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO documents (id, chat_id, name, path, content_type)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id, chat_id, name, path, content_type, created_at
                    """,
                    (doc_id, chat_id, filename, stored_path, content_type),
                )
                row = cur.fetchone()
            conn.commit()

        logger.info("Registered document %s for chat %s", doc_id, chat_id)
        return dict(row) if row else {}

    def get(self, document_id: str, chat_id: str) -> Optional[dict[str, Any]]:
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM documents WHERE id = %s AND chat_id = %s",
                    (document_id, chat_id),
                )
                row = cur.fetchone()
        return dict(row) if row else None

    def list_for_chat(self, chat_id: str) -> list[dict[str, Any]]:
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM documents WHERE chat_id = %s ORDER BY created_at DESC",
                    (chat_id,),
                )
                rows = cur.fetchall()
        return [dict(r) for r in rows]

    def read_bytes(self, stored_path: str) -> bytes:
        return self._storage.read(stored_path)
