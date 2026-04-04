from __future__ import annotations

import logging
import uuid
from typing import Any

from supabase import create_client, Client

from .config import SupabaseConfig

logger = logging.getLogger(__name__)


class SupabaseDocumentStore:
    def __init__(self, config: SupabaseConfig) -> None:
        if not config.url or not config.key:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_KEY environment variables must be set."
            )
        self._client: Client = create_client(config.url, config.key)
        self._bucket = config.bucket
        self._table = config.table

    def upload_document(
        self,
        file_bytes: bytes,
        filename: str,
        user_id: str,
        content_type: str = "application/octet-stream",
    ) -> dict[str, Any]:
        doc_id = str(uuid.uuid4())
        storage_path = f"{user_id}/{doc_id}/{filename}"

        self._client.storage.from_(self._bucket).upload(
            path=storage_path,
            file=file_bytes,
            file_options={"content-type": content_type},
        )
        logger.info("Uploaded file to storage: %s", storage_path)

        row = {
            "id": doc_id,
            "user_id": user_id,
            "filename": filename,
            "storage_path": storage_path,
            "content_type": content_type,
        }
        self._client.table(self._table).insert(row).execute()
        logger.info("Inserted document metadata: %s", doc_id)

        return row

    def get_document_url(self, storage_path: str) -> str:
        return self._client.storage.from_(self._bucket).get_public_url(storage_path)

    def download_document(self, storage_path: str) -> bytes:
        return self._client.storage.from_(self._bucket).download(storage_path)


class SupabasePipelineHistory:
    """Persists pipeline run history to the pipeline_runs table."""

    def __init__(self, config: SupabaseConfig) -> None:
        if not config.url or not config.key:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_KEY environment variables must be set."
            )
        self._client: Client = create_client(config.url, config.key)

    def create_run(
        self,
        user_id: str,
        document_id: str,
        config_snapshot: dict[str, Any],
    ) -> str:
        run_id = str(uuid.uuid4())
        self._client.table("pipeline_runs").insert({
            "id": run_id,
            "user_id": user_id,
            "document_id": document_id,
            "status": "CREATED",
            "config": config_snapshot,
        }).execute()
        logger.info("Created pipeline_run: %s", run_id)
        return run_id

    def update_status(self, run_id: str, status: str) -> None:
        self._client.table("pipeline_runs").update(
            {"status": status}
        ).eq("id", run_id).execute()

    def complete_run(
        self,
        run_id: str,
        step_times: dict[str, float],
        llm_usage: dict[str, int],
        neo4j_stats: dict[str, int],
    ) -> None:
        self._client.table("pipeline_runs").update({
            "status": "COMPLETED",
            "step_times": step_times,
            "llm_usage": llm_usage,
            "neo4j_stats": neo4j_stats,
            "completed_at": "now()",
        }).eq("id", run_id).execute()
        logger.info("Pipeline run %s marked COMPLETED.", run_id)

    def fail_run(self, run_id: str, error: str) -> None:
        self._client.table("pipeline_runs").update({
            "status": "FAILED",
            "error": error,
            "completed_at": "now()",
        }).eq("id", run_id).execute()
        logger.info("Pipeline run %s marked FAILED.", run_id)
