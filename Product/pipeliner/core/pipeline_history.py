from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from .db import connection

logger = logging.getLogger(__name__)


class PipelineHistory:
    """Persists pipeline_runs lifecycle events to Postgres."""

    def create_run(
        self,
        chat_id: str,
        document_id: str,
        config_snapshot: dict[str, Any],
    ) -> str:
        run_id = str(uuid.uuid4())
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO pipeline_runs (id, chat_id, document_id, status, config)
                    VALUES (%s, %s, %s, 'CREATED', %s)
                    """,
                    (run_id, chat_id, document_id, json.dumps(config_snapshot)),
                )
            conn.commit()
        logger.info("Created pipeline_run: %s", run_id)
        return run_id

    def update_status(self, run_id: str, status: str) -> None:
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE pipeline_runs SET status = %s WHERE id = %s",
                    (status, run_id),
                )
            conn.commit()

    def complete_run(
        self,
        run_id: str,
        step_times: dict[str, float],
        llm_usage: dict[str, int],
        neo4j_stats: dict[str, int],
    ) -> None:
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE pipeline_runs
                    SET status = 'COMPLETED',
                        step_times = %s,
                        llm_usage = %s,
                        neo4j_stats = %s,
                        completed_at = now()
                    WHERE id = %s
                    """,
                    (
                        json.dumps(step_times),
                        json.dumps(llm_usage),
                        json.dumps(neo4j_stats),
                        run_id,
                    ),
                )
            conn.commit()
        logger.info("Pipeline run %s marked COMPLETED.", run_id)

    def fail_run(self, run_id: str, error: str) -> None:
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE pipeline_runs
                    SET status = 'FAILED', error = %s, completed_at = now()
                    WHERE id = %s
                    """,
                    (error, run_id),
                )
            conn.commit()
        logger.info("Pipeline run %s marked FAILED.", run_id)

    def cancel_run(self, run_id: str, reason: str = "Cancelled by user") -> None:
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE pipeline_runs
                    SET status = 'CANCELLED', error = %s, completed_at = now()
                    WHERE id = %s
                    """,
                    (reason, run_id),
                )
            conn.commit()
        logger.info("Pipeline run %s marked CANCELLED.", run_id)
