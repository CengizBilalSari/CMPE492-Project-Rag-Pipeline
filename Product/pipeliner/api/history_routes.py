from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException

from core.db import connection

router = APIRouter(prefix="/api/history", tags=["history"])


def _rows(cur) -> list[dict]:
    return [dict(r) for r in cur.fetchall()]


@router.get("/documents")
async def list_documents(x_chat_id: str = Header(..., alias="X-Chat-Id")):
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, chat_id, name, path, content_type, created_at
                FROM documents
                WHERE chat_id = %s
                ORDER BY created_at DESC
                """,
                (x_chat_id,),
            )
            return _rows(cur)


@router.get("/pipeline-runs")
async def list_pipeline_runs(x_chat_id: str = Header(..., alias="X-Chat-Id")):
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT pr.id, pr.chat_id, pr.document_id, pr.status, pr.config, pr.step_times, pr.llm_usage, pr.neo4j_stats, pr.error, pr.started_at, pr.completed_at, d.name AS document_name, 'custom' as pipeline_type
                FROM pipeline_runs pr
                LEFT JOIN documents d ON d.id = pr.document_id
                WHERE pr.chat_id = %s
                
                UNION ALL
                
                SELECT ms.id, ms.chat_id, ms.document_id, ms.status, NULL::jsonb as config, NULL::jsonb as step_times, NULL::jsonb as llm_usage, NULL::jsonb as neo4j_stats, ms.error, ms.created_at as started_at, ms.completed_at, d.name AS document_name, 'ms-graphrag' as pipeline_type
                FROM ms_graphrag_runs ms
                LEFT JOIN documents d ON d.id = ms.document_id
                WHERE ms.chat_id = %s
                
                ORDER BY started_at DESC
                """,
                (x_chat_id, x_chat_id),
            )
            return _rows(cur)


@router.get("/evaluation-jobs")
async def list_evaluation_jobs(x_chat_id: str = Header(..., alias="X-Chat-Id")):
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT *
                FROM evaluation_jobs
                WHERE chat_id = %s
                ORDER BY created_at DESC
                """,
                (x_chat_id,),
            )
            return _rows(cur)


@router.get("/evaluation-jobs/{job_id}/details")
async def get_evaluation_job_details(
    job_id: str,
    x_chat_id: str = Header(..., alias="X-Chat-Id"),
):
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM evaluation_jobs WHERE id = %s AND chat_id = %s",
                (job_id, x_chat_id),
            )
            job = cur.fetchone()
            if not job:
                raise HTTPException(status_code=404, detail="Job not found")

            cur.execute(
                "SELECT * FROM evaluation_results WHERE job_id = %s",
                (job_id,),
            )
            results = _rows(cur)

            cur.execute(
                "SELECT * FROM qa_pairs WHERE job_id = %s ORDER BY id",
                (job_id,),
            )
            qa_pairs = _rows(cur)

            if qa_pairs:
                pair_ids = [p["id"] for p in qa_pairs]
                cur.execute(
                    "SELECT * FROM qa_evaluations WHERE qa_pair_id = ANY(%s)",
                    (pair_ids,),
                )
                evaluations = _rows(cur)
                by_pair: dict = {}
                for e in evaluations:
                    by_pair.setdefault(e["qa_pair_id"], []).append(e)
                for p in qa_pairs:
                    p["qa_evaluations"] = by_pair.get(p["id"], [])

    return {
        "job": dict(job),
        "results": results,
        "qa_pairs": qa_pairs,
    }
