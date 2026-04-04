from __future__ import annotations

import logging
import os
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, Header, HTTPException, UploadFile
from supabase import create_client

from app.models.evaluation import (
    EvaluationJobStatus,
    EvaluationResultMetrics,
    EvaluationResultsResponse,
)
from app.services import evaluation_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/evaluate", tags=["evaluation"])


@router.post("/start")
async def start_evaluation(
    background_tasks: BackgroundTasks,
    search_types: str = Form(...),
    question_source: str = Form(...),
    x_user_id: str = Header(..., alias="X-User-Id"),
    file: Optional[UploadFile] = File(None),
):
    """Start an evaluation job.

    - search_types: comma-separated list, e.g. "global,local,lazy,ppr,rag,no-retriever"
    - question_source: "custom" (upload CSV) or "auto" (generate from Supabase docs)
    - X-User-Id header: user id from localStorage
    - file: CSV with 'question' and 'ground_truth_answer' columns (required when question_source="custom")
    """
    types_list: List[str] = [s.strip() for s in search_types.split(",") if s.strip()]
    if not types_list:
        raise HTTPException(status_code=400, detail="search_types must not be empty.")

    valid = {"global", "local", "lazy", "ppr", "rag", "no-retriever"}
    invalid = set(types_list) - valid
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid search types: {invalid}. Choose from: {valid}",
        )

    if question_source not in ("custom", "auto"):
        raise HTTPException(status_code=400, detail="question_source must be 'custom' or 'auto'.")

    uploaded_csv_bytes: Optional[bytes] = None
    document_texts: Optional[List[str]] = None

    if question_source == "custom":
        if not file:
            raise HTTPException(status_code=400, detail="CSV file is required when question_source='custom'.")
        uploaded_csv_bytes = await file.read()

    elif question_source == "auto":
        supabase_url = os.getenv("SUPABASE_URL", "")
        supabase_key = os.getenv("SUPABASE_KEY", "")
        if not supabase_url or not supabase_key:
            raise HTTPException(status_code=500, detail="SUPABASE_URL and SUPABASE_KEY must be set for auto mode.")

        client = create_client(supabase_url, supabase_key)
        # Fetch only this user's documents
        docs_resp = client.table("documents").select("storage_path").eq("user_id", x_user_id).execute()
        docs = docs_resp.data or []
        if not docs:
            raise HTTPException(status_code=400, detail="No documents found for this user.")

        document_texts = []
        for doc in docs:
            path = doc["storage_path"]
            try:
                file_bytes = client.storage.from_("documents").download(path)
                text = file_bytes.decode("utf-8", errors="replace")
                if text.strip():
                    document_texts.append(text)
            except Exception as e:
                logger.warning("Failed to download %s: %s", path, e)

        if not document_texts:
            raise HTTPException(status_code=400, detail="Could not extract text from any documents.")

    job_id = evaluation_service.create_job(
        user_id=x_user_id,
        search_types=types_list,
        question_source=question_source,
    )

    background_tasks.add_task(
        evaluation_service.start_evaluation_job,
        job_id=job_id,
        search_types=types_list,
        question_source=question_source,
        uploaded_csv_bytes=uploaded_csv_bytes,
        document_texts=document_texts,
    )

    return {"job_id": job_id, "status": "pending"}


@router.get("/status/{job_id}", response_model=EvaluationJobStatus)
async def get_evaluation_status(job_id: str):
    job = evaluation_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return EvaluationJobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress"),
        error=job.get("error"),
    )


@router.get("/results/{job_id}", response_model=EvaluationResultsResponse)
async def get_evaluation_results(job_id: str):
    job = evaluation_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed yet. Status: {job['status']}")

    rows = evaluation_service.get_results(job_id)
    results = [
        EvaluationResultMetrics(
            search_type=r["search_type"],
            token_cost=r["token_cost"],
            time_per_request=r["time_per_request"],
            answer_accuracy=r["answer_accuracy"],
            context_relevance=r["context_relevance"],
        )
        for r in rows
    ]
    return EvaluationResultsResponse(job_id=job_id, results=results)
