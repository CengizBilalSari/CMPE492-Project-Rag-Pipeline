from __future__ import annotations

import io
import logging
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, Header, HTTPException, UploadFile

from app.models.evaluation import (
    EvaluationJobStatus,
    EvaluationResultMetrics,
    EvaluationResultsResponse,
)
from app.services import evaluation_service
from app.services.db import connection, read_document_bytes

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/evaluate", tags=["evaluation"])


def _extract_text(file_bytes: bytes, content_type: str) -> str:
    if "pdf" in (content_type or ""):
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            raise HTTPException(status_code=500, detail="pypdf not installed.")
    return file_bytes.decode("utf-8", errors="replace")


@router.post("/start")
async def start_evaluation(
    background_tasks: BackgroundTasks,
    search_types: str = Form(...),
    question_source: str = Form(...),
    llm_provider: str = Form("lmstudio"),
    llm_model: str = Form("gpt-4o"),
    doc_id: Optional[str] = Form(None),
    x_chat_id: str = Header(..., alias="X-Chat-Id"),
    file: Optional[UploadFile] = File(None),
):
    """Start an evaluation job.

    - search_types: comma-separated list, e.g. "global,local,lazy,ppr,rag,no-retriever"
    - question_source: "custom" (upload CSV) or "auto" (generate from documents)
    - X-Chat-Id header: chat id from login
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
        with connection() as conn:
            with conn.cursor() as cur:
                if doc_id:
                    cur.execute(
                        "SELECT path, content_type FROM documents WHERE id = %s AND chat_id = %s",
                        (doc_id, x_chat_id),
                    )
                else:
                    cur.execute(
                        "SELECT path, content_type FROM documents WHERE chat_id = %s",
                        (x_chat_id,),
                    )
                docs = [dict(r) for r in cur.fetchall()]

        if not docs:
            raise HTTPException(status_code=400, detail="No documents found for this chat.")

        document_texts = []
        for doc in docs:
            try:
                file_bytes = read_document_bytes(doc["path"])
                text = _extract_text(file_bytes, doc.get("content_type", ""))
                if text.strip():
                    document_texts.append(text)
            except Exception as e:
                logger.warning("Failed to read %s: %s", doc["path"], e)

        if not document_texts:
            raise HTTPException(status_code=400, detail="Could not extract text from any documents.")

    job_id = evaluation_service.create_job(
        chat_id=x_chat_id,
        search_types=types_list,
        question_source=question_source,
        document_id=doc_id,
    )

    background_tasks.add_task(
        evaluation_service.start_evaluation_job,
        job_id=job_id,
        search_types=types_list,
        question_source=question_source,
        uploaded_csv_bytes=uploaded_csv_bytes,
        document_texts=document_texts,
        llm_provider=llm_provider,
        llm_model=llm_model,
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
