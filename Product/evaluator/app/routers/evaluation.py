from __future__ import annotations

import io
import json
import logging
import os
from typing import List, Optional

import queue
import threading

from fastapi import APIRouter, BackgroundTasks, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.models.evaluation import (
    EvaluationJobStatus,
    EvaluationResultMetrics,
    EvaluationResultsResponse,
)
from app.services import evaluation_service
from app.services.db import connection, read_document_bytes

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/evaluate", tags=["evaluation"])

_LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434/v1")

_OPENAI_MODELS = ["gpt-4o-mini", "gpt-4o"]
_LMSTUDIO_MODELS = ["deepseek/deepseek-r1-0528-qwen3-8b", "llama-3-22b-instruct-v0.1", "google/gemma-4-31b"]
_OLLAMA_MODELS = ["llama3:latest", "mistral:latest", "phi3:latest", "gemma:latest", "qwen2:latest"]


@router.get("/config")
async def get_evaluate_config():
    """Return available LLM providers and their models, fetching LMStudio/Ollama dynamically."""
    import httpx

    lmstudio_models = list(_LMSTUDIO_MODELS)
    ollama_models = list(_OLLAMA_MODELS)

    async with httpx.AsyncClient(timeout=2.0) as client:
        try:
            resp = await client.get(f"{_LMSTUDIO_BASE_URL.rstrip('/')}/models")
            if resp.status_code == 200:
                data = resp.json()
                fetched = [m["id"] for m in data.get("data", [])]
                if fetched:
                    lmstudio_models = fetched
        except Exception as e:
            logger.warning("Could not fetch LMStudio models dynamically: %s", e)

        try:
            base_url = _OLLAMA_BASE_URL.split("/v1")[0].rstrip("/")
            resp = await client.get(f"{base_url}/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                fetched = [m.get("name") for m in (data.get("models") or []) if m.get("name")]
                if fetched:
                    ollama_models = fetched
        except Exception as e:
            logger.warning("Could not fetch Ollama models dynamically: %s", e)

    return {
        "llm_providers": {
            "openai": _OPENAI_MODELS,
            "lmstudio": lmstudio_models,
            "ollama": ollama_models,
        }
    }


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
    llm_provider: str = Form("openai"),
    llm_model: str = Form("gpt-4o-mini"),
    doc_id: Optional[str] = Form(None),
    qa_pairs: Optional[str] = Form(None),
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

    valid = {"global", "local", "lazy", "ppr", "rag", "no-retriever", "ms-graphrag-global", "ms-graphrag-local"}
    invalid = set(types_list) - valid
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid search types: {invalid}. Choose from: {valid}",
        )

    if question_source not in ("custom", "auto", "manual", "auto_modified"):
        raise HTTPException(status_code=400, detail="question_source must be 'custom', 'auto', 'manual', or 'auto_modified'.")

    uploaded_csv_bytes: Optional[bytes] = None
    document_texts: Optional[List[str]] = None
    parsed_qa_pairs: Optional[List[dict]] = None

    if qa_pairs:
        try:
            parsed_qa_pairs = json.loads(qa_pairs)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="qa_pairs must be valid JSON")

    if question_source == "custom":
        if not file:
            raise HTTPException(status_code=400, detail="CSV file is required when question_source='custom'.")
        uploaded_csv_bytes = await file.read()

    elif question_source == "auto" and not parsed_qa_pairs:
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
        provider=llm_provider,
        model=llm_model,
    )

    background_tasks.add_task(
        evaluation_service.start_evaluation_job,
        job_id=job_id,
        search_types=types_list,
        question_source=question_source,
        uploaded_csv_bytes=uploaded_csv_bytes,
        document_texts=document_texts,
        provided_qa_pairs=parsed_qa_pairs,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )

    return {"job_id": job_id, "status": "pending"}


@router.post("/generate-questions")
async def generate_questions(
    llm_provider: str = Form("openai"),
    llm_model: str = Form("gpt-4o-mini"),
    doc_id: str = Form(...),
    num_questions: int = Form(10),
    x_chat_id: str = Header(..., alias="X-Chat-Id"),
):
    """Generate questions from a document synchronously for review."""
    if num_questions < 1 or num_questions > 20:
        raise HTTPException(status_code=400, detail="num_questions must be between 1 and 20.")

    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT path, content_type FROM documents WHERE id = %s AND chat_id = %s",
                (doc_id, x_chat_id),
            )
            doc = cur.fetchone()

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    try:
        file_bytes = read_document_bytes(dict(doc)["path"])
        text = _extract_text(file_bytes, dict(doc).get("content_type", ""))
        if not text.strip():
            raise ValueError("Document text is empty.")
    except Exception as e:
        logger.warning("Failed to extract text: %s", e)
        raise HTTPException(status_code=500, detail="Could not extract text from document.")

    from app.evaluation.pipeline import QuestionGenerator
    progress_queue = queue.Queue()

    def run():
        try:
            generator = QuestionGenerator(provider=llm_provider, model=llm_model)
            qa_pairs = generator.generate_from_text(
                text,
                num_questions=num_questions,
                on_progress=lambda msg: progress_queue.put({"type": "progress", "message": msg}),
            )
            progress_queue.put({"type": "complete", "questions": qa_pairs})
        except Exception as e:
            logger.error("Failed to generate questions: %s", e)
            progress_queue.put({"type": "error", "message": str(e)})

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    def event_stream():
        while True:
            try:
                event = progress_queue.get(timeout=300)
                yield f"data: {json.dumps(event)}\n\n"
                if event["type"] in ("complete", "error"):
                    break
            except queue.Empty:
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")


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


@router.get("/details/{job_id}")
async def get_evaluation_details(job_id: str):
    """Return per-question evaluation results with reasoning for each search type."""
    job = evaluation_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed yet. Status: {job['status']}")

    details = evaluation_service.get_detailed_results(job_id)
    return {"job_id": job_id, "details": details}
