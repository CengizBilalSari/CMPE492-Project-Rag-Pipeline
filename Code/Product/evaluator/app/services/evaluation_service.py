from __future__ import annotations

import csv
import io
import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from supabase import create_client, Client

from app.evaluation.evaluator import RAGEvaluator, EvalRow
from app.evaluation.pipeline import QuestionGenerator
from app.evaluation.rag import GraphRAGSearchClient
from app.models.evaluation import EvaluationResultMetrics

load_dotenv()
logger = logging.getLogger(__name__)


def _get_supabase() -> Client:
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_KEY", "")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set.")
    return create_client(url, key)


# ── Job lifecycle ────────────────────────────────────────


def create_job(user_id: str, search_types: List[str], question_source: str, document_id: Optional[str] = None) -> str:
    """Insert a new evaluation_jobs row and return its id."""
    job_id = str(uuid.uuid4())
    db = _get_supabase()
    payload = {
        "id": job_id,
        "user_id": user_id,
        "search_types": search_types,
        "question_source": question_source,
        "status": "pending",
    }
    if document_id:
        payload["document_id"] = document_id

    db.table("evaluation_jobs").insert(payload).execute()
    return job_id


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Read job status from Supabase."""
    db = _get_supabase()
    resp = db.table("evaluation_jobs").select("*").eq("id", job_id).execute()
    rows = resp.data or []
    return rows[0] if rows else None


def get_results(job_id: str) -> List[Dict[str, Any]]:
    """Read aggregated evaluation_results for a job."""
    db = _get_supabase()
    resp = db.table("evaluation_results").select("*").eq("job_id", job_id).execute()
    return resp.data or []


def _update_job(job_id: str, **fields) -> None:
    db = _get_supabase()
    db.table("evaluation_jobs").update(fields).eq("id", job_id).execute()


# ── Background task ──────────────────────────────────────


def start_evaluation_job(
    job_id: str,
    search_types: List[str],
    question_source: str,
    uploaded_csv_bytes: Optional[bytes] = None,
    document_texts: Optional[List[str]] = None,
    llm_provider: str = "openai",
    llm_model: str = "gpt-4o",
) -> None:
    """Run the full evaluation lifecycle. Called from a BackgroundTask."""
    db = _get_supabase()

    try:
        # ── Step 1: Prepare QA pairs ──────────────────────
        qa_rows: List[Dict[str, str]] = []

        if question_source == "custom" and uploaded_csv_bytes:
            _update_job(job_id, status="generating_questions", progress="Parsing uploaded CSV...")
            qa_rows = _parse_csv(uploaded_csv_bytes)
            if not qa_rows:
                raise ValueError("CSV has no valid rows with 'question' and 'ground_truth_answer'.")

        elif question_source == "auto":
            _update_job(job_id, status="generating_questions", progress="Auto-generating questions...")

            if not document_texts:
                raise ValueError("No document texts provided for auto generation.")

            generator = QuestionGenerator(provider=llm_provider, model=llm_model)
            for i, text in enumerate(document_texts):
                _update_job(job_id, progress=f"Generating questions from document {i+1}/{len(document_texts)}...")
                pairs = generator.generate_from_text(text, num_questions=10)
                qa_rows.extend(pairs)

            if not qa_rows:
                raise ValueError("Auto-generation produced no QA pairs.")
        else:
            raise ValueError(f"Invalid question_source: {question_source}")

        # Persist QA pairs to qa_pairs table
        qa_pair_ids: List[str] = []
        for row in qa_rows:
            pair_id = str(uuid.uuid4())
            qa_pair_ids.append(pair_id)
            db.table("qa_pairs").insert({
                "id": pair_id,
                "job_id": job_id,
                "question": row["question"],
                "ground_truth_answer": row["ground_truth_answer"],
                "source": question_source,
            }).execute()

        logger.info("Persisted %d QA pairs for job %s.", len(qa_rows), job_id)

        # ── Step 2: Evaluate each search type ─────────────
        _update_job(job_id, status="evaluating")

        search_client = GraphRAGSearchClient(
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
            llm_provider=llm_provider,
            llm_model=llm_model,
        )
        evaluator = RAGEvaluator(
            search_client=search_client, 
            judge_provider=llm_provider, 
            judge_model=llm_model
        )

        for st in search_types:
            _update_job(job_id, progress=f"Evaluating search type: {st}...")
            logger.info("Evaluating search_type=%s with %d questions", st, len(qa_rows))

            eval_rows = evaluator.run(qa_rows, search_type=st)
            metrics = RAGEvaluator.aggregate(eval_rows)

            # Persist aggregated result
            db.table("evaluation_results").insert({
                "job_id": job_id,
                "search_type": st,
                "token_cost": metrics["total_tokens"],
                "prompt_tokens": metrics["total_prompt_tokens"],
                "completion_tokens": metrics["total_completion_tokens"],
                "time_per_request": metrics["avg_latency_ms"],
                "answer_accuracy": metrics["avg_answer_correctness"],
                "context_relevance": metrics["avg_context_relevance"],
            }).execute()

            # Persist per-question evaluations
            for er, pair_id in zip(eval_rows, qa_pair_ids):
                db.table("qa_evaluations").insert({
                    "job_id": job_id,
                    "qa_pair_id": pair_id,
                    "search_type": st,
                    "rag_answer": er.rag_answer,
                    "retrieved_contexts": json.dumps(er.retrieved_contexts or []),
                    "answer_correctness_score": er.answer_correctness_score,
                    "answer_correctness_reason": er.answer_correctness_reason,
                    "context_relevance_score": er.context_relevance_score,
                    "context_relevance_reason": er.context_relevance_reason,
                    "latency_ms": er.latency_ms,
                    "prompt_tokens": er.prompt_tokens,
                    "completion_tokens": er.completion_tokens,
                    "total_tokens": er.total_tokens,
                }).execute()

        # ── Step 3: Mark completed ────────────────────────
        _update_job(job_id, status="completed", progress="Done.", completed_at="now()")
        logger.info("Evaluation job %s completed.", job_id)

    except Exception as e:
        logger.error("Evaluation job %s failed: %s", job_id, e, exc_info=True)
        _update_job(job_id, status="failed", error=str(e), completed_at="now()")


def _parse_csv(csv_bytes: bytes) -> List[Dict[str, str]]:
    text = csv_bytes.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    rows = []
    for row in reader:
        q = row.get("question", "").strip()
        a = row.get("ground_truth_answer", "").strip()
        if q and a:
            rows.append({"question": q, "ground_truth_answer": a})
    return rows
