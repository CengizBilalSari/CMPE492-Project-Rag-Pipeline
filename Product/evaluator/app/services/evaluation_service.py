from __future__ import annotations

import csv
import io
import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from app.evaluation.evaluator import RAGEvaluator, EvalRow
from app.evaluation.pipeline import QuestionGenerator
from app.evaluation.rag import GraphRAGSearchClient
from app.models.evaluation import EvaluationResultMetrics
from app.services.db import connection

load_dotenv()
logger = logging.getLogger(__name__)


# ── Job lifecycle ────────────────────────────────────────


def create_job(
    chat_id: str,
    search_types: List[str],
    question_source: str,
    document_id: Optional[str] = None,
) -> str:
    """Insert a new evaluation_jobs row and return its id."""
    job_id = str(uuid.uuid4())
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO evaluation_jobs (id, chat_id, search_types, question_source, document_id, status)
                VALUES (%s, %s, %s, %s, %s, 'pending')
                """,
                (job_id, chat_id, search_types, question_source, document_id),
            )
        conn.commit()
    return job_id


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM evaluation_jobs WHERE id = %s", (job_id,))
            row = cur.fetchone()
    return dict(row) if row else None


def get_results(job_id: str) -> List[Dict[str, Any]]:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM evaluation_results WHERE job_id = %s", (job_id,))
            return [dict(r) for r in cur.fetchall()]


def _update_job(job_id: str, **fields) -> None:
    if not fields:
        return
    sets = ", ".join(f"{k} = %s" for k in fields)
    values = list(fields.values()) + [job_id]
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"UPDATE evaluation_jobs SET {sets} WHERE id = %s", values)
        conn.commit()


def _mark_completed(job_id: str, progress: str = "Done.") -> None:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE evaluation_jobs
                SET status = 'completed', progress = %s, completed_at = now()
                WHERE id = %s
                """,
                (progress, job_id),
            )
        conn.commit()


def _mark_failed(job_id: str, error: str) -> None:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE evaluation_jobs
                SET status = 'failed', error = %s, completed_at = now()
                WHERE id = %s
                """,
                (error, job_id),
            )
        conn.commit()


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
    try:
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

        # Persist QA pairs
        qa_pair_ids: List[str] = []
        with connection() as conn:
            with conn.cursor() as cur:
                for row in qa_rows:
                    pair_id = str(uuid.uuid4())
                    qa_pair_ids.append(pair_id)
                    cur.execute(
                        """
                        INSERT INTO qa_pairs (id, job_id, question, ground_truth_answer, source)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (pair_id, job_id, row["question"], row["ground_truth_answer"], question_source),
                    )
            conn.commit()

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
            judge_model=llm_model,
        )

        for st in search_types:
            _update_job(job_id, progress=f"Evaluating search type: {st}...")
            logger.info("Evaluating search_type=%s with %d questions", st, len(qa_rows))

            eval_rows = evaluator.run(qa_rows, search_type=st)
            metrics = RAGEvaluator.aggregate(eval_rows)

            with connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO evaluation_results
                            (job_id, search_type, token_cost, time_per_request, answer_accuracy, context_relevance)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (
                            job_id,
                            st,
                            metrics["total_tokens"],
                            metrics["avg_latency_ms"],
                            metrics["avg_answer_correctness"],
                            metrics["avg_context_relevance"],
                        ),
                    )

                    for er, pair_id in zip(eval_rows, qa_pair_ids):
                        cur.execute(
                            """
                            INSERT INTO qa_evaluations (
                                job_id, qa_pair_id, search_type, rag_answer, retrieved_contexts,
                                answer_correctness_score, answer_correctness_reason,
                                context_relevance_score, context_relevance_reason,
                                latency_ms, prompt_tokens, completion_tokens, total_tokens
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                job_id,
                                pair_id,
                                st,
                                er.rag_answer,
                                json.dumps(er.retrieved_contexts or []),
                                er.answer_correctness_score,
                                er.answer_correctness_reason,
                                er.context_relevance_score,
                                er.context_relevance_reason,
                                er.latency_ms,
                                er.prompt_tokens,
                                er.completion_tokens,
                                er.total_tokens,
                            ),
                        )
                conn.commit()

        _mark_completed(job_id)
        logger.info("Evaluation job %s completed.", job_id)

    except Exception as e:
        logger.error("Evaluation job %s failed: %s", job_id, e, exc_info=True)
        _mark_failed(job_id, str(e))


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
