import csv
import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

try:
    from groq import Groq
except Exception:
    Groq = None

from .models import EvalConfig
from .rag import RAGClient

load_dotenv()
logger = logging.getLogger(__name__)



@dataclass
class EvalRow:
    question: str
    ground_truth_answer: str
    rag_answer: str = ""
    retrieved_contexts: Optional[List[str]] = None
    answer_correctness_score: Optional[float] = None
    answer_correctness_reason: Optional[str] = None
    context_relevance_score: Optional[float] = None
    context_relevance_reason: Optional[str] = None
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    question_type: str = ""
    source: str = ""
    persona: Optional[str] = None
    task: Optional[str] = None
    chunk_id: Optional[int] = None



JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluation judge for a Retrieval-Augmented Generation (RAG) system.
You will receive a question, the ground truth answer, the RAG system's answer, and the retrieved contexts.

Evaluate TWO dimensions and return a JSON object:

1. **answer_correctness** (0-10): How correct and complete is the RAG answer compared to the ground truth?
   - 0: Completely wrong or irrelevant
   - 5: Partially correct, missing key details
   - 10: Fully correct and complete

2. **context_relevance** (0-10): How helpful are the retrieved contexts for answering the question?
   - 0: Contexts are completely irrelevant to the question
   - 5: Some contexts are relevant but key information is missing
   - 10: Contexts contain all the information needed to answer correctly

Return ONLY a JSON object with this exact structure:
{
  "answer_correctness": {"score": <0-10>, "reason": "<brief explanation>"},
  "context_relevance": {"score": <0-10>, "reason": "<brief explanation>"}
}"""


class LLMJudge:
    def __init__(self, provider: str, model: str):
        provider = provider.lower()
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Missing OPENAI_API_KEY in .env file")
            self.client = OpenAI(api_key=api_key)
        elif provider == "groq":
            if Groq is None:
                raise ImportError("groq is not installed.")
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("Missing GROQ_API_KEY in .env file")
            self.client = Groq(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        self.model = model

    def evaluate(
        self,
        question: str,
        ground_truth: str,
        rag_answer: str,
        retrieved_contexts: List[str],
    ) -> Dict:
        contexts_str = "\n---\n".join(retrieved_contexts) if retrieved_contexts else "(no contexts retrieved)"
        user_msg = (
            f"**Question:** {question}\n\n"
            f"**Ground Truth Answer:** {ground_truth}\n\n"
            f"**RAG System Answer:** {rag_answer}\n\n"
            f"**Retrieved Contexts:**\n{contexts_str}"
        )

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {}

        ac = data.get("answer_correctness", {})
        cr = data.get("context_relevance", {})
        return {
            "answer_correctness_score": _safe_float(ac.get("score")),
            "answer_correctness_reason": ac.get("reason", ""),
            "context_relevance_score": _safe_float(cr.get("score")),
            "context_relevance_reason": cr.get("reason", ""),
        }


# Mock RAG: uses an LLM to fabricate an answer + contexts
class MockRAGClient:
    """Simulates a RAG endpoint by asking an LLM to answer the question."""

    def __init__(self, provider: str, model: str):
        provider = provider.lower()
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Missing OPENAI_API_KEY in .env file")
            self.client = OpenAI(api_key=api_key)
        elif provider == "groq":
            if Groq is None:
                raise ImportError("groq is not installed.")
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("Missing GROQ_API_KEY in .env file")
            self.client = Groq(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        self.model = model

    def query(self, question: str) -> Dict:
        """Return a mock RAG response with answer + retrieved_contexts."""
        system_prompt = (
            "You are a mock RAG system. Given the question and any available "
            "context, produce a JSON object with two keys:\n"
            '  "answer": your answer to the question,\n'
            '  "retrieved_contexts": a list of 2-3 short context passages '
            "that a retrieval system might have returned.\n"
            "Return ONLY the JSON object."
        )
        user_msg = f"Question: {question}"
      
        start = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
        )
        latency_ms = (time.perf_counter() - start) * 1000

        raw = resp.choices[0].message.content
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"answer": raw, "retrieved_contexts": []}

        raw_contexts = data.get("retrieved_contexts", [])
        contexts = [_normalize_context(c) for c in raw_contexts]

        return {
            "answer": data.get("answer", ""),
            "retrieved_contexts": contexts,
            "latency_ms": latency_ms,
            "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0),
            "completion_tokens": getattr(resp.usage, "completion_tokens", 0),
        }


class RAGEndpointClient:

    def __init__(self, endpoint_url: str, http_method: str, top_k: int):
        self.client = RAGClient(endpoint_url, http_method, top_k)

    def query(self, question: str, **_kwargs) -> Dict:
        start = time.perf_counter()
        result = self.client.retrieve(question)
        latency_ms = (time.perf_counter() - start) * 1000

        result["latency_ms"] = latency_ms
        return result



class RAGEvaluator:
    def __init__(self, config: EvalConfig):
        self.config = config

        if config.rag_endpoint_url:
            self._rag = RAGEndpointClient(
                config.rag_endpoint_url, config.rag_http_method, config.rag_top_k
            )
        else:
            logger.info("No rag_endpoint_url — running in MOCK mode (LLM answers)")
            self._rag = MockRAGClient(config.llm_provider, config.llm_model)

        self._judge = LLMJudge(config.llm_provider, config.llm_model)

    def run(self, qa_rows: List[Dict]) -> List[EvalRow]:
        eval_rows: List[EvalRow] = []

        for i, row in enumerate(qa_rows):
            question = row.get("question", "")
            ground_truth = row.get("answer", "")

            logger.info("Querying RAG for question %s/%s", i + 1, len(qa_rows))
            result = self._rag.query(question)

            rag_answer = result["answer"]
            contexts = result.get("retrieved_contexts", [])

            logger.info("Judging question %s/%s", i + 1, len(qa_rows))
            scores = self._judge.evaluate(
                question=question,
                ground_truth=ground_truth,
                rag_answer=rag_answer,
                retrieved_contexts=contexts,
            )

            er = EvalRow(
                question=question,
                ground_truth_answer=ground_truth,
                rag_answer=rag_answer,
                retrieved_contexts=contexts,
                answer_correctness_score=scores["answer_correctness_score"],
                answer_correctness_reason=scores["answer_correctness_reason"],
                context_relevance_score=scores["context_relevance_score"],
                context_relevance_reason=scores["context_relevance_reason"],
                latency_ms=result.get("latency_ms", 0),
                prompt_tokens=result.get("prompt_tokens", 0),
                completion_tokens=result.get("completion_tokens", 0),
                total_tokens=result.get("prompt_tokens", 0) + result.get("completion_tokens", 0),
                question_type=row.get("question_type", ""),
                source=row.get("source", ""),
                persona=row.get("persona"),
                task=row.get("task"),
                chunk_id=int(row["chunk_id"]) if row.get("chunk_id") else None,
            )
            eval_rows.append(er)

        return eval_rows

    def write_csv(self, rows: List[EvalRow], path: str) -> None:
        fieldnames = [
            "question", "ground_truth_answer", "rag_answer", "retrieved_contexts",
            "answer_correctness_score", "answer_correctness_reason",
            "context_relevance_score", "context_relevance_reason",
            "latency_ms", "prompt_tokens", "completion_tokens", "total_tokens",
            "question_type", "source", "persona", "task", "chunk_id",
        ]
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for er in rows:
                d = asdict(er)
                d["retrieved_contexts"] = json.dumps(d.get("retrieved_contexts") or [])
                writer.writerow(d)
        logger.info("Wrote %s evaluation rows to %s", len(rows), path)

    @staticmethod
    def aggregate(rows: List[EvalRow]) -> Dict[str, float]:
        n = len(rows) or 1
        return {
            "avg_answer_correctness": sum((r.answer_correctness_score or 0) for r in rows) / n,
            "avg_context_relevance": sum((r.context_relevance_score or 0) for r in rows) / n,
            "avg_latency_ms": sum(r.latency_ms for r in rows) / n,
            "total_prompt_tokens": sum(r.prompt_tokens for r in rows),
            "total_completion_tokens": sum(r.completion_tokens for r in rows),
            "total_tokens": sum(r.total_tokens for r in rows),
            "num_questions": len(rows),
        }



def _normalize_context(ctx) -> str:
    if isinstance(ctx, dict):
        return ctx.get("content") or ctx.get("text") or str(ctx)
    return str(ctx)


def _safe_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None
