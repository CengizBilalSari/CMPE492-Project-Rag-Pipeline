from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from .rag import GraphRAGSearchClient

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class EvalRow:
    question: str
    ground_truth_answer: str
    search_type: str = ""
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
    def __init__(self, provider: str = "openai", model: str = "gpt-4o"):
        self.provider = provider
        self.model = model
        
        if self.provider == "lmstudio":
            self.base_url = os.getenv("LMSTUDIO_BASE_URL")
            if not self.base_url:
                raise ValueError("Missing LMSTUDIO_BASE_URL in .env file for lmstudio provider")
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Missing OPENAI_API_KEY in .env file")
            self.client = OpenAI(api_key=api_key)

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

        judge_prompt_tokens = 0
        judge_completion_tokens = 0

        if self.provider == "lmstudio":
            import httpx
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                "temperature": 0.0,
                "max_tokens": 2048
            }
            res = httpx.post(self.base_url, json=payload, timeout=120.0)
            res.raise_for_status()
            res_json = res.json()
            raw = res_json["choices"][0]["message"]["content"] or ""
            usage = res_json.get("usage", {})
            judge_prompt_tokens = usage.get("prompt_tokens", 0)
            judge_completion_tokens = usage.get("completion_tokens", 0)
        else:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                response_format={"type": "json_object"}
            )
            raw = resp.choices[0].message.content or ""
            if resp.usage:
                judge_prompt_tokens = resp.usage.prompt_tokens or 0
                judge_completion_tokens = resp.usage.completion_tokens or 0
        import re
        clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
            
        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", clean, flags=re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse fallback JSON: {e}")
                    data = {}
            else:
                data = {}

        ac = data.get("answer_correctness", {})
        cr = data.get("context_relevance", {})
        return {
            "answer_correctness_score": _safe_float(ac.get("score")),
            "answer_correctness_reason": ac.get("reason", ""),
            "context_relevance_score": _safe_float(cr.get("score")),
            "context_relevance_reason": cr.get("reason", ""),
            "judge_prompt_tokens": judge_prompt_tokens,
            "judge_completion_tokens": judge_completion_tokens,
        }


class RAGEvaluator:
    """Evaluates a specific search_type against a list of QA pairs."""

    def __init__(
        self,
        search_client: GraphRAGSearchClient,
        judge_provider: str = "openai",
        judge_model: str = "gpt-4o",
    ):
        self._search_client = search_client
        self._judge = LLMJudge(provider=judge_provider, model=judge_model)

    def run(self, qa_rows: List[Dict], search_type: str) -> List[EvalRow]:
        eval_rows: List[EvalRow] = []

        for i, row in enumerate(qa_rows):
            question = row.get("question", "")
            ground_truth = row.get("ground_truth_answer", "")

            logger.info(
                "Querying [%s] for question %d/%d", search_type, i + 1, len(qa_rows),
            )
            result = self._search_client.query(question, search_type=search_type)

            rag_answer = result["answer"]
            contexts = result.get("retrieved_contexts", [])

            logger.info("Judging question %d/%d", i + 1, len(qa_rows))
            scores = self._judge.evaluate(
                question=question,
                ground_truth=ground_truth,
                rag_answer=rag_answer,
                retrieved_contexts=contexts,
            )

            search_pt = result.get("prompt_tokens", 0)
            search_ct = result.get("completion_tokens", 0)
            judge_pt = scores.get("judge_prompt_tokens", 0)
            judge_ct = scores.get("judge_completion_tokens", 0)
            total_pt = search_pt + judge_pt
            total_ct = search_ct + judge_ct

            er = EvalRow(
                question=question,
                ground_truth_answer=ground_truth,
                search_type=search_type,
                rag_answer=rag_answer,
                retrieved_contexts=contexts,
                answer_correctness_score=scores["answer_correctness_score"],
                answer_correctness_reason=scores["answer_correctness_reason"],
                context_relevance_score=scores["context_relevance_score"],
                context_relevance_reason=scores["context_relevance_reason"],
                latency_ms=result.get("latency_ms", 0),
                prompt_tokens=total_pt,
                completion_tokens=total_ct,
                total_tokens=total_pt + total_ct,
            )
            eval_rows.append(er)

        return eval_rows

    @staticmethod
    def aggregate(rows: List[EvalRow]) -> Dict[str, float]:
        n = len(rows) or 1
        return {
            "avg_answer_correctness": sum((r.answer_correctness_score or 0) for r in rows) / n,
            "avg_context_relevance": sum((r.context_relevance_score or 0) for r in rows) / n,
            "avg_latency_ms": sum(r.latency_ms for r in rows) / n,
            "total_tokens": sum(r.total_tokens for r in rows),
            "total_prompt_tokens": sum(r.prompt_tokens for r in rows),
            "total_completion_tokens": sum(r.completion_tokens for r in rows),
            "num_questions": len(rows),
        }


def _safe_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None
