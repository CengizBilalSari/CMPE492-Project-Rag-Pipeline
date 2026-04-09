from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class EvaluationStartRequest(BaseModel):
    search_types: List[str]
    question_source: str  # "auto" or "custom"


class EvaluationJobStatus(BaseModel):
    job_id: str
    status: str  # "pending" | "generating_questions" | "evaluating" | "completed" | "failed"
    progress: Optional[str] = None
    error: Optional[str] = None


class EvaluationResultMetrics(BaseModel):
    search_type: str
    token_cost: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    time_per_request: float
    answer_accuracy: float
    context_relevance: float


class EvaluationResultsResponse(BaseModel):
    job_id: str
    results: List[EvaluationResultMetrics]
