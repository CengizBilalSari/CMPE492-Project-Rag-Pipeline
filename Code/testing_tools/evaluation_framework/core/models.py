from dataclasses import dataclass
from typing import List, Optional


@dataclass
class QAEntry:
    question_type: str
    source: str
    question: str
    answer: str
    persona: Optional[str] = None
    task: Optional[str] = None
    chunk_id: Optional[int] = None
    chunk: Optional[str] = None
    file_path: Optional[str] = None
    retrieved_contexts: Optional[List[str]] = None


@dataclass
class PipelineConfig:
    chunk_size: int = 300
    chunk_overlap: int = 50
    splitter_type: str = "token"  # token | char | sentence
    llm_provider: str = "groq"  # openai | groq
    llm_model: str = "llama-3.3-70b-versatile" # groq: llama-3.3-70b-versatile | openai: gpt-4o
    personas_k: int = 3
    tasks_n: int = 2
    qas_m: int = 2
    rag_endpoint_url: Optional[str] = None
    rag_http_method: str = "post"  # post | get
    rag_top_k: int = 5
    output_csv_path: str = "qa_library.csv"
