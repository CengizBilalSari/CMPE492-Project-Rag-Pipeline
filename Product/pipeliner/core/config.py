from __future__ import annotations

import os
from enum import Enum
from typing import Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

load_dotenv()


class LLMProvider(str, Enum):
    OPENAI = "openai"
    LMSTUDIO = "lmstudio"
    OLLAMA = "ollama"


LMSTUDIO_MODELS = [
    "deepseek/deepseek-r1-0528-qwen3-8b",
    "llama-3-22b-instruct-v0.1",
    "google/gemma-4-31b",
]

OPENAI_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
]

OLLAMA_MODELS = [
    "llama3:latest",
    "mistral:latest",
    "phi3:latest",
    "gemma:latest",
    "qwen2:latest"
]

LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434/v1")

# ── Available Embedding Models (sentence-transformers compatible) ─────────
# Each entry: (model_name, dimensions, description)
EMBEDDING_MODELS = [
    "all-MiniLM-L6-v2",              # 384 dims — fast, lightweight, classic default
    "all-MiniLM-L12-v2",             # 384 dims — slightly better quality than L6
    "all-mpnet-base-v2",             # 768 dims — best quality classic sentence-transformers
    "BAAI/bge-small-en-v1.5",        # 384 dims — strong retrieval performance, small
    "BAAI/bge-base-en-v1.5",         # 768 dims — strong retrieval performance, medium
    "BAAI/bge-large-en-v1.5",        # 1024 dims — strongest BGE, large
    "nomic-ai/nomic-embed-text-v1.5", # 768 dims — open-source, strong MTEB scores
]

EMBEDDING_MODEL_INFO = {
    "all-MiniLM-L6-v2":              {"dimensions": 384,  "description": "Fast, lightweight, classic default"},
    "all-MiniLM-L12-v2":             {"dimensions": 384,  "description": "Slightly better quality than L6"},
    "all-mpnet-base-v2":             {"dimensions": 768,  "description": "Best quality classic model"},
    "BAAI/bge-small-en-v1.5":        {"dimensions": 384,  "description": "Strong retrieval, small footprint"},
    "BAAI/bge-base-en-v1.5":         {"dimensions": 768,  "description": "Strong retrieval, balanced"},
    "BAAI/bge-large-en-v1.5":        {"dimensions": 1024, "description": "Strongest BGE, large"},
    "nomic-ai/nomic-embed-text-v1.5": {"dimensions": 768,  "description": ""},
}


class LLMConfig(BaseModel):
    provider: LLMProvider = LLMProvider.OPENAI
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 4096

    @model_validator(mode="after")
    def validate_model_for_provider(self) -> "LLMConfig":
        # We no longer strictly validate models against hardcoded lists
        # because Ollama and LMStudio models are fetched dynamically.
        return self


class ChunkingConfig(BaseModel):
    strategy: str = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 200


class EmbeddingConfig(BaseModel):
    model: str = "all-MiniLM-L6-v2"

    @model_validator(mode="after")
    def validate_embedding_model(self) -> "EmbeddingConfig":
        if self.model not in EMBEDDING_MODELS:
            raise ValueError(
                f"Embedding model '{self.model}' is not available. "
                f"Choose from: {EMBEDDING_MODELS}"
            )
        return self


class Neo4jConfig(BaseModel):
    uri: str = Field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    user: str = Field(default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"))
    password: str = Field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", "password"))
    database: str = "neo4j"


class ExtractionConfig(BaseModel):
    max_concurrency: int = 5


class EntityResolutionConfig(BaseModel):
    enabled: bool = True
    use_llm: bool = True
    k_neighbors: int = 10
    similarity_threshold: float = 0.85
    word_distance_threshold: int = 3
    max_concurrency: int = 5


class CommunityDetectionConfig(BaseModel):
    algorithm: str = "louvain"
    resolution: float = 1.0
    level: int = -1


class PostgresConfig(BaseModel):
    dsn: str = Field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL",
            "postgresql://graphrag:graphrag@postgres:5432/graphrag",
        )
    )


class StorageConfig(BaseModel):
    base_path: str = Field(default_factory=lambda: os.getenv("STORAGE_BASE", "/docs"))


class PipelineConfig(BaseModel):
    llm: LLMConfig = LLMConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    neo4j: Neo4jConfig = Neo4jConfig()
    extraction: ExtractionConfig = ExtractionConfig()
    entity_resolution: EntityResolutionConfig = EntityResolutionConfig()
    community_detection: CommunityDetectionConfig = CommunityDetectionConfig()
    postgres: PostgresConfig = PostgresConfig()
    storage: StorageConfig = StorageConfig()
