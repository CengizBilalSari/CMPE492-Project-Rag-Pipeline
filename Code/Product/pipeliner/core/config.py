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


LMSTUDIO_MODELS = [
    "deepseek/deepseek-r1-0528-qwen3-8b",
    "llama-3-22b-instruct-v0.1",
    "google/gemma-4-31b",
]

OPENAI_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
]

LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")


class LLMConfig(BaseModel):
    provider: LLMProvider = LLMProvider.OPENAI
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 4096

    @model_validator(mode="after")
    def validate_model_for_provider(self) -> "LLMConfig":
        if self.provider == LLMProvider.OPENAI and self.model not in OPENAI_MODELS:
            raise ValueError(
                f"Model '{self.model}' is not available for openai. "
                f"Choose from: {OPENAI_MODELS}"
            )
        if self.provider == LLMProvider.LMSTUDIO and self.model not in LMSTUDIO_MODELS:
            raise ValueError(
                f"Model '{self.model}' is not available for lmstudio. "
                f"Choose from: {LMSTUDIO_MODELS}"
            )
        return self


class ChunkingConfig(BaseModel):
    strategy: str = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 200


class EmbeddingConfig(BaseModel):
    model: str = "all-MiniLM-L6-v2"


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


class SupabaseConfig(BaseModel):
    url: str = Field(default_factory=lambda: os.getenv("SUPABASE_URL", ""))
    key: str = Field(default_factory=lambda: os.getenv("SUPABASE_KEY", ""))
    bucket: str = "documents"
    table: str = "documents"


class PipelineConfig(BaseModel):
    llm: LLMConfig = LLMConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    neo4j: Neo4jConfig = Neo4jConfig()
    extraction: ExtractionConfig = ExtractionConfig()
    entity_resolution: EntityResolutionConfig = EntityResolutionConfig()
    community_detection: CommunityDetectionConfig = CommunityDetectionConfig()
    supabase: SupabaseConfig = SupabaseConfig()
