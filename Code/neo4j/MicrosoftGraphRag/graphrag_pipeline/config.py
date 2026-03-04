from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv()

class ChunkingConfig(BaseModel):
    strategy: str = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 200


class LLMConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 4096


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


class GlobalSearchConfig(BaseModel):
    max_token_per_batch: int = 2000
    max_concurrency: int = 5
    top_k: int = 10


class PipelineConfig(BaseModel):
    chunking: ChunkingConfig = ChunkingConfig()
    llm: LLMConfig = LLMConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    neo4j: Neo4jConfig = Neo4jConfig()
    extraction: ExtractionConfig = ExtractionConfig()
    entity_resolution: EntityResolutionConfig = EntityResolutionConfig()
    community_detection: CommunityDetectionConfig = CommunityDetectionConfig()
    global_search: GlobalSearchConfig = GlobalSearchConfig()



def load_config(path: Optional[str] = None) -> PipelineConfig:
    if path is None:
        path = str(_PROJECT_ROOT / "config.yaml")
    data: dict[str, Any] = {}
    config_path = Path(path)
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    return PipelineConfig(**data)


