"""
Configuration module for the GraphRAG pipeline.
Loads settings from config.yaml and environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Load .env from the Code/ directory (where NEO4J / LLM keys live)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
_CODE_DIR = _PROJECT_ROOT.parents[2]  # Code/
load_dotenv(_CODE_DIR / ".env")


# ---------------------------------------------------------------------------
# Pydantic config models
# ---------------------------------------------------------------------------
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
    provider: str = "openai"
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
    k_neighbors: int = 10
    similarity_threshold: float = 0.85
    word_distance_threshold: int = 3
    max_concurrency: int = 5


class CommunityDetectionConfig(BaseModel):
    algorithm: str = "louvain"
    resolution: float = 1.0


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration."""

    chunking: ChunkingConfig = ChunkingConfig()
    llm: LLMConfig = LLMConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    neo4j: Neo4jConfig = Neo4jConfig()
    extraction: ExtractionConfig = ExtractionConfig()
    entity_resolution: EntityResolutionConfig = EntityResolutionConfig()
    community_detection: CommunityDetectionConfig = CommunityDetectionConfig()


# ---------------------------------------------------------------------------
# Loader helper
# ---------------------------------------------------------------------------
def load_config(path: Optional[str] = None, overrides: Optional[dict[str, Any]] = None) -> PipelineConfig:
    """Load pipeline config from a YAML file with optional dict overrides.

    Args:
        path: Path to a YAML config file.  Defaults to config.yaml next to this module.
        overrides: Dict of overrides merged on top of the YAML values.

    Returns:
        A validated ``PipelineConfig`` instance.
    """
    if path is None:
        path = str(_PROJECT_ROOT / "config.yaml")

    data: dict[str, Any] = {}
    config_path = Path(path)
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

    if overrides:
        _deep_merge(data, overrides)

    return PipelineConfig(**data)


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge *override* into *base* in-place."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
