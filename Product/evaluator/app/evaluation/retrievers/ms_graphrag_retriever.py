"""Microsoft GraphRAG retriever wrapper.

Uses the ``graphrag`` library's query engine (GlobalSearch / LocalSearch)
to answer questions against parquet files produced by ``graphrag index``.

This module is imported lazily from ``rag.py`` only when the user selects
``ms-graphrag-global`` or ``ms-graphrag-local`` as a search type.

Supports OpenAI, LMStudio, and Ollama providers.  LMStudio / Ollama are
treated as OpenAI-compatible endpoints with a custom ``api_base``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _resolve_provider(llm_provider: str):
    """Return (model_provider, api_base) suitable for graphrag LanguageModelConfig.

    Microsoft GraphRAG uses ``model_provider="openai"`` for all
    OpenAI-compatible endpoints.  For LMStudio / Ollama we just
    redirect via ``api_base``.
    """
    if llm_provider == "lmstudio":
        api_base = os.getenv("LMSTUDIO_BASE_URL", "http://host.docker.internal:1234/v1")
        api_base = api_base.rstrip("/").removesuffix("/v1")
        return "openai", api_base
    elif llm_provider == "ollama":
        api_base = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434/v1")
        api_base = api_base.rstrip("/").removesuffix("/v1")
        return "openai", api_base
    else:
        return "openai", ""


class MSGraphRAGRetriever:
    """Wraps Microsoft GraphRAG's Global and Local search engines.

    Parameters
    ----------
    workspace_path : str
        Absolute path to the graphrag workspace (contains ``output/`` with parquet files).
    llm_provider : str
        Provider name: ``"openai"``, ``"lmstudio"``, or ``"ollama"``.
    llm_model : str
        Model name (e.g. ``gpt-4o``, ``llama3``, etc.).
    api_key : str
        API key (required for OpenAI; can be empty for local providers).
    """

    def __init__(
        self,
        workspace_path: str,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o",
        api_key: str = "",
    ) -> None:
        self._workspace = Path(workspace_path)
        self._output_dir = self._workspace / "output"
        self._llm_provider = llm_provider
        self._llm_model = llm_model
        self._api_key = api_key or os.getenv("OPENAI_API_KEY", "") or "sk-dummy"
        self._model_provider, self._api_base = _resolve_provider(llm_provider)

    def _make_llm_config(self):
        """Build a LanguageModelConfig with the correct provider/base."""
        from graphrag.config.models.language_model_config import LanguageModelConfig

        kwargs = dict(
            api_key=self._api_key,
            model=self._llm_model,
            model_provider=self._model_provider,
            max_retries=5,
        )
        if self._api_base:
            kwargs["api_base"] = self._api_base
        return LanguageModelConfig(**kwargs)

    def _make_embedding_config(self):
        """Build a LanguageModelConfig for embeddings."""
        from graphrag.config.models.language_model_config import LanguageModelConfig

        # For local providers, use a small default; for OpenAI use their model
        embed_model = "text-embedding-3-small" if self._llm_provider == "openai" else self._llm_model
        kwargs = dict(
            api_key=self._api_key,
            model=embed_model,
            model_provider=self._model_provider,
            max_retries=5,
        )
        if self._api_base:
            kwargs["api_base"] = self._api_base
        return LanguageModelConfig(**kwargs)

    async def search(self, question: str, method: str = "global") -> Dict[str, Any]:
        """Run a search query and return results.

        Parameters
        ----------
        question : str
            The question to answer.
        method : str
            Either ``"global"`` or ``"local"``.

        Returns
        -------
        dict with keys: answer, contexts, latency_ms, prompt_tokens, completion_tokens
        """
        t0 = time.perf_counter()

        try:
            if method == "global":
                answer, contexts, prompt_tokens, completion_tokens = await self._global_search(question)
            else:
                answer, contexts, prompt_tokens, completion_tokens = await self._local_search(question)
        except Exception as exc:
            logger.error("MS GraphRAG %s search failed: %s", method, exc, exc_info=True)
            return {
                "answer": f"MS GraphRAG search failed: {exc}",
                "contexts": [],
                "latency_ms": (time.perf_counter() - t0) * 1000,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }

        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "answer": answer,
            "contexts": contexts,
            "latency_ms": latency_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

    # ── Internal search implementations ───────────────────────────

    async def _global_search(self, question: str):
        """Execute a Global Search using community reports."""
        from graphrag.language_model.manager import ModelManager
        from graphrag.query.indexer_adapters import (
            read_indexer_communities,
            read_indexer_entities,
            read_indexer_reports,
        )
        from graphrag.query.structured_search.global_search.community_context import (
            GlobalCommunityContext,
        )
        from graphrag.query.structured_search.global_search.search import GlobalSearch
        from graphrag.tokenizer.get_tokenizer import get_tokenizer

        # Load parquet data
        community_df = pd.read_parquet(self._output_dir / "communities.parquet")
        entity_df = pd.read_parquet(self._output_dir / "entities.parquet")
        report_df = pd.read_parquet(self._output_dir / "community_reports.parquet")

        communities = read_indexer_communities(community_df, report_df)
        reports = read_indexer_reports(report_df, community_df, community_level=2)
        entities = read_indexer_entities(entity_df, community_df, community_level=2)

        # Set up LLM using user's provider selection
        config = self._make_llm_config()
        model = ModelManager().get_or_create_chat_model(
            name="ms_global_search",
            config=config,
        )
        tokenizer = get_tokenizer(config)

        # Build context
        context_builder = GlobalCommunityContext(
            community_reports=reports,
            communities=communities,
            entities=entities,
            tokenizer=tokenizer,
        )

        # Create search engine
        search_engine = GlobalSearch(
            model=model,
            context_builder=context_builder,
            tokenizer=tokenizer,
            max_data_tokens=12_000,
            map_llm_params={"max_tokens": 1000, "temperature": 0.0},
            reduce_llm_params={"max_tokens": 2000, "temperature": 0.0},
            allow_general_knowledge=False,
            json_mode=True,
            concurrent_coroutines=8,
            response_type="multiple paragraphs",
        )

        result = await search_engine.search(question)

        # Extract contexts from community reports used
        contexts = []
        if hasattr(result, "context_data") and "reports" in result.context_data:
            reports_data = result.context_data["reports"]
            if hasattr(reports_data, "iterrows"):
                for _, row in reports_data.iterrows():
                    if "summary" in row:
                        contexts.append(str(row["summary"]))
                    elif "full_content" in row:
                        contexts.append(str(row["full_content"])[:500])

        prompt_tokens = getattr(result, "prompt_tokens", 0)
        output_tokens = getattr(result, "output_tokens", 0)

        return result.response, contexts, prompt_tokens, output_tokens

    async def _local_search(self, question: str):
        """Execute a Local Search using entity-centric subgraph."""
        from graphrag.language_model.manager import ModelManager
        from graphrag.query.indexer_adapters import (
            read_indexer_communities,
            read_indexer_entities,
            read_indexer_reports,
            read_indexer_relationships,
            read_indexer_text_units,
        )
        from graphrag.query.structured_search.local_search.mixed_context import (
            LocalSearchMixedContext,
        )
        from graphrag.query.structured_search.local_search.search import LocalSearch
        from graphrag.tokenizer.get_tokenizer import get_tokenizer

        # Load parquet data
        entity_df = pd.read_parquet(self._output_dir / "entities.parquet")
        relationship_df = pd.read_parquet(self._output_dir / "relationships.parquet")
        text_unit_df = pd.read_parquet(self._output_dir / "text_units.parquet")
        community_df = pd.read_parquet(self._output_dir / "communities.parquet")
        report_df = pd.read_parquet(self._output_dir / "community_reports.parquet")

        entities = read_indexer_entities(entity_df, community_df, community_level=2)
        relationships = read_indexer_relationships(relationship_df)
        text_units = read_indexer_text_units(text_unit_df)
        reports = read_indexer_reports(report_df, community_df, community_level=2)
        communities = read_indexer_communities(community_df, report_df)

        # Set up LLM and embedding model using user's provider selection
        config = self._make_llm_config()
        model = ModelManager().get_or_create_chat_model(
            name="ms_local_search",
            config=config,
        )

        embedding_config = self._make_embedding_config()
        embedding_model = ModelManager().get_or_create_embedding_model(
            name="ms_local_embedding",
            config=embedding_config,
        )

        tokenizer = get_tokenizer(config)

        # Build context
        context_builder = LocalSearchMixedContext(
            community_reports=reports,
            text_units=text_units,
            entities=entities,
            relationships=relationships,
            communities=communities,
            entity_text_embeddings=embedding_model,
            tokenizer=tokenizer,
        )

        # Create search engine
        search_engine = LocalSearch(
            model=model,
            context_builder=context_builder,
            tokenizer=tokenizer,
            max_data_tokens=12_000,
            llm_params={"max_tokens": 2000, "temperature": 0.0},
            response_type="multiple paragraphs",
        )

        result = await search_engine.search(question)

        # Extract contexts
        contexts = []
        if hasattr(result, "context_data"):
            for key in ("entities", "relationships", "sources"):
                data = result.context_data.get(key)
                if data is not None and hasattr(data, "iterrows"):
                    for _, row in data.head(5).iterrows():
                        ctx_text = " | ".join(str(v) for v in row.values if v)
                        if ctx_text:
                            contexts.append(ctx_text[:500])

        prompt_tokens = getattr(result, "prompt_tokens", 0)
        output_tokens = getattr(result, "output_tokens", 0)

        return result.response, contexts, prompt_tokens, output_tokens
