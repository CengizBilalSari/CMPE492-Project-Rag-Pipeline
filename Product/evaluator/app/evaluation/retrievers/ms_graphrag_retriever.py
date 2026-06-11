"""Microsoft GraphRAG retriever wrapper for Version 3.0.x.

Uses the ``graphrag.api`` query engine (GlobalSearch / LocalSearch)
to answer questions against parquet files produced by ``graphrag index``.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import tiktoken
from graphrag.config.load_config import load_config
import graphrag.api as api

logger = logging.getLogger(__name__)

class MSGraphRAGRetriever:
    """Wraps Microsoft GraphRAG's Global and Local search engines (v3.0.x API).

    Parameters
    ----------
    workspace_path : str
        Absolute path to the graphrag workspace (contains ``output/`` and ``settings.yaml``).
    llm_provider : str
        Provider name: ``"openai"``, ``"lmstudio"``, or ``"ollama"``.
    llm_model : str
        Model name (e.g. ``gpt-4o``).
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
        
        # In GraphRAG 3.0, settings are unified and loaded directly from the workspace
        try:
            self.config = load_config(self._workspace)
            
            # Inject evaluator overrides into the config so it matches your test parameters
            if hasattr(self.config, "models") and self.config.models:
                chat_config = self.config.models.get("default_chat_model")
                if chat_config:
                    chat_config.api_key = self._api_key
                    chat_config.model = self._llm_model
                    
                    # Handle local providers by redirecting the API base
                    if self._llm_provider == "lmstudio":
                        base_url = os.getenv("LMSTUDIO_BASE_URL", "http://host.docker.internal:1234/v1")
                        chat_config.api_base = base_url.rstrip("/").removesuffix("/v1")
                    elif self._llm_provider == "ollama":
                        base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434/v1")
                        chat_config.api_base = base_url.rstrip("/").removesuffix("/v1")
                        
            # Fallback for older config formats just in case
            elif hasattr(self.config, "llm") and self.config.llm:
                self.config.llm.api_key = self._api_key
                self.config.llm.model = self._llm_model
                if self._llm_provider == "lmstudio":
                    base_url = os.getenv("LMSTUDIO_BASE_URL", "http://host.docker.internal:1234/v1")
                    self.config.llm.api_base = base_url.rstrip("/").removesuffix("/v1")
                elif self._llm_provider == "ollama":
                    base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434/v1")
                    self.config.llm.api_base = base_url.rstrip("/").removesuffix("/v1")

        except Exception as e:
            logger.error(f"Could not load settings.yaml from {self._workspace}. Ensure the file exists. Error: {e}")
            self.config = None

    def _load_parquet(self, table_name: str) -> pd.DataFrame:
        """Helper to load parquet files, checking both v3 pipeline names and standard names."""
        # v3.0 default pipeline uses 'create_final_...' prefixes
        new_path = self._output_dir / f"create_final_{table_name}.parquet"
        old_path = self._output_dir / f"{table_name}.parquet"
        
        if new_path.exists():
            return pd.read_parquet(new_path)
        elif old_path.exists():
            return pd.read_parquet(old_path)
        else:
            logger.warning(f"Could not find {table_name} parquet at {new_path} or {old_path}")
            return pd.DataFrame()

    async def search(self, question: str, method: str = "global") -> Dict[str, Any]:
        """Run a search query and return results."""
        
        # Include ALL expected keys so rag.py doesn't crash if config is missing
        if not self.config:
            return {
                "answer": "GraphRAG configuration missing (settings.yaml).",
                "contexts": [],
                "latency_ms": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }

        t0 = time.perf_counter()

        try:
            if method == "global":
                answer, contexts = await self._global_search(question)
            else:
                answer, contexts = await self._local_search(question)
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
        
        # Estimate token usage for the final generation step
        prompt_text = question + "\n" + "\n".join(contexts)
        prompt_tokens = self._count_tokens(prompt_text)
        completion_tokens = self._count_tokens(answer)
        
        return {
            "answer": answer,
            "contexts": contexts,
            "latency_ms": latency_ms,
            "prompt_tokens": prompt_tokens, 
            "completion_tokens": completion_tokens,
        }

    def _count_tokens(self, text: str) -> int:
        """Estimate token count using tiktoken (cl100k_base used by OpenAI models)."""
        if not text:
            return 0
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception as exc:
            logger.warning(f"Token counting failed: {exc}")
            return 0

    # ── Internal v3 API search implementations ───────────────────────────

    async def _global_search(self, question: str):
        """Execute a Global Search using the v3.0 API."""
        # Removed the 'nodes' loading since it's no longer accepted
        entities = self._load_parquet("entities")
        communities = self._load_parquet("communities")
        community_reports = self._load_parquet("community_reports")

        response, context_data = await api.global_search(
            config=self.config,
            # Removed nodes=nodes
            entities=entities,
            communities=communities,
            community_reports=community_reports,
            community_level=2,
            dynamic_community_selection=False,  # <--- Added required argument
            response_type="Multiple Paragraphs",
            query=question,
        )

        contexts = self._parse_context_data(context_data)
        return str(response), contexts

    async def _local_search(self, question: str):
        """Execute a Local Search using the v3.0 API."""
        entities = self._load_parquet("entities")
        text_units = self._load_parquet("text_units")
        relationships = self._load_parquet("relationships")
        community_reports = self._load_parquet("community_reports")
        communities = self._load_parquet("communities")
        
        # Load covariates, but explicitly set to None if the file is missing/empty
        covariates = self._load_parquet("covariates")
        if covariates is not None and covariates.empty:
            covariates = None

        response, context_data = await api.local_search(
            config=self.config,
            entities=entities,
            text_units=text_units,
            relationships=relationships,
            community_reports=community_reports,
            communities=communities,     
            covariates=covariates,       # <--- Now safely passes None
            community_level=2,
            response_type="Multiple Paragraphs",
            query=question,
        )

        contexts = self._parse_context_data(context_data)
        return str(response), contexts

    def _parse_context_data(self, context_data: Any) -> list:
        """Extracts context strings from the API's returned context object."""
        contexts = []
        if isinstance(context_data, list):
            for c in context_data:
                contexts.append(str(c)[:500])
        elif isinstance(context_data, dict):
            for k, v in context_data.items():
                if isinstance(v, pd.DataFrame):
                    # Handle DataFrames safely
                    for _, row in v.head(5).iterrows():
                        ctx_text = " | ".join(str(val) for val in row.values if pd.notna(val) and val)
                        if ctx_text:
                            contexts.append(ctx_text[:500])
                else:
                    contexts.append(f"{k}: {str(v)[:500]}")
        return contexts