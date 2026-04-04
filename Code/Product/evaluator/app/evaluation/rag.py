from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, List

import neo4j
import numpy as np
from openai import AsyncOpenAI

from .llm_base import LLMInterface
from .retrievers import GlobalRetriever, LocalRetriever, PPRRerankerRetriever, LazyRetriever

logger = logging.getLogger(__name__)


class OpenAILLM(LLMInterface):
    """Async OpenAI LLM for retriever use."""

    def __init__(self, provider: str = "openai", model: str = "gpt-4o", temperature: float = 0.0, max_tokens: int = 4096) -> None:
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        self.provider = provider
        
        if self.provider == "lmstudio":
            base_url = os.getenv("LMSTUDIO_BASE_URL", "").rstrip("/")
            if not base_url:
                raise ValueError("LMSTUDIO_BASE_URL environment variable is not set.")
            self.base_url = base_url
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set.")
            self._client = AsyncOpenAI(api_key=api_key, max_retries=0)

    async def ainvoke(self, prompt: str, system_prompt=None) -> str:
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.time()

        if self.provider == "lmstudio":
            import httpx
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.7 if self.temperature == 0.0 else self.temperature,
                "max_tokens": self.max_tokens if self.max_tokens > 0 else 4096,
            }
            async def _call():
                endpoint = self.base_url if self.base_url.endswith("/chat/completions") else f"{self.base_url}/chat/completions"
                async with httpx.AsyncClient() as client:
                    resp = await client.post(endpoint, json=payload, timeout=180.0)
                    resp.raise_for_status()
                    return resp.json()

            response = await self._execute_with_retry("LMStudio", _call)
            duration = time.time() - start
            content = response["choices"][0]["message"]["content"] or ""
            pt = response.get("usage", {}).get("prompt_tokens", 0)
            ct = response.get("usage", {}).get("completion_tokens", 0)
        else:
            async def _call():
                return await self._client.chat.completions.create(
                    model=self.model, messages=messages,
                    temperature=self.temperature, max_tokens=self.max_tokens,
                )

            response = await self._execute_with_retry("OpenAI", _call)
            duration = time.time() - start
            content = response.choices[0].message.content or ""
            pt = getattr(response.usage, "prompt_tokens", 0) if response.usage else 0
            ct = getattr(response.usage, "completion_tokens", 0) if response.usage else 0

        self._update_and_log_usage(pt, ct, duration)
        return content


async def _hf_embed(texts: list[str], model_name: str = "all-MiniLM-L6-v2", show_progress: bool = False) -> list[list[float]]:
    from sentence_transformers import SentenceTransformer

    _cache = _hf_embed.__dict__.setdefault("_model_cache", {})
    if model_name not in _cache:
        _cache[model_name] = SentenceTransformer(model_name)
    st_model = _cache[model_name]
    embeddings = st_model.encode(texts, show_progress_bar=show_progress)
    return embeddings.tolist()


VALID_SEARCH_TYPES = {"global", "local", "lazy", "ppr", "rag", "no-retriever"}


class GraphRAGSearchClient:
    """Unified client that executes any of the 6 search modes against Neo4j.

    Modes:
        global       – MAP-REDUCE over community summaries (GlobalRetriever)
        local        – Entity-centric subgraph + chunks (LocalRetriever)
        lazy         – Multi-step query refinement (LazyRetriever)
        ppr          – Personalized PageRank reranking (PPRRerankerRetriever)
        rag          – Baseline vector-similarity over chunks (no graph algorithms)
        no-retriever – Direct LLM invocation (no retrieval at all)
    """

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        neo4j_database: str = "neo4j",
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o",
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self._neo4j_uri = neo4j_uri
        self._neo4j_user = neo4j_user
        self._neo4j_password = neo4j_password
        self._neo4j_database = neo4j_database
        self._llm_provider = llm_provider
        self._llm_model = llm_model
        self._embedding_model = embedding_model

    def query(self, question: str, search_type: str = "global") -> Dict[str, Any]:
        search_type = search_type.lower()
        if search_type not in VALID_SEARCH_TYPES:
            raise ValueError(f"Unknown search_type '{search_type}'. Choose from: {VALID_SEARCH_TYPES}")

        t0 = time.perf_counter()
        answer, contexts = asyncio.run(self._search(question, search_type))
        latency_ms = (time.perf_counter() - t0) * 1000

        return {
            "answer": answer or "",
            "retrieved_contexts": contexts,
            "latency_ms": latency_ms,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

    async def _search(self, question: str, mode: str):
        llm = OpenAILLM(provider=self._llm_provider, model=self._llm_model)
        driver = neo4j.GraphDatabase.driver(
            self._neo4j_uri, auth=(self._neo4j_user, self._neo4j_password),
        )

        async def embed_fn(texts):
            return await _hf_embed(texts, model_name=self._embedding_model, show_progress=False)

        try:
            if mode == "no-retriever":
                answer = await llm.ainvoke(question)
                return answer, []

            if mode == "rag":
                answer, contexts = await self._baseline_rag(driver, llm, embed_fn, question)
                return answer, contexts

            if mode == "local":
                retriever = LocalRetriever(
                    driver=driver, llm=llm, embedding_fn=embed_fn,
                    top_k_entities=5, hop_depth=1, max_chunks=10,
                    database=self._neo4j_database,
                )
                answer = await retriever.search(question)
                contexts = await self._fetch_local_contexts(driver, embed_fn, question)

            elif mode == "lazy":
                retriever = LazyRetriever(
                    driver=driver, llm=llm, embedding_fn=embed_fn,
                    max_subqueries=5, max_chunks=10,
                    database=self._neo4j_database,
                )
                answer = await retriever.search(question)
                contexts = await self._fetch_local_contexts(driver, embed_fn, question)

            elif mode == "ppr":
                retriever = PPRRerankerRetriever(
                    driver=driver, llm=llm, embedding_fn=embed_fn,
                    top_k_entities=10, max_chunks=15,
                    database=self._neo4j_database,
                )
                answer = await retriever.search(question)
                contexts = await self._fetch_local_contexts(driver, embed_fn, question)

            else:  # global
                retriever = GlobalRetriever(
                    driver=driver, llm=llm, token_limit=1000,
                    max_concurrency=1, top_k=10,
                    database=self._neo4j_database,
                    embedding_fn=embed_fn, top_communities=20,
                )
                answer = await retriever.search(question)
                contexts = self._fetch_global_contexts(driver, embed_fn, question)
                # contexts is a coroutine for global
                if asyncio.iscoroutine(contexts):
                    contexts = await contexts

            # Fall back to direct LLM if retriever found nothing
            _no_context_phrases = (
                "i couldn't find any relevant",
                "no community summaries found",
                "no relevant entities found",
            )
            if answer.strip().lower().startswith(_no_context_phrases):
                answer = await llm.ainvoke(question)
                contexts = []

            return answer, contexts

        finally:
            driver.close()
            if hasattr(llm, "_client") and hasattr(llm._client, "close"):
                await llm._client.close()

    async def _baseline_rag(self, driver, llm, embed_fn, question: str):
        """Baseline RAG: simple vector similarity over chunks, no graph algorithms."""
        qemb = (await embed_fn([question]))[0]

        # Fetch all chunks and find most similar by embedding entity mentions
        records, _, _ = driver.execute_query(
            """
            MATCH (c:Chunk)
            RETURN c.text AS text
            LIMIT 100
            """,
            database_=self._neo4j_database,
        )
        chunk_texts = [r["text"] for r in records if r["text"]]

        if not chunk_texts:
            answer = await llm.ainvoke(question)
            return answer, []

        chunk_embeddings = await embed_fn(chunk_texts)
        emb = np.array(chunk_embeddings)
        q_norm = np.array(qemb) / (np.linalg.norm(qemb) + 1e-9)
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
        scores = (emb / norms) @ q_norm
        top_idx = np.argsort(scores)[::-1][:10]
        contexts = [chunk_texts[i] for i in top_idx]

        context_str = "\n\n".join([f"--- Chunk {i+1} ---\n{c}" for i, c in enumerate(contexts)])
        prompt = (
            f"Answer the following question using ONLY the provided context.\n\n"
            f"--- CONTEXT ---\n{context_str}\n\n"
            f"--- QUESTION ---\n{question}\n\n"
            f"--- ANSWER ---"
        )
        answer = await llm.ainvoke(prompt)
        return answer, contexts

    async def _fetch_local_contexts(self, driver, embed_fn, question: str) -> List[str]:
        """Fetch chunk texts associated with top similar entities."""
        qemb = (await embed_fn([question]))[0]
        records, _, _ = driver.execute_query(
            """
            MATCH (e:Entity)
            WHERE e.text_embedding IS NOT NULL
            WITH e, gds.similarity.cosine(e.text_embedding, $qemb) AS score
            WHERE score > 0
            RETURN e.name AS name ORDER BY score DESC LIMIT 5
            """,
            qemb=qemb, database_=self._neo4j_database,
        )
        seed_entities = [r["name"] for r in records]
        if not seed_entities:
            return []

        chunk_records, _, _ = driver.execute_query(
            """
            MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
            WHERE e.name IN $seeds
            RETURN DISTINCT c.text AS text LIMIT 10
            """,
            seeds=seed_entities, database_=self._neo4j_database,
        )
        return [r["text"].replace("\x00", "") for r in chunk_records if r["text"]]

    async def _fetch_global_contexts(self, driver, embed_fn, question: str) -> List[str]:
        """Return top community summaries as contexts."""
        records, _, _ = driver.execute_query(
            "MATCH (cs:CommunitySummary) RETURN cs.text AS text",
            database_=self._neo4j_database,
        )
        all_summaries = [r["text"] for r in records if r["text"]]
        if not all_summaries:
            return []

        if len(all_summaries) > 20:
            embeddings = await embed_fn([question] + all_summaries)
            emb = np.array(embeddings)
            q_norm = emb[0] / (np.linalg.norm(emb[0]) + 1e-9)
            s_embs = emb[1:]
            norms = np.linalg.norm(s_embs, axis=1, keepdims=True) + 1e-9
            scores = (s_embs / norms) @ q_norm
            top_idx = np.argsort(scores)[::-1][:20]
            return [all_summaries[i] for i in top_idx]
        return all_summaries
