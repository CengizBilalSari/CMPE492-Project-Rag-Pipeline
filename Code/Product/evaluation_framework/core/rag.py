import asyncio
import sys
import time
from pathlib import Path
from typing import Dict, List

try:
    import requests
except Exception:
    requests = None


class RAGClient:
    def __init__(self, endpoint_url: str, http_method: str = "post", top_k: int = 5):
        if requests is None:
            raise ImportError("requests is not installed. Install with `pip install requests`.")
        self.endpoint_url = endpoint_url
        self.http_method = http_method.lower()
        self.top_k = top_k

    def retrieve(self, query: str) -> Dict:
        payload = {"query": query, "top_k": self.top_k}
        if self.http_method == "get":
            resp = requests.get(self.endpoint_url, params=payload, timeout=30)
        else:
            resp = requests.post(self.endpoint_url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        contexts: List[str] = []
        if isinstance(data, dict):
            if "contexts" in data:
                contexts = data["contexts"]
            elif "documents" in data:
                contexts = data["documents"]
            elif "results" in data:
                contexts = data["results"]
        elif isinstance(data, list):
            contexts = data

        answer = ""
        if isinstance(data, dict) and "answer" in data:
            answer = data["answer"]
        elif contexts:
            answer = contexts[0]

        token_usage = {}
        if isinstance(data, dict) and "token_usage" in data:
            token_usage = data["token_usage"]

        return {
            "answer": answer,
            "retrieved_contexts": contexts,
            "prompt_tokens": token_usage.get("prompt_tokens", 0),
            "completion_tokens": token_usage.get("completion_tokens", 0),
            "total_tokens": token_usage.get("total_tokens", 0),
        }


class GraphRAGClient:
    """Calls the GraphRAG retrievers directly.

    question_type "global" → GlobalRetriever
    question_type "local"  → LocalRetriever
    anything else          → GlobalRetriever

    A fresh LLM client and Neo4j driver are created per query so that
    each asyncio.run() call gets a clean event loop with no stale connections.
    """

    def __init__(self, graphrag_dir: str) -> None:
        graphrag_dir = str(Path(graphrag_dir).resolve())
        if graphrag_dir not in sys.path:
            sys.path.insert(0, graphrag_dir)

        from graphrag_pipeline.config import load_config
        self._config = load_config()

    def query(self, question: str, question_type: str = "global", **_kwargs) -> Dict:
        mode = "local" if question_type.lower() == "local" else "global"
        t0 = time.perf_counter()
        answer, contexts = asyncio.run(self._search(question, mode))
        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "answer": answer or "",
            "retrieved_contexts": contexts,
            "latency_ms": latency_ms,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

    async def _search(self, question: str, mode: str):
        from graphrag_pipeline.llm import get_llm
        from graphrag_pipeline.retrievers import GlobalRetriever, LocalRetriever
        from graphrag_pipeline.utils import hf_embed
        import neo4j as _neo4j
        import numpy as np

        cfg = self._config
        llm = get_llm(
            provider=cfg.llm.provider,
            model=cfg.llm.model,
            temperature=cfg.llm.temperature,
            max_tokens=cfg.llm.max_tokens,
        )
        driver = _neo4j.GraphDatabase.driver(
            cfg.neo4j.uri,
            auth=(cfg.neo4j.user, cfg.neo4j.password),
        )
        try:
            if mode == "local":

                async def embed_fn(texts):
                    return await hf_embed(texts, model_name=cfg.embedding.model, show_progress=False)
                retriever = LocalRetriever(
                    driver=driver,
                    llm=llm,
                    embedding_fn=embed_fn,
                    top_k_entities=cfg.local_search.top_k_entities,
                    hop_depth=cfg.local_search.hop_depth,
                    max_chunks=cfg.local_search.max_chunks,
                    database=cfg.neo4j.database,
                )
                answer = await retriever.search(question)
                # Fetch the same source chunks the retriever used
                qemb = (await embed_fn([question]))[0]
                records, _, _ = driver.execute_query(
                    """
                    MATCH (e:Entity)
                    WHERE e.text_embedding IS NOT NULL
                    WITH e, gds.similarity.cosine(e.text_embedding, $qemb) AS score
                    WHERE score > 0
                    RETURN e.name AS name ORDER BY score DESC LIMIT $top_k
                    """,
                    qemb=qemb, top_k=cfg.local_search.top_k_entities,
                    database_=cfg.neo4j.database,
                )
                seed_entities = [r["name"] for r in records]
                chunk_records, _, _ = driver.execute_query(
                    """
                    MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
                    WHERE e.name IN $seeds
                    RETURN DISTINCT c.text AS text LIMIT $max_chunks
                    """,
                    seeds=seed_entities, max_chunks=cfg.local_search.max_chunks,
                    database_=cfg.neo4j.database,
                )
                contexts = [r["text"].replace("\x00", "") for r in chunk_records if r["text"]]
            else:
                async def embed_fn_global(texts):
                    return await hf_embed(texts, model_name=cfg.embedding.model, show_progress=False)
                retriever = GlobalRetriever(
                    driver=driver,
                    llm=llm,
                    token_limit=cfg.global_search.max_token_per_batch,
                    max_concurrency=cfg.global_search.max_concurrency,
                    top_k=cfg.global_search.top_k,
                    database=cfg.neo4j.database,
                    embedding_fn=embed_fn_global,
                    top_communities=cfg.global_search.top_communities,
                )
                answer = await retriever.search(question)
                # Return the top community summaries as contexts
                all_summaries = retriever._fetch_all_summaries()
                if all_summaries and len(all_summaries) > cfg.global_search.top_communities:
                    embeddings = await embed_fn_global([question] + all_summaries)
                    emb = np.array(embeddings)
                    q_norm = emb[0] / (np.linalg.norm(emb[0]) + 1e-9)
                    s_embs = emb[1:]
                    scores = (s_embs / (np.linalg.norm(s_embs, axis=1, keepdims=True) + 1e-9)) @ q_norm
                    top_idx = np.argsort(scores)[::-1][:cfg.global_search.top_communities]
                    contexts = [all_summaries[i] for i in top_idx]
                else:
                    contexts = all_summaries
            # If retriever couldn't find relevant context, fall back to direct LLM
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
            # Close the async HTTP client before the event loop ends to avoid
            # "Event loop is closed" errors from httpx cleanup
            if hasattr(llm, "_client") and hasattr(llm._client, "close"):
                await llm._client.close()
