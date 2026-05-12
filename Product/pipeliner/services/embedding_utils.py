from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL_CACHE = {}


async def hf_embed(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    show_progress: bool = False,
    progress_callback=None,
    batch_size: int = 32,
) -> list[list[float]]:
    from sentence_transformers import SentenceTransformer

    if model_name not in _EMBEDDING_MODEL_CACHE:
        logger.info("Loading embedding model '%s' into memory...", model_name)
        _EMBEDDING_MODEL_CACHE[model_name] = SentenceTransformer(model_name)

    st_model = _EMBEDDING_MODEL_CACHE[model_name]
    if not texts:
        return []

    if not progress_callback:
        embeddings = st_model.encode(texts, show_progress_bar=show_progress)
        return embeddings.tolist()

    total = len(texts)
    embeddings: list[list[float]] = []
    for start in range(0, total, batch_size):
        batch = texts[start:start + batch_size]
        batch_emb = st_model.encode(batch, show_progress_bar=False)
        for emb in batch_emb:
            embeddings.append(emb.tolist() if hasattr(emb, "tolist") else list(emb))
        progress_callback(min(start + len(batch), total), total)

    return embeddings
