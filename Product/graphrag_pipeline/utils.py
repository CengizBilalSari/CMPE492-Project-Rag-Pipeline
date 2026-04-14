from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL_CACHE = {}

async def hf_embed(texts: list[str], model_name: str = "all-MiniLM-L6-v2", show_progress: bool = False) -> list[list[float]]:
    from sentence_transformers import SentenceTransformer
    
    if model_name not in _EMBEDDING_MODEL_CACHE:
        logger.info("Loading embedding model '%s' into memory...", model_name)
        _EMBEDDING_MODEL_CACHE[model_name] = SentenceTransformer(model_name)
        
    st_model = _EMBEDDING_MODEL_CACHE[model_name]
    embeddings = st_model.encode(texts, show_progress_bar=show_progress)
    return embeddings.tolist()
