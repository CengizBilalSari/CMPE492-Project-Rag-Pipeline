from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure the pipeliner package root is on sys.path so that
# `from core.config import ...` and `from services.* import ...` resolve.
_PACKAGE_ROOT = Path(__file__).resolve().parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.document_routes import router as document_router
from api.pipeline_routes import router as pipeline_router
from api.history_routes import router as history_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title="Pipeliner — GraphRAG Backend",
    version="0.1.0",
    description="Standalone FastAPI backend for the GraphRAG knowledge-graph pipeline.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(document_router)
app.include_router(pipeline_router)
app.include_router(history_router)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
