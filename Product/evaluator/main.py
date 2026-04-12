from __future__ import annotations

import logging
import sys
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.evaluation import router as evaluation_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title="Evaluator — GraphRAG Evaluation Backend",
    version="0.1.0",
    description="Standalone FastAPI backend for evaluating GraphRAG search methods.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(evaluation_router)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
