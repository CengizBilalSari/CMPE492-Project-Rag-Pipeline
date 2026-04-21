from __future__ import annotations

import io
import json
import logging
import traceback

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from core.config import (
    PipelineConfig,
    EMBEDDING_MODELS,
    EMBEDDING_MODEL_INFO,
    OPENAI_MODELS,
    LMSTUDIO_MODELS,
)
from core.document_store import DocumentStore
from services.graph_pipeline import GraphRAGPipeline

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/pipeline/config")
async def get_pipeline_config():
    """Return available models, embedding options, and chunking strategies."""
    return {
        "llm_providers": {
            "openai": OPENAI_MODELS,
            "lmstudio": LMSTUDIO_MODELS,
        },
        "embedding_models": [
            {
                "name": name,
                "dimensions": EMBEDDING_MODEL_INFO[name]["dimensions"],
                "description": EMBEDDING_MODEL_INFO[name]["description"],
            }
            for name in EMBEDDING_MODELS
        ],
        "chunking_strategies": [
            "sentence", "token", "character",
            "recursive", "semantic", "propositional",
        ],
    }


@router.websocket("/ws/pipeline/run")
async def run_pipeline(ws: WebSocket):
    """
    WebSocket endpoint for running the GraphRAG pipeline.

    Expects a JSON message on connect:
    {
        "chat_id": "...",
        "document_id": "...",       // required — used to fetch the file from local storage
        "document_text": "...",     // optional — if provided, skips the storage fetch
        "doc_title": "...",
        "doc_source": "...",
        "config": { ... }
    }
    """
    await ws.accept()

    try:
        raw = await ws.receive_text()
        payload = json.loads(raw)

        chat_id = payload.get("chat_id", "")
        document_id = payload.get("document_id", "")
        document_text = payload.get("document_text", "")
        doc_title = payload.get("doc_title", "")
        doc_source = payload.get("doc_source", "")

        if not chat_id:
            await ws.send_json({"type": "error", "message": "chat_id is required."})
            await ws.close()
            return

        if not document_text and document_id:
            try:
                store = DocumentStore()
                doc = store.get(document_id, chat_id)
                if not doc:
                    await ws.send_json({"type": "error", "message": f"Document '{document_id}' not found."})
                    await ws.close()
                    return

                content_type = doc.get("content_type", "")
                if not doc_title:
                    doc_title = doc.get("name", "untitled")

                file_bytes = store.read_bytes(doc["path"])

                if "pdf" in content_type:
                    try:
                        import pypdf
                        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                        document_text = "\n".join(
                            page.extract_text() or "" for page in reader.pages
                        )
                    except ImportError:
                        await ws.send_json({"type": "error", "message": "pypdf is not installed. Cannot extract PDF text."})
                        await ws.close()
                        return
                else:
                    document_text = file_bytes.decode("utf-8", errors="replace")

            except Exception as fetch_err:
                logger.error("Failed to fetch document: %s", fetch_err)
                await ws.send_json({"type": "error", "message": f"Failed to fetch document: {fetch_err}"})
                await ws.close()
                return

        if not document_text:
            await ws.send_json({"type": "error", "message": "document_text is required (or provide a valid document_id)."})
            await ws.close()
            return

        config_overrides = payload.get("config", {})
        config = PipelineConfig(**config_overrides)

        pipeline = GraphRAGPipeline(config, chat_id=chat_id, document_id=document_id)

        try:
            async for status_msg in pipeline.run(
                document_text=document_text,
                doc_title=doc_title,
                doc_source=doc_source,
            ):
                await ws.send_json({"type": "status", "message": status_msg})
        finally:
            pipeline.close()

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except json.JSONDecodeError:
        await ws.send_json({"type": "error", "message": "Invalid JSON payload."})
        await ws.close()
    except Exception as e:
        logger.error("Pipeline error: %s\n%s", e, traceback.format_exc())
        try:
            await ws.send_json({"type": "error", "message": str(e)})
            await ws.close()
        except Exception:
            pass
