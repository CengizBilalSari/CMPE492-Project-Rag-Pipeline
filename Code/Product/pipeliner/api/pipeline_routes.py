from __future__ import annotations

import json
import logging
import traceback

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from core.config import PipelineConfig
from services.graph_pipeline import GraphRAGPipeline

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/pipeline/run")
async def run_pipeline(ws: WebSocket):
    """
    WebSocket endpoint for running the GraphRAG pipeline.

    Expects a JSON message on connect with the following shape:
    {
        "user_id": "...",
        "document_id": "...",
        "document_text": "...",
        "doc_title": "...",
        "doc_source": "...",
        "config": {                 // optional overrides
            "llm": {"provider": "lmstudio", "model": "google/gemma-4-31b"},
            "chunking": {"strategy": "recursive", "chunk_size": 1000},
            ...
        }
    }

    The server streams back JSON messages:
    {"type": "status", "message": "Step 1/6: Chunking..."}
    {"type": "status", "message": "Pipeline completed."}
    {"type": "error",  "message": "..."}
    """
    await ws.accept()

    try:
        raw = await ws.receive_text()
        payload = json.loads(raw)

        user_id = payload.get("user_id", "")
        document_id = payload.get("document_id", "")
        document_text = payload.get("document_text", "")
        doc_title = payload.get("doc_title", "")
        doc_source = payload.get("doc_source", "")

        if not document_text:
            await ws.send_json({"type": "error", "message": "document_text is required."})
            await ws.close()
            return

        config_overrides = payload.get("config", {})
        config = PipelineConfig(**config_overrides)

        pipeline = GraphRAGPipeline(config, user_id=user_id, document_id=document_id)

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
