from __future__ import annotations

import asyncio
import io
import json
import logging
import traceback
from contextlib import suppress

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from core.config import (
    PipelineConfig,
    EMBEDDING_MODELS,
    EMBEDDING_MODEL_INFO,
    OPENAI_MODELS,
    LMSTUDIO_MODELS,
    OLLAMA_MODELS,
    OLLAMA_BASE_URL,
    LMSTUDIO_BASE_URL,
)
from core.document_store import DocumentStore
from services.graph_pipeline import GraphRAGPipeline

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/pipeline/config")
async def get_pipeline_config():
    """Return available models, embedding options, and chunking strategies."""
    import httpx
    
    ollama_models = list(OLLAMA_MODELS)
    lmstudio_models = list(LMSTUDIO_MODELS)
    
    # Try fetching dynamically available models
    async with httpx.AsyncClient(timeout=2.0) as client:
        try:
            # OLLAMA_BASE_URL usually ends in /v1, strip it to reach /api/tags
            base_url = OLLAMA_BASE_URL.split("/v1")[0].rstrip("/")
            resp = await client.get(f"{base_url}/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                fetched_models = data.get("models") or []
                fetched = [m.get("name") for m in fetched_models if m.get("name")]
                if fetched:
                    ollama_models = fetched
        except Exception as e:
            logger.warning(f"Could not fetch Ollama models dynamically: {e}")

        try:
            resp = await client.get(f"{LMSTUDIO_BASE_URL.rstrip('/')}/models")
            if resp.status_code == 200:
                data = resp.json()
                fetched = [m["id"] for m in data.get("data", [])]
                if fetched:
                    lmstudio_models = fetched
        except Exception as e:
            logger.warning(f"Could not fetch LMStudio models dynamically: {e}")

    return {
        "llm_providers": {
            "openai": OPENAI_MODELS,
            "lmstudio": lmstudio_models,
            "ollama": ollama_models,
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

        pipeline_type = payload.get("pipeline_type", "custom")

        if pipeline_type == "ms-graphrag":
            from services.ms_graphrag_pipeline import MSGraphRAGPipeline
            pipeline = MSGraphRAGPipeline(config, chat_id=chat_id, document_id=document_id)
        else:
            pipeline = GraphRAGPipeline(config, chat_id=chat_id, document_id=document_id)

        queue: asyncio.Queue[tuple[str, str | None]] = asyncio.Queue()

        async def _produce():
            try:
                async for status_msg in pipeline.run(
                    document_text=document_text,
                    doc_title=doc_title,
                    doc_source=doc_source,
                ):
                    await queue.put(("status", status_msg))
            except asyncio.CancelledError:
                raise
            except Exception as e:
                await queue.put(("error", str(e)))
            finally:
                await queue.put(("done", None))

        producer_task = asyncio.create_task(_produce())

        try:
            while True:
                msg_type, msg = await queue.get()
                if msg_type == "done":
                    break
                await ws.send_json({"type": msg_type, "message": msg})
        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected.")
        finally:
            pipeline.cancel()
            if not producer_task.done():
                producer_task.cancel()
            with suppress(asyncio.CancelledError):
                await producer_task
            pipeline.close()
            with suppress(Exception):
                await ws.close()

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

class PullRequest(BaseModel):
    model: str

@router.get("/api/ollama/recommendations")
async def get_ollama_recommendations():
    tiers = [
        {
            "id": "tier1",
            "name": "Tier 1: Lightweight",
            "description": "Best for <8GB RAM or older machines. Fast but lower reasoning.",
            "min_ram_gb": 0,
            "max_ram_gb": 12,
            "models": [
                {"name": "phi3:mini", "size": "3.8B", "desc": "Good at following strict instructions."},
                {"name": "llama3.2:3b", "size": "3B", "desc": "Lightweight Llama model, surprisingly smart."},
                {"name": "gemma2:2b", "size": "2B", "desc": "Google's small powerhouse for basic tasks."},
                {"name": "qwen2.5:1.5b", "size": "1.5B", "desc": "Extremely fast, good for low-end hardware."}
            ]
        },
        {
            "id": "tier2",
            "name": "Tier 2: Balanced",
            "description": "Best for 16GB RAM. The gold standard for local GraphRAG.",
            "min_ram_gb": 12,
            "max_ram_gb": 24,
            "models": [
                {"name": "llama3.1:latest", "size": "8B", "desc": "Excellent local extraction and reasoning."},
                {"name": "llama3:latest", "size": "8B", "desc": "Classic Llama 3 8B model."},
                {"name": "qwen2.5:7b", "size": "7B", "desc": "Incredible reasoning and coding capabilities."},
                {"name": "mistral-nemo:latest", "size": "12B", "desc": "Massive 128k context window, great for graphs."},
                {"name": "gemma2:9b", "size": "9B", "desc": "Punches above its weight class in logic."},
                {"name": "mistral:latest", "size": "7B", "desc": "Solid alternative to Llama 3."}
            ]
        },
        {
            "id": "tier3",
            "name": "Tier 3: Heavyweight",
            "description": "Best for 32GB+ RAM. Ultimate local quality.",
            "min_ram_gb": 24,
            "max_ram_gb": 999,
            "models": [
                {"name": "qwen2.5:32b", "size": "32B", "desc": "Near GPT-4 performance for local GraphRAG."},
                {"name": "command-r:latest", "size": "35B", "desc": "Optimized specifically for RAG workflows."},
                {"name": "mixtral:8x7b", "size": "47B", "desc": "Fast Mixture of Experts model."},
                {"name": "llama3.1:70b", "size": "70B", "desc": "Requires heavy hardware but unmatched quality."}
            ]
        }
    ]
            
    return {"tiers": tiers}

@router.post("/api/ollama/pull")
async def pull_ollama_model(req: PullRequest):
    import httpx
    from fastapi.responses import StreamingResponse
    
    base_url = OLLAMA_BASE_URL.split("/v1")[0].rstrip("/")
    
    async def stream_pull():
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", f"{base_url}/api/pull", json={"name": req.model}) as r:
                    async for chunk in r.aiter_bytes():
                        yield chunk
        except Exception as e:
            yield f'{{"error":"{str(e)}"}}\n'.encode("utf-8")
                    
    return StreamingResponse(stream_pull(), media_type="application/x-ndjson")

