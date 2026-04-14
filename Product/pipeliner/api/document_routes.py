from __future__ import annotations

import logging

from fastapi import APIRouter, File, Header, UploadFile, HTTPException

from core.document_store import DocumentStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])

ALLOWED_CONTENT_TYPES = {"application/pdf", "application/txt", "text/txt", "text/plain", "text/csv"}


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    x_chat_id: str = Header(..., alias="X-Chat-Id"),
):
    """Upload a document to local storage and register its metadata in Postgres."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")

    content_type = file.content_type or ""
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{content_type}'. Allowed: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if content_type == "text/plain":
        content_type = "text/txt"

    try:
        store = DocumentStore()
        result = store.upload(
            file_bytes=file_bytes,
            filename=file.filename,
            chat_id=x_chat_id,
            content_type=content_type,
        )
    except Exception as e:
        logger.error("Failed to upload document: %s", e)
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

    return {
        "id": str(result["id"]),
        "name": result["name"],
        "path": result["path"],
    }
