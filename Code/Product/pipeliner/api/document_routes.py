from __future__ import annotations

import logging

from fastapi import APIRouter, File, Header, UploadFile, HTTPException

from core.config import SupabaseConfig
from core.supabase_client import SupabaseDocumentStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])

ALLOWED_CONTENT_TYPES = {"application/pdf", "application/txt", "text/txt"}


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    x_user_id: str = Header(..., alias="X-User-Id"),
):
    """Upload a document to Supabase storage and register its metadata."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")

    content_type = file.content_type or ""
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{content_type}'. Allowed: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}",
        )

    try:
        store = SupabaseDocumentStore(SupabaseConfig())
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = store.upload_document(
            file_bytes=file_bytes,
            filename=file.filename,
            user_id=x_user_id,
            content_type=content_type,
        )
    except Exception as e:
        logger.error("Failed to upload document: %s", e)
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

    return {
        "id": result["id"],
        "filename": result["filename"],
        "storage_path": result["storage_path"],
    }
