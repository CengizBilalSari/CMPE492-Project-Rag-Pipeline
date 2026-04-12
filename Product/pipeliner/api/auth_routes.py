from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.db import connection

router = APIRouter(prefix="/api/auth", tags=["auth"])
logger = logging.getLogger(__name__)


# ── Schemas ──────────────────────────────────────────────────────────────────

class CreateChatRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)


class ChatResponse(BaseModel):
    chat_id: str
    name: str


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/chats", response_model=list[ChatResponse])
async def list_chats() -> list[ChatResponse]:
    """Return all chat bases."""
    chats = []
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT chat_id, name
                FROM chats
                ORDER BY created_at DESC
                """
            )
            for row in cur.fetchall():
                chats.append(ChatResponse(
                    chat_id=str(row["chat_id"]),
                    name=row["name"]
                ))
    return chats


@router.post("/chats", response_model=ChatResponse)
async def create_chat(payload: CreateChatRequest) -> ChatResponse:
    """Create a new chat base."""
    name = payload.name.strip()

    if not name:
        raise HTTPException(status_code=400, detail="Chat name must not be empty.")

    with connection() as conn:
        with conn.cursor() as cur:
            # Check for name collision globally
            cur.execute("SELECT 1 FROM chats WHERE name = %s", (name,))
            if cur.fetchone():
                raise HTTPException(status_code=409, detail=f"Chat name '{name}' already exists.")

            cur.execute(
                """
                INSERT INTO chats (name)
                VALUES (%s)
                RETURNING chat_id
                """,
                (name,)
            )
            chat_id = str(cur.fetchone()["chat_id"])
        conn.commit()

    if not chat_id:
        raise HTTPException(status_code=500, detail="Failed to create chat.")

    logger.info("Created chat: name=%s chat_id=%s", name, chat_id)
    return ChatResponse(chat_id=str(chat_id), name=name)
