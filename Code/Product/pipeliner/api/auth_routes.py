from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.db import connection

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=100)


class LoginResponse(BaseModel):
    chat_id: str
    username: str


@router.post("/login", response_model=LoginResponse)
async def login(payload: LoginRequest) -> LoginResponse:
    """Look up a chat_id by username, or create a new chat if the username is new."""
    username = payload.username.strip()
    if not username:
        raise HTTPException(status_code=400, detail="Username must not be empty.")

    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chats (username)
                VALUES (%s)
                ON CONFLICT (username) DO UPDATE SET username = EXCLUDED.username
                RETURNING chat_id, username
                """,
                (username,),
            )
            row = cur.fetchone()
        conn.commit()

    if not row:
        raise HTTPException(status_code=500, detail="Failed to resolve chat.")

    logger.info("Login: username=%s chat_id=%s", row["username"], row["chat_id"])
    return LoginResponse(chat_id=str(row["chat_id"]), username=row["username"])
