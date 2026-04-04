from __future__ import annotations

import os
from fastapi import APIRouter, Header, HTTPException
from supabase import create_client

router = APIRouter(prefix="/api/history", tags=["history"])


def _get_supabase():
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_KEY", "")
    if not url or not key:
        raise HTTPException(status_code=500, detail="Supabase not configured.")
    return create_client(url, key)


@router.get("/documents")
async def list_documents(x_user_id: str = Header(..., alias="X-User-Id")):
    db = _get_supabase()
    resp = db.table("documents").select("*").eq("user_id", x_user_id).order("created_at", desc=True).execute()
    return resp.data or []


@router.get("/pipeline-runs")
async def list_pipeline_runs(x_user_id: str = Header(..., alias="X-User-Id")):
    db = _get_supabase()
    resp = (
        db.table("pipeline_runs")
        .select("*, documents(filename)")
        .eq("user_id", x_user_id)
        .order("started_at", desc=True)
        .execute()
    )
    return resp.data or []


@router.get("/evaluation-jobs")
async def list_evaluation_jobs(x_user_id: str = Header(..., alias="X-User-Id")):
    db = _get_supabase()
    resp = (
        db.table("evaluation_jobs")
        .select("*")
        .eq("user_id", x_user_id)
        .order("created_at", desc=True)
        .execute()
    )
    return resp.data or []
