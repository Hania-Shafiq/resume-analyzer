"""Extra API routes: skill evidence depth and PDF overlay."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from app.evidence import evidence_counts, skill_evidence
from app.overlay import build_overlay
from app.skills import analyze_job_description

extra_router = APIRouter(tags=["Evidence & Overlay"])

MAX_OVERLAY_BYTES = 5 * 1024 * 1024


class EvidenceRequest(BaseModel):
    resume_text: str
    skills: Optional[List[str]] = None


@extra_router.post("/resume/evidence")
def resume_evidence(body: EvidenceRequest) -> Dict[str, Any]:
    """Classify each skill as high_depth / demonstrated / listed.

    Response: ``{skill: {level, multiplier, evidence, signals}, ..., "counts": {...}}``.
    """
    if not body.resume_text.strip():
        raise HTTPException(status_code=400, detail="resume_text must not be empty.")
    evidence = skill_evidence(body.resume_text, body.skills)
    return {**evidence, "counts": evidence_counts(evidence)}


@extra_router.post("/resume/overlay")
async def resume_overlay(
    file: UploadFile = File(...),
    jd_text: str = Form(...),
    include_semantic: bool = Form(False),
) -> Dict[str, Any]:
    """Render PDF pages with colour-coded skill/semantic annotations."""
    if not jd_text.strip():
        raise HTTPException(status_code=400, detail="jd_text must not be empty.")
    name = (file.filename or "").lower()
    if not name.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    raw = await file.read(MAX_OVERLAY_BYTES + 1)
    if len(raw) > MAX_OVERLAY_BYTES:
        raise HTTPException(status_code=400, detail="File too large. Maximum is 5 MB.")
    if not raw or b"%PDF-" not in raw[:1024]:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid PDF.")
    try:
        analysis = analyze_job_description(jd_text)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="jd_text must not be empty.")
    jd = analysis.to_dict()
    jd["raw_text"] = jd_text
    try:
        return await run_in_threadpool(build_overlay, raw, jd, None, include_semantic)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
