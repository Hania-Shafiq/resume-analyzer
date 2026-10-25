"""Blind screening API router for candidate anonymization and dossier generation."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.experience import extract_experience
from app.preprocess import light_clean
from app.redact import anonymize
from app.skills import extract_education, extract_experience_years, extract_skills

blind_router = APIRouter(tags=["Blind Screening"])

DISCLAIMER_TEXT = (
    "Redaction is heuristic and does not guarantee anonymity. "
    "Location redaction can hide legitimate right-to-work information, "
    "which is why it is configurable."
)

DEGREE_HIERARCHY: Dict[str, int] = {
    "phd": 5,
    "doctorate": 5,
    "doctor of philosophy": 5,
    "master's": 4,
    "master": 4,
    "masters": 4,
    "mba": 4,
    "m.s": 4,
    "m.sc": 4,
    "bachelor's": 3,
    "bachelor": 3,
    "bachelors": 3,
    "b.s": 3,
    "b.sc": 3,
    "b.e": 3,
    "b.tech": 3,
    "associate": 2,
    "associate's": 2,
    "high school": 1,
    "matric": 1,
    "hssc": 1,
    "ssc": 1,
    "a-level": 1,
    "o-level": 1,
}


def _get_degree_rank(deg_str: str) -> int:
    clean = deg_str.strip().lower()
    for name, rank in DEGREE_HIERARCHY.items():
        if name in clean:
            return rank
    return 0


class AnonymizeRequest(BaseModel):
    resume_text: str = Field(..., description="Full resume text to be anonymized.")
    options: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional toggles to disable redaction for specific types (e.g. {'location': False}).",
    )


class DossierRequest(BaseModel):
    resume_text: str = Field(..., description="Full resume text for dossier creation.")


@blind_router.post("/resume/anonymize")
def redact_resume(body: AnonymizeRequest) -> Dict[str, Any]:
    """Anonymize a resume text by redacting PII according to configurable options.

    Returns
    -------
    dict:
        {
            "text": str,
            "redactions": list[{"type": str, "count": int}],
            "audit": list[{"type": str, "placeholder": str, "line": int, "rule": str}],
            "residual_flags": list[{"type": str, "line": int, "match": str}]
        }
    """
    if not isinstance(body.resume_text, str) or not body.resume_text.strip():
        raise HTTPException(status_code=400, detail="resume_text must not be empty.")

    return anonymize(body.resume_text, options=body.options)


@blind_router.post("/resume/dossier")
def create_blind_dossier(body: DossierRequest) -> Dict[str, Any]:
    """Create an anonymized candidate dossier for blind screening.

    Returns
    -------
    dict:
        {
            "candidate_ref": str,
            "anonymized_text": str,
            "skills": list[str],
            "experience": int,
            "education_level": str | None,
            "redaction_summary": list[dict],
            "audit": list[dict],
            "residual_flags": list[dict],
            "disclaimer": str
        }
    """
    if not isinstance(body.resume_text, str) or not body.resume_text.strip():
        raise HTTPException(status_code=400, detail="resume_text must not be empty.")

    # 1. Candidate reference from SHA-256
    text_bytes = body.resume_text.encode("utf-8")
    candidate_ref = "C-" + hashlib.sha256(text_bytes).hexdigest()[:8]

    # 2. Run anonymizer
    anon_result = anonymize(body.resume_text)
    anonymized_text = anon_result["text"]

    # 3. Skills extraction
    skills: List[str] = []
    try:
        skills = extract_skills(light_clean(anonymized_text))
    except Exception:
        skills = []

    # 4. Experience years from existing experience extractor
    experience_years = 0
    try:
        exp_summary = extract_experience(body.resume_text)
        experience_years = exp_summary.professional_months // 12
        if experience_years == 0:
            explicit = extract_experience_years(body.resume_text)
            if explicit is not None:
                experience_years = explicit
    except Exception:
        experience_years = 0

    # 5. Highest degree education level from existing extractor
    education_level: Optional[str] = None
    try:
        edu_entries = extract_education(body.resume_text)
        if edu_entries:
            best_entry = max(edu_entries, key=lambda e: _get_degree_rank(e.degree))
            education_level = best_entry.degree
    except Exception:
        education_level = None

    return {
        "candidate_ref": candidate_ref,
        "anonymized_text": anonymized_text,
        "skills": skills,
        "experience": experience_years,
        "education_level": education_level,
        "redaction_summary": anon_result["redactions"],
        "audit": anon_result["audit"],
        "residual_flags": anon_result["residual_flags"],
        "disclaimer": DISCLAIMER_TEXT,
    }
