"""Router for STAR bullet optimization endpoint."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.preprocess import light_clean
from app.skills import analyze_job_description, extract_skills
from app.star import split_bullets, suggest_star, weak_bullet_reasons

star_router = APIRouter(tags=["STAR Optimizer"])

DISCLAIMER_TEXT = "Suggestions give structure only; fill every placeholder with true facts."


class StarRewriteRequest(BaseModel):
    resume_text: str
    jd_text: Optional[str] = None
    max_suggestions: int = Field(default=5, ge=1, le=50)


@star_router.post("/resume/star-rewrite")
def rewrite_star_bullets(body: StarRewriteRequest) -> Dict[str, Any]:
    """Analyze resume experience bullets and propose STAR optimizations.

    If jd_text is provided, identifies missing skills from the job description
    and conditionally suggests them where category matches exist.
    """
    if not isinstance(body.resume_text, str) or not body.resume_text.strip():
        raise HTTPException(status_code=400, detail="resume_text must not be empty.")

    # 1. Split bullets and evaluate total weak bullets in the resume
    bullets = split_bullets(body.resume_text)
    weak_count = 0
    for b in bullets:
        reasons = weak_bullet_reasons(b)
        if ("no_action_verb" in reasons) or any(r.startswith("weak_opener:") for r in reasons):
            weak_count += 1

    # 2. If jd_text is given, extract missing required and preferred skills
    missing_skills: Optional[List[str]] = None
    if body.jd_text and body.jd_text.strip():
        jd_analysis = analyze_job_description(body.jd_text)
        resume_skills = set(extract_skills(light_clean(body.resume_text)))

        required_missing = [s for s in jd_analysis.required_skills if s.lower() not in resume_skills]
        preferred_missing = [s for s in jd_analysis.preferred_skills if s.lower() not in resume_skills]

        deduped: List[str] = []
        seen = set()
        for s in required_missing + preferred_missing:
            s_clean = s.strip().lower()
            if s_clean and s_clean not in seen:
                seen.add(s_clean)
                deduped.append(s_clean)
        missing_skills = deduped

    # 3. Generate deterministic STAR suggestions
    suggestions = suggest_star(
        resume_text=body.resume_text,
        missing_skills=missing_skills,
        max_suggestions=body.max_suggestions,
    )

    # 4. Count suggestions that require real experience (missing skills)
    missing_skill_count = sum(
        1 for s in suggestions if s.get("requires_real_experience") is True
    )

    return {
        "suggestions": suggestions,
        "counts": {
            "weak_bullets": weak_count,
            "suggestions": len(suggestions),
            "missing_skill_suggestions": missing_skill_count,
        },
        "disclaimer": DISCLAIMER_TEXT,
    }
