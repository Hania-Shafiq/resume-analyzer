"""Main application entry point for the FastAPI backend.

Endpoints
---------
GET  /                  Health check
POST /resume/upload     Parse resume (PDF or DOCX) → extracted text + skills
POST /job/analyze       Analyze a job description → required/preferred skills, exp, edu
POST /match             Score a resume text against a job description
GET  /skill-gap         Identify skills present in JD but missing from resume
GET  /recommendations   Suggest skills the candidate should learn
"""

import io
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.parser import extract_text
from app.preprocess import light_clean
from app.skills import (
    analyze_job_description,
    extract_skills,
    extract_education,
    extract_experience_years,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Resume Analyzer API",
    description=(
        "AI-powered resume parsing, skill extraction, job-description analysis, "
        "semantic matching, and skill-gap detection."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class JDRequest(BaseModel):
    job_title: str = ""
    description: str


class MatchRequest(BaseModel):
    resume_text: str
    job_description: str


class SkillGapRequest(BaseModel):
    resume_text: str
    job_description: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def root():
    """Health check — confirms the API is running."""
    return {"status": "ok", "message": "Resume Analyzer API is running. Visit /docs for Swagger UI."}


@app.post("/resume/upload", tags=["Resume"])
async def upload_resume(file: UploadFile = File(...)):
    """Upload a PDF or DOCX resume and extract its text and skills.

    Returns
    -------
    JSON with keys: filename, char_count, skills, education, experience_years, text_preview.
    """
    # Read uploaded bytes
    raw_bytes = await file.read()

    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Extract raw text using the parser
    try:
        raw_text = extract_text(raw_bytes, filename=file.filename)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # Light-clean for further NLP processing
    cleaned = light_clean(raw_text)

    # Extract structured information
    skills      = extract_skills(cleaned)
    education   = extract_education(cleaned)
    experience  = extract_experience_years(cleaned)

    return {
        "filename":         file.filename,
        "char_count":       len(cleaned),
        "skills":           skills,
        "education":        [{"degree": e.degree, "field": e.field} for e in education],
        "experience_years": experience,
        "text_preview":     cleaned[:500] + ("..." if len(cleaned) > 500 else ""),
    }


@app.post("/job/analyze", tags=["Job Description"])
def analyze_jd(body: JDRequest):
    """Analyze a job description text and return structured requirements.

    Returns
    -------
    JSON matching JDAnalysis.to_dict() plus the raw title prefix if provided.
    """
    if not body.description.strip():
        raise HTTPException(status_code=400, detail="Job description text must not be empty.")

    # Prepend the title so title/category inference can use it
    full_jd = (body.job_title + "\n" + body.description) if body.job_title else body.description

    try:
        result = analyze_job_description(full_jd)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return result.to_dict()


@app.post("/match", tags=["Matching"])
def match_resume_to_jd(body: MatchRequest):
    """Score a resume against a job description.

    Scoring formula (Phase 3 full implementation pending):
      0.40 × skill_overlap  +  0.40 × semantic_similarity
      + 0.10 × experience_fit  +  0.10 × education_fit

    Currently returns skill-overlap score only (Phase 1).
    """
    if not body.resume_text.strip():
        raise HTTPException(status_code=400, detail="resume_text must not be empty.")
    if not body.job_description.strip():
        raise HTTPException(status_code=400, detail="job_description must not be empty.")

    resume_skills = set(extract_skills(light_clean(body.resume_text)))
    jd_analysis   = analyze_job_description(body.job_description)
    required      = set(jd_analysis.required_skills)
    preferred     = set(jd_analysis.preferred_skills)

    # Skill overlap scores
    req_matched   = resume_skills & required
    pref_matched  = resume_skills & preferred

    req_score  = len(req_matched)  / max(len(required),  1)
    pref_score = len(pref_matched) / max(len(preferred), 1)

    # Weighted skill score (required counts more)
    skill_score = round(0.80 * req_score + 0.20 * pref_score, 4)

    return {
        "skill_overlap_score":       skill_score,
        "required_matched":          sorted(req_matched),
        "preferred_matched":         sorted(pref_matched),
        "required_missing":          sorted(required - resume_skills),
        "preferred_missing":         sorted(preferred - resume_skills),
        "total_required":            len(required),
        "total_preferred":           len(preferred),
        "note": "Full semantic + experience + education scoring coming in Phase 3.",
    }


@app.get("/skill-gap", tags=["Skill Gap"])
def skill_gap(resume_text: str, job_description: str):
    """Return skills required by the JD that are absent from the resume.

    Query params
    ------------
    resume_text      : cleaned resume text
    job_description  : raw JD text
    """
    if not resume_text.strip() or not job_description.strip():
        raise HTTPException(status_code=400, detail="Both resume_text and job_description are required.")

    resume_skills   = set(extract_skills(light_clean(resume_text)))
    jd              = analyze_job_description(job_description)
    required_gap    = sorted(set(jd.required_skills)  - resume_skills)
    preferred_gap   = sorted(set(jd.preferred_skills) - resume_skills)

    return {
        "required_gap":  required_gap,
        "preferred_gap": preferred_gap,
        "gap_count":     len(required_gap) + len(preferred_gap),
    }


@app.get("/recommendations", tags=["Recommendations"])
def recommendations(resume_text: str, job_description: str = ""):
    """Suggest skills to learn based on skill gaps.

    If job_description is provided, gaps are JD-specific.
    Otherwise returns a generic list of high-demand skills missing from the resume.
    """
    if not resume_text.strip():
        raise HTTPException(status_code=400, detail="resume_text must not be empty.")

    resume_skills = set(extract_skills(light_clean(resume_text)))

    if job_description.strip():
        jd           = analyze_job_description(job_description)
        gap_required = sorted(set(jd.required_skills)  - resume_skills)
        gap_preferred= sorted(set(jd.preferred_skills) - resume_skills)
    else:
        # Generic high-demand skills (static list for Phase 1)
        high_demand   = ["python", "docker", "kubernetes", "machine learning",
                         "postgresql", "aws", "fastapi", "git", "ci/cd",
                         "natural language processing", "scikit-learn", "pandas"]
        gap_required  = [s for s in high_demand if s not in resume_skills]
        gap_preferred = []

    return {
        "recommended_to_learn":   gap_required[:10],
        "nice_to_add":            gap_preferred[:5],
        "note": "Course-link integration planned for Phase 4.",
    }
