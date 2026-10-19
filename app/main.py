"""Main application entry point for the FastAPI backend.

Endpoints
---------
GET  /                  Health check
POST /resume/upload     Parse resume (PDF or DOCX) → extracted text + skills
POST /job/analyze       Analyze a job description → required/preferred skills, exp, edu
POST /match             Score a resume text against a job description
GET  /skill-gap         Identify skills present in JD but missing from resume
GET  /recommendations   Suggest skills the candidate should learn
POST /recommend-roles   Predict the best-fit job roles for a resume (with probability %)
POST /resume/bulk-analyze  Bulk upload + analyze multiple resumes against a JD
POST /rank              Rank analyzed candidates with Top-N shortlisting
POST /export/csv        Export ranked shortlist as CSV download
"""

import csv
import hashlib
import io
import os
from pathlib import Path
from typing import Annotated, List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.parser import (
    extract_text,
    extract_email,
    infer_candidate_name,
    extract_candidate_name,
    extract_phone,
    extract_links,
)
from app.preprocess import light_clean
from app.ranker import rank_candidates
from app.recommender import load_model, recommend_roles
from app.experience import extract_experience
from app.scoring import compute_scores
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

from app.routers_star import star_router; app.include_router(star_router)


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

class RoleRecommendRequest(BaseModel):
    resume_text: str = Field(..., description="Full resume text (e.g. `extracted_text` from /resume/upload).")
    top_k: int = Field(4, ge=1, le=50, description="How many roles to return (capped at the number of known roles).")

    model_config = {
        "json_schema_extra": {
            "example": {
                "resume_text": "Data analyst with 4 years of experience building Power BI dashboards and SQL reports.",
                "top_k": 4,
            }
        }
    }


class RoleScore(BaseModel):
    role: str
    probability_percent: float = Field(..., description="Probability in percent, 0-100.")


class RoleRecommendResponse(BaseModel):
    recommendations: list[RoleScore]
    model: str | None = Field(None, description="Name of the classifier that produced the result.")


class RankRequest(BaseModel):
    results: list
    top_n: int | None = None


class ExportRequest(BaseModel):
    ranked: list


# ---------------------------------------------------------------------------
# Constants — upload limits
# ---------------------------------------------------------------------------
MAX_FILES = 200
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt"}


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
    JSON with keys: filename, char_count, skills, education, experience_years, text_preview, extracted_text.
    """
    # Read uploaded bytes
    raw_bytes = await file.read()

    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Size check
    if len(raw_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(raw_bytes) / (1024*1024):.1f} MB). Maximum is {MAX_FILE_SIZE / (1024*1024):.0f} MB.",
        )

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
    exp_summary = extract_experience(raw_text)

    # Fallback: if explicit phrase (e.g. "5 years of experience") wasn't in text,
    # compute experience years from parsed work history date intervals
    if experience is None and exp_summary.professional_months > 0:
        experience = exp_summary.professional_months // 12

    email       = extract_email(raw_text) or extract_email(cleaned)
    candidate_name = extract_candidate_name(raw_text, file.filename or "")
    phone       = extract_phone(raw_text) or extract_phone(cleaned)
    links       = extract_links(raw_text)

    return {
        "filename":         file.filename,
        "candidate_name":   candidate_name,
        "email":            email,
        "phone":            phone,
        "links":            links,
        "char_count":       len(cleaned),
        "skills":           skills,
        "education":        [{"degree": e.degree, "field": e.field} for e in education],
        "experience_years": experience,
        "professional_months": exp_summary.professional_months,
        "freelance_months": exp_summary.freelance_months,
        "professional_experience_str": exp_summary.professional_str,
        "freelance_experience_str": exp_summary.freelance_str,
        "combined_experience_str": exp_summary.combined_str,
        "text_preview":     cleaned[:500] + ("..." if len(cleaned) > 500 else ""),
        "extracted_text":   cleaned,
        "raw_text":         raw_text,
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
    """Score a resume against a job description using configurable scoring modes.

    Scoring modes (configured via the SCORING_MODE environment variable, defaults to 'skills_only'):
      - skills_only: Evaluates taxonomy skill overlap (0.80 required + 0.20 preferred,
        or 1.0 required when preferred is absent).
      - hybrid: Weighted combination of skill overlap (0.70) and normalized semantic
        similarity (0.30) via Sentence-Transformers (all-MiniLM-L6-v2).
      - proposal: Multidimensional composite score combining skill overlap (0.40),
        semantic similarity (0.40), experience timeline fit (0.10), and education
        tier fit (0.10).

    Returns
    -------
    JSON object containing:
      - skill_overlap_score: float (0.0 to 1.0)
      - final_score: float (0.0 to 1.0), equal to skill_overlap_score in skills_only mode
      - semantic_similarity: float (0.0 to 1.0) or None if in skills_only mode
      - scoring_mode: str ('skills_only', 'hybrid', or 'proposal')
      - score_breakdown: dict of component values, weights, and weighted contributions
      - required_matched, preferred_matched, required_missing, preferred_missing: skill lists
      - total_required, total_preferred: counts
    """
    if not body.resume_text.strip():
        raise HTTPException(status_code=400, detail="resume_text must not be empty.")
    if not body.job_description.strip():
        raise HTTPException(status_code=400, detail="job_description must not be empty.")

    cleaned_resume = light_clean(body.resume_text)
    resume_skills = set(extract_skills(cleaned_resume))
    jd_analysis   = analyze_job_description(body.job_description)
    required      = set(jd_analysis.required_skills)
    preferred     = set(jd_analysis.preferred_skills)

    # Skill overlap lists
    req_matched   = sorted(resume_skills & required)
    pref_matched  = sorted(resume_skills & preferred)
    req_missing   = sorted(required - resume_skills)
    pref_missing  = sorted(preferred - resume_skills)

    # Candidate profile for experience and education fit
    edu_list   = extract_education(cleaned_resume)
    exp_years  = extract_experience_years(cleaned_resume)
    profile = {
        "experience_years": exp_years,
        "education": [{"degree": e.degree, "field": e.field} for e in edu_list],
    }

    scores = compute_scores(
        resume_text=body.resume_text,
        resume_skills=resume_skills,
        jd_analysis=jd_analysis,
        resume_profile=profile,
    )

    return {
        "skill_overlap_score":       scores["skill_overlap_score"],
        "final_score":               scores["final_score"],
        "semantic_similarity":       scores["semantic_similarity"],
        "scoring_mode":              scores["scoring_mode"],
        "score_breakdown":           scores["breakdown"],
        "skill_recency":             scores.get("skill_recency"),
        "required_matched":          req_matched,
        "preferred_matched":         pref_matched,
        "required_missing":          req_missing,
        "preferred_missing":         pref_missing,
        "total_required":            len(required),
        "total_preferred":           len(preferred),
        "note":                      "Semantic, experience and education scoring planned for a future release.",
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
        "note": "Course-link integration planned for a future release.",
    }


@app.post("/recommend-roles", response_model=RoleRecommendResponse, tags=["Recommendations"])
def recommend_job_roles(body: RoleRecommendRequest):
    """Predict which job roles best fit a resume.

    Uses the classifier trained in ``notebooks/03_job_classifier.ipynb``
    (``app/models/job_classifier.joblib``).

    Returns
    -------
    JSON: ``{"recommendations": [{"role": str, "probability_percent": float}, ...], "model": str}``
    sorted by probability, highest first.

    Errors
    ------
    400 : resume_text is empty.
    422 : invalid body, or no usable text after cleaning.
    503 : the trained model file does not exist yet (run the notebook).
    """
    if not body.resume_text.strip():
        raise HTTPException(status_code=400, detail="resume_text must not be empty.")

    try:
        roles = recommend_roles(body.resume_text, top_k=body.top_k)
        model_name = load_model().get("model_name")  # already cached by the call above
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return {"recommendations": roles, "model": model_name}


# ---------------------------------------------------------------------------
# Bulk Upload + Analyze
# ---------------------------------------------------------------------------

@app.post("/resume/bulk-analyze", tags=["Bulk Analysis"])
async def bulk_analyze(
    files: List[UploadFile] = File(...),
    job_description: str = Form(...),
):
    """Upload multiple resumes and score each against a job description.

    Accepts PDF, DOCX, and TXT files. Max 200 files, 10 MB each.
    Duplicate files (by SHA-256 hash) are detected and skipped.
    One failed file does NOT stop the rest.

    Returns
    -------
    JSON with ``summary`` (counts) and ``results`` (per-file details).
    """
    if not job_description.strip():
        raise HTTPException(status_code=400, detail="job_description must not be empty.")

    if len(files) > MAX_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum is {MAX_FILES}, got {len(files)}.",
        )

    # Pre-analyze the JD once (shared across all resumes)
    try:
        jd_analysis = analyze_job_description(job_description)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=f"JD analysis failed: {exc}")

    required = set(jd_analysis.required_skills)
    preferred = set(jd_analysis.preferred_skills)

    results: list[dict] = []
    seen_hashes: dict[str, str] = {}  # hash → first filename
    summary = {
        "total_uploaded": len(files),
        "total_analyzed": 0,
        "total_failed": 0,
        "total_duplicates": 0,
    }

    scoring_mode = os.getenv("SCORING_MODE", "skills_only")
    restore_encode = None
    if scoring_mode != "skills_only":
        try:
            from app.matcher import get_model, split_sentences
            model = get_model()
            jd_chunks = split_sentences(job_description)
            jd_emb = model.encode(jd_chunks, normalize_embeddings=True, show_progress_bar=False)
            jd_chunks_tuple = tuple(jd_chunks)
            orig_encode = model.encode

            def cached_encode(sentences, *args, **kwargs):
                if tuple(sentences) == jd_chunks_tuple:
                    return jd_emb
                return orig_encode(sentences, *args, **kwargs)

            model.encode = cached_encode
            restore_encode = (model, orig_encode)
        except Exception:
            pass

    try:
        for file in files:
            entry: dict = {"filename": file.filename, "status": "failed", "error": None}

            try:
                # --- Validate extension ---
                ext = Path(file.filename or "").suffix.lower()
                if ext not in ALLOWED_EXTENSIONS:
                    entry["error"] = (
                        f"Unsupported format '{ext}'. "
                        f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
                    )
                    summary["total_failed"] += 1
                    results.append(entry)
                    continue

                # --- Read bytes ---
                raw_bytes = await file.read()

                if not raw_bytes:
                    entry["error"] = "File is empty (0 bytes)."
                    summary["total_failed"] += 1
                    results.append(entry)
                    continue

                # --- Size check ---
                if len(raw_bytes) > MAX_FILE_SIZE:
                    entry["error"] = (
                        f"File too large ({len(raw_bytes) / (1024*1024):.1f} MB). "
                        f"Maximum is {MAX_FILE_SIZE / (1024*1024):.0f} MB."
                    )
                    summary["total_failed"] += 1
                    results.append(entry)
                    continue

                # --- Duplicate detection ---
                content_hash = hashlib.sha256(raw_bytes).hexdigest()
                if content_hash in seen_hashes:
                    entry["status"] = "duplicate"
                    entry["error"] = f"Duplicate of {seen_hashes[content_hash]}"
                    entry["content_hash"] = content_hash
                    summary["total_duplicates"] += 1
                    results.append(entry)
                    continue

                seen_hashes[content_hash] = file.filename

                # --- Parse ---
                raw_text = extract_text(raw_bytes, filename=file.filename)
                cleaned = light_clean(raw_text)

                # --- Extract structured info ---
                skills = extract_skills(cleaned)
                education = extract_education(cleaned)
                experience = extract_experience_years(cleaned)
                exp_summary = extract_experience(raw_text)

                # --- Score against JD ---
                resume_skills = set(skills)
                req_matched = sorted(resume_skills & required)
                pref_matched = sorted(resume_skills & preferred)
                req_missing = sorted(required - resume_skills)
                pref_missing = sorted(preferred - resume_skills)

                profile = {
                    "experience_years": experience,
                    "professional_months": exp_summary.professional_months,
                    "education": [{"degree": e.degree, "field": e.field} for e in education],
                }
                jd_dict = jd_analysis.to_dict()
                jd_dict["raw_text"] = job_description

                scores = compute_scores(
                    resume_text=cleaned,
                    resume_skills=resume_skills,
                    jd_analysis=jd_dict,
                    resume_profile=profile,
                )

                match_score = scores["final_score"]

                # Derive candidate name from resume text (fallback to filename)
                candidate_name = extract_candidate_name(raw_text, file.filename or "")

                email = extract_email(raw_text) or extract_email(cleaned)

                entry.update({
                    "status": "analyzed",
                    "candidate_name": candidate_name,
                    "email": email,
                    "content_hash": content_hash,
                    "skills": skills,
                    "experience_years": experience,
                    "professional_months": exp_summary.professional_months,
                    "freelance_months": exp_summary.freelance_months,
                    "professional_experience_str": exp_summary.professional_str,
                    "freelance_experience_str": exp_summary.freelance_str,
                    "combined_experience_str": exp_summary.combined_str,
                    "has_internships": exp_summary.has_internships,
                    "internship_note": exp_summary.internship_note,
                    "education": [
                        {"degree": e.degree, "field": e.field} for e in education
                    ],
                    "match_score": match_score,
                    "final_score": scores["final_score"],
                    "semantic_similarity": scores["semantic_similarity"],
                    "scoring_mode": scores["scoring_mode"],
                    "score_breakdown": scores["breakdown"],
                    "skill_recency": scores.get("skill_recency"),
                    "required_matched": req_matched,
                    "required_missing": req_missing,
                    "preferred_matched": pref_matched,
                    "preferred_missing": pref_missing,
                    "total_required": len(required),
                    "total_preferred": len(preferred),
                    "skills_match_count": len(req_matched) + len(pref_matched),
                    "error": None,
                })
                summary["total_analyzed"] += 1

            except Exception as exc:
                entry["error"] = str(exc)
                summary["total_failed"] += 1

            results.append(entry)
    finally:
        if restore_encode is not None:
            restore_encode[0].encode = restore_encode[1]

    return {"summary": summary, "results": results}


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

@app.post("/rank", tags=["Ranking"])
def rank_resumes(body: RankRequest):
    """Rank analyzed candidates by match score with Top-N shortlisting.

    Tie-breaking: score DESC → skills_match_count DESC →
    professional_months DESC → freelance_months DESC → candidate_name ASC.
    """
    try:
        return rank_candidates(body.results, top_n=body.top_n)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ---------------------------------------------------------------------------
# CSV Export
# ---------------------------------------------------------------------------

@app.post("/export/csv", tags=["Export"])
def export_csv(body: ExportRequest):
    """Export a ranked shortlist as a downloadable CSV file."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Rank", "Candidate", "Email", "Score (%)",
        "Professional Experience", "Freelance Experience",
        "Required Matched", "Required Missing",
        "Preferred Matched", "Preferred Missing",
        "Education",
    ])

    for entry in body.ranked:
        edu_list = entry.get("education") or []
        edu_str = "; ".join(
            e.get("degree", "") + (" in " + e["field"] if e.get("field") else "")
            for e in edu_list
        ) if edu_list else "N/A"

        prof_str = entry.get("professional_experience_str") or (
            f"{entry['experience_years']} yrs" if entry.get("experience_years") is not None else "Not indicated"
        )
        free_str = entry.get("freelance_experience_str") or "None"

        writer.writerow([
            entry.get("rank", ""),
            entry.get("candidate_name", entry.get("filename", "")),
            entry.get("email") or "Not indicated",
            int(round((entry.get("match_score") or 0) * 100)),
            prof_str,
            free_str,
            ", ".join(entry.get("required_matched") or []),
            ", ".join(entry.get("required_missing") or []),
            ", ".join(entry.get("preferred_matched") or []),
            ", ".join(entry.get("preferred_missing") or []),
            edu_str,
        ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=shortlist_export.csv"},
    )

