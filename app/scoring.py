"""Scoring module for the Resume Analyzer.

Provides configurable scoring modes ("skills_only", "hybrid", "proposal")
combining skill taxonomy overlap, dense semantic embedding similarity,
experience timeline fit, and education tier fit.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Presets and Configuration
# ---------------------------------------------------------------------------

DEFAULT_SEMANTIC_LOW: float = 0.20
DEFAULT_SEMANTIC_HIGH: float = 0.65

DEFAULT_EVIDENCE_MULTIPLIERS: dict[str, float] = {
    "high_depth": 1.0,
    "demonstrated": 0.85,
    "listed": 0.5,
}

SCORING_CONFIG: dict[str, Any] = {
    "skills_only": {
        "skill": 1.0,
        "semantic": 0.0,
        "experience": 0.0,
        "education": 0.0,
    },
    "hybrid": {
        "skill": 0.70,
        "semantic": 0.30,
        "experience": 0.0,
        "education": 0.0,
    },
    "proposal": {
        "skill": 0.40,
        "semantic": 0.40,
        "experience": 0.10,
        "education": 0.10,
    },
    "composite": {
        "skill": 0.40,
        "semantic": 0.30,
        "experience": 0.15,
        "recency": 0.15,
    },
    "evidence_multipliers": {
        "high_depth": 1.0,
        "demonstrated": 0.85,
        "listed": 0.5,
    },
}

SCORING_PRESETS = {"skills_only", "hybrid", "proposal", "composite"}

DEGREE_RANKS: dict[str, int] = {
    "phd": 5,
    "doctorate": 5,
    "doctor": 5,
    "master's": 4,
    "masters": 4,
    "master": 4,
    "m.s": 4,
    "m.sc": 4,
    "mba": 4,
    "bachelor's": 3,
    "bachelors": 3,
    "bachelor": 3,
    "b.s": 3,
    "b.sc": 3,
    "b.tech": 3,
    "b.e": 3,
    "associate": 2,
    "high school": 1,
    "secondary": 1,
    "intermediate": 1,
    "matric": 1,
    "hssc": 1,
    "ssc": 1,
    "fsc": 1,
    "a-level": 1,
    "o-level": 1,
}


# ---------------------------------------------------------------------------
# Helpers for Extraction and Degree Hierarchy
# ---------------------------------------------------------------------------

def get_degree_rank(deg: Any) -> int:
    """Map a degree representation to a hierarchical integer rank (1-5)."""
    if not deg:
        return 0
    if isinstance(deg, (int, float)):
        return int(deg)
    if hasattr(deg, "degree"):
        deg = getattr(deg, "degree")
    elif isinstance(deg, dict) and "degree" in deg:
        deg = deg["degree"]
    if not isinstance(deg, str):
        return 0
    clean = deg.strip().lower()
    for name in sorted(DEGREE_RANKS.keys(), key=len, reverse=True):
        if name in clean:
            return DEGREE_RANKS[name]
    return 0


def extract_candidate_degree_level(resume_profile: Any) -> int:
    """Determine highest academic degree level achieved by the candidate."""
    if not resume_profile:
        return 0
    if isinstance(resume_profile, (int, float)):
        return int(resume_profile)
    if isinstance(resume_profile, str):
        return get_degree_rank(resume_profile)
    if isinstance(resume_profile, list):
        ranks = [get_degree_rank(item) for item in resume_profile]
        return max(ranks) if ranks else 0
    if hasattr(resume_profile, "education"):
        return extract_candidate_degree_level(getattr(resume_profile, "education"))
    if isinstance(resume_profile, dict):
        for key in ("education", "education_requirement", "degrees", "education_entries"):
            if key in resume_profile and resume_profile[key]:
                return extract_candidate_degree_level(resume_profile[key])
        for key in ("degree", "highest_degree", "degree_level", "education_level"):
            if key in resume_profile and resume_profile[key]:
                return get_degree_rank(resume_profile[key])
    return 0


def extract_jd_required_degree_level(jd_analysis: Any) -> int:
    """Determine minimum degree level required by the job description (0 if none)."""
    if not jd_analysis:
        return 0
    edu_req = None
    if hasattr(jd_analysis, "education_requirement"):
        edu_req = getattr(jd_analysis, "education_requirement")
    elif isinstance(jd_analysis, dict):
        edu_req = jd_analysis.get("education_requirement")

    if not edu_req:
        return 0
    if isinstance(edu_req, (int, float)):
        return int(edu_req)
    if isinstance(edu_req, str):
        return get_degree_rank(edu_req)
    if isinstance(edu_req, list):
        ranks = [get_degree_rank(item) for item in edu_req if get_degree_rank(item) > 0]
        return min(ranks) if ranks else 0
    return get_degree_rank(edu_req)


def extract_candidate_years(resume_profile: Any) -> float:
    """Extract candidate years of experience from profile data."""
    if not resume_profile:
        return 0.0
    if isinstance(resume_profile, (int, float)):
        return max(0.0, float(resume_profile))
    if isinstance(resume_profile, dict):
        if resume_profile.get("experience_years") is not None:
            try:
                return max(0.0, float(resume_profile["experience_years"]))
            except (ValueError, TypeError):
                pass
        if resume_profile.get("professional_months") is not None:
            try:
                return max(0.0, float(resume_profile["professional_months"]) / 12.0)
            except (ValueError, TypeError):
                pass
        if resume_profile.get("experience") is not None:
            try:
                return max(0.0, float(resume_profile["experience"]))
            except (ValueError, TypeError):
                pass
    if hasattr(resume_profile, "experience_years") and getattr(resume_profile, "experience_years") is not None:
        try:
            return max(0.0, float(getattr(resume_profile, "experience_years")))
        except (ValueError, TypeError):
            pass
    if hasattr(resume_profile, "professional_months") and getattr(resume_profile, "professional_months") is not None:
        try:
            return max(0.0, float(getattr(resume_profile, "professional_months")) / 12.0)
        except (ValueError, TypeError):
            pass
    return 0.0


def extract_jd_required_years(jd_analysis: Any) -> float | None:
    """Extract minimum experience years stated by the JD, or None if omitted."""
    if not jd_analysis:
        return None
    min_exp = None
    if hasattr(jd_analysis, "min_experience_years"):
        min_exp = getattr(jd_analysis, "min_experience_years")
    elif isinstance(jd_analysis, dict):
        min_exp = jd_analysis.get("min_experience_years")
    if min_exp is not None:
        try:
            val = float(min_exp)
            if val > 0:
                return val
        except (ValueError, TypeError):
            pass
    # Resilient fallback: scan raw JD text if min_experience_years was omitted/None
    raw_jd = extract_jd_text(jd_analysis)
    if raw_jd:
        from app.skills import extract_experience_years
        fallback_exp = extract_experience_years(raw_jd)
        if fallback_exp is not None and fallback_exp > 0:
            return float(fallback_exp)
    return None


def extract_jd_text(jd_analysis: Any) -> str:
    """Obtain or synthesize raw text from the JD analysis object."""
    if isinstance(jd_analysis, str):
        return jd_analysis
    if isinstance(jd_analysis, dict):
        for key in ("raw_text", "jd_text", "job_description", "description", "text"):
            val = jd_analysis.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    for attr in ("raw_text", "jd_text", "job_description", "description", "text"):
        val = getattr(jd_analysis, attr, None)
        if isinstance(val, str) and val.strip():
            return val.strip()

    # Fallback: synthesize text from attributes
    parts = []
    title = getattr(jd_analysis, "job_title", None) or (
        jd_analysis.get("job_title") if isinstance(jd_analysis, dict) else None
    )
    if title:
        parts.append(str(title))
    req = getattr(jd_analysis, "required_skills", None) or (
        jd_analysis.get("required_skills") if isinstance(jd_analysis, dict) else None
    )
    if req:
        parts.append("Required skills: " + ", ".join(str(s) for s in req))
    pref = getattr(jd_analysis, "preferred_skills", None) or (
        jd_analysis.get("preferred_skills") if isinstance(jd_analysis, dict) else None
    )
    if pref:
        parts.append("Preferred skills: " + ", ".join(str(s) for s in pref))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main Scoring Function
# ---------------------------------------------------------------------------

def compute_scores(
    resume_text: str | None,
    resume_skills: Any,
    jd_analysis: Any,
    resume_profile: Any = None,
    mode: str | None = None,
    semantic_low: float | None = None,
    semantic_high: float | None = None,
) -> dict[str, Any]:
    """Compute multidimensional candidate-job match score.

    Parameters
    ----------
    resume_text : str or None
        Raw or cleaned resume text.
    resume_skills : iterable of str, or None
        Canonical skills extracted from the resume.
    jd_analysis : JDAnalysis, dict, or str
        Structured job description analysis (or raw text).
    resume_profile : dict, list, or None, optional
        Candidate structured profile containing experience and education.
    mode : {"skills_only", "hybrid", "proposal"}, optional
        Scoring preset. Defaults to env var SCORING_MODE or "skills_only".
    semantic_low : float, optional
        Lower bound for cosine similarity normalization (default: 0.20).
    semantic_high : float, optional
        Upper bound for cosine similarity normalization (default: 0.65).

    Returns
    -------
    dict with keys:
        - skill_overlap_score: float in [0.0, 1.0]
        - semantic_similarity: float in [0.0, 1.0] or None
        - experience_fit: float in [0.0, 1.0]
        - education_fit: float in [0.0, 1.0]
        - final_score: float in [0.0, 1.0]
        - scoring_mode: str ("skills_only", "hybrid", "proposal")
        - weights_used: dict[str, float]
        - breakdown: dict of component values, weights, and contributions
    """
    # ── 1. Resolve scoring mode and bounds ─────────────────────────────────
    active_mode = mode or os.getenv("SCORING_MODE", "skills_only")
    if active_mode not in SCORING_PRESETS:
        raise ValueError(
            f"Unknown scoring mode '{active_mode}'. Allowed modes: {sorted(SCORING_PRESETS)}"
        )

    weights = SCORING_CONFIG[active_mode].copy()

    if semantic_low is None:
        semantic_low = float(os.getenv("SEMANTIC_SCORE_LOW", DEFAULT_SEMANTIC_LOW))
    if semantic_high is None:
        semantic_high = float(os.getenv("SEMANTIC_SCORE_HIGH", DEFAULT_SEMANTIC_HIGH))

    # Resolve resume text
    raw_resume_text = resume_text or ""
    if not raw_resume_text and isinstance(resume_profile, dict):
        raw_resume_text = (
            resume_profile.get("extracted_text")
            or resume_profile.get("raw_text")
            or ""
        )

    # ── 2. Skill Component (Plain vs Evidence-Adjusted Coverage) ───────────
    if isinstance(resume_skills, (list, set, tuple)):
        res_skills = {str(s).strip().lower() for s in resume_skills if str(s).strip()}
    else:
        res_skills = set()

    req_list = []
    pref_list = []
    if hasattr(jd_analysis, "required_skills"):
        req_list = getattr(jd_analysis, "required_skills") or []
    elif isinstance(jd_analysis, dict):
        req_list = jd_analysis.get("required_skills") or []

    if hasattr(jd_analysis, "preferred_skills"):
        pref_list = getattr(jd_analysis, "preferred_skills") or []
    elif isinstance(jd_analysis, dict):
        pref_list = jd_analysis.get("preferred_skills") or []

    req_skills = {str(s).strip().lower() for s in req_list if str(s).strip()}
    pref_skills = {str(s).strip().lower() for s in pref_list if str(s).strip()}

    req_matched = sorted(res_skills & req_skills)
    pref_matched = sorted(res_skills & pref_skills)

    req_score = len(req_matched) / max(len(req_skills), 1)
    pref_score = len(pref_matched) / max(len(pref_skills), 1)

    if req_skills and pref_skills:
        plain_coverage = round(0.80 * req_score + 0.20 * pref_score, 4)
    elif req_skills:
        plain_coverage = round(req_score, 4)
    elif pref_skills:
        plain_coverage = round(pref_score, 4)
    else:
        plain_coverage = 0.0

    # Evidence multiplier check (app/evidence.py)
    breakdown_notes: list[str] = []
    multipliers_map: dict[str, Any] = {}
    evidence_available = False

    try:
        from app.evidence import skill_multipliers
        all_matched = req_matched + pref_matched
        mults = skill_multipliers(raw_resume_text, all_matched)
        if isinstance(mults, dict):
            multipliers_map = mults
            evidence_available = True
        else:
            breakdown_notes.append("evidence_unavailable")
    except ImportError:
        breakdown_notes.append("evidence_unavailable")
    except Exception as exc:
        logger.warning("skill_multipliers failed (%s); using default 1.0", exc)
        breakdown_notes.append("evidence_unavailable")

    def _resolve_multiplier(skill_name: str) -> float:
        if not evidence_available:
            return 1.0
        ev_cfg = SCORING_CONFIG.get("evidence_multipliers", DEFAULT_EVIDENCE_MULTIPLIERS)
        val = multipliers_map.get(skill_name)
        if val is None:
            val = multipliers_map.get(skill_name.lower())
        if val is None:
            for k, v in multipliers_map.items():
                if isinstance(k, str) and k.strip().lower() == skill_name.strip().lower():
                    val = v
                    break
        if val is None:
            return 1.0
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            lvl = val.strip().lower()
            return float(ev_cfg.get(lvl, 1.0))
        return 1.0

    adj_req_credit = sum(_resolve_multiplier(s) for s in req_matched)
    adj_pref_credit = sum(_resolve_multiplier(s) for s in pref_matched)

    adj_req_score = adj_req_credit / max(len(req_skills), 1)
    adj_pref_score = adj_pref_credit / max(len(pref_skills), 1)

    if req_skills and pref_skills:
        adjusted_coverage = round(0.80 * adj_req_score + 0.20 * adj_pref_score, 4)
    elif req_skills:
        adjusted_coverage = round(adj_req_score, 4)
    elif pref_skills:
        adjusted_coverage = round(adj_pref_score, 4)
    else:
        adjusted_coverage = 0.0

    # In composite mode, skill coverage incorporates evidence multipliers
    if active_mode == "composite":
        skill_coverage_val = adjusted_coverage
    else:
        skill_coverage_val = plain_coverage

    skill_overlap_score = plain_coverage

    # ── 3. Experience Fit Component ─────────────────────────────────────────
    required_years = extract_jd_required_years(jd_analysis)
    if required_years is not None and required_years > 0:
        candidate_years = extract_candidate_years(resume_profile)
        if candidate_years == 0.0 and raw_resume_text:
            from app.skills import extract_experience_years
            fb_cv_years = extract_experience_years(raw_resume_text)
            if fb_cv_years is not None and fb_cv_years > 0:
                candidate_years = float(fb_cv_years)
        experience_fit = round(min(candidate_years / required_years, 1.0), 4)
    else:
        experience_fit = 1.0

    # ── 4. Education Fit Component ──────────────────────────────────────────
    required_degree_rank = extract_jd_required_degree_level(jd_analysis)
    if required_degree_rank > 0:
        candidate_degree_rank = extract_candidate_degree_level(resume_profile)
        education_fit = 1.0 if candidate_degree_rank >= required_degree_rank else 0.5
    else:
        education_fit = 1.0

    # ── 5. Semantic Component (Lazy Loaded) ─────────────────────────────────
    semantic_similarity_val: float | None = None
    if active_mode != "skills_only":
        try:
            from app.matcher import normalize_similarity, semantic_similarity
            raw_jd_text = extract_jd_text(jd_analysis)

            cos = float(semantic_similarity(raw_resume_text, raw_jd_text))
            semantic_similarity_val = round(
                normalize_similarity(cos, low=semantic_low, high=semantic_high), 4
            )
        except Exception as exc:
            logger.warning(
                "Semantic similarity computation failed (%s); falling back to skills_only mode",
                exc,
            )
            active_mode = "skills_only"
            weights = SCORING_CONFIG["skills_only"].copy()
            semantic_similarity_val = None
            skill_coverage_val = plain_coverage

    # ── 6. Skill Recency Component (Lazy Loaded for Composite Mode) ─────────
    skill_recency_table: dict[str, dict[str, Any]] | None = None
    recency_val: float | None = None

    if active_mode == "composite":
        try:
            from app.recency import skill_last_used, skill_recency_score
            all_target_skills = list(res_skills | req_skills | pref_skills)
            last_used = skill_last_used(raw_resume_text, skills=all_target_skills)
            rec_result = skill_recency_score(req_matched, pref_matched, last_used)
            recency_val = rec_result["score"]
            skill_recency_table = rec_result["per_skill"]
        except Exception as exc:
            logger.warning(
                "Skill recency computation failed (%s); using neutral recency value 0.5",
                exc,
            )
            recency_val = 0.5
            skill_recency_table = {
                s: {"last_used_year": None, "weight": 0.5, "source": "undated"}
                for s in (res_skills | req_skills | pref_skills)
            }

    # ── 7. Breakdown and Final Score ────────────────────────────────────────
    skill_weight = weights["skill"]
    breakdown = {
        "skill": {
            "value": skill_coverage_val,
            "plain_coverage": plain_coverage,
            "adjusted_coverage": adjusted_coverage,
            "weight": skill_weight,
            "weighted_contribution": round(skill_coverage_val * skill_weight, 4),
        },
        "semantic": {
            "value": semantic_similarity_val,
            "weight": weights["semantic"],
            "weighted_contribution": (
                round(semantic_similarity_val * weights["semantic"], 4)
                if semantic_similarity_val is not None
                else 0.0
            ),
        },
        "experience": {
            "value": experience_fit,
            "weight": weights["experience"],
            "weighted_contribution": round(experience_fit * weights["experience"], 4),
        },
    }

    if breakdown_notes:
        breakdown["skill"]["notes"] = list(breakdown_notes)
        breakdown["notes"] = list(breakdown_notes)

    # Education component (only if configured in preset, e.g. proposal)
    if "education" in weights:
        edu_w = weights["education"]
        breakdown["education"] = {
            "value": education_fit,
            "weight": edu_w,
            "weighted_contribution": round(education_fit * edu_w, 4),
        }

    # Recency component (for composite mode)
    if active_mode == "composite":
        rec_w = weights.get("recency", 0.15)
        rec_v = recency_val if recency_val is not None else 0.5
        breakdown["recency"] = {
            "value": rec_v,
            "weight": rec_w,
            "weighted_contribution": round(rec_v * rec_w, 4),
        }

    contrib_sum = sum(
        item["weighted_contribution"]
        for item in breakdown.values()
        if isinstance(item, dict) and "weighted_contribution" in item
    )
    final_score = round(contrib_sum, 4)
    final_score = max(0.0, min(1.0, final_score))

    return {
        "skill_overlap_score": skill_overlap_score,
        "semantic_similarity": semantic_similarity_val,
        "experience_fit": experience_fit,
        "education_fit": education_fit,
        "final_score": final_score,
        "scoring_mode": active_mode,
        "weights_used": weights,
        "breakdown": breakdown,
        "skill_recency": skill_recency_table,
    }
