"""Skill recency module for the Resume Analyzer (Feature 6).

Calculates how recently candidate skills were applied across professional roles,
decaying older skill experience using exponential half-life weighting while
treating active/present roles with maximum weight and undated skills neutrally.
"""

from __future__ import annotations

import datetime
import logging
import re
from typing import Any, Iterable

from app.experience import (
    RE_MONTH_YEAR_RANGE,
    RE_NUM_MONTH_RANGE,
    RE_YEAR_RANGE,
    _PRESENT_PATTERN,
    parse_date_endpoints,
)
from app.skills import extract_skills

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration and Constants
# ---------------------------------------------------------------------------

RECENCY_UNDATED: float = 0.5

RECENCY_CONFIG: dict[str, float] = {
    "half_life": 3.0,
    "floor": 0.05,
    "undated": 0.5,
    "neutral": 0.5,
}

# Section heading patterns
RE_EDUCATION_HEADER = re.compile(
    r"^[\W_]*(?:education(?:al)?(?:\s+background|\s+history)?|academic(?:s|\s+background|\s+history)?|qualifications|degrees|educational\s+qualifications)[\W_]*$",
    re.IGNORECASE,
)
RE_SKILLS_HEADER = re.compile(
    r"^[\W_]*(?:technical\s+skills|skills(?:\s*&(?:\s*abilities|\s*competencies))?|core\s+competencies|technologies|tools\s*&\s*technologies|key\s+skills|programming\s+languages|skillset|proficiencies|areas\s+of\s+expertise|technical\s+proficiencies)[\W_]*$",
    re.IGNORECASE,
)
RE_EXPERIENCE_HEADER = re.compile(
    r"^[\W_]*(?:work\s+experience|professional\s+experience|experience|employment(?:\s+history)?|work\s+history|career\s+history|relevant\s+experience|freelance\s+experience|freelance\s+work|projects|personal\s+projects|selected\s+projects)[\W_]*$",
    re.IGNORECASE,
)
RE_OTHER_HEADER = re.compile(
    r"^[\W_]*(?:summary|professional\s+summary|executive\s+summary|profile|about\s+me|objective|career\s+objective|certifications?|publications?|awards?|honors?|languages?|interests?|references?|hobbies)[\W_]*$",
    re.IGNORECASE,
)
RE_INLINE_SKILLS = re.compile(
    r"^[\W_]*(?:technical\s+skills|skills|technologies|core\s+competencies|tools)\s*[:\-]\s*(.+)$",
    re.IGNORECASE,
)

# Degree lines outside education section
RE_DEGREE_LINE = re.compile(
    r"\b(?:bachelor(?:'s)?|b\.?s\.?c?|b\.?tech|b\.?e\.?|master(?:'s)?|m\.?s\.?c?|m\.?tech|m\.?b\.?a\.?|ph\.?d\.?|doctorate|associate(?:'s)?|diploma|high\s+school)\b",
    re.IGNORECASE,
)

# Bullet list prefix pattern
RE_BULLET = re.compile(r"^[\s•\-*+—–▪▫■◦►>]|\b\d+[\.\)]\s+")


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _resolve_today(today: datetime.date | datetime.datetime | int | str | None) -> datetime.date:
    """Resolve an optional today parameter into a datetime.date."""
    if today is None:
        return datetime.date.today()
    if isinstance(today, datetime.datetime):
        return today.date()
    if isinstance(today, datetime.date):
        return today
    if isinstance(today, (int, float)):
        return datetime.date(int(today), 1, 1)
    if isinstance(today, str):
        try:
            return datetime.date.fromisoformat(today)
        except Exception:
            return datetime.date(int(today), 1, 1)
    return datetime.date.today()


def _is_bullet_line(line: str) -> bool:
    """Return True if line begins with a bullet marker or list numeral."""
    return bool(RE_BULLET.match(line.strip()))


def _parse_line_date(
    line: str,
    now: datetime.date,
) -> tuple[int, int, bool, str] | None:
    """Parse date range from a line using app.experience date-range regexes.

    Returns:
        (start_year, end_year, is_present, matched_substring) or None.
    """
    # 1. Month Year range (e.g. "Jan 2021 - Present", "March 2020 to Dec 2022")
    m_my = RE_MONTH_YEAR_RANGE.search(line)
    if m_my:
        s_str = m_my.group(1) + " " + m_my.group(2)
        e_str = m_my.group(3)
        iv = parse_date_endpoints(s_str, e_str, now=now)
        if iv:
            is_present = bool(re.search(_PRESENT_PATTERN, e_str, re.IGNORECASE))
            end_year = now.year if is_present else iv.end_year
            return (iv.start_year, end_year, is_present, m_my.group(0))

    # 2. MM/YYYY range (e.g. "01/2021 - Present", "01/2020 - 05/2023")
    m_num = RE_NUM_MONTH_RANGE.search(line)
    if m_num:
        s_str = m_num.group(1) + "/" + m_num.group(2)
        e_str = m_num.group(3)
        iv = parse_date_endpoints(s_str, e_str, now=now)
        if iv:
            is_present = bool(re.search(_PRESENT_PATTERN, e_str, re.IGNORECASE))
            end_year = now.year if is_present else iv.end_year
            return (iv.start_year, end_year, is_present, m_num.group(0))

    # 3. YYYY range (e.g. "2020 - 2023", "2021 - Present")
    m_yr = RE_YEAR_RANGE.search(line)
    if m_yr:
        s_str = m_yr.group(1)
        e_str = m_yr.group(2)
        iv = parse_date_endpoints(s_str, e_str, now=now)
        if iv:
            is_present = bool(re.search(_PRESENT_PATTERN, e_str, re.IGNORECASE))
            end_year = now.year if is_present else iv.end_year
            return (iv.start_year, end_year, is_present, m_yr.group(0))

    # 4. Standalone single year on a short line (e.g. "(2024)", "2024")
    if len(line.strip()) <= 35:
        m_single = re.search(r"\b([12]\d{3})\b", line)
        if m_single:
            y = int(m_single.group(1))
            if 1970 <= y <= now.year + 2:
                return (y, y, False, m_single.group(0))

    return None


def _skill_matches_line(
    skill: str,
    line: str,
    extracted_line_skills: set[str] | None = None,
) -> bool:
    """Check if a skill is mentioned in a text line with word boundary precision."""
    clean_skill = skill.strip().lower()
    if not clean_skill:
        return False

    if extracted_line_skills and clean_skill in extracted_line_skills:
        return True

    # Punctuation/symbol boundary matching
    pattern = rf"(?<![a-zA-Z0-9]){re.escape(clean_skill)}(?![a-zA-Z0-9])"
    return bool(re.search(pattern, line, re.IGNORECASE))


def _split_resume_entries(
    resume_text: str,
    now: datetime.date,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Segment a resume into experience entries, skills section lines, and education lines.

    Handles single-line headers (title | company | date) and two-line layouts
    (title on line 1, date on line 2), as well as undated experience blocks.
    """
    lines = [l.strip() for l in resume_text.splitlines()]
    entries: list[dict[str, Any]] = []
    skills_lines: list[str] = []
    education_lines: list[str] = []

    current_section: str | None = None
    pending_titles: list[str] = []
    current_entry: dict[str, Any] | None = None

    for raw_line in lines:
        if not raw_line:
            continue

        # Check section boundaries
        if RE_EDUCATION_HEADER.match(raw_line):
            current_section = "education"
            current_entry = None
            pending_titles = []
            continue
        elif RE_SKILLS_HEADER.match(raw_line):
            current_section = "skills"
            current_entry = None
            pending_titles = []
            continue
        elif RE_EXPERIENCE_HEADER.match(raw_line):
            current_section = "experience"
            current_entry = None
            pending_titles = []
            continue
        elif RE_OTHER_HEADER.match(raw_line):
            current_section = "other"
            current_entry = None
            pending_titles = []
            continue

        # Inline skills (e.g. "Skills: Python, Docker, Go")
        m_inline = RE_INLINE_SKILLS.match(raw_line)
        if m_inline:
            skills_lines.append(raw_line)
            continue

        # Lines inside Education section are ignored for experience
        if current_section == "education":
            education_lines.append(raw_line)
            continue

        # Lines inside Skills section are undated skills evidence
        if current_section == "skills":
            skills_lines.append(raw_line)
            continue

        # Degree lines outside Education section are ignored for experience
        if RE_DEGREE_LINE.search(raw_line):
            education_lines.append(raw_line)
            continue

        # Evaluate date range on current line
        date_res = _parse_line_date(raw_line, now=now)
        if date_res:
            sy, ey, is_pres, match_str = date_res
            residual = raw_line.replace(match_str, "").strip("()[] |-–—•\t")

            if len(residual) <= 3:
                # Two-line layout: date range sits on its own line
                if pending_titles:
                    current_entry = {
                        "start_year": sy,
                        "end_year": ey,
                        "is_present": is_pres,
                        "source": "present_role" if is_pres else "dated_role",
                        "lines": list(pending_titles) + [raw_line],
                    }
                    entries.append(current_entry)
                    pending_titles = []
                elif current_entry and current_entry.get("source") == "undated":
                    current_entry["start_year"] = sy
                    current_entry["end_year"] = ey
                    current_entry["is_present"] = is_pres
                    current_entry["source"] = "present_role" if is_pres else "dated_role"
                    current_entry["lines"].append(raw_line)
                else:
                    current_entry = {
                        "start_year": sy,
                        "end_year": ey,
                        "is_present": is_pres,
                        "source": "present_role" if is_pres else "dated_role",
                        "lines": [raw_line],
                    }
                    entries.append(current_entry)
            else:
                # Title and date on the same line
                current_entry = {
                    "start_year": sy,
                    "end_year": ey,
                    "is_present": is_pres,
                    "source": "present_role" if is_pres else "dated_role",
                    "lines": [raw_line],
                }
                entries.append(current_entry)
                pending_titles = []
        else:
            # Non-date line
            if _is_bullet_line(raw_line):
                if current_entry is not None:
                    current_entry["lines"].append(raw_line)
                elif pending_titles:
                    # Undated experience entry
                    current_entry = {
                        "start_year": None,
                        "end_year": None,
                        "is_present": False,
                        "source": "undated",
                        "lines": list(pending_titles) + [raw_line],
                    }
                    entries.append(current_entry)
                    pending_titles = []
                else:
                    skills_lines.append(raw_line)
            else:
                # Potential title / company line
                if current_entry is not None:
                    current_entry = None
                pending_titles.append(raw_line)

    # Capture any trailing undated experience block
    if pending_titles and current_section == "experience":
        entries.append({
            "start_year": None,
            "end_year": None,
            "is_present": False,
            "source": "undated",
            "lines": list(pending_titles),
        })

    return entries, skills_lines, education_lines


# ---------------------------------------------------------------------------
# Core Recency Functions
# ---------------------------------------------------------------------------

def skill_last_used(
    resume_text: str,
    skills: Iterable[str] | None = None,
    today: datetime.date | datetime.datetime | int | str | None = None,
) -> dict[str, dict[str, Any]]:
    """Determine the last-used calendar year, evidence line, and source for each skill.

    Parameters
    ----------
    resume_text : str
        Extracted resume plain text.
    skills : iterable of str, optional
        Target skills to query. If None, extracts canonical skills automatically.
    today : date, datetime, int, or None, optional
        Reference evaluation date (injected for deterministic testing).

    Returns
    -------
    dict
        Mapping of {skill: {"last_used_year": int or None,
                            "source": "present_role" | "dated_role" | "undated",
                            "evidence": str or None}}
    """
    now = _resolve_today(today)

    if not resume_text or not resume_text.strip():
        if skills is None:
            return {}
        return {
            s: {"last_used_year": None, "source": "undated", "evidence": None}
            for s in skills
        }

    if skills is None:
        target_skills = extract_skills(resume_text)
    else:
        target_skills = list(skills)

    entries, skills_lines, edu_lines = _split_resume_entries(resume_text, now=now)

    # Pre-extract canonical skills per line to optimize matching & handle aliases
    line_cache: dict[str, set[str]] = {}
    for entry in entries:
        for line in entry["lines"]:
            if line not in line_cache:
                line_cache[line] = set(extract_skills(line))
    for line in skills_lines:
        if line not in line_cache:
            line_cache[line] = set(extract_skills(line))
    for line in edu_lines:
        if line not in line_cache:
            line_cache[line] = set(extract_skills(line))

    results: dict[str, dict[str, Any]] = {}

    for skill in target_skills:
        best_year: int | None = None
        best_source: str = "undated"
        best_evidence: str | None = None

        # 1. Inspect experience entries (dated and undated)
        for entry in entries:
            ey = entry["end_year"]
            is_pres = entry["is_present"]
            src = entry["source"]

            matched_in_entry = False
            for line in entry["lines"]:
                if _skill_matches_line(skill, line, line_cache.get(line)):
                    matched_in_entry = True
                    entry_evidence = line
                    break

            if matched_in_entry:
                if src == "undated":
                    if best_year is None:
                        best_source = "undated"
                        best_evidence = entry_evidence
                else:
                    # Dated role: compare end years, with present_role taking precedence on ties
                    is_better = False
                    if best_year is None:
                        is_better = True
                    elif ey is not None and ey > best_year:
                        is_better = True
                    elif ey is not None and ey == best_year and is_pres and best_source != "present_role":
                        is_better = True

                    if is_better:
                        best_year = ey
                        best_source = src
                        best_evidence = entry_evidence

        # 2. Inspect skills section if not found in any dated experience entry
        if best_year is None:
            for line in skills_lines:
                if _skill_matches_line(skill, line, line_cache.get(line)):
                    best_source = "undated"
                    best_evidence = line
                    break

        # 3. Inspect education section (ignored for dates; treated as undated)
        if best_year is None and best_evidence is None:
            for line in edu_lines:
                if _skill_matches_line(skill, line, line_cache.get(line)):
                    best_source = "undated"
                    best_evidence = line
                    break

        results[skill] = {
            "last_used_year": best_year,
            "source": best_source,
            "evidence": best_evidence,
        }

    return results


def recency_weight(
    last_used_year: int | None,
    today: datetime.date | datetime.datetime | int | str | None = None,
    half_life: float | None = None,
    floor: float | None = None,
) -> float:
    """Calculate exponential recency weight for a skill given its last used year.

    Formula:
        recency_weight = max(floor, 0.5 ** (age / half_life))
        with age = max(0, today.year - last_used_year)

    Undated skills return the neutral value RECENCY_UNDATED (0.5).

    Parameters
    ----------
    last_used_year : int or None
        The most recent calendar year the skill was used.
    today : date, int, or None, optional
        Reference date (defaults to current date).
    half_life : float, optional
        Half-life in years (defaults to RECENCY_CONFIG["half_life"]).
    floor : float, optional
        Minimum weight floor (defaults to RECENCY_CONFIG["floor"]).

    Returns
    -------
    float
        Weight in [0.0, 1.0].
    """
    hl = half_life if half_life is not None else RECENCY_CONFIG.get("half_life", 3.0)
    fl = floor if floor is not None else RECENCY_CONFIG.get("floor", 0.05)
    undated_val = RECENCY_CONFIG.get("undated", RECENCY_CONFIG.get("neutral", RECENCY_UNDATED))

    if last_used_year is None:
        return float(undated_val)

    now = _resolve_today(today)
    age = max(0, now.year - int(last_used_year))
    raw_weight = 0.5 ** (age / hl)
    weight = max(fl, raw_weight)
    return round(min(1.0, float(weight)), 4)


def skill_recency_score(
    matched_required: Iterable[str],
    matched_preferred: Iterable[str],
    last_used: dict[str, dict[str, Any]],
    today: datetime.date | datetime.datetime | int | str | None = None,
) -> dict[str, Any]:
    """Compute overall recency score across matched required and preferred skills.

    Applies the 0.80 required / 0.20 preferred weighting structure matching
    the skill overlap component:
      - Both matched: 0.80 * mean(req_weights) + 0.20 * mean(pref_weights)
      - Required only: mean(req_weights)
      - Preferred only: mean(pref_weights)
      - Neither matched: 0.0

    Parameters
    ----------
    matched_required : iterable of str
        Skills required by the job that the candidate matched.
    matched_preferred : iterable of str
        Skills preferred by the job that the candidate matched.
    last_used : dict
        Mapping returned by skill_last_used().
    today : date, int, or None, optional
        Reference evaluation date.

    Returns
    -------
    dict
        {"score": float in [0.0, 1.0],
         "per_skill": {skill: {"last_used_year": int or None,
                               "weight": float,
                               "source": str}}}
    """
    req_skills = list(matched_required)
    pref_skills = list(matched_preferred)

    def _lookup_skill_info(s: str) -> dict[str, Any]:
        if s in last_used:
            return last_used[s]
        s_lower = s.strip().lower()
        if s_lower in last_used:
            return last_used[s_lower]
        for k, v in last_used.items():
            if isinstance(k, str) and k.strip().lower() == s_lower:
                return v
        return {"last_used_year": None, "source": "undated", "evidence": None}

    # Build per_skill dictionary for all evaluated and present skills
    all_keys = list(dict.fromkeys(req_skills + pref_skills + list(last_used.keys())))
    per_skill: dict[str, dict[str, Any]] = {}

    for s in all_keys:
        info = _lookup_skill_info(s)
        ly = info.get("last_used_year")
        src = info.get("source", "undated")
        w = recency_weight(ly, today=today)
        per_skill[s] = {
            "last_used_year": ly,
            "weight": w,
            "source": src,
        }

    # Calculate group means
    req_weights = [per_skill[s]["weight"] for s in req_skills if s in per_skill]
    pref_weights = [per_skill[s]["weight"] for s in pref_skills if s in per_skill]

    req_mean = sum(req_weights) / len(req_weights) if req_weights else 0.0
    pref_mean = sum(pref_weights) / len(pref_weights) if pref_weights else 0.0

    if req_weights and pref_weights:
        final_score = round(0.80 * req_mean + 0.20 * pref_mean, 4)
    elif req_weights:
        final_score = round(req_mean, 4)
    elif pref_weights:
        final_score = round(pref_mean, 4)
    else:
        final_score = 0.0

    return {
        "score": max(0.0, min(1.0, final_score)),
        "per_skill": per_skill,
    }
