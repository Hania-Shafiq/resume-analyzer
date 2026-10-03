"""Skills extraction module for the Resume Analyzer.

Hybrid dictionary + NLP approach
---------------------------------
Why not keywords-only?
  • A pure keyword scan (str.find / re.search) is case-sensitive by default and
    fails on aliases ("js" vs "javascript") unless you maintain a giant list of
    regexes — fragile and slow to update.
  • It cannot understand context: "Python" the country vs "Python" the language.
  • spaCy PhraseMatcher compiles all patterns into a highly optimised Aho-Corasick
    automaton, so matching 5 000 phrases across a 1 000-word resume takes < 5 ms —
    faster than a naive regex loop.

Why not NER only?
  • Generic NER models (even en_core_web_lg) are trained on news corpora, not job
    postings.  They mis-label "Flask" as a person name and miss "k8s" entirely.
  • A curated dictionary gives us *recall control*: every entry is a confirmed skill.

Hybrid advantage
  • Dictionary   → high precision (only real skills are matched).
  • PhraseMatcher → handles aliases, case-insensitivity, and multi-word phrases in
                    a single linear-time pass.
  • spaCy pipeline → gives us sentence boundaries and POS context so future
                     versions can filter "Python Island" (PROPN/LOC) vs "Python"
                     (PROPN used as a tool).
  • Regex + heuristics → best approach for experience years and education degree,
                          because these follow predictable surface patterns.
"""

import json
import re
from pathlib import Path
from typing import NamedTuple

import spacy
from spacy.matcher import PhraseMatcher

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DATA_DIR   = Path(__file__).parent / "data"
_SKILLS_JSON = _DATA_DIR / "skills.json"

# ---------------------------------------------------------------------------
# Module-level singletons (loaded once, reused across calls)
# ---------------------------------------------------------------------------
_nlp:     spacy.language.Language | None = None
_matcher: PhraseMatcher | None = None
# Maps every surface form (lower-case) → canonical skill name
_alias_to_canonical: dict[str, str] = {}
# Maps canonical name → category
_canonical_to_category: dict[str, str] = {}


def _load_resources() -> tuple[spacy.language.Language, PhraseMatcher]:
    """Load spaCy + PhraseMatcher from disk on first call; cache thereafter."""
    global _nlp, _matcher, _alias_to_canonical, _canonical_to_category

    if _nlp is not None and _matcher is not None:
        return _nlp, _matcher

    # ── 1. Load spaCy (disable unused pipes for speed) ──────────────────────
    try:
        _nlp = spacy.load("en_core_web_sm", disable=["ner", "attribute_ruler"])
    except OSError as exc:
        raise OSError(
            "spaCy model 'en_core_web_sm' not found. "
            "Run:  python -m spacy download en_core_web_sm"
        ) from exc

    # ── 2. Load skill taxonomy ───────────────────────────────────────────────
    if not _SKILLS_JSON.exists():
        raise FileNotFoundError(f"Skill taxonomy not found: {_SKILLS_JSON}")

    with _SKILLS_JSON.open(encoding="utf-8") as fh:
        taxonomy = json.load(fh)

    # ── 3. Build alias → canonical + category mappings ──────────────────────
    for category, skills in taxonomy["skills"].items():
        for canonical, meta in skills.items():
            _canonical_to_category[canonical] = category
            # The canonical name itself is a valid surface form
            _alias_to_canonical[canonical.lower()] = canonical
            # Register every alias too
            for alias in meta.get("aliases", []):
                _alias_to_canonical[alias.lower()] = canonical

    # ── 4. Build PhraseMatcher (LOWER attribute → case-insensitive) ──────────
    _matcher = PhraseMatcher(_nlp.vocab, attr="LOWER")
    for surface_form in _alias_to_canonical:
        doc = _nlp.make_doc(surface_form)  # fast tokenisation, no full pipeline
        _matcher.add(surface_form, [doc])

    return _nlp, _matcher


# ===========================================================================
# Public API
# ===========================================================================

def extract_skills(text: str) -> list[str]:
    """Extract canonical skill names from *text* using a hybrid NLP approach.

    Steps
    -----
    1. Load (or reuse) the spaCy model and PhraseMatcher.
    2. Tokenise *text* with spaCy (no parsing/NER needed).
    3. Run PhraseMatcher in a single linear pass — handles case-insensitivity
       and multi-word skills (e.g. "natural language processing") atomically.
    4. Each match is looked up in the alias→canonical table to normalise the name.
    5. Deduplicate while preserving first-occurrence order.

    Parameters
    ----------
    text : str
        Cleaned resume or job-description text.

    Returns
    -------
    list[str]
        Deduplicated list of canonical skill names in order of first appearance.
        Example: ["python", "fastapi", "docker", "machine learning"]

    Raises
    ------
    TypeError  : if *text* is not a string.
    ValueError : if *text* is empty.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}.")
    if not text.strip():
        raise ValueError("text must not be empty.")

    nlp, matcher = _load_resources()

    # Tokenise only — no full pipeline needed for matching
    doc = nlp.make_doc(text)

    # Run the PhraseMatcher; returns list of (match_id, start, end) tuples
    matches = matcher(doc)

    seen: set[str] = set()
    results: list[str] = []

    for _match_id, start, end in matches:
        # The match_id key is the surface form we registered
        surface_form = doc[start:end].text.lower()
        canonical = _alias_to_canonical.get(surface_form)
        if canonical and canonical not in seen:
            seen.add(canonical)
            results.append(canonical)

    return results


def skill_category(canonical_name: str) -> str | None:
    """Return the category for a canonical skill name, or None if not found."""
    _load_resources()
    return _canonical_to_category.get(canonical_name)


# ---------------------------------------------------------------------------
# Education extraction
# ---------------------------------------------------------------------------

# Ordered from most to least specific so the first match wins
_DEGREE_PATTERNS: list[tuple[str, str]] = [
    # PhD
    (r"\b(ph\.?\s*d\.?|doctor(?:ate)?(?: of philosophy)?)\b", "PhD"),
    # Master's
    (r"\b(m\.?\s*(?:sc|s|tech|eng|ba|b\.?a)\.?|master(?:'s)?(?: of (?:science|arts|engineering|technology|business))?|mba)\b", "Master's"),
    # Bachelor's
    (r"\b(b\.?\s*(?:sc|s|tech|eng|ba|e|cs|com)\.?|bachelor(?:'s)?(?: of (?:science|arts|engineering|technology|commerce))?|undergraduate)\b", "Bachelor's"),
    # Associate
    (r"\b(associate(?:'s)?(?: degree)?|a\.?\s*s\.?|a\.?\s*a\.?)\b", "Associate"),
    # High School
    (r"\b(high school diploma|secondary school|hssc|matric(?:ulation)?|a-levels?|o-levels?|ssc)\b", "High School"),
]

_COMPILED_DEGREE = [(re.compile(pat, re.IGNORECASE), label) for pat, label in _DEGREE_PATTERNS]

# Common fields of study to extract alongside degree
_FIELD_PATTERN = re.compile(
    r"\b(?:in|of)\s+([A-Z][a-zA-Z\s&]+?)(?:\s*(?:from|at|,|\.|\n|$))",
    re.MULTILINE,
)


class EducationEntry(NamedTuple):
    """Parsed education record."""

    degree: str        # e.g. "Bachelor's", "Master's", "PhD"
    field: str | None  # e.g. "Computer Science", "Data Science"
    raw: str           # the matched sentence / line for debugging


def extract_education(text: str) -> list[EducationEntry]:
    """Extract education degree entries from resume text.

    Uses an ordered set of regex patterns (most specific first) to detect
    degree level, then looks ahead in the same sentence for a field of study.

    Parameters
    ----------
    text : str
        Cleaned resume text.

    Returns
    -------
    list[EducationEntry]
        One entry per detected degree. Deduplicated by degree level.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}.")

    seen_degrees: set[str] = set()
    results: list[EducationEntry] = []

    # Work line by line for accuracy — degrees are usually on their own line
    for line in text.splitlines():
        line_stripped = line.strip()
        if not line_stripped:
            continue

        for compiled_pat, degree_label in _COMPILED_DEGREE:
            if compiled_pat.search(line_stripped):
                if degree_label in seen_degrees:
                    break  # already captured this degree level

                # Try to find field of study in the same line
                field_match = _FIELD_PATTERN.search(line_stripped)
                field = field_match.group(1).strip() if field_match else None

                results.append(EducationEntry(
                    degree=degree_label,
                    field=field,
                    raw=line_stripped,
                ))
                seen_degrees.add(degree_label)
                break  # stop after first matching degree level for this line

    return results


# ---------------------------------------------------------------------------
# Experience years extraction
# ---------------------------------------------------------------------------

# Patterns that express years of experience in job postings / resume summaries
_EXP_PATTERNS: list[re.Pattern] = [re.compile(pat, re.IGNORECASE) for pat in [
    # "5+ years of experience", "5 years experience"
    r"(\d+)\+?\s*(?:to\s*\d+)?\s*years?\s*(?:of\s*)?(?:professional\s*)?experience",
    # "over 3 years", "more than 7 years"
    r"(?:over|more than|at least|nearly)\s*(\d+)\s*years?",
    # "experience of 4+ years"
    r"experience\s*(?:of\s*)?(\d+)\+?\s*years?",
    # Range: "3-5 years"
    r"(\d+)\s*[-–]\s*(\d+)\s*years?",
]]


def extract_experience_years(text: str) -> int | None:
    """Estimate total years of professional experience mentioned in *text*.

    Scans for common patterns such as:
      - "5+ years of experience"
      - "over 3 years"
      - "3-5 years of experience"  → returns the lower bound (3)

    If multiple mentions are found, returns the **maximum** to be conservative
    (the candidate is likely quoting total years across sections).

    Parameters
    ----------
    text : str
        Cleaned resume text.

    Returns
    -------
    int | None
        Estimated years as an integer, or None if no pattern matched.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}.")

    found_years: list[int] = []

    for pat in _EXP_PATTERNS:
        for m in pat.finditer(text):
            # Group 1 is always the primary year number
            try:
                found_years.append(int(m.group(1)))
            except (IndexError, ValueError):
                pass

    return max(found_years) if found_years else None
