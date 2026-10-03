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


# ===========================================================================
# Job Description Analysis
# ===========================================================================

# ---------------------------------------------------------------------------
# Section-heading regexes
# "Required", "Requirements", "Must have", "You must", "Qualifications"
# → everything in this block is a hard requirement.
# ---------------------------------------------------------------------------
_REQUIRED_HEADING = re.compile(
    r"^\s*(?:"
    r"requirements?|required(?: skills?| qualifications?)?|"
    r"must[- ]have|must[- ]haves?|"
    r"you (must|will|should)|"
    r"minimum qualifications?|"
    r"essential skills?|"
    r"what you(?:'ll)? need|"
    r"what we(?:'re)? looking for"
    r")\s*[:\-]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# "Preferred", "Nice to have", "Bonus", "Plus", "Good to have", "Desirable"
# → everything in this block is optional / preferred.
_PREFERRED_HEADING = re.compile(
    r"^\s*(?:"
    r"preferred(?: skills?| qualifications?)?|"
    r"nice[- ]to[- ]have[s]?|"
    r"bonus(?: points?| skills?)?|"
    r"plus(?:es)?|"
    r"good[- ]to[- ]have|"
    r"desirable(?: skills?)?|"
    r"advantageous|"
    r"optional(?: skills?)?"
    r")\s*[:\-]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Any heading that starts a new top-level section (used to detect section end)
_ANY_SECTION_HEADING = re.compile(
    r"^\s*(?:"
    r"about(?: the)?(?: role| job| company| us|position)?|"
    r"responsibilities|duties|role overview|"
    r"requirements?|required|must[- ]have|minimum qualifications?|"
    r"preferred|nice[- ]to[- ]have|bonus|"
    r"benefits?|compensation|salary|"
    r"how to apply|application process|"
    r"what you(?:'ll)? (do|bring|need)|"
    r"what we(?: offer| expect)?|"
    r"who you are|about you"
    r")\s*[:\-]?\s*$",
    re.IGNORECASE,
)

# Inline signals inside bullet/sentence text (used when no headings present)
_REQUIRED_INLINE = re.compile(
    r"\b(required|must\s+have|must\s+know|mandatory|essential|"
    r"you\s+must|minimum|at\s+least|expect(?:ed)?)\b",
    re.IGNORECASE,
)
_PREFERRED_INLINE = re.compile(
    r"\b(preferred?|nice[- ]to[- ]have|bonus|plus|"
    r"desirable|advantage(?:ous)?|ideally|optional|"
    r"good[- ]to[- ]have|familiarity\s+with)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Job title / category inference (rough heuristic from JD title line)
# ---------------------------------------------------------------------------
_TITLE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(machine learning|ml engineer|ai engineer|data scientist|nlp engineer)\b", re.I), "machine_learning"),
    (re.compile(r"\b(data engineer|data analyst|analytics engineer|bi developer)\b", re.I), "data_engineering"),
    (re.compile(r"\b(backend|back-end|back end|api developer|server[- ]side)\b", re.I), "backend"),
    (re.compile(r"\b(frontend|front-end|front end|ui developer|react developer|vue developer)\b", re.I), "frontend"),
    (re.compile(r"\b(full[- ]?stack)\b", re.I), "fullstack"),
    (re.compile(r"\b(devops|site reliability|sre|platform engineer|cloud engineer|infrastructure)\b", re.I), "devops"),
    (re.compile(r"\b(mobile|android|ios|flutter|react native)\b", re.I), "mobile"),
    (re.compile(r"\b(software engineer|software developer|sde|swe)\b", re.I), "software_engineering"),
    (re.compile(r"\b(product manager|pm\b|program manager)\b", re.I), "product_management"),
    (re.compile(r"\b(qa|quality assurance|test engineer|sdet|automation engineer)\b", re.I), "qa_testing"),
]


def _split_jd_into_sections(text: str) -> dict[str, list[str]]:
    """Split JD text into labelled sections: 'required', 'preferred', 'other'.

    Strategy
    --------
    1. Walk through lines.
    2. When a heading regex matches, switch the current active section.
    3. Non-heading lines are accumulated into the active section's bucket.
    4. If *no* section headings are found at all, fall back to inline-signal
       classification: scan each bullet/sentence for required/preferred keywords.

    Returns
    -------
    dict with keys "required", "preferred", "other"; values are lists of text lines.
    """
    sections: dict[str, list[str]] = {"required": [], "preferred": [], "other": []}
    current: str = "other"
    found_any_heading = False

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        if _REQUIRED_HEADING.match(stripped):
            current = "required"
            found_any_heading = True
            continue  # heading line itself is not content
        if _PREFERRED_HEADING.match(stripped):
            current = "preferred"
            found_any_heading = True
            continue
        # Any other top-level section resets to "other"
        if _ANY_SECTION_HEADING.match(stripped) and current in ("required", "preferred"):
            current = "other"
            continue

        sections[current].append(stripped)

    # ── Fallback: no headings detected — classify by inline signals ──────────
    if not found_any_heading:
        reclassified: dict[str, list[str]] = {"required": [], "preferred": [], "other": []}
        for line in sections["other"]:
            if _PREFERRED_INLINE.search(line):
                reclassified["preferred"].append(line)
            elif _REQUIRED_INLINE.search(line):
                reclassified["required"].append(line)
            else:
                # No signal → treat as required (conservative default)
                reclassified["required"].append(line)
        return reclassified

    return sections


def _infer_job_title_and_category(text: str) -> tuple[str | None, str | None]:
    """Guess job title and functional category from the first non-empty lines."""
    # Use the first 3 non-empty lines as the "title zone"
    title_zone = "\n".join(
        line.strip() for line in text.splitlines() if line.strip()
    )[:300]

    detected_category: str | None = None
    for pattern, category in _TITLE_PATTERNS:
        if pattern.search(title_zone):
            detected_category = category
            break

    # Extract the first non-empty line as a raw title guess
    first_line = next(
        (line.strip() for line in text.splitlines() if line.strip()), None
    )
    return first_line, detected_category


class JDAnalysis:
    """Structured result from analyze_job_description().

    Attributes
    ----------
    required_skills : list[str]
        Canonical skills that are hard requirements (must-have).
    preferred_skills : list[str]
        Canonical skills that are nice-to-have / preferred.
    min_experience_years : int | None
        Minimum years of experience, or None if not mentioned.
    education_requirement : list[EducationEntry]
        Detected education requirements (degree level + field).
    job_title : str | None
        First non-empty line of the JD used as a raw title hint.
    job_category : str | None
        Inferred functional category (e.g. 'backend', 'machine_learning').
    """

    __slots__ = (
        "required_skills",
        "preferred_skills",
        "min_experience_years",
        "education_requirement",
        "job_title",
        "job_category",
    )

    def __init__(
        self,
        required_skills: list[str],
        preferred_skills: list[str],
        min_experience_years: int | None,
        education_requirement: list[EducationEntry],
        job_title: str | None,
        job_category: str | None,
    ) -> None:
        self.required_skills = required_skills
        self.preferred_skills = preferred_skills
        self.min_experience_years = min_experience_years
        self.education_requirement = education_requirement
        self.job_title = job_title
        self.job_category = job_category

    def to_dict(self) -> dict:
        """Return a plain dict representation (JSON-serialisable)."""
        return {
            "required_skills": self.required_skills,
            "preferred_skills": self.preferred_skills,
            "min_experience_years": self.min_experience_years,
            "education_requirement": [
                {"degree": e.degree, "field": e.field, "raw": e.raw}
                for e in self.education_requirement
            ],
            "job_title": self.job_title,
            "job_category": self.job_category,
        }

    def __repr__(self) -> str:
        return (
            f"JDAnalysis(title={self.job_title!r}, category={self.job_category!r}, "
            f"required={len(self.required_skills)}, preferred={len(self.preferred_skills)}, "
            f"min_exp={self.min_experience_years}yr)"
        )


def analyze_job_description(jd_text: str) -> JDAnalysis:
    """Parse a job description and extract structured hiring requirements.

    This is the primary entry point for JD analysis. It orchestrates:

    1. **Section splitting** — identifies "Required" vs "Preferred" blocks
       using heading-level regexes. Falls back to inline keyword signals
       ("must have", "nice to have") if no section headings are found.

    2. **Skill extraction** — runs :func:`extract_skills` independently on
       the required and preferred text blocks, so each skill is tagged with
       the correct importance tier.

    3. **Experience parsing** — runs :func:`extract_experience_years` on the
       full JD text (experience is rarely confined to one section).

    4. **Education parsing** — runs :func:`extract_education` on the full
       JD text (same rationale as experience).

    5. **Title / category inference** — lightweight regex scan of the first
       few lines to guess the job family.

    Parameters
    ----------
    jd_text : str
        Raw or lightly cleaned job description text.

    Returns
    -------
    JDAnalysis
        A structured object with the following attributes:

        - ``required_skills``      — canonical skills that are hard requirements.
        - ``preferred_skills``     — canonical skills that are nice-to-have.
        - ``min_experience_years`` — integer or None.
        - ``education_requirement``— list of :class:`EducationEntry`.
        - ``job_title``            — first line of JD (raw title hint).
        - ``job_category``         — inferred functional category string or None.

        Call ``.to_dict()`` for a JSON-serialisable dict.

    Raises
    ------
    TypeError  : if *jd_text* is not a string.
    ValueError : if *jd_text* is empty or blank.
    """
    if not isinstance(jd_text, str):
        raise TypeError(f"Expected str, got {type(jd_text).__name__}.")
    if not jd_text.strip():
        raise ValueError("jd_text must not be empty.")

    # ── Step 1: Split into required / preferred / other sections ────────────
    sections = _split_jd_into_sections(jd_text)

    required_text  = "\n".join(sections["required"])
    preferred_text = "\n".join(sections["preferred"])
    # "other" feeds into required as a catch-all (conservative)
    other_text     = "\n".join(sections["other"])

    # ── Step 2: Extract skills per section ──────────────────────────────────
    # We merge "other" into required (safe default: unknown section = required)
    combined_required = (required_text + "\n" + other_text).strip()

    required_skills  = extract_skills(combined_required) if combined_required else []
    preferred_skills = extract_skills(preferred_text)    if preferred_text    else []

    # A skill found in *both* sections is truly required — remove from preferred
    required_set = set(required_skills)
    preferred_skills = [s for s in preferred_skills if s not in required_set]

    # ── Step 3: Experience + Education from full JD ──────────────────────────
    min_exp  = extract_experience_years(jd_text)
    edu_reqs = extract_education(jd_text)

    # ── Step 4: Title + category inference ──────────────────────────────────
    job_title, job_category = _infer_job_title_and_category(jd_text)

    return JDAnalysis(
        required_skills=required_skills,
        preferred_skills=preferred_skills,
        min_experience_years=min_exp,
        education_requirement=edu_reqs,
        job_title=job_title,
        job_category=job_category,
    )
