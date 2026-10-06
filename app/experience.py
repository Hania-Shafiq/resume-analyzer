"""Experience extraction and classification module.

Extracts professional work experience and freelancing experience separately,
parses varied date formats, merges overlapping calendar date intervals, and
formats durations as 'X yrs Y mos'.
"""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DateInterval:
    """A calendar date interval represented in months from year 0."""
    start_year: int
    start_month: int
    end_year: int
    end_month: int

    @property
    def start_index(self) -> int:
        return self.start_year * 12 + (self.start_month - 1)

    @property
    def end_index(self) -> int:
        # Inclusive end month (+1 index)
        return self.end_year * 12 + self.end_month


@dataclass
class ExperienceEntry:
    title: str = ""
    company: str = ""
    start_date_str: str = ""
    end_date_str: str = ""
    interval: Optional[DateInterval] = None
    duration_months: int = 0
    bucket: str = "professional"  # "professional" | "freelance"
    is_internship: bool = False
    confidence: str = "high"  # "high" | "low"
    raw_text: str = ""


@dataclass
class ExperienceSummary:
    professional_months: int = 0
    freelance_months: int = 0
    professional_str: str = "Not indicated"
    freelance_str: str = "Not indicated"
    combined_str: Optional[str] = None
    has_internships: bool = False
    internship_note: Optional[str] = None
    entries: List[ExperienceEntry] = field(default_factory=list)
    has_any_experience: bool = False

    def to_dict(self) -> dict:
        return {
            "professional_months": self.professional_months,
            "freelance_months": self.freelance_months,
            "professional_str": self.professional_str,
            "freelance_str": self.freelance_str,
            "combined_str": self.combined_str,
            "has_internships": self.has_internships,
            "internship_note": self.internship_note,
            "has_any_experience": self.has_any_experience,
            "entries_count": len(self.entries),
        }


# ---------------------------------------------------------------------------
# Constants & Regex Patterns
# ---------------------------------------------------------------------------

MONTH_MAP = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

FREELANCE_KEYWORDS = [
    "freelance",
    "freelancer",
    "freelancing",
    "self-employed",
    "self employed",
    "independent contractor",
    "contractor",
    "fiverr",
    "upwork",
    "toptal",
    "consultant (self)",
    "self consultant",
    "independent consultant",
    "client project",
    "client work",
]

INTERNSHIP_KEYWORDS = [
    "intern",
    "internship",
    "trainee",
    "graduate trainee",
    "apprentice",
]

SECTION_HEADERS = [
    "work experience",
    "professional experience",
    "experience",
    "employment history",
    "work history",
    "career history",
    "employment",
    "projects",
    "freelance experience",
    "freelance work",
]

NEXT_SECTION_HEADERS = [
    "education",
    "skills",
    "technical skills",
    "certifications",
    "licenses",
    "publications",
    "awards",
    "languages",
    "interests",
    "references",
    "summary",
    "profile",
]

# Date range regexes
# 1) Month Year – Month Year (e.g. Jan 2021 – Mar 2023, January 2021 to Present)
_MONTHS_PATTERN = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
_SEP_PATTERN = r"(?:\s*[-–—/to]+\s*)"
_PRESENT_PATTERN = r"(?:present|current|now|till\s*date|ongoing|active)"

RE_MONTH_YEAR_RANGE = re.compile(
    rf"\b({_MONTHS_PATTERN})\s+['’]?([12]\d{{3}})\s*{_SEP_PATTERN}\s*({_MONTHS_PATTERN}\s+['’]?[12]\d{{3}}|{_PRESENT_PATTERN})\b",
    re.IGNORECASE,
)

# 2) MM/YYYY - MM/YYYY (e.g. 01/2021 – 03/2023 or 1/2021 - Present)
RE_NUM_MONTH_RANGE = re.compile(
    rf"\b(0?[1-9]|1[0-2])[/-]([12]\d{{3}})\s*{_SEP_PATTERN}\s*((?:0?[1-9]|1[0-2])[/-][12]\d{{3}}|{_PRESENT_PATTERN})\b",
    re.IGNORECASE,
)

# 3) YYYY - YYYY (e.g. 2020 – 2023 or 2021 - Present)
RE_YEAR_RANGE = re.compile(
    rf"\b([12]\d{{3}})\s*{_SEP_PATTERN}\s*([12]\d{{3}}|{_PRESENT_PATTERN})\b",
    re.IGNORECASE,
)

# 4) Explicit duration strings: "2 years", "6 months", "1 year 4 months"
RE_EXPLICIT_DURATION = re.compile(
    r"\b(?:(\d+)\s*(?:years?|yrs?))?\s*(?:and\s*)?(?:(\d+)\s*(?:months?|mos?))?\b",
    re.IGNORECASE,
)

# Fallback summary statements: "3+ years of experience"
_EXP_SUMMARY_PATTERNS = [
    re.compile(r"(\d+)\+?\s*(?:to\s*\d+)?\s*years?\s*(?:of\s*)?(?:professional\s*)?experience", re.IGNORECASE),
    re.compile(r"(?:over|more than|at least|nearly)\s*(\d+)\s*years?", re.IGNORECASE),
    re.compile(r"experience\s*(?:of\s*)?(\d+)\+?\s*years?", re.IGNORECASE),
    re.compile(r"(\d+)\s*[-–]\s*(\d+)\s*years?\s*(?:of\s*)?experience", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _parse_month(m_str: str) -> int:
    m_clean = m_str.strip().lower()
    return MONTH_MAP.get(m_clean, 1)


def parse_date_endpoints(start_str: str, end_str: str, now: Optional[datetime.date] = None) -> Optional[DateInterval]:
    """Parse start and end strings into a DateInterval."""
    if now is None:
        now = datetime.date.today()

    cur_year = now.year
    cur_month = now.month

    # End date handling
    end_str_clean = end_str.strip().lower()
    is_present = bool(re.search(_PRESENT_PATTERN, end_str_clean))

    # Match Month Year
    m_my_start = re.match(rf"^({_MONTHS_PATTERN})\s+['’]?([12]\d{{3}})$", start_str.strip(), re.IGNORECASE)
    if m_my_start:
        sy = int(m_my_start.group(2))
        sm = _parse_month(m_my_start.group(1))
        if is_present:
            ey, em = cur_year, cur_month
        else:
            m_my_end = re.match(rf"^({_MONTHS_PATTERN})\s+['’]?([12]\d{{3}})$", end_str.strip(), re.IGNORECASE)
            if m_my_end:
                ey = int(m_my_end.group(2))
                em = _parse_month(m_my_end.group(1))
            else:
                ey, em = sy, sm
        if (ey > sy) or (ey == sy and em >= sm):
            return DateInterval(sy, sm, ey, em)
        return None

    # Match MM/YYYY
    m_num_start = re.match(r"^(0?[1-9]|1[0-2])[/-]([12]\d{3})$", start_str.strip())
    if m_num_start:
        sm = int(m_num_start.group(1))
        sy = int(m_num_start.group(2))
        if is_present:
            ey, em = cur_year, cur_month
        else:
            m_num_end = re.match(r"^(0?[1-9]|1[0-2])[/-]([12]\d{3})$", end_str.strip())
            if m_num_end:
                em = int(m_num_end.group(1))
                ey = int(m_num_end.group(2))
            else:
                ey, em = sy, sm
        if (ey > sy) or (ey == sy and em >= sm):
            return DateInterval(sy, sm, ey, em)
        return None

    # Match YYYY
    m_y_start = re.match(r"^([12]\d{3})$", start_str.strip())
    if m_y_start:
        sy = int(m_y_start.group(1))
        sm = 1
        if is_present:
            ey, em = cur_year, cur_month
        else:
            m_y_end = re.match(r"^([12]\d{3})$", end_str.strip())
            if m_y_end:
                ey = int(m_y_end.group(1))
                em = 12
            else:
                ey, em = sy, 12
        if (ey > sy) or (ey == sy and em >= sm):
            return DateInterval(sy, sm, ey, em)
        return None

    return None


def merge_intervals(intervals: List[DateInterval]) -> int:
    """Merge overlapping intervals and return total duration in months."""
    if not intervals:
        return 0

    ranges = sorted([(iv.start_index, iv.end_index) for iv in intervals], key=lambda x: x[0])
    merged = []
    cur_start, cur_end = ranges[0]

    for s, e in ranges[1:]:
        if s <= cur_end:
            cur_end = max(cur_end, e)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    merged.append((cur_start, cur_end))

    total_months = 0
    for s, e in merged:
        # e - s is number of months covered
        duration = max(1, e - s)
        total_months += duration

    return total_months


def format_duration(total_months: int) -> str:
    """Format months count into 'X yrs Y mos' string."""
    if total_months <= 0:
        return "None"

    years = total_months // 12
    months = total_months % 12

    if years > 0 and months > 0:
        return f"{years} yr{'s' if years != 1 else ''} {months} mo{'s' if months != 1 else ''}"
    elif years > 0 and months == 0:
        return f"{years} yr{'s' if years != 1 else ''}"
    else:
        return f"{months} mo{'s' if months != 1 else ''}"


def classify_bucket(text_context: str) -> Tuple[str, bool, str]:
    """Classify an experience entry into professional vs freelance.

    Returns:
        (bucket: "professional" | "freelance", is_internship: bool, confidence: "high" | "low")
    """
    lower = text_context.lower()

    # Check for freelance indicators
    is_freelance = any(kw in lower for kw in FREELANCE_KEYWORDS)
    is_internship = any(kw in lower for kw in INTERNSHIP_KEYWORDS)

    if is_freelance:
        return "freelance", is_internship, "high"

    # Default is professional
    # If ambiguous (no clear company indicators or vague phrasing): flag low confidence
    has_company_clue = any(s in text_context for s in ["|", " at ", "@", ",", "Inc", "LLC", "Ltd", "Corp", "Company", "Technologies", "Studio", "Lab"])
    confidence = "high" if has_company_clue else "low"

    return "professional", is_internship, confidence


# ---------------------------------------------------------------------------
# Section and Entry Extraction
# ---------------------------------------------------------------------------

def extract_experience(text: str, now: Optional[datetime.date] = None) -> ExperienceSummary:
    """Extract, classify, and calculate Professional and Freelance experience."""
    if not isinstance(text, str) or not text.strip():
        return ExperienceSummary()

    lines = text.splitlines()
    entries: List[ExperienceEntry] = []

    # Step 1: Scan lines for date ranges
    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue

        # Try Month Year Range
        m_my = RE_MONTH_YEAR_RANGE.search(line_clean)
        if m_my:
            s_str, e_str = m_my.group(1) + " " + m_my.group(2), m_my.group(3)
            interval = parse_date_endpoints(s_str, e_str, now=now)
            if interval:
                bucket, is_intern, conf = classify_bucket(line_clean)
                parts = [p.strip() for p in line_clean.split("|") if p.strip()]
                title = parts[0] if parts else line_clean
                company = parts[1] if len(parts) > 1 else ""
                entries.append(ExperienceEntry(
                    title=title,
                    company=company,
                    start_date_str=s_str,
                    end_date_str=e_str,
                    interval=interval,
                    bucket=bucket,
                    is_internship=is_intern,
                    confidence=conf,
                    raw_text=line_clean,
                ))
                continue

        # Try MM/YYYY Range
        m_num = RE_NUM_MONTH_RANGE.search(line_clean)
        if m_num:
            s_str, e_str = m_num.group(1) + "/" + m_num.group(2), m_num.group(3)
            interval = parse_date_endpoints(s_str, e_str, now=now)
            if interval:
                bucket, is_intern, conf = classify_bucket(line_clean)
                parts = [p.strip() for p in line_clean.split("|") if p.strip()]
                title = parts[0] if parts else line_clean
                company = parts[1] if len(parts) > 1 else ""
                entries.append(ExperienceEntry(
                    title=title,
                    company=company,
                    start_date_str=s_str,
                    end_date_str=e_str,
                    interval=interval,
                    bucket=bucket,
                    is_internship=is_intern,
                    confidence=conf,
                    raw_text=line_clean,
                ))
                continue

        # Try YYYY Range
        m_yr = RE_YEAR_RANGE.search(line_clean)
        if m_yr:
            s_str, e_str = m_yr.group(1), m_yr.group(2)
            interval = parse_date_endpoints(s_str, e_str, now=now)
            if interval:
                bucket, is_intern, conf = classify_bucket(line_clean)
                parts = [p.strip() for p in line_clean.split("|") if p.strip()]
                title = parts[0] if parts else line_clean
                company = parts[1] if len(parts) > 1 else ""
                entries.append(ExperienceEntry(
                    title=title,
                    company=company,
                    start_date_str=s_str,
                    end_date_str=e_str,
                    interval=interval,
                    bucket=bucket,
                    is_internship=is_intern,
                    confidence=conf,
                    raw_text=line_clean,
                ))
                continue

    # Step 2: Separate entries by bucket and merge overlapping ranges
    prof_intervals = [e.interval for e in entries if e.bucket == "professional" and e.interval]
    free_intervals = [e.interval for e in entries if e.bucket == "freelance" and e.interval]

    prof_months = merge_intervals(prof_intervals)
    free_months = merge_intervals(free_intervals)

    has_internships = any(e.is_internship for e in entries)

    # Step 3: Fallback check if no dated intervals were found
    if prof_months == 0 and free_months == 0:
        found_fallback_years = []
        for pat in _EXP_SUMMARY_PATTERNS:
            for m in pat.finditer(text):
                try:
                    found_fallback_years.append(int(m.group(1)))
                except (IndexError, ValueError):
                    pass

        if found_fallback_years:
            fb_years = max(found_fallback_years)
            # Default fallback to professional
            prof_months = fb_years * 12
            entries.append(ExperienceEntry(
                title="Professional Experience (summary)",
                duration_months=prof_months,
                bucket="professional",
                confidence="low",
                raw_text=f"{fb_years} years of experience",
            ))

    # Step 4: Format strings
    has_any = (prof_months > 0) or (free_months > 0) or bool(entries)
    if not has_any:
        # Check if the document mentions experience section without valid dates/numbers
        has_sec = any(h in text.lower() for h in SECTION_HEADERS)
        if has_sec:
            prof_str = "None"
            free_str = "None"
        else:
            prof_str = "Not indicated"
            free_str = "Not indicated"
    else:
        prof_str = format_duration(prof_months)
        free_str = format_duration(free_months)

    intern_note = "Includes internship experience" if has_internships else None

    # Optional combined secondary figure
    total_months = prof_months + free_months
    combined_str = format_duration(total_months) if (prof_months > 0 and free_months > 0) else None

    return ExperienceSummary(
        professional_months=prof_months,
        freelance_months=free_months,
        professional_str=prof_str,
        freelance_str=free_str,
        combined_str=combined_str,
        has_internships=has_internships,
        internship_note=intern_note,
        entries=entries,
        has_any_experience=has_any,
    )
