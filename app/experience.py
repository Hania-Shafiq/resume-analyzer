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
    full_time_months: int = 0
    internship_months: int = 0
    professional_str: str = "Not indicated"
    freelance_str: str = "Not indicated"
    full_time_str: str = "Not indicated"
    internship_str: str = "Not indicated"
    combined_str: Optional[str] = None
    has_internships: bool = False
    internship_note: Optional[str] = None
    entries: List[ExperienceEntry] = field(default_factory=list)
    has_any_experience: bool = False

    def to_dict(self) -> dict:
        return {
            "professional_months": self.professional_months,
            "freelance_months": self.freelance_months,
            "full_time_months": self.full_time_months,
            "internship_months": self.internship_months,
            "professional_str": self.professional_str,
            "freelance_str": self.freelance_str,
            "full_time_str": self.full_time_str,
            "internship_str": self.internship_str,
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

EDUCATION_SECTION_HEADERS = [
    "education",
    "academic background",
    "academics",
    "qualifications",
    "academic qualifications",
    "educational background",
    "education and certifications",
    "education & certifications",
    "education and training",
]

_ALL_SECTION_HEADERS = set(
    SECTION_HEADERS + NEXT_SECTION_HEADERS + EDUCATION_SECTION_HEADERS + [
        "academic background", "education & certifications", "work", "job history"
    ]
)

_EDU_DEGREE_ABBR = r"(?<![a-zA-Z0-9])(?:b\.?\s*s\.?(?:c)?|b\.?\s*e\.?|m\.?\s*s\.?(?:c)?|m\.?\s*e\.?|ph\.?\s*d\.?)(?![a-zA-Z0-9])"
_EDU_KEYWORDS = r"\b(?:university|college|institute|bachelor(?:'s)?|master(?:'s)?|doctorate|diploma)\b"
EDU_LINE_PATTERN = re.compile(rf"{_EDU_DEGREE_ABBR}|{_EDU_KEYWORDS}", re.IGNORECASE)

COMMON_TITLE_KEYWORDS = {
    "engineer", "developer", "architect", "manager", "lead", "director",
    "consultant", "analyst", "designer", "specialist", "intern", "trainee",
    "associate", "officer", "administrator", "scientist", "programmer",
    "coordinator", "head", "vp", "president", "technician",
}

# Date range regexes
# 1) Month Year – Month Year (e.g. Jan 2021 – Mar 2023, January 2021 to Present)
_MONTHS_PATTERN = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
_SEP_PATTERN = r"(?:\s*(?:[-–—/]|to|until|till)+\s*)"
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

# Unified endpoint and mixed range regexes
_DATE_ENDPOINT_PATTERN = rf"(?:{_MONTHS_PATTERN}\s+['’]?[12]\d{{3}}|(?:0?[1-9]|1[0-2])[/-][12]\d{{3}}|[12]\d{{3}})"
RE_COMBINED_DATE_RANGE = re.compile(
    rf"\b({_DATE_ENDPOINT_PATTERN})\s*{_SEP_PATTERN}\s*({_DATE_ENDPOINT_PATTERN}|{_PRESENT_PATTERN})\b",
    re.IGNORECASE,
)

# 4) Since / From (e.g. Since 2022, Since Jan 2021)
RE_SINCE_RANGE = re.compile(
    rf"\b(?:since|from)\s+(?:({_MONTHS_PATTERN})\s+['’]?([12]\d{{3}})|(0?[1-9]|1[0-2])[/-]([12]\d{{3}})|([12]\d{{3}}))(?!\s*(?:[-–—/]|to|until|till)\s*(?:[12]\d{{3}}|{_MONTHS_PATTERN}|{_PRESENT_PATTERN}))\b",
    re.IGNORECASE,
)

# 5) Explicit duration strings: "2 years", "6 months", "1 year 4 months"
RE_EXPLICIT_DURATION = re.compile(
    r"\b(?:(\d+)\s*(?:years?|yrs?))?\s*(?:and\s*)?(?:(\d+)\s*(?:months?|mos?))?\b",
    re.IGNORECASE,
)

# Fallback summary statements: "3+ years of experience"
_EXP_SUMMARY_PATTERNS = [
    re.compile(r"(\d{1,2})\+?\s*(?:to\s*\d{1,2})?\s*(?:years?|yrs?|yr)\s*(?:of\s*)?(?:professional|full[- ]time|relevant|industry)?\s*experience", re.IGNORECASE),
    re.compile(r"(?:over|more than|at least|nearly|minimum|min\.?)\s*(\d{1,2})\+?\s*(?:years?|yrs?|yr)", re.IGNORECASE),
    re.compile(r"\bexperience\b[*_]*\s*[:\-–]?\s*[*_]*\s*(?:of\s*)?(\d{1,2})\+?\s*(?:years?|yrs?|yr)?", re.IGNORECASE),
    re.compile(r"(\d{1,2})\s*[-–]\s*(\d{1,2})\s*(?:years?|yrs?|yr)\s*(?:of\s*)?experience", re.IGNORECASE),
    re.compile(r"(\d{1,2})\+?\s*(?:years?|yrs?|yr)\b\s*(?:of\s*)?[a-zA-Z\s&/\-]{0,30}experience", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _parse_month(m_str: str) -> int:
    m_clean = m_str.strip().lower()
    return MONTH_MAP.get(m_clean, 1)


def _is_section_header(line: str) -> Optional[str]:
    """Check if a line looks like a known section header."""
    norm = re.sub(r"^[#*\-=_:\s\d\.]+|[#*\-=_:\s\d\.]+$", "", line).strip().lower()
    if not norm:
        return None
    for h in EDUCATION_SECTION_HEADERS:
        if norm == h or norm.startswith(f"{h}:"):
            return "education"
    for h in SECTION_HEADERS:
        if norm == h or norm.startswith(f"{h}:"):
            return "experience"
    for h in NEXT_SECTION_HEADERS:
        if norm == h or norm.startswith(f"{h}:"):
            return "other"
    if norm in _ALL_SECTION_HEADERS:
        return "other"
    return None


def _is_date_only_line(line: str, matched_str: str) -> bool:
    """Check if the line contains only a date range (ignoring bullets, spaces, delimiters)."""
    rem = line.replace(matched_str, "")
    rem_clean = re.sub(r"[\s\-\–\—\•\*\:\,\(\)\[\]\|\./]", "", rem)
    return len(rem_clean) == 0


def _is_date_line(line: str) -> bool:
    """Check if a line matches any date range pattern."""
    return bool(
        RE_MONTH_YEAR_RANGE.search(line)
        or RE_NUM_MONTH_RANGE.search(line)
        or RE_YEAR_RANGE.search(line)
        or RE_SINCE_RANGE.search(line)
    )


def _extract_title_company_from_preceding_lines(
    lines: List[str], current_idx: int
) -> Tuple[str, str]:
    """Scan backwards up to 2 non-empty lines for job title and company."""
    prev_lines = []
    for j in range(current_idx - 1, -1, -1):
        cand = lines[j].strip()
        if not cand:
            continue
        if _is_section_header(cand) or _is_date_line(cand):
            break
        prev_lines.append(cand)
        if len(prev_lines) == 2:
            break

    if not prev_lines:
        return "", ""

    if len(prev_lines) == 1:
        p0 = prev_lines[0]
        if "|" in p0:
            parts = [p.strip() for p in p0.split("|") if p.strip()]
            return (parts[0], parts[1] if len(parts) > 1 else "")
        elif " at " in p0:
            parts = p0.split(" at ", 1)
            return (parts[0].strip(), parts[1].strip())
        elif ", " in p0:
            parts = p0.split(", ", 1)
            return (parts[0].strip(), parts[1].strip())
        else:
            return (p0, "")

    # len(prev_lines) == 2: prev_lines[0] is line -1, prev_lines[1] is line -2
    p_minus_1 = prev_lines[0]
    p_minus_2 = prev_lines[1]

    if "|" in p_minus_1:
        parts = [p.strip() for p in p_minus_1.split("|") if p.strip()]
        return (parts[0], parts[1] if len(parts) > 1 else "")
    elif " at " in p_minus_1:
        parts = p_minus_1.split(" at ", 1)
        return (parts[0].strip(), parts[1].strip())

    p1_is_title = any(w in p_minus_1.lower() for w in COMMON_TITLE_KEYWORDS)
    p2_is_title = any(w in p_minus_2.lower() for w in COMMON_TITLE_KEYWORDS)

    if p1_is_title and not p2_is_title:
        return (p_minus_1, p_minus_2)
    else:
        return (p_minus_2, p_minus_1)


def _parse_single_date(d_str: str, is_end: bool, cur_year: int, cur_month: int) -> Optional[Tuple[int, int]]:
    """Parse a single date endpoint string into (year, month)."""
    cleaned = d_str.strip()
    if not cleaned:
        return None
    if is_end and re.search(_PRESENT_PATTERN, cleaned.lower()):
        return (cur_year, cur_month)
    # 1. Month Year: e.g. "Jan 2021", "January 2021", "Jan '21"
    m_my = re.match(rf"^({_MONTHS_PATTERN})\s+['’]?([12]\d{{3}})$", cleaned, re.IGNORECASE)
    if m_my:
        return (int(m_my.group(2)), _parse_month(m_my.group(1)))
    # 2. MM/YYYY or MM-YYYY
    m_num = re.match(r"^(0?[1-9]|1[0-2])[/-]([12]\d{3})$", cleaned)
    if m_num:
        return (int(m_num.group(2)), int(m_num.group(1)))
    # 3. YYYY: e.g. "2021"
    m_y = re.match(r"^([12]\d{3})$", cleaned)
    if m_y:
        return (int(m_y.group(1)), 12 if is_end else 1)
    return None


def parse_date_endpoints(start_str: str, end_str: str, now: Optional[datetime.date] = None) -> Optional[DateInterval]:
    """Parse start and end strings into a DateInterval."""
    if now is None:
        now = datetime.date.today()

    cur_year = now.year
    cur_month = now.month

    # Handle 'Since YYYY' or 'From YYYY'
    start_str_clean = re.sub(r"^(?:since|from)\s+", "", start_str.strip(), flags=re.IGNORECASE).strip()
    if not end_str or not end_str.strip():
        end_str = "Present"
    start_str = start_str_clean

    start_parsed = _parse_single_date(start_str, is_end=False, cur_year=cur_year, cur_month=cur_month)
    end_parsed = _parse_single_date(end_str, is_end=True, cur_year=cur_year, cur_month=cur_month)

    if start_parsed is None:
        return None

    sy, sm = start_parsed
    if end_parsed is None:
        ey, em = (sy, 12) if re.match(r"^([12]\d{3})$", start_str.strip()) else (sy, sm)
    else:
        ey, em = end_parsed

    if (ey > sy) or (ey == sy and em >= sm):
        return DateInterval(sy, sm, ey, em)
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
    current_section = "other"

    # Step 1: Scan lines for date ranges
    for idx, line in enumerate(lines):
        line_clean = line.strip()
        if not line_clean:
            continue

        sec_header = _is_section_header(line_clean)
        if sec_header is not None:
            current_section = sec_header
            continue

        # Look for date ranges in this line
        matched = False
        s_str, e_str, match_text = "", "", ""

        # 1) Combined & Mixed Date Range (handles Month-Year, MM/YYYY, and YYYY combinations)
        m_comb = RE_COMBINED_DATE_RANGE.search(line_clean)
        if m_comb:
            s_str = m_comb.group(1)
            e_str = m_comb.group(2)
            match_text = m_comb.group(0)
            matched = True
        else:
            # 2) Since / From Range (e.g. Since 2022)
            m_since = RE_SINCE_RANGE.search(line_clean)
            if m_since:
                if m_since.group(1) and m_since.group(2):
                    s_str = m_since.group(1) + " " + m_since.group(2)
                elif m_since.group(3) and m_since.group(4):
                    s_str = m_since.group(3) + "/" + m_since.group(4)
                else:
                    s_str = m_since.group(5)
                e_str = "Present"
                match_text = m_since.group(0)
                matched = True

        if not matched:
            continue

        # Education filtering:
        # If in an education section, or line contains degree/university words: skip
        if current_section == "education" or EDU_LINE_PATTERN.search(line_clean):
            continue

        interval = parse_date_endpoints(s_str, e_str, now=now)
        if not interval:
            continue

        is_date_only = _is_date_only_line(line_clean, match_text)

        if is_date_only:
            title, company = _extract_title_company_from_preceding_lines(lines, idx)
            # If the inferred title or company has degree/university words, skip
            if EDU_LINE_PATTERN.search(title) or EDU_LINE_PATTERN.search(company):
                continue
            if not title:
                title = line_clean
                company = ""
            context = f"{title} | {company} | {line_clean}"
        else:
            if "|" in line_clean:
                parts = [p.strip() for p in line_clean.split("|") if p.strip()]
                title = parts[0] if parts else line_clean
                company = parts[1] if len(parts) > 1 else ""
            elif " at " in line_clean:
                clean_no_date = line_clean.replace(match_text, "").strip(" -–—,()")
                if " at " in clean_no_date:
                    parts = clean_no_date.split(" at ", 1)
                    title = parts[0].strip()
                    company = parts[1].strip()
                else:
                    title = clean_no_date or line_clean
                    company = ""
            else:
                title = line_clean
                company = ""
            context = line_clean

        bucket, is_intern, conf = classify_bucket(context)
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

    # Step 2: Separate entries by bucket and merge overlapping ranges
    prof_intervals = [e.interval for e in entries if e.bucket == "professional" and e.interval]
    free_intervals = [e.interval for e in entries if e.bucket == "freelance" and e.interval]
    full_time_intervals = [e.interval for e in entries if e.bucket == "professional" and not e.is_internship and e.interval]
    internship_intervals = [e.interval for e in entries if e.is_internship and e.interval]

    prof_months = merge_intervals(prof_intervals)
    free_months = merge_intervals(free_intervals)
    full_time_months = merge_intervals(full_time_intervals)
    internship_months = merge_intervals(internship_intervals)

    has_internships = any(e.is_internship for e in entries)

    # Step 3: Fallback check if no dated intervals were found
    if prof_months == 0 and free_months == 0:
        found_fallback_years = []
        cleaned_text = re.sub(r"[*_#`~]", " ", text)
        for pat in _EXP_SUMMARY_PATTERNS:
            for m in pat.finditer(cleaned_text):
                try:
                    found_fallback_years.append(int(m.group(1)))
                except (IndexError, ValueError):
                    pass

        if found_fallback_years:
            fb_years = max(found_fallback_years)
            # Default fallback to professional
            prof_months = fb_years * 12
            full_time_months = fb_years * 12
            entries.append(ExperienceEntry(
                title="Professional Experience (summary)",
                duration_months=prof_months,
                bucket="professional",
                confidence="low",
                raw_text=f"{fb_years} years of experience",
            ))

    # Step 4: Format strings
    has_any = (prof_months > 0) or (free_months > 0) or bool(entries)
    has_sec = any(h in text.lower() for h in SECTION_HEADERS)
    if not has_any:
        # Check if the document mentions experience section without valid dates/numbers
        if has_sec:
            prof_str = "None"
            free_str = "None"
        else:
            prof_str = "Not indicated"
            free_str = "Not indicated"
    else:
        prof_str = format_duration(prof_months)
        free_str = format_duration(free_months)

    full_time_str = format_duration(full_time_months) if full_time_months > 0 else ("None" if has_sec else "Not indicated")
    internship_str = format_duration(internship_months) if internship_months > 0 else ("None" if has_sec else "Not indicated")
    intern_note = "Includes internship experience" if has_internships else None

    # Optional combined secondary figure
    total_months = prof_months + free_months
    combined_str = format_duration(total_months) if (prof_months > 0 and free_months > 0) else None

    return ExperienceSummary(
        professional_months=prof_months,
        freelance_months=free_months,
        full_time_months=full_time_months,
        internship_months=internship_months,
        professional_str=prof_str,
        freelance_str=free_str,
        full_time_str=full_time_str,
        internship_str=internship_str,
        combined_str=combined_str,
        has_internships=has_internships,
        internship_note=intern_note,
        entries=entries,
        has_any_experience=has_any,
    )
