"""Blind-screening anonymizer module for resume redaction.

Redaction is heuristic and does not guarantee anonymity. Location redaction
can hide legitimate right-to-work information, which is why it is configurable
and can be enabled or disabled via options.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Default configuration and lists
# ---------------------------------------------------------------------------

DEFAULT_OPTIONS: Dict[str, Any] = {
    "email": True,
    "phone": True,
    "link": True,
    "name": True,
    "pronoun": True,
    "year": True,
    "location": True,
    "personal": True,
}

COMMON_LOCATIONS: Set[str] = {
    # Pakistani major cities
    "karachi", "lahore", "islamabad", "rawalpindi", "faisalabad",
    "peshawar", "quetta", "multan", "sialkot", "gujranwala",
    "hyderabad", "sukkur", "bahawalpur", "sargodha", "abbottabad",
    "mirpur", "gilgit", "gwadar", "pakistan",
    # Common international countries & tech hubs
    "united states", "usa", "us", "united kingdom", "uk", "canada",
    "australia", "germany", "france", "uae", "united arab emirates",
    "dubai", "abu dhabi", "saudi arabia", "qatar", "singapore",
    "london", "new york", "san francisco", "seattle", "toronto",
    "vancouver", "berlin", "tokyo", "sydney", "melbourne", "california",
    "texas", "washington", "ontario",
}

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

_PHONE_PATTERN = re.compile(
    r"""(?x)
    (?<![\w\+])
    (?:
        # Pakistani +92 or 0092
        (?:\+92|0092)[\s.-]?(?:\(0\))?[\s.-]?\(?3\d{2}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b
        |
        # Pakistani 03xx
        (?:\(03\d{2}\)|03\d{2})[\s.-]?\d{3}[\s.-]?\d{4}\b
        |
        (?:\(03\d{2}\)|03\d{2})[\s.-]?\d{7}\b
        |
        # International formats
        \+[1-9]\d{0,2}[\s.-]?(?:\(\d{1,4}\)|\d{1,4})[\s.-]?(?:\d{1,4}[\s.-]?){1,3}\d{2,4}\b
    )
    """
)

_URL_PATTERN = re.compile(
    r"""(?xi)
    (?<![@\w])
    (?:
        https?://[^\s<>'"()]+
        |
        www\.[^\s<>'"()]+
        |
        \b(?:linkedin\.com|github\.com)/[^\s<>'"()]+
        |
        \b(?:[a-zA-Z0-9-]+\.)+(?:com|org|net|io|dev|me|ai|co|app|tech|info|edu|gov|pk|uk|ca|de|fr|xyz)(?:/[^\s<>'"()]*)?
    )
    """
)

_EDUCATION_SECTION_RE = re.compile(
    r"^\s*(education|academic\s+background|academic\s+qualifications|academics|degrees)\b",
    re.IGNORECASE,
)

_EXPERIENCE_SECTION_RE = re.compile(
    r"^\s*(experience|work\s+experience|employment\s+history|professional\s+experience|projects|work\s+history)\b",
    re.IGNORECASE,
)

_OTHER_SECTION_RE = re.compile(
    r"^\s*(skills|technical\s+skills|certifications|publications|awards|languages|interests|references|contact)\b",
    re.IGNORECASE,
)

_DEGREE_KEYWORD_RE = re.compile(
    r"\b(bachelor(?:'s)?|b\.?s\.?|b\.?sc\.?|b\.?a\.?|b\.?e\.?|b\.?tech|master(?:'s)?|m\.?s\.?|m\.?sc\.?|m\.?a\.?|m\.?tech|mba|ph\.?d\.?|doctorate|associate(?:'s)?|matric(?:ulation)?|intermediate|fsc|ics|fa|hssc|ssc|a-?levels?|o-?levels?|degree|graduated|graduation|convocation|class\s+of)\b",
    re.IGNORECASE,
)

_STREET_ADDRESS_RE = re.compile(
    r"""(?xi)
    \b(?:
        \d+[\w\s,.-]+(?:Street|St\.?|Road|Rd\.?|Avenue|Ave\.?|Boulevard|Blvd\.?|Drive|Dr\.?|Lane|Ln\.?|Way|Court|Ct\.?|Sector\s+[A-Za-z0-9-]+|Phase\s+[A-Za-z0-9-]+|Block\s+[A-Za-z0-9-]+)
        |
        House\s*#?\s*[\w\d-]+[\w\s,.-]+
        |
        Plot\s*#?\s*[\w\d-]+[\w\s,.-]+
        |
        Flat\s*#?\s*[\w\d-]+[\w\s,.-]+
        |
        Apt\s*#?\s*[\w\d-]+[\w\s,.-]+
    )\b
    """
)

_PAK_CV_FIELDS_RE = [
    # Father's name
    (re.compile(r"(?i)\b(father(?:'s)?\s*name|father\s*name)\s*[:=-]\s*([^\n\r]+)"), "Father's Name: [PERSONAL]"),
    # CNIC / National ID
    (re.compile(r"(?i)\b(cnic|nic|national\s*id(?:\s*card)?)\s*(?:no\.?|number)?\s*[:=-]?\s*([0-9]{5}[- ]?[0-9]{7}[- ]?[0-9]|[0-9]{13})"), "CNIC: [PERSONAL]"),
    # Standalone CNIC format
    (re.compile(r"\b\d{5}-\d{7}-\d\b"), "[PERSONAL]"),
    # Domicile
    (re.compile(r"(?i)\b(domicile|prc)\s*[:=-]\s*([^\n\r]+)"), "Domicile: [PERSONAL]"),
    # Nationality
    (re.compile(r"(?i)\b(nationality|citizenship)\s*[:=-]\s*([^\n\r]+)"), "Nationality: [PERSONAL]"),
    # Religion
    (re.compile(r"(?i)\b(religion|faith|sect)\s*[:=-]\s*([^\n\r]+)"), "Religion: [PERSONAL]"),
    # Date of Birth
    (re.compile(r"(?i)\b(date\s*of\s*birth|dob|birth\s*date|d\.o\.b\.?)\s*[:=-]\s*([^\n\r]+)"), "Date of Birth: [PERSONAL]"),
    # Marital Status
    (re.compile(r"(?i)\b(marital\s*status)\s*[:=-]\s*([^\n\r]+)"), "Marital Status: [PERSONAL]"),
    # Gender
    (re.compile(r"(?i)\b(gender|sex)\s*[:=-]\s*([^\n\r]+)"), "Gender: [PERSONAL]"),
    # Photo mentions
    (re.compile(r"(?i)\[?(photo|picture|photograph|headshot)\s*(attached|included|here|size)?\]?"), "[PERSONAL]"),
    (re.compile(r"(?i)\bpassport\s*size\s*photograph\b"), "[PERSONAL]"),
]

_PERSONAL_STANDALONE_WORDS_RE = re.compile(
    r"(?i)\b(muslim|islam|christian|christianity|hindu|hinduism|sikh|jewish|ahmadi|sunni|shia|single|married|divorced|widowed|unmarried|pakistani)\b"
)

_JOB_TITLE_KEYWORDS_RE = re.compile(
    r"\b(engineer|developer|manager|lead|architect|consultant|analyst|specialist|officer|director|intern|scientist|administrator|designer|technician|supervisor|coordinator|head|associate|programmer|expert|practitioner)\b",
    re.IGNORECASE,
)

# Pronoun replacement map (lowercase form -> neutral replacement)
_PRONOUN_MAP: Dict[str, str] = {
    "himself": "themselves",
    "herself": "themselves",
    "hers": "theirs",
    "his": "their",
    "he": "they",
    "she": "they",
    "him": "them",
}


def _get_spacy_pipeline():
    """Lazy load spaCy en_core_web_sm model."""
    try:
        from app.parser import _get_spacy_nlp
        nlp = _get_spacy_nlp()
        if nlp is not None:
            return nlp
    except Exception:
        pass
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except Exception:
        return None


def _load_known_skills() -> Set[str]:
    """Load canonical and alias skill names from repository taxonomy."""
    skills = {"jade", "python", "java", "ruby", "rust", "go", "c", "r", "swift", "julia", "perl"}
    try:
        from app.skills import _load_resources
        _, matcher = _load_resources()
        from app.skills import _alias_to_canonical
        skills.update(_alias_to_canonical.keys())
    except Exception:
        pass
    return skills


def _normalize_options(options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize user options with defaults and aliases."""
    opts = dict(DEFAULT_OPTIONS)
    if not options or not isinstance(options, dict):
        return opts

    # email switch
    if "email" in options:
        opts["email"] = bool(options["email"])
    elif "emails" in options:
        opts["email"] = bool(options["emails"])

    # phone switch
    if "phone" in options:
        opts["phone"] = bool(options["phone"])
    elif "phones" in options:
        opts["phone"] = bool(options["phones"])

    # link switch
    for k in ("link", "links", "url", "urls"):
        if k in options:
            opts["link"] = bool(options[k])
            break

    # name switch
    if "name" in options:
        opts["name"] = bool(options["name"])
    elif "names" in options:
        opts["name"] = bool(options["names"])

    # pronoun switch
    for k in ("pronoun", "pronouns", "gender_pronouns", "gender"):
        if k in options:
            opts["pronoun"] = bool(options[k])
            break

    # year switch
    for k in ("year", "years", "graduation_year", "graduation_years", "education_year"):
        if k in options:
            opts["year"] = bool(options[k])
            break

    # location switch
    if "location" in options:
        opts["location"] = bool(options["location"])
    elif "locations" in options and isinstance(options["locations"], bool):
        opts["location"] = bool(options["locations"])

    # custom locations list
    if "custom_locations" in options and isinstance(options["custom_locations"], (list, set)):
        opts["custom_locations"] = set(str(c).lower() for c in options["custom_locations"])
    elif "locations" in options and isinstance(options["locations"], (list, set)):
        opts["custom_locations"] = set(str(c).lower() for c in options["locations"])
    else:
        opts["custom_locations"] = set()

    # personal switch
    for k in ("personal", "personal_cues"):
        if k in options:
            opts["personal"] = bool(options[k])
            break

    return opts


# ---------------------------------------------------------------------------
# Redaction helpers
# ---------------------------------------------------------------------------

def _match_case(orig: str, replacement: str) -> str:
    """Preserve title/upper/lower casing in replacement."""
    if not orig or not replacement:
        return replacement
    if orig.isupper():
        return replacement.upper()
    if orig[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement.lower()


def _replace_pronouns_in_line(line: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Replace gendered pronouns with neutral pronouns while keeping reasonable grammar."""
    audit_entries: List[Dict[str, Any]] = []

    # 1. Verb agreement adjustments: 'he is' -> 'they are', 'she was' -> 'they were', 'he has' -> 'they have'
    verb_agreements = [
        (re.compile(r"\b(he|she)\s+is\b", re.IGNORECASE), "they are"),
        (re.compile(r"\b(he|she)\s+was\b", re.IGNORECASE), "they were"),
        (re.compile(r"\b(he|she)\s+has\b", re.IGNORECASE), "they have"),
    ]
    for pat, target in verb_agreements:
        def _va_sub(m):
            orig = m.group(0)
            replacement = _match_case(orig, target)
            audit_entries.append({
                "type": "pronoun",
                "placeholder": replacement.split()[0],
                "rule": "pronoun_grammar_agreement",
            })
            return replacement
        line = pat.sub(_va_sub, line)

    # 2. General pronoun replacement
    # Pattern matching pronouns
    pattern = re.compile(
        r"\b(himself|herself|hers|his|he|she|him|her)\b",
        re.IGNORECASE,
    )

    def _sub_pronoun(m: re.Match) -> str:
        word = m.group(1)
        lower = word.lower()

        if lower == "her":
            # Check context: if preceded by preposition with no following noun, objective 'them'
            before = line[:m.start()]
            after = line[m.end():]
            is_prep = bool(re.search(r"\b(to|with|for|by|from|at|against|promoted|managed|hired|interviewed)\s*$", before, re.IGNORECASE))
            has_following_noun = bool(re.match(r"^\s+[a-zA-Z]", after)) and not bool(re.match(r"^\s+(and|or|but|so|in|to|with|\.|\,|\;|\:|\)|$)", after))
            repl = "them" if (is_prep and not has_following_noun) else "their"
        else:
            repl = _PRONOUN_MAP.get(lower, "they")

        final_word = _match_case(word, repl)
        audit_entries.append({
            "type": "pronoun",
            "placeholder": final_word,
            "rule": "pronoun_neutralize",
        })
        return final_word

    line = pattern.sub(_sub_pronoun, line)
    return line, audit_entries


def _extract_name_candidates(text: str) -> List[str]:
    """Detect candidate names using first-line heuristic and spaCy PERSON in first 400 chars."""
    candidates: List[str] = []
    seen = set()

    # 1. Try parser extract_candidate_name
    try:
        from app.parser import extract_candidate_name
        name = extract_candidate_name(text, "")
        if name and name.strip() and name.strip() not in seen:
            cand = name.strip()
            if not _JOB_TITLE_KEYWORDS_RE.search(cand):
                seen.add(cand)
                candidates.append(cand)
    except Exception:
        pass

    # 2. First-line heuristic: Check first 5 non-empty lines
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in lines[:5]:
        if re.search(r"\d", line) or "@" in line or _URL_PATTERN.search(line):
            continue
        if re.search(r"\b(resume|cv|curriculum\s+vitae|profile|summary|education|skills|experience|contact)\b", line, re.IGNORECASE):
            continue
        if _JOB_TITLE_KEYWORDS_RE.search(line):
            continue
        words = line.split()
        if 1 <= len(words) <= 4 and len(line) <= 60:
            if line not in seen:
                seen.add(line)
                candidates.append(line)
            break

    # 3. spaCy PERSON entity in first 400 characters
    nlp = _get_spacy_pipeline()
    if nlp is not None:
        sample = text[:400]
        try:
            doc = nlp(sample)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    t = ent.text.strip()
                    if t and t not in seen and not re.search(r"\d", t) and "@" not in t:
                        if not _JOB_TITLE_KEYWORDS_RE.search(t):
                            seen.add(t)
                            candidates.append(t)
                            break
        except Exception:
            pass

    return candidates


# ---------------------------------------------------------------------------
# Core Anonymizer
# ---------------------------------------------------------------------------

def anonymize(text: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Redact personal identifiable information (PII) from resume text.

    Parameters
    ----------
    text : str
        Input resume text to be anonymized.
    options : dict, optional
        Configuration toggles to selectively disable redaction types:
        'email', 'phone', 'link', 'name', 'pronoun', 'year', 'location', 'personal'.

    Returns
    -------
    dict
        {
            "text": str,
            "redactions": list[{"type": str, "count": int}],
            "audit": list[{"type": str, "placeholder": str, "line": int, "rule": str}],
            "residual_flags": list[{"type": str, "line": int, "match": str}]
        }
    """
    if not isinstance(text, str) or not text:
        return {
            "text": "" if text is None else text,
            "redactions": [],
            "audit": [],
            "residual_flags": [],
        }

    opts = _normalize_options(options)
    audit: List[Dict[str, Any]] = []
    lines = text.splitlines()

    known_skills = _load_known_skills()
    custom_locs = COMMON_LOCATIONS.union(opts.get("custom_locations", set()))
    nlp = _get_spacy_pipeline()

    # Track sections to distinguish education graduation years vs work history dates
    in_education_section = False
    in_experience_section = False

    # Detect candidate names upfront for name redaction
    candidate_names: List[str] = []
    if opts["name"]:
        candidate_names = _extract_name_candidates(text)

    # First non-empty line index for header candidate name heuristic
    first_non_empty_idx = -1
    for idx, l in enumerate(lines):
        if l.strip():
            first_non_empty_idx = idx
            break

    # Identify header block range (lines before major sections or first 10 lines)
    header_block_limit = min(len(lines), 10)
    for idx, l in enumerate(lines[:12]):
        if _EXPERIENCE_SECTION_RE.search(l) or _EDUCATION_SECTION_RE.search(l) or _OTHER_SECTION_RE.search(l):
            header_block_limit = idx
            break

    redacted_lines: List[str] = []

    for line_idx, raw_line in enumerate(lines):
        line_num = line_idx + 1
        line = raw_line

        # Section tracking
        if _EDUCATION_SECTION_RE.search(line):
            in_education_section = True
            in_experience_section = False
        elif _EXPERIENCE_SECTION_RE.search(line):
            in_experience_section = True
            in_education_section = False
        elif _OTHER_SECTION_RE.search(line):
            in_education_section = False
            in_experience_section = False

        # --- 1. Email Redaction ---
        if opts["email"]:
            def _sub_email(m):
                audit.append({
                    "type": "email",
                    "placeholder": "[EMAIL]",
                    "line": line_num,
                    "rule": "regex_email",
                })
                return "[EMAIL]"
            line = _EMAIL_PATTERN.sub(_sub_email, line)

        # --- 2. Personal cues & Pakistani CV fields ---
        if opts["personal"]:
            for pat, repl in _PAK_CV_FIELDS_RE:
                if pat.search(line):
                    def _sub_pers(m):
                        audit.append({
                            "type": "personal",
                            "placeholder": "[PERSONAL]",
                            "line": line_num,
                            "rule": "pakistani_cv_fields",
                        })
                        return repl
                    line = pat.sub(_sub_pers, line)

            # Standalone religion / marital status / nationality on demographic lines
            if re.search(r"(?i)\b(personal\s+details|personal\s+information|demographics)\b", line) or line_idx < header_block_limit:
                def _sub_stand(m):
                    audit.append({
                        "type": "personal",
                        "placeholder": "[PERSONAL]",
                        "line": line_num,
                        "rule": "personal_cue_standalone",
                    })
                    return "[PERSONAL]"
                line = _PERSONAL_STANDALONE_WORDS_RE.sub(_sub_stand, line)

        # --- 3. Phone Redaction ---
        if opts["phone"]:
            def _sub_phone(m):
                audit.append({
                    "type": "phone",
                    "placeholder": "[PHONE]",
                    "line": line_num,
                    "rule": "regex_phone",
                })
                return "[PHONE]"
            line = _PHONE_PATTERN.sub(_sub_phone, line)

        # --- 4. URL / Link Redaction ---
        if opts["link"]:
            def _sub_link(m):
                audit.append({
                    "type": "link",
                    "placeholder": "[LINK]",
                    "line": line_num,
                    "rule": "regex_link",
                })
                return "[LINK]"
            line = _URL_PATTERN.sub(_sub_link, line)

        # --- 5. Graduation Year on Education Lines ---
        if opts["year"]:
            is_edu_line = in_education_section or bool(_DEGREE_KEYWORD_RE.search(line))
            # Work experience dates must be kept untouched
            is_work_line = in_experience_section and not bool(_DEGREE_KEYWORD_RE.search(line))
            has_present = bool(re.search(r"\b(present|current|till\s+date)\b", line, re.IGNORECASE))

            if is_edu_line and not is_work_line and not has_present:
                def _sub_year(m):
                    audit.append({
                        "type": "year",
                        "placeholder": "[YEAR]",
                        "line": line_num,
                        "rule": "education_graduation_year",
                    })
                    return "[YEAR]"
                line = re.sub(r"\b(19\d\d|20\d\d)\b", _sub_year, line)

        # --- 6. Location Cues (Header block and contact lines) ---
        if opts["location"]:
            is_contact_line = bool(re.search(
                r"\b(address|location|residence|residing|contact|postal|zip|city|street|block|sector|phase|p\.o\.?\s*box)\b",
                line,
                re.IGNORECASE,
            ))
            is_in_header = (line_idx < header_block_limit)

            if is_contact_line or is_in_header:
                # Street address replacement
                def _sub_addr(m):
                    audit.append({
                        "type": "location",
                        "placeholder": "[LOCATION]",
                        "line": line_num,
                        "rule": "location_street_address",
                    })
                    return "[LOCATION]"
                line = _STREET_ADDRESS_RE.sub(_sub_addr, line)

                # Configurable city / country list replacement
                for loc in custom_locs:
                    loc_pat = re.compile(rf"\b{re.escape(loc)}\b", re.IGNORECASE)
                    if loc_pat.search(line):
                        def _sub_loc(m):
                            audit.append({
                                "type": "location",
                                "placeholder": "[LOCATION]",
                                "line": line_num,
                                "rule": "location_known_city_country",
                            })
                            return "[LOCATION]"
                        line = loc_pat.sub(_sub_loc, line)

                # spaCy GPE and LOC entities on header/contact lines
                if nlp is not None and len(line.strip()) > 3:
                    try:
                        doc = nlp(line)
                        for ent in doc.ents:
                            if ent.label_ in ("GPE", "LOC") and ent.text.strip():
                                t = ent.text.strip()
                                # Do not redact if it's already a placeholder
                                if t not in ("[LOCATION]", "[NAME]", "[EMAIL]", "[PHONE]", "[LINK]", "[YEAR]", "[PERSONAL]"):
                                    audit.append({
                                        "type": "location",
                                        "placeholder": "[LOCATION]",
                                        "line": line_num,
                                        "rule": "location_spacy_entity",
                                    })
                                    line = re.sub(rf"\b{re.escape(t)}\b", "[LOCATION]", line)
                    except Exception:
                        pass

        # --- 7. Candidate Name Redaction ---
        if opts["name"]:
            # First line heuristic for candidate name
            if line_idx == first_non_empty_idx:
                stripped = line.strip()
                # If first non-empty line is 1-4 words and not a section title or already redacted or job title
                is_section_header = stripped.endswith(":") or bool(re.search(r"\b(summary|profile|overview|objective|about|experience|education|skills|resume|cv)\b", stripped, re.IGNORECASE))
                if (
                    stripped
                    and not stripped.startswith("[")
                    and len(stripped.split()) <= 4
                    and not is_section_header
                    and not _JOB_TITLE_KEYWORDS_RE.search(stripped)
                ):
                    audit.append({
                        "type": "name",
                        "placeholder": "[NAME]",
                        "line": line_num,
                        "rule": "candidate_name_first_line",
                    })
                    line = "[NAME]"
            else:
                for cand_name in candidate_names:
                    cand_clean = cand_name.strip()
                    if not cand_clean or cand_clean.lower() in ("curriculum vitae", "resume", "cv"):
                        continue

                    # If candidate name is a skill word (e.g. Jade or Python)
                    words = cand_clean.split()
                    if len(words) == 1 and cand_clean.lower() in known_skills:
                        # Only redact on header lines, preserve skill occurrences in body
                        if line_idx < header_block_limit and not re.search(r"\b(skills|technologies|tools)\b", line, re.IGNORECASE):
                            if re.search(rf"\b{re.escape(cand_clean)}\b", line, re.IGNORECASE):
                                audit.append({
                                    "type": "name",
                                    "placeholder": "[NAME]",
                                    "line": line_num,
                                    "rule": "candidate_name_skill_header",
                                })
                                line = re.sub(rf"\b{re.escape(cand_clean)}\b", "[NAME]", line, flags=re.IGNORECASE)
                    else:
                        # Multi-word name or non-skill name: replace everywhere
                        name_pat = re.compile(rf"\b{re.escape(cand_clean)}\b", re.IGNORECASE)
                        if name_pat.search(line):
                            audit.append({
                                "type": "name",
                                "placeholder": "[NAME]",
                                "line": line_num,
                                "rule": "candidate_name_match",
                            })
                            line = name_pat.sub("[NAME]", line)

        # --- 8. Pronoun Grammar Neutralization ---
        if opts["pronoun"]:
            line, pr_audits = _replace_pronouns_in_line(line)
            for item in pr_audits:
                item["line"] = line_num
                audit.append(item)

        redacted_lines.append(line)

    redacted_text = "\n".join(redacted_lines)

    # Compute per-type redaction counts
    type_order = ["email", "phone", "link", "name", "pronoun", "year", "location", "personal"]
    counts_by_type: Dict[str, int] = {t: 0 for t in type_order}
    for item in audit:
        t = item.get("type")
        if t in counts_by_type:
            counts_by_type[t] += 1
        else:
            counts_by_type[t] = counts_by_type.get(t, 0) + 1

    redactions_summary = [
        {"type": t, "count": counts_by_type[t]}
        for t in type_order
        if counts_by_type[t] > 0
    ]

    # --- 9. Residual Flags Second Pass ---
    residual_flags = find_residual_flags(redacted_text)

    return {
        "text": redacted_text,
        "redactions": redactions_summary,
        "audit": audit,
        "residual_flags": residual_flags,
    }


# ---------------------------------------------------------------------------
# Residual Flags Second Pass (Requirement 2)
# ---------------------------------------------------------------------------

def find_residual_flags(redacted_text: str) -> List[Dict[str, Any]]:
    """Scan redacted text to flag any residual unredacted PII or candidate name cues.

    Flags:
    - Email addresses
    - Phone numbers
    - URLs / Links
    - Four-digit years on education lines
    - Gendered pronouns
    - Title Case two-word sequence in the first 5 lines
    """
    flags: List[Dict[str, Any]] = []
    lines = redacted_text.splitlines()

    in_edu = False
    in_exp = False

    for idx, line in enumerate(lines):
        line_num = idx + 1
        line_clean = line.strip()
        if not line_clean:
            continue

        if _EDUCATION_SECTION_RE.search(line):
            in_edu = True
            in_exp = False
        elif _EXPERIENCE_SECTION_RE.search(line):
            in_exp = True
            in_edu = False
        elif _OTHER_SECTION_RE.search(line):
            in_edu = False
            in_exp = False

        # 1. Email check
        m_email = _EMAIL_PATTERN.search(line)
        if m_email and m_email.group(0) != "[EMAIL]":
            flags.append({"type": "email", "line": line_num, "match": m_email.group(0)})

        # 2. Phone check
        m_phone = _PHONE_PATTERN.search(line)
        if m_phone and m_phone.group(0) != "[PHONE]":
            flags.append({"type": "phone", "line": line_num, "match": m_phone.group(0)})

        # 3. URL check
        m_url = _URL_PATTERN.search(line)
        if m_url and m_url.group(0) != "[LINK]":
            flags.append({"type": "link", "line": line_num, "match": m_url.group(0)})

        # 4. Four-digit year on education line
        is_edu = in_edu or bool(_DEGREE_KEYWORD_RE.search(line))
        is_work = in_exp and not bool(_DEGREE_KEYWORD_RE.search(line))
        if is_edu and not is_work:
            m_year = re.search(r"\b(19\d\d|20\d\d)\b", line)
            if m_year and m_year.group(0) != "[YEAR]":
                flags.append({"type": "year", "line": line_num, "match": m_year.group(0)})

        # 5. Gendered pronoun check
        m_pro = re.search(r"\b(he|she|him|her|his|hers|himself|herself)\b", line, re.IGNORECASE)
        if m_pro:
            flags.append({"type": "pronoun", "line": line_num, "match": m_pro.group(0)})

        # 6. Title Case two-word sequence in the first 5 lines
        if idx < 5:
            # Exclude lines that start with placeholder or known headers
            if not line_clean.startswith("[NAME]"):
                matches = re.finditer(r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b", line)
                for m in matches:
                    cand = m.group(1)
                    if cand.lower() not in ("curriculum vitae", "software engineer", "summary profile"):
                        flags.append({"type": "name", "line": line_num, "match": cand})

    return flags
