"""Rule-based STAR bullet optimizer.

Deterministic, non-LLM, offline optimizer that analyzes resume bullets,
identifies weakness patterns (e.g. weak openers, missing action verbs,
missing metrics), and constructs structured STAR improvements adhering
to strict anti-hallucination and truthfulness constraints.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import spacy

from app.skills import extract_skills, skill_category

TEMPLATES_PATH = Path(__file__).resolve().parent / "data" / "star_templates.json"

# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------

STRONG_ACTION_VERBS: Set[str] = {
    # Architecture, Engineering & Design
    "architected", "designed", "engineered", "authored", "structured",
    "formulated", "modeled", "prototyped",
    # Implementation & Building
    "built", "developed", "implemented", "constructed", "created",
    "programmed", "coded", "fabricated", "assembled",
    # Leadership & Ownership
    "owned", "led", "spearheaded", "directed", "orchestrated",
    "founded", "established", "championed", "steered",
    # Delivery, Deployment & Release
    "deployed", "delivered", "launched", "published", "released",
    "shipped", "dispatched", "provisioned", "migrated",
    # Optimization & Acceleration
    "optimized", "accelerated", "streamlined", "enhanced", "boosted",
    "maximized", "refactored", "upgraded", "consolidated",
    # Reduction & Mitigation
    "reduced", "eliminated", "decreased", "minimized", "mitigated",
    "curtailed", "trimmed",
    # Growth & Scaling
    "scaled", "expanded", "increased", "multiplied", "grown",
    # Automation & Integration
    "automated", "integrated", "configured", "executed", "collaborated",
    "contributed", "supported", "maintained", "administered",
    # Present / Base forms
    "build", "develop", "design", "architect", "engineer", "implement",
    "lead", "spearhead", "deploy", "optimize", "automate", "scale",
    "launch", "deliver", "refactor", "integrate", "orchestrate",
}

WEAK_OPENER_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("responsible for", re.compile(r"^(?:was\s+|i\s+was\s+)?responsible\s+for\b", re.IGNORECASE)),
    ("worked on", re.compile(r"^(?:i\s+)?worked\s+on\b", re.IGNORECASE)),
    ("helped with", re.compile(r"^(?:i\s+)?helped\s+with\b", re.IGNORECASE)),
    ("involved in", re.compile(r"^(?:was\s+|i\s+was\s+)?involved\s+in\b", re.IGNORECASE)),
    ("duties included", re.compile(r"^(?:my\s+)?duties\s+included\b", re.IGNORECASE)),
    ("assisted in", re.compile(r"^(?:i\s+)?assisted\s+in\b", re.IGNORECASE)),
    ("participated in", re.compile(r"^(?:i\s+)?participated\s+in\b", re.IGNORECASE)),
]

CATEGORY_TRIGGER_WORDS: Dict[str, List[str]] = {
    "devops_tools": [
        "deployment", "deploying", "deploy", "deployed",
        "container", "containers", "containerized", "containerization",
        "pipeline", "pipelines", "ci/cd", "cicd", "continuous integration",
        "infrastructure", "cluster", "clusters", "monitoring", "logging",
        "orchestration", "automation", "automated", "server", "servers",
        "build", "builds", "release", "releases",
    ],
    "web": [
        "api", "apis", "rest", "graphql", "backend", "frontend", "full-stack", "fullstack",
        "endpoint", "endpoints", "web", "website", "application", "ui", "ux",
        "interface", "client-side", "server-side", "microservice", "microservices",
    ],
    "databases": [
        "database", "databases", "query", "queries", "schema", "schemas",
        "sql", "nosql", "table", "tables", "storage", "warehouse", "warehousing",
        "backup", "backups", "migration", "migrations", "indexing",
    ],
    "machine_learning": [
        "model", "models", "training", "inference", "dataset", "datasets",
        "neural", "ai", "algorithm", "algorithms", "classification", "regression",
        "nlp", "computer vision", "deep learning", "evaluation",
    ],
    "cloud": [
        "cloud", "aws", "azure", "gcp", "hosting", "virtual machine",
        "bucket", "serverless", "lambda", "instance", "instances",
    ],
    "programming": [
        "code", "coding", "script", "scripts", "scripting", "logic",
        "refactor", "refactoring", "functions", "library", "module",
    ],
    "soft_skills": [
        "team", "stakeholders", "clients", "agile", "scrum", "sprint",
        "communication", "documentation", "presentation", "mentoring",
    ],
}

METRIC_RE = re.compile(
    r"\b(?:"
    r"\d+[\d,.]*(?:%|\+|k|m|b|x)?"
    r"|\$\d+[\d,.]*"
    r"|\d+\s*(?:percent|users?|customers?|clients?|requests?|ms|seconds?|minutes?|hours?|days?|weeks?|months?|years?|nodes?|servers?|instances?|endpoints?|services?|tps|qps)"
    r")\b",
    re.IGNORECASE,
)

EXPERIENCE_PROJECT_HEADERS = {
    "experience", "work experience", "professional experience",
    "employment history", "work history", "career history",
    "projects", "personal projects", "technical projects", "key projects",
    "freelance experience", "freelance work", "employment",
}

IGNORE_SECTION_HEADERS = {
    "education", "academic background", "academics", "qualifications",
    "skills", "technical skills", "certifications", "licenses",
    "contact", "contact info", "contact information",
    "languages", "interests", "awards", "publications", "references",
    "summary", "profile", "professional summary", "about",
}

EDU_WORDS_RE = re.compile(
    r"\b(?:b\.?s\.?|b\.?e\.?|m\.?s\.?|bsc|msc|bachelor|master|ph\.?d\.?|doctorate|degree|diploma|university|college|institute)\b",
    re.IGNORECASE,
)

CONTACT_RE = re.compile(
    r"(?:\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b|\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b|https?://|www\.)",
    re.IGNORECASE,
)

_TEMPLATES_CACHE: Optional[Dict[str, Any]] = None


def _load_templates() -> Dict[str, Any]:
    global _TEMPLATES_CACHE
    if _TEMPLATES_CACHE is not None:
        return _TEMPLATES_CACHE
    if TEMPLATES_PATH.exists():
        with open(TEMPLATES_PATH, "r", encoding="utf-8") as fh:
            _TEMPLATES_CACHE = json.load(fh)
    else:
        _TEMPLATES_CACHE = {
            "opener_verb_map": {
                "responsible for": "Owned",
                "worked on": "Built",
                "helped with": "Collaborated to deliver",
                "involved in": "Contributed to",
                "duties included": "Executed",
                "assisted in": "Supported",
                "participated in": "Engaged in",
            },
            "default_verb": "Engineered",
            "placeholders": [
                "[result: what improved, by how much, measured how]",
                "[N%]",
                "[N hours]",
                "[N users]",
            ],
            "honesty_notes": {
                "missing": "Only include this skill if you have genuine hands-on experience using it in this role. Never invent tools or accomplishments.",
                "present": "Verify that all outcome placeholders accurately reflect your genuine metrics and achievements.",
            },
        }
    return _TEMPLATES_CACHE


# ---------------------------------------------------------------------------
# 1. Bullet Splitting
# ---------------------------------------------------------------------------

def split_bullets(resume_text: str) -> List[str]:
    """Extract experience and project bullets/sentences from resume text.

    Ignores Education, contact info, and section headers. Uses spaCy sentence
    splitting for prose paragraphs and line/bullet marker splitting for lists.
    """
    if not isinstance(resume_text, str) or not resume_text.strip():
        return []

    lines = resume_text.splitlines()
    bullets: List[str] = []

    # Check if text contains explicit section headers
    has_explicit_headers = False
    for line in lines:
        cleaned = re.sub(r"^[#*\-=_:\s\d\.]+|[#*\-=_:\s\d\.]+$", "", line).strip().lower()
        if cleaned in EXPERIENCE_PROJECT_HEADERS or cleaned in IGNORE_SECTION_HEADERS:
            has_explicit_headers = True
            break

    # If headers are present, track sections; otherwise default to experience
    current_section = "other" if has_explicit_headers else "experience"
    bullet_marker_re = re.compile(r"^[\s\•\*\-\–\—\▪\▫\◦\d\.\)\(\]]+\s*")

    raw_experience_blocks: List[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        clean_header = re.sub(r"^[#*\-=_:\s\d\.]+|[#*\-=_:\s\d\.]+$", "", stripped).strip().lower()

        # Check section transition
        if clean_header in EXPERIENCE_PROJECT_HEADERS:
            current_section = "experience"
            continue
        elif clean_header in IGNORE_SECTION_HEADERS:
            current_section = "other"
            continue

        if current_section != "experience":
            continue

        # Skip contact information lines or education degree lines
        if CONTACT_RE.search(stripped) or EDU_WORDS_RE.search(stripped):
            continue

        # Skip pure date lines (e.g. '2020 - 2022', 'Jan 2021 - Present')
        if re.match(r"^[\d\s/\-–—to\.,()presentcurrentnow]+$", stripped, re.IGNORECASE):
            continue

        raw_experience_blocks.append(stripped)

    if not raw_experience_blocks:
        return []

    # Process blocks: separate explicit bullets vs. paragraphs
    nlp = None
    for block in raw_experience_blocks:
        is_bullet = bool(re.match(r"^[\•\*\-\–\—\▪\▫\◦]|\b\d+[\.\)]\s+", block))
        if is_bullet:
            cleaned = bullet_marker_re.sub("", block).strip()
            if cleaned and len(cleaned.split()) >= 3:
                bullets.append(cleaned)
        else:
            # Paragraph or list line without leading bullet symbol
            if len(block.split()) >= 3:
                # If block has multiple sentences, split with spaCy
                if "." in block or ";" in block:
                    if nlp is None:
                        nlp = spacy.load("en_core_web_sm", disable=["ner"])
                    doc = nlp(block)
                    for sent in doc.sents:
                        s_text = sent.text.strip()
                        s_text = bullet_marker_re.sub("", s_text).strip()
                        if s_text and len(s_text.split()) >= 3:
                            bullets.append(s_text)
                else:
                    bullets.append(block)

    return bullets


# ---------------------------------------------------------------------------
# 2. Weakness Detection
# ---------------------------------------------------------------------------

def weak_bullet_reasons(bullet: str) -> List[str]:
    """Identify weakness patterns in a resume bullet.

    Returns a list of reasons such as:
      - 'weak_opener:<phrase>' (e.g. 'weak_opener:responsible for')
      - 'no_action_verb'
      - 'no_metric'
      - 'no_skill_mentioned'
      - 'too_short'

    A bullet is considered weak when it has 'no_action_verb' or a weak opener.
    """
    if not isinstance(bullet, str) or not bullet.strip():
        return ["too_short", "no_action_verb", "no_metric", "no_skill_mentioned"]

    reasons: List[str] = []
    clean_bullet = bullet.strip()
    words = clean_bullet.split()

    if len(words) < 6:
        reasons.append("too_short")

    # 1. Weak Opener
    weak_opener_found = None
    for name, pat in WEAK_OPENER_PATTERNS:
        if pat.search(clean_bullet):
            reasons.append(f"weak_opener:{name}")
            weak_opener_found = name
            break

    # 2. Action Verb
    # If starting with weak opener, it does not start with a strong action verb
    starts_with_strong_verb = False
    if not weak_opener_found and words:
        first_token = re.sub(r"[^\w]", "", words[0].lower())
        second_token = re.sub(r"[^\w]", "", words[1].lower()) if len(words) > 1 else ""

        if first_token in STRONG_ACTION_VERBS:
            starts_with_strong_verb = True
        elif first_token in {"successfully", "effectively", "consistently", "collaboratively", "independently"} and second_token in STRONG_ACTION_VERBS:
            starts_with_strong_verb = True

    if not starts_with_strong_verb:
        reasons.append("no_action_verb")

    # 3. Metric
    if not METRIC_RE.search(clean_bullet):
        reasons.append("no_metric")

    # 4. Skill
    skills = extract_skills(clean_bullet)
    if not skills:
        reasons.append("no_skill_mentioned")

    return reasons


# ---------------------------------------------------------------------------
# 3. STAR Optimizer
# ---------------------------------------------------------------------------

def _infer_bullet_category(bullet: str) -> Optional[str]:
    """Infer the technical skills category of a bullet using domain keywords."""
    lower = bullet.lower()
    counts: Dict[str, int] = {}
    for cat, triggers in CATEGORY_TRIGGER_WORDS.items():
        score = sum(1 for t in triggers if re.search(rf"\b{re.escape(t)}\b", lower))
        if score > 0:
            counts[cat] = score

    if counts:
        return max(counts.keys(), key=lambda k: counts[k])
    return None


def _format_skill_name(skill: str) -> str:
    """Format canonical skill for display in sentences."""
    s = skill.strip()
    acronyms = {"aws", "gcp", "sql", "api", "ui", "ux", "ci/cd", "nlp", "ml", "k8s"}
    if s.lower() in acronyms:
        return s.upper()
    return s.title() if not any(c.isupper() for c in s[1:]) else s


def suggest_star(
    resume_text: str,
    jd_analysis: Optional[Any] = None,
    missing_skills: Optional[Union[List[str], Set[str]]] = None,
    max_suggestions: int = 5,
) -> List[Dict[str, Any]]:
    """Generate rule-based STAR optimizations for weak bullets.

    Adheres strictly to the anti-hallucination constraint: never invents
    employers, tools not used, or numerical statistics.
    """
    if not isinstance(resume_text, str) or not resume_text.strip():
        return []

    templates = _load_templates()
    opener_verb_map = templates.get("opener_verb_map", {})
    default_verb = templates.get("default_verb", "Engineered")

    # Resolve candidate missing skills from JD analysis or direct list
    all_missing_skills: List[str] = []
    if missing_skills is not None:
        all_missing_skills.extend([s.lower() for s in missing_skills if isinstance(s, str)])
    elif jd_analysis is not None:
        if hasattr(jd_analysis, "required_skills"):
            all_missing_skills.extend([s.lower() for s in getattr(jd_analysis, "required_skills", [])])
            all_missing_skills.extend([s.lower() for s in getattr(jd_analysis, "preferred_skills", [])])
        elif isinstance(jd_analysis, dict):
            all_missing_skills.extend([s.lower() for s in jd_analysis.get("required_skills", [])])
            all_missing_skills.extend([s.lower() for s in jd_analysis.get("preferred_skills", [])])

    bullets = split_bullets(resume_text)
    suggestions: List[Dict[str, Any]] = []

    for idx, bullet in enumerate(bullets):
        reasons = weak_bullet_reasons(bullet)
        is_weak = ("no_action_verb" in reasons) or any(r.startswith("weak_opener:") for r in reasons)
        if not is_weak:
            continue

        # Extract weak opener (if any)
        weak_opener = None
        for r in reasons:
            if r.startswith("weak_opener:"):
                weak_opener = r.split(":", 1)[1]
                break

        strong_verb = opener_verb_map.get(weak_opener, default_verb)

        # Extract subject from original wording
        subject = bullet
        if weak_opener:
            for name, pat in WEAK_OPENER_PATTERNS:
                if name == weak_opener:
                    subject = pat.sub("", bullet).strip()
                    break

        subject = subject.rstrip(". \t\n;:,")
        if not subject:
            subject = bullet.rstrip(". \t\n;:,")

        # 3a. Target skill selection
        bullet_skills = extract_skills(bullet)
        target_skill: Optional[str] = None
        skill_status = "present"

        if bullet_skills:
            target_skill = bullet_skills[0]
            skill_status = "present"
        else:
            # Check if missing JD skill matches bullet category
            bullet_cat = _infer_bullet_category(bullet)
            if bullet_cat and all_missing_skills:
                for candidate in all_missing_skills:
                    c_cat = skill_category(candidate)
                    if c_cat == bullet_cat:
                        target_skill = candidate
                        skill_status = "missing"
                        break

        # 3b. Build four STAR parts
        situation = f"In an engineering setting addressing {subject}"
        task = f"Tasked with delivering {subject}"

        if skill_status == "missing" and target_skill:
            skill_disp = _format_skill_name(target_skill)
            action = f"{strong_verb} {subject} using {skill_disp}"
        else:
            action = f"{strong_verb} {subject}"

        result_placeholder = "[result: what improved, by how much, measured how]"
        result = result_placeholder

        # 3d. Build rewritten bullet
        if skill_status == "missing" and target_skill:
            skill_disp = _format_skill_name(target_skill)
            rewritten_bullet = (
                f"If you have used {skill_disp} on this work: "
                f"{action}, achieving {result}."
            )
            requires_real_experience = True
            honesty_note = (
                f"Only include {target_skill} if you have genuine hands-on experience "
                "using it in this role. Never invent tools or accomplishments."
            )
        else:
            rewritten_bullet = f"{action}, achieving {result}."
            requires_real_experience = False
            honesty_note = (
                "Verify that all outcome placeholders accurately reflect your "
                "genuine metrics and achievements."
            )

        # Identify placeholders present in the output
        placeholders = [result_placeholder]

        # 3c. Anti-hallucination verification: Ensure NO new digits were introduced
        orig_digits = set(re.findall(r"\d", bullet))
        suggested_digits = (
            set(re.findall(r"\d", situation))
            | set(re.findall(r"\d", task))
            | set(re.findall(r"\d", action))
            | set(re.findall(r"\d", result))
            | set(re.findall(r"\d", rewritten_bullet))
        )
        new_digits = suggested_digits - orig_digits
        if new_digits:
            # Strip any unverified numerical additions
            for d in new_digits:
                situation = situation.replace(d, "")
                task = task.replace(d, "")
                action = action.replace(d, "")
                rewritten_bullet = rewritten_bullet.replace(d, "")

        # Compute weakness score for deterministic ranking
        weakness_score = len(reasons) + (2 if weak_opener else 0)

        suggestions.append({
            "_sort_key": (-weakness_score, idx),
            "original": bullet,
            "reasons": reasons,
            "target_skill": target_skill,
            "skill_status": skill_status,
            "star": {
                "situation": situation,
                "task": task,
                "action": action,
                "result": result,
            },
            "rewritten_bullet": rewritten_bullet,
            "placeholders": placeholders,
            "requires_real_experience": requires_real_experience,
            "honesty_note": honesty_note,
        })

    # Sort deterministically: highest weakness score first, then position in resume
    suggestions.sort(key=lambda s: s["_sort_key"])

    # Remove temporary internal sort key and truncate to max_suggestions
    clean_suggestions = []
    for item in suggestions[:max_suggestions]:
        item.pop("_sort_key", None)
        clean_suggestions.append(item)

    return clean_suggestions
