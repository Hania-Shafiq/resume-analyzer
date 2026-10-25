"""Skill evidence depth (Feature 3).

Classifies how strongly the resume *demonstrates* each skill:

* ``high_depth``   - one sentence has a strong action verb AND a measurable metric
* ``demonstrated`` - one sentence has a strong action verb OR a measurable metric
* ``listed``       - only appears in keyword lists / a Skills section / bare mentions

The module only depends on ``app.skills`` (no import of ``app.main`` or
``app.scoring``), so ``skill_multipliers`` can be imported lazily by the scoring
module without circular imports.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Dict, List, Tuple

from app import skills as _skills
from app.skills import extract_skills

# Single source of truth for the multipliers.
EVIDENCE_MULTIPLIERS: Dict[str, float] = {
    "high_depth": 1.0,
    "demonstrated": 0.85,
    "listed": 0.5,
}

_LEVEL_RANK = {"listed": 0, "demonstrated": 1, "high_depth": 2}
_MAX_CHARS = 300_000

STRONG_VERBS = frozenset({
    "build", "develop", "design", "deploy", "implement", "optimize", "optimise",
    "reduce", "migrate", "automate", "lead", "scale", "integrate", "architect",
    "create", "engineer", "launch", "deliver", "improve", "refactor", "manage",
    "configure", "maintain", "ship", "establish", "spearhead", "orchestrate",
    "increase", "accelerate", "streamline", "containerize", "containerise",
    "write", "train", "analyze", "analyse", "redesign", "rewrite", "provision",
    "administer", "monitor", "secure", "debug", "test", "release", "publish",
    "mentor", "drive", "cut", "boost", "lower", "enhance", "extend", "modernize",
    "upgrade", "consolidate", "execute", "own", "found", "construct", "program",
    "clean", "extract", "visualize", "visualise", "transform", "query", "process",
})

_IRREGULAR = {
    "built": "build", "led": "lead", "wrote": "write", "written": "write",
    "drove": "drive", "driven": "drive", "shipped": "ship",
}


def _forms(base: str) -> set:
    forms = {base, base + "s", base + "es", base + "ing", base + "ed", base + "d"}
    if base.endswith("e"):
        forms.add(base[:-1] + "ing")
    if base.endswith("y"):
        forms |= {base[:-1] + "ies", base[:-1] + "ied"}
    forms |= {base + base[-1] + "ed", base + base[-1] + "ing"}
    return forms


_FORM_TO_BASE: Dict[str, str] = {}
for _b in STRONG_VERBS:
    for _f in _forms(_b):
        _FORM_TO_BASE.setdefault(_f, _b)
_FORM_TO_BASE.update(_IRREGULAR)

_BULLET_RE = re.compile(r"^[\s\u2022\u25aa\u25ab\u25cf\u25e6\u2023\u00b7>*+\-\u2013\u2014]+|^\d{1,2}[.)]\s+")
_SKILLS_HEADER_RE = re.compile(
    r"^\W*(?:technical\s+skills|key\s+skills|core\s+skills|skills(?:\s*(?:&|and)\s*"
    r"(?:abilities|competencies|tools|technologies))?|core\s+competencies|technologies|"
    r"tools(?:\s*(?:&|and)\s*technologies)?|tech\s+stack|skill\s*set|proficiencies|"
    r"technical\s+proficiencies|areas\s+of\s+expertise)\W*$",
    re.IGNORECASE,
)
_OTHER_HEADER_RE = re.compile(
    r"^\W*(?:work\s+experience|professional\s+experience|experience|employment"
    r"(?:\s+history)?|work\s+history|projects?|personal\s+projects|education|"
    r"academic\s+background|summary|professional\s+summary|profile|objective|"
    r"certifications?|awards?|publications?|languages?|interests?|references?|"
    r"achievements|volunteer(?:ing)?(?:\s+experience)?|contact)\W*$",
    re.IGNORECASE,
)
_INLINE_SKILLS_RE = re.compile(
    r"^\W*(?:technical\s+skills|key\s+skills|core\s+skills|core\s+competencies|skills|"
    r"technologies|tools|tech\s+stack|stack)\s*[:\-\u2013\u2014]\s*\S",
    re.IGNORECASE,
)
_LABEL_RE = re.compile(
    r"\b(?:key\s+|core\s+|technical\s+)?(?:skills|technologies|tools|tech\s+stack|"
    r"competencies|stack)\s*:",
    re.IGNORECASE,
)

_METRIC_RE = re.compile(
    r"(?<![\w.])(?:"
    r"\$\s?\d[\d,]*(?:\.\d+)?\s?(?:[kKmMbB]\b|million|billion|thousand)?"
    r"|\d[\d,]*(?:\.\d+)?\s?%"
    r"|\d+(?:\.\d+)?\s?[xX]"
    r"|\d[\d,]*(?:\.\d+)?\s?(?:[kKmMbB]\b\+?|million|billion|thousand)?\+?"
    r")(?!\w)"
)
_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")
_DURATION_AFTER_RE = re.compile(r"^\s*\+?\s*(?:years?|yrs?|months?|mos?|weeks?)\b", re.IGNORECASE)


@lru_cache(maxsize=1)
def _get_nlp():
    import spacy

    try:
        return spacy.load("en_core_web_sm", disable=["ner"])
    except OSError as exc:  # pragma: no cover - environment dependent
        raise OSError(
            "spaCy model 'en_core_web_sm' not found. Run:  python -m spacy download en_core_web_sm"
        ) from exc


@lru_cache(maxsize=1)
def _aliases_by_canonical() -> Dict[str, List[str]]:
    _skills._load_resources()
    inv: Dict[str, set] = {}
    for alias, canon in _skills._alias_to_canonical.items():
        inv.setdefault(canon, set()).add(alias)
    return {k: sorted(v, key=len, reverse=True) for k, v in inv.items()}


def _canonical(name: str) -> str:
    _skills._load_resources()
    low = name.strip().lower()
    return _skills._alias_to_canonical.get(low, low)


def skill_aliases(skill: str) -> List[str]:
    """All lower-case surface forms of a canonical skill (longest first)."""
    skill = skill.strip().lower()
    aliases = set(_aliases_by_canonical().get(skill, []))
    aliases.add(skill)
    return sorted(aliases, key=len, reverse=True)


@lru_cache(maxsize=2048)
def skill_pattern(skill: str) -> "re.Pattern[str]":
    """Whole-word, case-insensitive regex matching any alias of *skill*."""
    parts = []
    for alias in skill_aliases(skill):
        tokens = alias.split()
        if tokens:
            parts.append(r"\s+".join(re.escape(t) for t in tokens))
    return re.compile(r"(?<!\w)(?:" + "|".join(parts) + r")(?!\w)", re.IGNORECASE)


def _split_units(text: str) -> List[Tuple[str, bool]]:
    units: List[Tuple[str, bool]] = []
    in_skills = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if len(line) <= 60 and _SKILLS_HEADER_RE.match(line):
            in_skills = True
            continue
        if len(line) <= 60 and _OTHER_HEADER_RE.match(line):
            in_skills = False
            continue
        body = _BULLET_RE.sub("", line).strip()
        if not body:
            continue
        units.append((body, in_skills or bool(_INLINE_SKILLS_RE.match(body))))
    return units


def _strong_verbs(sent) -> list:
    out = []
    first = sent[0] if len(sent) else None
    for tok in sent:
        low = tok.lower_
        lemma = tok.lemma_.lower()
        base = lemma if lemma in STRONG_VERBS else _FORM_TO_BASE.get(low)
        if not base:
            continue
        verbish = (
            tok.pos_ in ("VERB", "AUX")
            or tok.tag_.startswith("VB")
            or tok.i == sent.start
            or (first is not None and first.pos_ == "ADV" and tok.i == sent.start + 1)
        )
        if verbish:
            out.append((tok, base))
    return out


def _metrics(text: str, skill_spans: List[Tuple[int, int]]) -> List[str]:
    found = []
    for m in _METRIC_RE.finditer(text):
        a, b = m.span()
        if any(a < e and s < b for s, e in skill_spans):
            continue  # number belongs to a skill name (e.g. "Python 3")
        raw = m.group(0).strip()
        if _YEAR_RE.match(raw):
            continue
        if _DURATION_AFTER_RE.match(text[b:b + 14]):
            continue  # "5 years of experience" is not an achievement metric
        found.append(raw.replace(" ", ""))
    return found


def _governing_verb(sent, verbs, spans):
    doc = sent.doc
    for verb, base in verbs:
        for a, b in spans:
            span = doc.char_span(sent.start_char + a, sent.start_char + b, alignment_mode="expand")
            if span is not None and any(verb.is_ancestor(t) for t in span):
                return verb, base
    return None


def _analyze_sentence(sent, patterns) -> Dict[str, Dict[str, Any]]:
    text = sent.text
    spans: Dict[str, List[Tuple[int, int]]] = {}
    for skill, pat in patterns.items():
        found = [m.span() for m in pat.finditer(text)]
        if found:
            spans[skill] = found
    if not spans:
        return {}

    label = _LABEL_RE.search(text)
    all_spans = [s for v in spans.values() for s in v]
    n_distinct = len(spans)
    verbs = _strong_verbs(sent)
    metrics = _metrics(text, all_spans)

    remaining = text
    for a, b in sorted(all_spans, reverse=True):
        remaining = remaining[:a] + " " + remaining[b:]
    list_like = len(re.findall(r"[^\W\d_]{2,}", remaining)) <= 1

    out: Dict[str, Dict[str, Any]] = {}
    for skill, sp in spans.items():
        eff = [s for s in sp if label is None or s[0] < label.start()]
        if not eff:
            out[skill] = {"level": "listed", "signals": ["list_only"], "evidence": None}
            continue
        governed = _governing_verb(sent, verbs, eff) if verbs else None
        chosen = governed
        if chosen is None and verbs and n_distinct < 4:
            chosen = verbs[0]
        signals: List[str] = []
        if chosen is not None:
            signals.append("action_verb:" + chosen[1])
        metric_ok = bool(metrics) and (chosen is not None or (not list_like and n_distinct <= 3))
        if metric_ok:
            signals.extend("metric:" + m for m in metrics[:2])
        if chosen is not None and metric_ok:
            level = "high_depth"
        elif chosen is not None or metric_ok:
            level = "demonstrated"
        else:
            level = "listed"
            signals = ["list_only" if list_like or n_distinct >= 4 else "mention_only"]
        out[skill] = {
            "level": level,
            "signals": signals,
            "evidence": " ".join(text.split()) if level != "listed" else None,
            "governed": governed is not None,
        }
    return out


def _resolve_targets(text: str, skills) -> List[str]:
    if skills is None:
        try:
            names = extract_skills(text)
        except (ValueError, TypeError):
            return []
    else:
        names = [s for s in skills if isinstance(s, str) and s.strip()]
    seen, out = set(), []
    for name in names:
        canon = _canonical(name)
        if canon not in seen:
            seen.add(canon)
            out.append(canon)
    return out


def skill_evidence(resume_text: str, skills=None) -> Dict[str, Dict[str, Any]]:
    """Return ``{skill: {"level", "multiplier", "evidence", "signals"}}``.

    Skills that never occur in the resume text are omitted. Keys are canonical
    lower-case skill names (the same form ``extract_skills`` returns).
    """
    if not isinstance(resume_text, str):
        raise TypeError(f"Expected str, got {type(resume_text).__name__}.")
    if not resume_text.strip():
        return {}
    text = resume_text[:_MAX_CHARS]
    targets = _resolve_targets(text, skills)
    if not targets:
        return {}
    patterns = {s: skill_pattern(s) for s in targets}

    best: Dict[str, Dict[str, Any]] = {}

    def update(skill: str, rec: Dict[str, Any]) -> None:
        cur = best.get(skill)
        if cur is None or _LEVEL_RANK[rec["level"]] > _LEVEL_RANK[cur["level"]] or (
            _LEVEL_RANK[rec["level"]] == _LEVEL_RANK[cur["level"]]
            and rec.get("governed") and not cur.get("governed")
        ):
            best[skill] = rec

    units = _split_units(text)
    prose = [u[0] for u in units if not u[1]]
    docs = iter(_get_nlp().pipe(prose, batch_size=32)) if prose else iter(())

    for body, is_skills_line in units:
        if is_skills_line:
            for skill, pat in patterns.items():
                if pat.search(body):
                    update(skill, {"level": "listed", "signals": ["list_only"], "evidence": None})
            continue
        doc = next(docs)
        for sent in doc.sents:
            for skill, rec in _analyze_sentence(sent, patterns).items():
                update(skill, rec)

    result: Dict[str, Dict[str, Any]] = {}
    for skill in targets:
        rec = best.get(skill)
        if rec is None:
            continue
        result[skill] = {
            "level": rec["level"],
            "multiplier": EVIDENCE_MULTIPLIERS[rec["level"]],
            "evidence": rec["evidence"] if rec["level"] != "listed" else None,
            "signals": list(rec["signals"]),
        }
    return result


def skill_multipliers(resume_text: str, skills=None) -> Dict[str, float]:
    """Return ``{skill: multiplier}`` using the same classification as skill_evidence."""
    return {s: info["multiplier"] for s, info in skill_evidence(resume_text, skills).items()}


def evidence_counts(evidence: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    counts = {"high_depth": 0, "demonstrated": 0, "listed": 0}
    for info in evidence.values():
        counts[info["level"]] += 1
    return counts
