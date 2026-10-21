"""PDF overlay (Feature 7): annotate a resume PDF with JD-relevant evidence.

Categories (precedence: unverified > required > preferred; semantic last):
  required   #10b981  required JD skill, evidence high_depth/demonstrated
  preferred  #f59e0b  preferred JD skill, evidence high_depth/demonstrated
  semantic   #3b82f6  sentence semantically close to a JD sentence, no highlighted skill
  unverified #f43f5e  skill that is only *listed* (no supporting evidence)
Skill level is classified once per skill; every occurrence of the skill gets the
resulting category. No OCR is performed and no document text is logged.
"""

from __future__ import annotations

import base64
import re
from typing import Any, Dict, List, Optional

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz

from app.evidence import skill_aliases, skill_evidence, skill_pattern
from app.skills import extract_skills

SEMANTIC_THRESHOLD = 0.55
MAX_BOXES_PER_PAGE = 200
_TEXT_PAGE_CAP = 30

CATEGORY_COLORS = {
    "required": "#10b981",
    "preferred": "#f59e0b",
    "semantic": "#3b82f6",
    "unverified": "#f43f5e",
}
LEGEND = [
    {"category": "required", "label": "Required skill", "color": "#10b981",
     "description": "Required by the job and backed by evidence (action verb and/or metric)."},
    {"category": "preferred", "label": "Preferred skill", "color": "#f59e0b",
     "description": "Preferred by the job and backed by evidence (action verb and/or metric)."},
    {"category": "semantic", "label": "Semantic match", "color": "#3b82f6",
     "description": "Sentence closely matching a job-description sentence."},
    {"category": "unverified", "label": "Unverified (listed only)", "color": "#f43f5e",
     "description": "Skill that is only listed, without supporting evidence."},
]


def _as_list(value) -> List[str]:
    return [str(s).strip().lower() for s in (value or []) if str(s).strip()]


def _jd_fields(jd_analysis: Any):
    if isinstance(jd_analysis, dict):
        req, pref = jd_analysis.get("required_skills"), jd_analysis.get("preferred_skills")
        raw = jd_analysis.get("raw_text") or jd_analysis.get("jd_text") or ""
    else:
        req = getattr(jd_analysis, "required_skills", None)
        pref = getattr(jd_analysis, "preferred_skills", None)
        raw = getattr(jd_analysis, "raw_text", "") or ""
    req, pref = _as_list(req), _as_list(pref)
    if not raw:
        lines = []
        if req:
            lines.append("Required skills: " + ", ".join(req))
        if pref:
            lines.append("Preferred skills: " + ", ".join(pref))
        raw = "\n".join(lines)
    return req, pref, raw


def _is_whole_word_hit(words, rect, alias: str) -> bool:
    text = " ".join(w[4] for w in words if fitz.Rect(w[:4]).intersects(rect)).lower()
    parts = [re.escape(p) for p in alias.lower().split()]
    if not parts:
        return False
    return re.search(r"(?<!\w)" + r"\s+".join(parts) + r"(?!\w)", text) is not None


def _norm_box(rect, page) -> Optional[List[float]]:
    r = fitz.Rect(rect) * page.rotation_matrix
    pr = page.rect
    w, h = pr.width or 1.0, pr.height or 1.0
    x0, y0 = (r.x0 - pr.x0) / w, (r.y0 - pr.y0) / h
    x1, y1 = (r.x1 - pr.x0) / w, (r.y1 - pr.y0) / h

    def clamp(v: float) -> float:
        return max(0.0, min(1.0, v))

    box = [round(clamp(min(x0, x1)), 5), round(clamp(min(y0, y1)), 5),
           round(clamp(max(x0, x1)), 5), round(clamp(max(y0, y1)), 5)]
    return box if box[2] > box[0] and box[3] > box[1] else None


def _semantic_sentences(jd_text: str, resume_text: str, highlighted: List[str]) -> List[str]:
    # Lazy import: the API must run without sentence-transformers in skills_only mode.
    from app.matcher import compute_similarity_matrix, normalize_similarity, split_sentences

    jd_chunks = split_sentences(jd_text)
    res_chunks = split_sentences(resume_text)[:300]
    if not jd_chunks or not res_chunks:
        return []
    sim = compute_similarity_matrix(jd_chunks, res_chunks)
    best = sim.max(axis=0)
    pats = [skill_pattern(s) for s in highlighted]
    out = []
    for sentence, cos in zip(res_chunks, best):
        if normalize_similarity(float(cos)) < SEMANTIC_THRESHOLD:
            continue
        if any(p.search(sentence) for p in pats):
            continue
        out.append(sentence)
    return out


def build_overlay(
    pdf_bytes: bytes,
    jd_analysis: Any,
    resume_skills: Optional[List[str]] = None,
    include_semantic: bool = False,
    max_pages: int = 5,
    zoom: float = 1.5,
) -> Dict[str, Any]:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        raise ValueError("The uploaded file is not a readable PDF.") from None
    try:
        if doc.page_count == 0:
            raise ValueError("The PDF document contains no pages.")
        n_render = min(doc.page_count, max(1, int(max_pages)))
        texts = [doc.load_page(i).get_text("text") for i in range(min(doc.page_count, _TEXT_PAGE_CAP))]
        full_text = "\n".join(texts)
        has_text = bool(full_text.strip())

        required, preferred, jd_raw = _jd_fields(jd_analysis)
        category_of: Dict[str, str] = {}
        evidence: Dict[str, Dict[str, Any]] = {}
        if has_text:
            if resume_skills is None:
                try:
                    resume_skills = extract_skills(full_text)
                except ValueError:
                    resume_skills = []
            evidence = skill_evidence(full_text, list(resume_skills))
            for skill, info in evidence.items():
                if info["level"] == "listed":
                    category_of[skill] = "unverified"
                elif skill in required:
                    category_of[skill] = "required"
                elif skill in preferred:
                    category_of[skill] = "preferred"

        semantic_unavailable = False
        semantic_sentences: List[str] = []
        if include_semantic and has_text:
            try:
                semantic_sentences = _semantic_sentences(jd_raw, full_text, list(category_of))
            except Exception:
                semantic_unavailable = True
                semantic_sentences = []

        order = {"unverified": 0, "required": 1, "preferred": 2}
        ordered_skills = sorted(category_of, key=lambda s: (order[category_of[s]], s))
        used_semantic: set = set()
        pages: List[Dict[str, Any]] = []
        counts = {c: 0 for c in CATEGORY_COLORS}
        mat = fitz.Matrix(zoom, zoom)

        for pno in range(n_render):
            page = doc.load_page(pno)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            image_b64 = base64.b64encode(pix.tobytes("png")).decode("ascii")
            boxes: List[Dict[str, Any]] = []
            seen: set = set()

            def add(rect, category, skill, matched, tooltip, _boxes=boxes, _seen=seen, _page=page):
                if len(_boxes) >= MAX_BOXES_PER_PAGE:
                    return
                key = tuple(round(v, 1) for v in (rect.x0, rect.y0, rect.x1, rect.y1))
                if key in _seen:
                    return
                bbox = _norm_box(rect, _page)
                if bbox is None:
                    return
                _seen.add(key)
                _boxes.append({"bbox": bbox, "category": category, "skill": skill,
                               "text": (matched or "")[:80], "tooltip": tooltip})
                counts[category] += 1

            if has_text and page.get_text("text").strip():
                words = page.get_text("words")
                for skill in ordered_skills:
                    cat = category_of[skill]
                    info = evidence[skill]
                    if cat == "unverified":
                        why = f"{skill}: listed only - no supporting evidence in the resume"
                    else:
                        why = (f"{skill}: {cat} skill, evidence {info['level']} "
                               f"({', '.join(info['signals'])})")
                    for alias in skill_aliases(skill):
                        for rect in page.search_for(alias):
                            if not _is_whole_word_hit(words, rect, alias):
                                continue
                            matched = page.get_textbox(rect).strip() or alias
                            add(rect, cat, skill, matched, why)
                for sentence in semantic_sentences:
                    if sentence in used_semantic:
                        continue
                    rects = page.search_for(sentence[:60]) or page.search_for(sentence[:30])
                    if rects:
                        used_semantic.add(sentence)
                        add(rects[0], "semantic", None, sentence[:80],
                            "Semantically similar to a sentence in the job description")

            pages.append({
                "page": pno + 1,
                "width": round(page.rect.width, 2),
                "height": round(page.rect.height, 2),
                "image_base64": image_b64,
                "boxes": boxes,
            })

        return {
            "has_text_layer": has_text,
            "semantic_unavailable": semantic_unavailable,
            "total_pages": doc.page_count,
            "pages": pages,
            "counts": counts,
            "legend": [dict(item) for item in LEGEND],
        }
    finally:
        doc.close()
