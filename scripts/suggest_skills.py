"""Suggest new skill candidates by scanning resume documents.

Scans .txt, .pdf, and .docx files in a folder, extracts noun chunks and
capitalized or technical tokens not present in the current skills taxonomy,
ranks candidates by frequency across files, and outputs candidates.csv for
manual review.

NOTE: This script NEVER automatically modifies skills.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import spacy

try:
    from app.parser import extract_text
except ImportError:
    extract_text = None

# Default path to skills taxonomy
DEFAULT_SKILLS_JSON = Path(__file__).resolve().parent.parent / "app" / "data" / "skills.json"

# Common non-skill stopwords and resume boilerplate to exclude
COMMON_IGNORE_WORDS = {
    "resume", "curriculum vitae", "cv", "experience", "education",
    "skills", "work experience", "summary", "profile", "professional summary",
    "projects", "certifications", "languages", "references", "contact",
    "responsibilities", "responsible", "duties", "achievements",
    "bachelor", "master", "phd", "degree", "diploma", "university",
    "college", "institute", "school", "high school", "academy",
    "year", "years", "month", "months", "time", "date", "present",
    "team", "member", "role", "position", "lead", "developer",
    "engineer", "manager", "intern", "company", "client", "job",
    "work", "project", "task", "candidate", "applicant", "author",
    "email", "phone", "address", "city", "country", "location",
    "name", "first name", "last name", "page", "pages",
}

DETERMINERS = {"the", "a", "an", "this", "that", "these", "those", "my", "our", "their", "his", "her", "its"}


def load_taxonomy_terms(skills_json_path: Optional[Union[str, Path]] = None) -> Set[str]:
    """Load all canonical and alias skill terms (lowercased) from skills.json."""
    path = Path(skills_json_path) if skills_json_path else DEFAULT_SKILLS_JSON
    if not path.exists():
        return set()

    with open(path, "r", encoding="utf-8") as fh:
        taxonomy = json.load(fh)

    terms: Set[str] = set()
    for category, skills in taxonomy.get("skills", {}).items():
        for canonical, meta in skills.items():
            terms.add(canonical.lower())
            for alias in meta.get("aliases", []):
                terms.add(alias.lower())

    return terms


def extract_file_text(file_path: Path) -> str:
    """Extract plain text from .txt, .pdf, or .docx file."""
    ext = file_path.suffix.lower()
    if ext == ".txt":
        return file_path.read_text(encoding="utf-8", errors="ignore")
    elif ext in (".pdf", ".docx"):
        if extract_text is not None:
            try:
                return extract_text(file_path)
            except Exception:
                pass
        # Fallback reading
        if ext == ".docx":
            try:
                import docx
                doc = docx.Document(file_path)
                return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            except Exception:
                return ""
        elif ext == ".pdf":
            try:
                import fitz
                doc = fitz.open(file_path)
                return "\n".join(page.get_text() for page in doc)
            except Exception:
                return ""
    return ""


def clean_chunk_text(text: str) -> str:
    """Clean a noun chunk by stripping leading determiners and punctuation."""
    cleaned = text.strip(" \t\n\r\"'()[]{}.,:;!?-_/\\")
    words = cleaned.split()
    while words and words[0].lower() in DETERMINERS:
        words.pop(0)
    return " ".join(words).strip(" \t\n\r\"'()[]{}.,:;!?-_/\\")


def is_valid_candidate(term: str, existing_terms: Set[str]) -> bool:
    """Check if candidate term is a valid potential technical skill."""
    norm = term.lower()
    if not norm or len(norm) < 2 or len(norm) > 40:
        return False
    if norm in existing_terms:
        return False
    if norm in COMMON_IGNORE_WORDS:
        return False
    if re.match(r"^[\d\s\-_.,/]+$", norm):
        return False
    if re.match(r"^\d+\+?\s*(?:years?|yrs?|months?|mos?)$", norm):
        return False
    # Avoid generic English fragments
    words = norm.split()
    if len(words) > 4:
        return False
    if all(w in DETERMINERS or len(w) <= 1 for w in words):
        return False
    return True


def scan_text_for_candidates(
    text: str,
    existing_terms: Set[str],
    nlp: Optional[spacy.language.Language] = None,
) -> List[Tuple[str, str]]:
    """Scan text for noun chunks and technical/capitalized tokens.

    Returns:
        List of (candidate_term, example_sentence)
    """
    if not text.strip():
        return []

    if nlp is None:
        nlp = spacy.load("en_core_web_sm", disable=["ner"])

    doc = nlp(text)
    candidates: List[Tuple[str, str]] = []

    # 1. Noun chunks
    try:
        for chunk in doc.noun_chunks:
            cleaned = clean_chunk_text(chunk.text)
            if is_valid_candidate(cleaned, existing_terms):
                sent_text = chunk.sent.text.strip().replace("\n", " ")
                candidates.append((cleaned, sent_text))
    except Exception:
        pass

    # 2. Capitalized or technical tokens (CamelCase, technical acronyms, dots, slashes)
    tech_re = re.compile(
        r"^(?:[A-Z][a-z0-9]+[A-Z][a-zA-Z0-9]*|[A-Z]{2,}[0-9]*|[a-zA-Z0-9]+[\.\-_/][a-zA-Z0-9]+)$"
    )

    for token in doc:
        token_text = token.text.strip(" \t\n\r\"'()[]{}.,:;!?")
        if not token_text:
            continue
        # Check if technical pattern or PROPN
        if tech_re.match(token_text) or (token.pos_ == "PROPN" and token_text[0].isupper() and token.i > 0):
            if is_valid_candidate(token_text, existing_terms):
                try:
                    sent_text = token.sent.text.strip().replace("\n", " ")
                except Exception:
                    sent_text = text[:150]
                candidates.append((token_text, sent_text))

    return candidates


def suggest_skills(
    folder_path: Union[str, Path],
    output_csv: Optional[Union[str, Path]] = None,
    skills_json_path: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Union[str, int]]]:
    """Scan folder for resume documents and write candidates.csv ranked by frequency.

    Parameters
    ----------
    folder_path : str or Path
        Directory containing .txt, .pdf, or .docx files.
    output_csv : str or Path, optional
        Path for candidates.csv output. Defaults to <folder_path>/candidates.csv.
    skills_json_path : str or Path, optional
        Path to skills.json taxonomy.

    Returns
    -------
    List[dict]
        Ranked list of dicts: {"term": str, "count": int, "example_sentence": str}
    """
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    if output_csv is None:
        csv_path = folder / "candidates.csv"
    else:
        csv_path = Path(output_csv)

    existing_terms = load_taxonomy_terms(skills_json_path)
    nlp = spacy.load("en_core_web_sm", disable=["ner"])

    # Collect files
    supported_exts = {".txt", ".pdf", ".docx"}
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in supported_exts]

    term_counts: Counter[str] = Counter()
    term_files: Dict[str, Set[str]] = defaultdict(set)
    term_display: Dict[str, str] = {}
    term_examples: Dict[str, str] = {}

    for file_path in files:
        file_text = extract_file_text(file_path)
        if not file_text:
            continue

        file_candidates = scan_text_for_candidates(file_text, existing_terms, nlp=nlp)
        seen_in_file: Set[str] = set()
        for term, example_sentence in file_candidates:
            norm = term.lower()
            if norm in seen_in_file:
                continue
            seen_in_file.add(norm)

            term_counts[norm] += 1
            term_files[norm].add(file_path.name)

            if norm not in term_display or (term[0].isupper() and not term_display[norm][0].isupper()):
                term_display[norm] = term

            if norm not in term_examples:
                term_examples[norm] = example_sentence

    # Rank by frequency across files, then total count
    ranked_norms = sorted(
        term_counts.keys(),
        key=lambda n: (-len(term_files[n]), -term_counts[n], term_display.get(n, n)),
    )

    results: List[Dict[str, Union[str, int]]] = []
    for norm in ranked_norms:
        results.append({
            "term": term_display.get(norm, norm),
            "count": term_counts[norm],
            "example_sentence": term_examples.get(norm, ""),
        })

    # Write candidates.csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["term", "count", "example_sentence"])
        for item in results:
            writer.writerow([item["term"], item["count"], item["example_sentence"]])

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Suggest candidate skills from resumes.")
    parser.add_argument("folder", type=str, help="Folder containing resumes (.txt, .pdf, .docx)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output candidates.csv path")
    parser.add_argument("--taxonomy", "-t", type=str, default=None, help="Path to skills.json taxonomy")
    args = parser.parse_args()

    results = suggest_skills(args.folder, output_csv=args.output, skills_json_path=args.taxonomy)
    print(f"Extraction complete. Found {len(results)} candidate skills.")


if __name__ == "__main__":
    main()
