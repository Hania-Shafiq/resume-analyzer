"""Evaluate redaction recall and over-redaction on test resumes.

Reads labeled PII from tests/eval/pii_labels.csv, runs anonymize() on each
resume in tests/eval/resumes/, and reports per-type recall and over-redactions.
Writes results into docs/EVAL_NOTES.md under 'Anonymizer'.
"""

from __future__ import annotations

import csv
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.experience import extract_experience
from app.parser import extract_text
from app.preprocess import light_clean
from app.redact import anonymize
from app.skills import extract_skills

PII_LABELS_PATH = PROJECT_ROOT / "tests" / "eval" / "pii_labels.csv"
RESUMES_DIR = PROJECT_ROOT / "tests" / "eval" / "resumes"
EVAL_NOTES_PATH = PROJECT_ROOT / "docs" / "EVAL_NOTES.md"


def load_pii_labels(filepath: Path) -> List[Dict[str, str]]:
    """Load ground truth PII labels from CSV file.

    Returns empty list if file doesn't exist, is empty, or only contains header/examples.
    """
    if not filepath.exists() or filepath.stat().st_size == 0:
        return []

    rows: List[Dict[str, str]] = []
    with filepath.open(mode="r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Check for common column names
            file_key = next((k for k in ("resume_file", "filename", "file") if k in r), None)
            type_key = next((k for k in ("pii_type", "type") if k in r), None)
            val_key = next((k for k in ("pii_value", "value", "labeled_value") if k in r), None)

            if not file_key or not type_key or not val_key:
                continue

            resume_file = (r[file_key] or "").strip()
            pii_type = (r[type_key] or "").strip().lower()
            pii_value = (r[val_key] or "").strip()

            # Skip header repeats or empty values
            if not resume_file or not pii_type or not pii_value:
                continue

            # Check if example marker
            id_val = r.get("label_id", "") or r.get("id", "")
            if id_val.startswith("EXAMPLE") and not (RESUMES_DIR / resume_file).exists():
                continue

            rows.append({
                "resume_file": resume_file,
                "pii_type": pii_type,
                "pii_value": pii_value,
            })

    return rows


def read_resume_file(filename: str) -> str:
    """Read and extract raw text from PDF, DOCX, or TXT resume file."""
    path = RESUMES_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Resume file not found: {path}")

    ext = path.suffix.lower()
    if ext == ".txt":
        return path.read_text(encoding="utf-8", errors="replace")

    data = path.read_bytes()
    return extract_text(data, filename=filename)


def run_evaluation() -> int:
    """Run anonymizer evaluation against labeled ground truth."""
    if not PII_LABELS_PATH.exists():
        print(f"[Evaluation] Label file not found: {PII_LABELS_PATH}")
        print("Please provide tests/eval/pii_labels.csv with ground truth labels to evaluate redaction.")
        return 0

    labels = load_pii_labels(PII_LABELS_PATH)
    if not labels:
        print("[Evaluation] tests/eval/pii_labels.csv is empty or contains no valid test labels.")
        print("Stopping with a clear message: No made-up results.")
        return 0

    # Group labels by resume file
    labels_by_file: Dict[str, List[Dict[str, str]]] = {}
    for item in labels:
        labels_by_file.setdefault(item["resume_file"], []).append(item)

    total_labels = len(labels)
    eval_files_count = len(labels_by_file)

    tp_by_type: Dict[str, int] = {}
    total_by_type: Dict[str, int] = {}
    over_redacted_skills = 0
    over_redacted_dates = 0

    print(f"Evaluating {eval_files_count} resumes against {total_labels} labeled PII items...\n")

    for filename, file_labels in labels_by_file.items():
        try:
            orig_text = read_resume_file(filename)
        except Exception as exc:
            print(f"Warning: Could not read {filename}: {exc}")
            continue

        result = anonymize(orig_text)
        anon_text = result["text"]
        anon_text_lower = anon_text.lower()

        # Check recall: each labeled PII value should be gone from the output
        for lbl in file_labels:
            ptype = lbl["pii_type"]
            pval = lbl["pii_value"]
            total_by_type[ptype] = total_by_type.get(ptype, 0) + 1

            if pval.lower() not in anon_text_lower:
                tp_by_type[ptype] = tp_by_type.get(ptype, 0) + 1

        # Check over-redaction of skills
        try:
            orig_skills = set(extract_skills(light_clean(orig_text)))
            anon_skills = set(extract_skills(light_clean(anon_text)))
            missing_skills = orig_skills - anon_skills
            over_redacted_skills += len(missing_skills)
        except Exception:
            pass

        # Check over-redaction of work dates
        try:
            exp = extract_experience(orig_text)
            for entry in exp.entries:
                for date_str in (entry.start_date_str, entry.end_date_str):
                    if date_str and date_str.lower() not in ("present", "current", ""):
                        if date_str in orig_text and date_str not in anon_text:
                            # Verify if it was in experience section and removed
                            over_redacted_dates += 1
        except Exception:
            pass

    total_over_redaction = over_redacted_skills + over_redacted_dates

    # Print Report
    print("=" * 60)
    print("Anonymizer Redaction Evaluation Report")
    print("=" * 60)
    print(f"Resumes evaluated:        {eval_files_count}")
    print(f"Total labeled PII:        {total_labels}")
    print(f"Over-redaction count:     {total_over_redaction} (skills: {over_redacted_skills}, work dates: {over_redacted_dates})")
    print("-" * 60)
    print(f"{'PII Type':<15} {'Labeled':<10} {'Redacted (TP)':<15} {'Recall':<10}")
    print("-" * 60)

    total_tp = 0
    recalls = []
    for ptype, count in sorted(total_by_type.items()):
        tp = tp_by_type.get(ptype, 0)
        total_tp += tp
        recall = tp / count if count > 0 else 0.0
        recalls.append(recall)
        print(f"{ptype:<15} {count:<10} {tp:<15} {recall * 100:.1f}%")

    overall_recall = total_tp / total_labels if total_labels > 0 else 0.0
    macro_recall = sum(recalls) / len(recalls) if recalls else 0.0
    print("-" * 60)
    print(f"{'Overall (micro)':<15} {total_labels:<10} {total_tp:<15} {overall_recall * 100:.1f}%")
    print(f"{'Macro Recall':<15} {'-':<10} {'-':<15} {macro_recall * 100:.1f}%")
    print("=" * 60)

    # Format Markdown and write into docs/EVAL_NOTES.md
    md_lines = [
        "",
        "---",
        "",
        "## Anonymizer",
        "",
        "Evaluation of blind-screening anonymizer on labeled PII dataset (`tests/eval/pii_labels.csv`).",
        "",
        "| Metric | Value |",
        "| :--- | ---: |",
        f"| Resumes evaluated | {eval_files_count} |",
        f"| Total labeled PII items | {total_labels} |",
        f"| Micro-averaged recall | {overall_recall:.4f} |",
        f"| Macro-averaged recall | {macro_recall:.4f} |",
        f"| Over-redaction count (total) | {total_over_redaction} |",
        f"| Over-redacted skills | {over_redacted_skills} |",
        f"| Over-redacted work dates | {over_redacted_dates} |",
        "",
        "### Per-Type Recall Breakdown",
        "",
        "| PII Type | Labeled Count | Redacted (TP) | Recall |",
        "| :--- | ---: | ---: | ---: |",
    ]
    for ptype, count in sorted(total_by_type.items()):
        tp = tp_by_type.get(ptype, 0)
        rec = tp / count if count > 0 else 0.0
        md_lines.append(f"| {ptype} | {count} | {tp} | {rec:.4f} |")
    md_lines.append("")

    markdown_block = "\n".join(md_lines)

    if EVAL_NOTES_PATH.exists():
        content = EVAL_NOTES_PATH.read_text(encoding="utf-8", errors="replace")
        if "## Anonymizer" in content:
            # Replace existing ## Anonymizer section
            pattern = re.compile(r"\n---\s*\n## Anonymizer.*?(?=\n---\s*\n## |\Z)", re.DOTALL)
            if pattern.search(content):
                updated = pattern.sub(markdown_block, content)
            else:
                updated = content.rstrip() + "\n" + markdown_block
        else:
            updated = content.rstrip() + "\n" + markdown_block
        EVAL_NOTES_PATH.write_text(updated, encoding="utf-8")
        print(f"Updated results in {EVAL_NOTES_PATH} under 'Anonymizer'.")

    return 0


if __name__ == "__main__":
    sys.exit(run_evaluation())
