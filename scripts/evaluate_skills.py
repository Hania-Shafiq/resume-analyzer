#!/usr/bin/env python3
"""Skill Extraction Evaluation Script.

Evaluates the project's real skill extractor against ground-truth hand-labeled resumes.
Computes per-resume precision, recall, F1, as well as micro- and macro-averaged metrics.
Tracks false positives and false negatives, and exports detailed results to CSV.
Writes a summary markdown table into docs/EVAL_NOTES.md.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import re
import sys

# Ensure repository root is on sys.path so app modules import cleanly
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.parser import extract_text
from app.preprocess import light_clean
from app.skills import extract_skills

DEFAULT_CSV = REPO_ROOT / "tests" / "eval" / "skill_eval_template.csv"
DEFAULT_RESUMES_DIR = REPO_ROOT / "tests" / "eval" / "resumes"
DEFAULT_RESULTS_CSV = REPO_ROOT / "tests" / "eval" / "skill_eval_results.csv"
DEFAULT_EVAL_NOTES = REPO_ROOT / "docs" / "EVAL_NOTES.md"
SKILLS_JSON = REPO_ROOT / "app" / "data" / "skills.json"

SKILL_EXTRACTION_HEADING = "Skill extraction"
TOP_ERROR_N = 15


def load_canonical_skills(skills_json: Path = SKILLS_JSON) -> set[str]:
    """Load all known canonical skill names from skills.json."""
    if not skills_json.exists():
        return set()
    try:
        with open(skills_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        canonical: set[str] = set()
        for cat_skills in data.get("skills", {}).values():
            canonical.update(cat_skills.keys())
        return canonical
    except Exception:
        return set()


def parse_true_skills(raw_true: str) -> list[str]:
    """Parse semicolon-separated skill labels; preserve order, drop empties."""
    return [s.strip().lower() for s in raw_true.split(";") if s.strip()]


def compute_prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Return (precision, recall, f1) from contingency counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def aggregate_metrics(evaluated_records: list[dict]) -> dict:
    """Compute micro/macro averages plus resume and label counts."""
    total_evaluated = len(evaluated_records)
    total_labeled_skills = sum(len(r["true_skills"]) for r in evaluated_records)

    if total_evaluated == 0:
        macro_precision = macro_recall = macro_f1 = 0.0
    else:
        macro_precision = sum(r["precision"] for r in evaluated_records) / total_evaluated
        macro_recall = sum(r["recall"] for r in evaluated_records) / total_evaluated
        macro_f1 = sum(r["f1"] for r in evaluated_records) / total_evaluated

    total_tp = sum(len(r["tp"]) for r in evaluated_records)
    total_fp = sum(len(r["fp"]) for r in evaluated_records)
    total_fn = sum(len(r["fn"]) for r in evaluated_records)

    micro_precision, micro_recall, micro_f1 = compute_prf(total_tp, total_fp, total_fn)

    return {
        "resumes_evaluated": total_evaluated,
        "total_labeled_skills": total_labeled_skills,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
    }


def top_errors_with_ids(
    skill_to_ids: dict[str, list[str]],
    n: int = TOP_ERROR_N,
) -> list[tuple[str, int, list[str]]]:
    """Return top-n skills by occurrence count with the resume ids involved."""
    ranked = sorted(
        (
            (skill, len(sorted(set(ids))), sorted(set(ids)))
            for skill, ids in skill_to_ids.items()
        ),
        key=lambda item: (-item[1], item[0]),
    )
    return ranked[:n]


def format_summary_markdown(metrics: dict) -> str:
    """Build the Skill extraction section body (heading + summary table)."""
    rows = [
        ("Resumes evaluated", str(metrics["resumes_evaluated"])),
        ("Total labeled skills", str(metrics["total_labeled_skills"])),
        ("Micro-averaged precision", f"{metrics['micro_precision']:.4f}"),
        ("Micro-averaged recall", f"{metrics['micro_recall']:.4f}"),
        ("Micro-averaged F1", f"{metrics['micro_f1']:.4f}"),
        ("Macro-averaged precision", f"{metrics['macro_precision']:.4f}"),
        ("Macro-averaged recall", f"{metrics['macro_recall']:.4f}"),
        ("Macro-averaged F1", f"{metrics['macro_f1']:.4f}"),
    ]
    lines = [
        f"## {SKILL_EXTRACTION_HEADING}",
        "",
        "| Metric | Value |",
        "| :--- | ---: |",
    ]
    for name, value in rows:
        lines.append(f"| {name} | {value} |")
    lines.append("")
    return "\n".join(lines)


def update_eval_notes(eval_notes_path: Path, metrics: dict) -> None:
    """Replace or append the Skill extraction section in EVAL_NOTES.md."""
    section = format_summary_markdown(metrics)
    # Match ## Skill extraction through the next same-level heading or EOF
    section_pattern = re.compile(
        rf"^## {re.escape(SKILL_EXTRACTION_HEADING)}\s*\n"
        r"(?:.*?\n)*?"
        r"(?=(?:^---\s*\n\s*)?^## |\Z)",
        re.MULTILINE,
    )

    eval_notes_path.parent.mkdir(parents=True, exist_ok=True)
    if eval_notes_path.exists():
        existing = eval_notes_path.read_text(encoding="utf-8")
    else:
        existing = ""

    if section_pattern.search(existing):
        updated = section_pattern.sub(section.rstrip() + "\n\n", existing)
        # Collapse excess blank lines introduced at EOF
        updated = re.sub(r"\n{3,}\Z", "\n", updated)
    else:
        if existing and not existing.endswith("\n"):
            existing += "\n"
        if existing and not existing.endswith("\n\n"):
            existing += "\n"
        updated = existing + section

    eval_notes_path.write_text(updated, encoding="utf-8")


def print_error_table(
    title: str,
    rows: list[tuple[str, int, list[str]]],
    empty_message: str,
) -> None:
    print(f"\n{title}")
    print("-" * 80)
    if not rows:
        print(f"  {empty_message}")
        return
    print(f"  {'#':>2}  {'Skill':<28} {'Count':>5}  Resume IDs")
    for rank, (skill, count, ids) in enumerate(rows, 1):
        id_str = ", ".join(ids)
        print(f"  {rank:>2}. {skill:<28} {count:>5}  {id_str}")


def run_evaluation(
    csv_path: Path,
    resumes_dir: Path,
    output_path: Path,
    eval_notes_path: Path = DEFAULT_EVAL_NOTES,
) -> None:
    if not csv_path.exists():
        print(f"Error: Evaluation CSV not found at '{csv_path}'", file=sys.stderr)
        sys.exit(1)

    canonical_skills = load_canonical_skills()

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print(f"Error: '{csv_path}' is empty.", file=sys.stderr)
        sys.exit(1)

    evaluated_records: list[dict] = []
    skipped_empty_label = 0
    skipped_other: list[tuple[str, str, str]] = []
    fp_skill_ids: dict[str, list[str]] = defaultdict(list)
    fn_skill_ids: dict[str, list[str]] = defaultdict(list)
    non_canonical_reports: list[tuple[str, list[str]]] = []

    print("=" * 80)
    print("SKILL EXTRACTION EVALUATION")
    print(f"Input CSV    : {csv_path}")
    print(f"Resumes Dir  : {resumes_dir}")
    print(f"Total Rows   : {len(rows)}")
    print("=" * 80)

    for row in rows:
        resume_id = (row.get("resume_id") or "").strip()
        resume_file = (row.get("resume_file") or "").strip()
        raw_true = (row.get("true_skills") or "").strip()

        if not raw_true:
            skipped_empty_label += 1
            continue

        file_path = resumes_dir / resume_file
        if not file_path.exists():
            skipped_other.append(
                (resume_id, resume_file, f"File not found in {resumes_dir}")
            )
            continue

        true_list = parse_true_skills(raw_true)
        true_skills = set(true_list)

        non_canonical = sorted(s for s in true_skills if s not in canonical_skills)
        if non_canonical:
            non_canonical_reports.append((resume_id, non_canonical))
            print(
                f"  [Notice] {resume_id}: labels not in canonical skills.json: "
                f"{non_canonical}"
            )

        try:
            with open(file_path, "rb") as bf:
                raw_bytes = bf.read()
            raw_text = extract_text(raw_bytes, filename=resume_file)
            cleaned = light_clean(raw_text)
            pred_skills = set(extract_skills(cleaned))
        except Exception as exc:
            skipped_other.append(
                (resume_id, resume_file, f"Extraction failed: {exc}")
            )
            continue

        tp = true_skills & pred_skills
        fp = pred_skills - true_skills
        fn = true_skills - pred_skills

        precision, recall, f1 = compute_prf(len(tp), len(fp), len(fn))

        for skill in fp:
            fp_skill_ids[skill].append(resume_id)
        for skill in fn:
            fn_skill_ids[skill].append(resume_id)

        evaluated_records.append({
            "resume_id": resume_id,
            "resume_file": resume_file,
            "true_skills": true_skills,
            "pred_skills": pred_skills,
            "tp": sorted(tp),
            "fp": sorted(fp),
            "fn": sorted(fn),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    if skipped_empty_label:
        print(
            f"\nSkipped {skipped_empty_label} resume(s) with an empty "
            f"'true_skills' label cell."
        )

    if non_canonical_reports:
        print(
            f"\nReported {len(non_canonical_reports)} resume(s) with "
            f"non-canonical label(s) (evaluation continued)."
        )

    total_evaluated = len(evaluated_records)

    if total_evaluated == 0:
        print("\nNo labeled resumes were evaluated.")
        if skipped_other:
            print(f"\nSkipped {len(skipped_other)} additional row(s):")
            for rid, rfile, reason in skipped_other:
                print(f"  - {rid} ({rfile}): {reason}")
        print(
            "\nThe CSV contains no usable labels. Fill the 'true_skills' column "
            "with semicolon-separated canonical skill names "
            "(e.g. 'python;docker;postgresql;fastapi') and re-run."
        )
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Print Per-Resume Detailed Breakdown
    # -----------------------------------------------------------------------
    print("\nPER-RESUME RESULTS:")
    print("-" * 80)
    for rec in evaluated_records:
        print(f"[{rec['resume_id']}] {rec['resume_file']}")
        print(
            f"  Metrics : Precision={rec['precision']:.3f} | "
            f"Recall={rec['recall']:.3f} | F1={rec['f1']:.3f}"
        )
        print(
            f"  True ({len(rec['true_skills'])})      : "
            f"{', '.join(sorted(rec['true_skills'])) or 'None'}"
        )
        print(
            f"  Predicted ({len(rec['pred_skills'])}): "
            f"{', '.join(sorted(rec['pred_skills'])) or 'None'}"
        )
        print(f"  TP ({len(rec['tp'])}): {', '.join(rec['tp']) or 'None'}")
        print(f"  FP ({len(rec['fp'])}): {', '.join(rec['fp']) or 'None'}")
        print(f"  FN ({len(rec['fn'])}): {', '.join(rec['fn']) or 'None'}")
        print("-" * 80)

    metrics = aggregate_metrics(evaluated_records)

    print("\n" + "=" * 80)
    print("OVERALL SUMMARY METRICS")
    print("=" * 80)
    print(f"Total Resumes Evaluated : {metrics['resumes_evaluated']}")
    print(f"Total Labeled Skills    : {metrics['total_labeled_skills']}")
    if skipped_empty_label or skipped_other:
        print(
            f"Total Resumes Skipped   : "
            f"{skipped_empty_label + len(skipped_other)} "
            f"(empty labels={skipped_empty_label}, other={len(skipped_other)})"
        )
    print(f"Total True Positives    : {metrics['total_tp']}")
    print(f"Total False Positives   : {metrics['total_fp']}")
    print(f"Total False Negatives   : {metrics['total_fn']}")
    print("-" * 80)
    print(
        f"Macro-Averaged Precision: {metrics['macro_precision']:.4f} "
        f"({metrics['macro_precision']*100:.2f}%)"
    )
    print(
        f"Macro-Averaged Recall   : {metrics['macro_recall']:.4f} "
        f"({metrics['macro_recall']*100:.2f}%)"
    )
    print(
        f"Macro-Averaged F1       : {metrics['macro_f1']:.4f} "
        f"({metrics['macro_f1']*100:.2f}%)"
    )
    print("-" * 80)
    print(
        f"Micro-Averaged Precision: {metrics['micro_precision']:.4f} "
        f"({metrics['micro_precision']*100:.2f}%)"
    )
    print(
        f"Micro-Averaged Recall   : {metrics['micro_recall']:.4f} "
        f"({metrics['micro_recall']*100:.2f}%)"
    )
    print(
        f"Micro-Averaged F1       : {metrics['micro_f1']:.4f} "
        f"({metrics['micro_f1']*100:.2f}%)"
    )
    print("=" * 80)

    fp_rows = top_errors_with_ids(fp_skill_ids, TOP_ERROR_N)
    fn_rows = top_errors_with_ids(fn_skill_ids, TOP_ERROR_N)

    print_error_table(
        f"TOP {TOP_ERROR_N} FALSE POSITIVE SKILLS "
        "(Extracted but NOT in true labels):",
        fp_rows,
        "(None - zero false positives detected)",
    )
    print_error_table(
        f"TOP {TOP_ERROR_N} FALSE NEGATIVE SKILLS "
        "(In true labels but MISSED by extractor):",
        fn_rows,
        "(None - zero false negatives detected)",
    )

    # -----------------------------------------------------------------------
    # Save Results to CSV
    # -----------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f)
        writer.writerow([
            "resume_id",
            "resume_file",
            "true_skills",
            "extracted_skills",
            "tp_count",
            "fp_count",
            "fn_count",
            "true_positives",
            "false_positives",
            "false_negatives",
            "precision",
            "recall",
            "f1",
        ])
        for rec in evaluated_records:
            writer.writerow([
                rec["resume_id"],
                rec["resume_file"],
                ";".join(sorted(rec["true_skills"])),
                ";".join(sorted(rec["pred_skills"])),
                len(rec["tp"]),
                len(rec["fp"]),
                len(rec["fn"]),
                ";".join(rec["tp"]),
                ";".join(rec["fp"]),
                ";".join(rec["fn"]),
                round(rec["precision"], 4),
                round(rec["recall"], 4),
                round(rec["f1"], 4),
            ])

    print(f"\nSaved detailed evaluation results to: {output_path}")

    update_eval_notes(eval_notes_path, metrics)
    print(f"Updated skill-extraction summary in: {eval_notes_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate resume skill extraction against ground truth hand-labels."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Path to labels CSV (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--resumes-dir",
        type=Path,
        default=DEFAULT_RESUMES_DIR,
        help=f"Path to resume directory (default: {DEFAULT_RESUMES_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS_CSV,
        help=f"Path to save output results CSV (default: {DEFAULT_RESULTS_CSV})",
    )
    parser.add_argument(
        "--eval-notes",
        type=Path,
        default=DEFAULT_EVAL_NOTES,
        help=f"Path to EVAL_NOTES markdown file (default: {DEFAULT_EVAL_NOTES})",
    )
    args = parser.parse_args()
    run_evaluation(
        args.csv,
        args.resumes_dir,
        args.output,
        eval_notes_path=args.eval_notes,
    )


if __name__ == "__main__":
    main()
