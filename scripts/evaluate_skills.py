#!/usr/bin/env python3
"""Skill Extraction Evaluation Script.

Evaluates the project's real skill extractor against ground-truth hand-labeled resumes.
Computes per-resume precision, recall, F1, as well as micro- and macro-averaged metrics.
Tracks false positives and false negatives, and exports detailed results to CSV.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
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
SKILLS_JSON = REPO_ROOT / "app" / "data" / "skills.json"


def load_canonical_skills() -> set[str]:
    """Load all known canonical skill names from skills.json."""
    if not SKILLS_JSON.exists():
        return set()
    try:
        with open(SKILLS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        canonical = set()
        for cat_skills in data.get("skills", {}).values():
            canonical.update(cat_skills.keys())
        return canonical
    except Exception:
        return set()


def run_evaluation(
    csv_path: Path,
    resumes_dir: Path,
    output_path: Path,
) -> None:
    if not csv_path.exists():
        print(f"Error: Evaluation CSV not found at '{csv_path}'", file=sys.stderr)
        sys.exit(1)

    canonical_skills = load_canonical_skills()

    # Read rows
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print(f"Error: '{csv_path}' is empty.", file=sys.stderr)
        sys.exit(1)

    evaluated_records = []
    skipped_records = []
    all_fps: list[str] = []
    all_fns: list[str] = []

    print("=" * 80)
    print("SKILL EXTRACTION EVALUATION")
    print(f"Input CSV    : {csv_path}")
    print(f"Resumes Dir  : {resumes_dir}")
    print(f"Total Rows   : {len(rows)}")
    print("=" * 80)

    for row in rows:
        resume_id = row.get("resume_id", "").strip()
        resume_file = row.get("resume_file", "").strip()
        raw_true = row.get("true_skills", "").strip()

        if not raw_true:
            skipped_records.append((resume_id, resume_file, "true_skills column is empty"))
            continue

        file_path = resumes_dir / resume_file
        if not file_path.exists():
            skipped_records.append((resume_id, resume_file, f"File not found in {resumes_dir}"))
            continue

        # Parse ground-truth skills (semicolon-separated)
        true_skills = set(
            s.strip().lower() for s in raw_true.split(";") if s.strip()
        )

        # Warn if any user-labeled skill is not in canonical taxonomy
        non_canonical = [s for s in true_skills if s not in canonical_skills]
        if non_canonical:
            print(f"  [Notice] {resume_id}: Skills not in canonical skills.json: {non_canonical}")

        # Extract skills using real pipeline (extract_text -> light_clean -> extract_skills)
        try:
            with open(file_path, "rb") as bf:
                raw_bytes = bf.read()
            raw_text = extract_text(raw_bytes, filename=resume_file)
            cleaned = light_clean(raw_text)
            pred_skills = set(extract_skills(cleaned))
        except Exception as exc:
            skipped_records.append((resume_id, resume_file, f"Extraction failed: {exc}"))
            continue

        tp = true_skills & pred_skills
        fp = pred_skills - true_skills
        fn = true_skills - pred_skills

        precision = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) > 0 else 0.0
        recall = len(tp) / (len(tp) + len(fn)) if (len(tp) + len(fn)) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        all_fps.extend(fp)
        all_fns.extend(fn)

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

    total_evaluated = len(evaluated_records)

    if total_evaluated == 0:
        print("\nNo labeled resumes were evaluated.")
        if skipped_records:
            print(f"\nSkipped {len(skipped_records)} row(s):")
            for rid, rfile, reason in skipped_records:
                print(f"  - {rid} ({rfile}): {reason}")
        print("\nPlease fill the 'true_skills' column in your CSV with semicolon-separated")
        print("canonical skill names (e.g. 'python;docker;postgresql;fastapi') and re-run.")
        sys.exit(0)

    # -----------------------------------------------------------------------
    # Print Per-Resume Detailed Breakdown
    # -----------------------------------------------------------------------
    print("\nPER-RESUME RESULTS:")
    print("-" * 80)
    for rec in evaluated_records:
        print(f"[{rec['resume_id']}] {rec['resume_file']}")
        print(f"  Metrics : Precision={rec['precision']:.3f} | Recall={rec['recall']:.3f} | F1={rec['f1']:.3f}")
        print(f"  True ({len(rec['true_skills'])})      : {', '.join(sorted(rec['true_skills'])) or 'None'}")
        print(f"  Predicted ({len(rec['pred_skills'])}): {', '.join(sorted(rec['pred_skills'])) or 'None'}")
        print(f"  TP ({len(rec['tp'])}): {', '.join(rec['tp']) or 'None'}")
        print(f"  FP ({len(rec['fp'])}): {', '.join(rec['fp']) or 'None'}")
        print(f"  FN ({len(rec['fn'])}): {', '.join(rec['fn']) or 'None'}")
        print("-" * 80)

    # -----------------------------------------------------------------------
    # Aggregate Metrics (Macro and Micro)
    # -----------------------------------------------------------------------
    macro_precision = sum(r["precision"] for r in evaluated_records) / total_evaluated
    macro_recall = sum(r["recall"] for r in evaluated_records) / total_evaluated
    macro_f1 = sum(r["f1"] for r in evaluated_records) / total_evaluated

    total_tp = sum(len(r["tp"]) for r in evaluated_records)
    total_fp = sum(len(r["fp"]) for r in evaluated_records)
    total_fn = sum(len(r["fn"]) for r in evaluated_records)

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)) if (micro_precision + micro_recall) > 0 else 0.0

    print("\n" + "=" * 80)
    print("OVERALL SUMMARY METRICS")
    print("=" * 80)
    print(f"Total Resumes Evaluated : {total_evaluated}")
    if skipped_records:
        print(f"Total Resumes Skipped   : {len(skipped_records)} (unlabeled or missing)")
    print(f"Total True Positives    : {total_tp}")
    print(f"Total False Positives   : {total_fp}")
    print(f"Total False Negatives   : {total_fn}")
    print("-" * 80)
    print(f"Macro-Averaged Precision: {macro_precision:.4f} ({macro_precision*100:.2f}%)")
    print(f"Macro-Averaged Recall   : {macro_recall:.4f} ({macro_recall*100:.2f}%)")
    print(f"Macro-Averaged F1       : {macro_f1:.4f} ({macro_f1*100:.2f}%)")
    print("-" * 80)
    print(f"Micro-Averaged Precision: {micro_precision:.4f} ({micro_precision*100:.2f}%)")
    print(f"Micro-Averaged Recall   : {micro_recall:.4f} ({micro_recall*100:.2f}%)")
    print(f"Micro-Averaged F1       : {micro_f1:.4f} ({micro_f1*100:.2f}%)")
    print("=" * 80)

    # -----------------------------------------------------------------------
    # Top 10 False Positives and False Negatives
    # -----------------------------------------------------------------------
    fp_counts = Counter(all_fps).most_common(10)
    fn_counts = Counter(all_fns).most_common(10)

    print("\nTOP 10 MOST FREQUENT FALSE POSITIVE SKILLS (Extracted but NOT in true labels):")
    print("-" * 80)
    if fp_counts:
        for rank, (skill, count) in enumerate(fp_counts, 1):
            print(f"  {rank:>2}. {skill:<28} : {count} occurrence(s)")
    else:
        print("  (None - zero false positives detected)")

    print("\nTOP 10 MOST FREQUENT FALSE NEGATIVE SKILLS (In true labels but MISSED by extractor):")
    print("-" * 80)
    if fn_counts:
        for rank, (skill, count) in enumerate(fn_counts, 1):
            print(f"  {rank:>2}. {skill:<28} : {count} occurrence(s)")
    else:
        print("  (None - zero false negatives detected)")

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


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate resume skill extraction against ground truth hand-labels."
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
    args = parser.parse_args()
    run_evaluation(args.csv, args.resumes_dir, args.output)


if __name__ == "__main__":
    main()
