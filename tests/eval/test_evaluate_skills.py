"""Unit tests for scripts/evaluate_skills.py (no fabricated eval metrics)."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "evaluate_skills.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("evaluate_skills", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["evaluate_skills"] = mod
    spec.loader.exec_module(mod)
    return mod


eval_mod = _load_module()


def test_parse_true_skills_strips_and_lowercases():
    assert eval_mod.parse_true_skills(" Python; Docker ; ;SQL ") == [
        "python",
        "docker",
        "sql",
    ]


def test_compute_prf_perfect_and_zero():
    assert eval_mod.compute_prf(4, 0, 0) == (1.0, 1.0, 1.0)
    assert eval_mod.compute_prf(0, 0, 0) == (0.0, 0.0, 0.0)
    p, r, f1 = eval_mod.compute_prf(1, 1, 0)
    assert p == pytest.approx(0.5)
    assert r == pytest.approx(1.0)
    assert f1 == pytest.approx(2 / 3)


def test_aggregate_metrics_micro_macro_and_counts():
    records = [
        {
            "true_skills": {"a", "b"},
            "tp": ["a"],
            "fp": ["c"],
            "fn": ["b"],
            "precision": 0.5,
            "recall": 0.5,
            "f1": 0.5,
        },
        {
            "true_skills": {"a"},
            "tp": ["a"],
            "fp": [],
            "fn": [],
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        },
    ]
    m = eval_mod.aggregate_metrics(records)
    assert m["resumes_evaluated"] == 2
    assert m["total_labeled_skills"] == 3
    assert m["total_tp"] == 2
    assert m["total_fp"] == 1
    assert m["total_fn"] == 1
    assert m["micro_precision"] == pytest.approx(2 / 3)
    assert m["micro_recall"] == pytest.approx(2 / 3)
    assert m["macro_precision"] == pytest.approx(0.75)
    assert m["macro_recall"] == pytest.approx(0.75)
    assert m["macro_f1"] == pytest.approx(0.75)


def test_aggregate_metrics_empty():
    m = eval_mod.aggregate_metrics([])
    assert m["resumes_evaluated"] == 0
    assert m["total_labeled_skills"] == 0
    assert m["macro_precision"] == 0.0
    assert m["macro_recall"] == 0.0
    assert m["macro_f1"] == 0.0
    assert m["micro_precision"] == 0.0
    assert m["micro_recall"] == 0.0
    assert m["micro_f1"] == 0.0


def test_top_errors_with_ids_orders_by_count():
    skill_to_ids = {
        "docker": ["r1"],
        "python": ["r1", "r2", "r3"],
        "java": ["r2", "r4"],
    }
    top = eval_mod.top_errors_with_ids(skill_to_ids, n=2)
    assert top[0][0] == "python"
    assert top[0][1] == 3
    assert top[0][2] == ["r1", "r2", "r3"]
    assert top[1][0] == "java"
    assert top[1][1] == 2


def test_update_eval_notes_replaces_skill_extraction_section(tmp_path: Path):
    notes = tmp_path / "EVAL_NOTES.md"
    notes.write_text(
        "# Notes\n\n## Skill extraction\n\nold junk\n\n## Other\n\nkeep me\n",
        encoding="utf-8",
    )
    metrics = {
        "resumes_evaluated": 2,
        "total_labeled_skills": 5,
        "micro_precision": 0.8,
        "micro_recall": 0.7,
        "micro_f1": 0.7466,
        "macro_precision": 0.75,
        "macro_recall": 0.72,
        "macro_f1": 0.7348,
    }
    eval_mod.update_eval_notes(notes, metrics)
    text = notes.read_text(encoding="utf-8")
    assert "## Skill extraction" in text
    assert "| Resumes evaluated | 2 |" in text
    assert "| Total labeled skills | 5 |" in text
    assert "old junk" not in text
    assert "## Other" in text
    assert "keep me" in text


def test_load_canonical_skills_includes_known_names():
    canonical = eval_mod.load_canonical_skills()
    assert "python" in canonical
    assert "docker" in canonical


def _write_csv(path: Path, rows: list[tuple[str, str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["resume_id", "resume_file", "true_skills"])
        for row in rows:
            writer.writerow(row)


def test_run_evaluation_exits_1_when_no_labels(tmp_path: Path, capsys):
    csv_path = tmp_path / "labels.csv"
    resumes_dir = tmp_path / "resumes"
    resumes_dir.mkdir()
    (resumes_dir / "a.txt").write_text("Python Docker", encoding="utf-8")
    _write_csv(
        csv_path,
        [
            ("r1", "a.txt", ""),
            ("r2", "a.txt", ""),
        ],
    )
    out_csv = tmp_path / "out.csv"
    notes = tmp_path / "EVAL_NOTES.md"
    notes.write_text("# Notes\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        eval_mod.run_evaluation(csv_path, resumes_dir, out_csv, notes)
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "empty" in captured.out.lower() or "no labeled" in captured.out.lower()
    assert "Skipped 2 resume(s) with an empty" in captured.out
    # Must not invent metrics in docs
    assert "| Micro-averaged" not in notes.read_text(encoding="utf-8")


def test_run_evaluation_reports_non_canonical_and_writes_summary(
    tmp_path: Path, capsys
):
    csv_path = tmp_path / "labels.csv"
    resumes_dir = tmp_path / "resumes"
    resumes_dir.mkdir()
    resume_file = "sample.txt"
    (resumes_dir / resume_file).write_text(
        "Alex Engineer\nSkills: Python, Docker, PostgreSQL, FastAPI\n",
        encoding="utf-8",
    )
    # Includes one non-canonical label; must report, not crash
    _write_csv(
        csv_path,
        [
            (
                "resume_x",
                resume_file,
                "python;docker;postgresql;fastapi;not_a_real_skill_xyz",
            ),
        ],
    )
    out_csv = tmp_path / "out.csv"
    notes = tmp_path / "EVAL_NOTES.md"
    notes.write_text("# Notes\n\n## Prior\n\nbody\n", encoding="utf-8")

    eval_mod.run_evaluation(csv_path, resumes_dir, out_csv, notes)

    captured = capsys.readouterr()
    assert "not_a_real_skill_xyz" in captured.out
    assert "non-canonical" in captured.out.lower() or "not in canonical" in captured.out
    assert out_csv.exists()
    text = notes.read_text(encoding="utf-8")
    assert "## Skill extraction" in text
    assert "| Resumes evaluated | 1 |" in text
    assert "| Total labeled skills | 5 |" in text
    assert "## Prior" in text


def test_run_evaluation_skips_empty_labels_in_mixed_csv(tmp_path: Path, capsys):
    csv_path = tmp_path / "mixed.csv"
    resumes_dir = tmp_path / "resumes"
    resumes_dir.mkdir()
    (resumes_dir / "valid.txt").write_text(
        "Taylor\nSkills: Python, Docker\n", encoding="utf-8"
    )
    (resumes_dir / "empty.txt").write_text(
        "Jordan\nSkills: React\n", encoding="utf-8"
    )
    _write_csv(
        csv_path,
        [
            ("r_empty", "empty.txt", ""),
            ("r_valid", "valid.txt", "python;docker"),
        ],
    )
    out_csv = tmp_path / "out.csv"
    notes = tmp_path / "EVAL_NOTES.md"
    notes.write_text("# Notes\n\n## Skill extraction\n\nplaceholder\n", encoding="utf-8")

    eval_mod.run_evaluation(csv_path, resumes_dir, out_csv, notes)

    captured = capsys.readouterr()
    assert "Skipped 1 resume(s) with an empty 'true_skills' label cell." in captured.out
    assert "Total Resumes Evaluated : 1" in captured.out
    text = notes.read_text(encoding="utf-8")
    assert "| Resumes evaluated | 1 |" in text
    assert "| Total labeled skills | 2 |" in text


def test_main_cli_with_eval_notes_flag(tmp_path: Path):
    csv_path = tmp_path / "cli.csv"
    resumes_dir = tmp_path / "resumes"
    resumes_dir.mkdir()
    (resumes_dir / "cli_sample.txt").write_text(
        "Candidate\nSkills: Python\n", encoding="utf-8"
    )
    _write_csv(csv_path, [("r1", "cli_sample.txt", "python")])
    out_csv = tmp_path / "cli_out.csv"
    notes = tmp_path / "CLI_EVAL_NOTES.md"
    notes.write_text("# Heading\n", encoding="utf-8")

    test_args = [
        "evaluate_skills.py",
        "--csv",
        str(csv_path),
        "--resumes-dir",
        str(resumes_dir),
        "--output",
        str(out_csv),
        "--eval-notes",
        str(notes),
    ]
    with patch.object(sys, "argv", test_args):
        eval_mod.main()

    assert out_csv.exists()
    assert "## Skill extraction" in notes.read_text(encoding="utf-8")
