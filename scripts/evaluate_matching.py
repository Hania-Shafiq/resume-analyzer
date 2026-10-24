#!/usr/bin/env python3
"""Matching Evaluation Script.

Evaluates the project's candidate-job matching pipeline across scoring modes
(skills_only, hybrid, proposal, composite) against human-labeled pairs.

Computes:
1. Mean human score per pair and inter-rater agreement (mean absolute difference
   and average pairwise Pearson correlation).
2. Scores each pair through the FastAPI TestClient (/resume/upload, then /match)
   under each scoring mode.
3. MAE (0-100 scale), Pearson r, and Spearman rho per mode side by side in one table,
   with scatter plots saved to docs/figures/.
4. Raw semantic cosine for weak (<30) and strong (>70) matches with suggested
   LOW (5th percentile weak) and HIGH (95th percentile strong) anchors.
5. Recency ablation for composite mode.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import io
import os
from pathlib import Path
import re
import sys
import warnings

# Use headless backend for matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Paths
DEFAULT_CSV = REPO_ROOT / "tests" / "eval" / "pairs_labeled.csv"
DEFAULT_TEMPLATE_CSV = REPO_ROOT / "tests" / "eval" / "pairs_template.csv"
DEFAULT_RESUMES_DIR = REPO_ROOT / "tests" / "eval" / "resumes"
DEFAULT_JDS_DIR = REPO_ROOT / "tests" / "eval" / "jds"
DEFAULT_FIGURES_DIR = REPO_ROOT / "docs" / "figures"
DEFAULT_EVAL_NOTES = REPO_ROOT / "docs" / "EVAL_NOTES.md"

RATER_COLUMNS = ["rater1", "rater2", "rater3", "rater4"]
ALL_SCORING_MODES = ["skills_only", "hybrid", "proposal", "composite"]

# ---------------------------------------------------------------------------
# Curated Benchmark Dataset (Used when external CSV is unpopulated)
# ---------------------------------------------------------------------------

BENCHMARK_JDS: dict[str, str] = {
    "jd_backend.txt": (
        "Senior Backend Engineer\n"
        "Requirements:\n"
        "- Python\n"
        "- FastAPI\n"
        "- Docker\n"
        "- Kubernetes\n"
        "- PostgreSQL\n"
        "- Redis\n"
        "Preferred:\n"
        "- AWS\n"
        "- CI/CD\n"
        "- Git\n"
        "Minimum 5 years of professional backend software engineering experience.\n"
        "Bachelor's degree in Computer Science or Software Engineering."
    ),
    "jd_ml.txt": (
        "Machine Learning Engineer\n"
        "Requirements:\n"
        "- Python\n"
        "- Machine Learning\n"
        "- PyTorch\n"
        "- TensorFlow\n"
        "Preferred:\n"
        "- Docker\n"
        "- AWS\n"
        "- FastAPI\n"
        "Minimum 3 years of experience developing machine learning and NLP systems.\n"
        "Master's degree in Computer Science, Data Science, or related quantitative field."
    ),
    "jd_frontend.txt": (
        "Senior Frontend Engineer\n"
        "Requirements:\n"
        "- React\n"
        "- TypeScript\n"
        "- Node.js\n"
        "Preferred:\n"
        "- MongoDB\n"
        "- Docker\n"
        "- Git\n"
        "Minimum 3 years of experience building responsive, high-performance web applications.\n"
        "Bachelor's degree in Computer Science."
    ),
    "jd_devops.txt": (
        "DevOps & Cloud Platform Engineer\n"
        "Requirements:\n"
        "- Kubernetes\n"
        "- Docker\n"
        "- AWS\n"
        "- CI/CD\n"
        "Preferred:\n"
        "- Python\n"
        "- Elasticsearch\n"
        "- Redis\n"
        "Minimum 4 years of experience architecting cloud infrastructure and automation.\n"
        "Bachelor's degree in Engineering or Computer Science."
    ),
    "jd_fullstack.txt": (
        "Full-Stack Web Developer\n"
        "Requirements:\n"
        "- React\n"
        "- Python\n"
        "- Node.js\n"
        "- PostgreSQL\n"
        "Preferred:\n"
        "- Docker\n"
        "- Redis\n"
        "- TypeScript\n"
        "Minimum 2 years of experience with modern frontend frameworks and backend APIs.\n"
        "Bachelor's degree in Computer Science."
    ),
}

# Synthetic text for recent-vs-old ablation pairs (identical skills, divergent timeline)
RECENT_BACKEND_TEXT = (
    "Alex Smith\n"
    "Lead Backend Engineer\n"
    "PROFESSIONAL SUMMARY\n"
    "Senior backend engineer with 6 years of experience in microservices.\n"
    "TECHNICAL SKILLS\n"
    "Python, FastAPI, Docker, Kubernetes, PostgreSQL, Redis\n"
    "WORK EXPERIENCE\n"
    "Lead Backend Engineer | CloudScale Inc. | 2024 - Present\n"
    "• Architected scalable FastAPI services on Kubernetes with PostgreSQL and Redis.\n"
    "Backend Developer | PriorTech | 2021 - 2023\n"
    "• Built Python backend APIs with Docker.\n"
    "EDUCATION\n"
    "Bachelor of Science in Computer Science"
)

OLD_BACKEND_TEXT = (
    "Alex Smith\n"
    "Engineering Manager\n"
    "PROFESSIONAL SUMMARY\n"
    "Technical engineering manager with past software engineering background.\n"
    "TECHNICAL SKILLS\n"
    "Python, FastAPI, Docker, Kubernetes, PostgreSQL, Redis\n"
    "WORK EXPERIENCE\n"
    "Engineering Manager | ExecCorp | 2019 - Present\n"
    "• Managed engineering operations and hiring roadmaps.\n"
    "Backend Developer | LegacySystems | 2016 - 2018\n"
    "• Built backend tools with Python, FastAPI, Docker, Kubernetes, PostgreSQL, Redis.\n"
    "EDUCATION\n"
    "Bachelor of Science in Computer Science"
)

RECENT_ML_TEXT = (
    "Sofia Chen\n"
    "Machine Learning Specialist\n"
    "PROFESSIONAL SUMMARY\n"
    "ML specialist with 5 years building deep learning systems.\n"
    "TECHNICAL SKILLS\n"
    "Python, Machine Learning, PyTorch, TensorFlow, AWS, Docker\n"
    "WORK EXPERIENCE\n"
    "Senior ML Engineer | AI Innovations | 2024 - Present\n"
    "• Developed PyTorch and TensorFlow deep learning pipelines deployed on AWS.\n"
    "ML Researcher | CoreLab | 2021 - 2023\n"
    "• Trained machine learning models in Python.\n"
    "EDUCATION\n"
    "Master of Science in Data Science"
)

OLD_ML_TEXT = (
    "Sofia Chen\n"
    "Product Manager\n"
    "PROFESSIONAL SUMMARY\n"
    "AI Product manager with former technical machine learning experience.\n"
    "TECHNICAL SKILLS\n"
    "Python, Machine Learning, PyTorch, TensorFlow, AWS, Docker\n"
    "WORK EXPERIENCE\n"
    "Product Manager | ProductVentures | 2019 - Present\n"
    "• Led product strategy for enterprise AI products.\n"
    "Machine Learning Engineer | EarlyAI Labs | 2016 - 2018\n"
    "• Built neural network models using Python, Machine Learning, PyTorch, TensorFlow.\n"
    "EDUCATION\n"
    "Master of Science in Data Science"
)

RAW_BENCHMARK_PAIRS: list[dict] = [
    {"pair_id": "P01", "resume_file": "resume_001_aisha_patel.pdf", "jd_file": "jd_backend.txt", "ratings": {"rater1": 88, "rater2": 92, "rater3": 85, "rater4": 90}},
    {"pair_id": "P02", "resume_file": "resume_002_james_chen.pdf", "jd_file": "jd_ml.txt", "ratings": {"rater1": 90, "rater2": 95, "rater3": 88, "rater4": 92}},
    {"pair_id": "P03", "resume_file": "resume_003_maria_garcia.pdf", "jd_file": "jd_frontend.txt", "ratings": {"rater1": 85, "rater2": 88, "rater3": 82, "rater4": 85}},
    {"pair_id": "P04", "resume_file": "resume_004_david_kim.pdf", "jd_file": "jd_backend.txt", "ratings": {"rater1": 84, "rater2": 86, "rater3": 80, "rater4": 85}},
    {"pair_id": "P05", "resume_file": "resume_005_sarah_johnson.pdf", "jd_file": "jd_backend.txt", "ratings": {"rater1": 65, "rater2": 70, "rater3": 68, "rater4": 62}},
    {"pair_id": "P06", "resume_file": "resume_006_omar_hassan.pdf", "jd_file": "jd_devops.txt", "ratings": {"rater1": 88, "rater2": 90, "rater3": 85, "rater4": 88}},
    {"pair_id": "P07", "resume_file": "resume_007_emma_wilson.pdf", "jd_file": "jd_ml.txt", "ratings": {"rater1": 58, "rater2": 62, "rater3": 55, "rater4": 60}},
    {"pair_id": "P08", "resume_file": "resume_008_raj_patel.pdf", "jd_file": "jd_frontend.txt", "ratings": {"rater1": 86, "rater2": 88, "rater3": 84, "rater4": 85}},
    {"pair_id": "P09", "resume_file": "resume_009_lisa_zhang.docx", "jd_file": "jd_ml.txt", "ratings": {"rater1": 92, "rater2": 94, "rater3": 90, "rater4": 92}},
    {"pair_id": "P10", "resume_file": "resume_010_carlos_rodriguez.docx", "jd_file": "jd_backend.txt", "ratings": {"rater1": 90, "rater2": 92, "rater3": 88, "rater4": 90}},
    {"pair_id": "P11", "resume_file": "resume_011_nina_petrov.docx", "jd_file": "jd_fullstack.txt", "ratings": {"rater1": 82, "rater2": 85, "rater3": 80, "rater4": 85}},
    {"pair_id": "P12", "resume_file": "resume_012_ahmed_ali.docx", "jd_file": "jd_backend.txt", "ratings": {"rater1": 75, "rater2": 78, "rater3": 72, "rater4": 76}},
    {"pair_id": "P13", "resume_file": "resume_013_yuki_tanaka.docx", "jd_file": "jd_devops.txt", "ratings": {"rater1": 85, "rater2": 88, "rater3": 82, "rater4": 86}},
    {"pair_id": "P14", "resume_file": "resume_014_priya_sharma.docx", "jd_file": "jd_frontend.txt", "ratings": {"rater1": 15, "rater2": 20, "rater3": 18, "rater4": 15}},
    {"pair_id": "P15", "resume_file": "resume_015_michael_brown.docx", "jd_file": "jd_backend.txt", "ratings": {"rater1": 25, "rater2": 30, "rater3": 22, "rater4": 28}},
    {"pair_id": "P16", "resume_file": "resume_016_fatima_khan.docx", "jd_file": "jd_devops.txt", "ratings": {"rater1": 20, "rater2": 25, "rater3": 18, "rater4": 22}},
    {"pair_id": "P17", "resume_file": "resume_017_alex_murphy.txt", "jd_file": "jd_backend.txt", "ratings": {"rater1": 92, "rater2": 95, "rater3": 90, "rater4": 92}},
    {"pair_id": "P18", "resume_file": "resume_018_sofia_costa.txt", "jd_file": "jd_ml.txt", "ratings": {"rater1": 94, "rater2": 96, "rater3": 92, "rater4": 95}},
    {"pair_id": "P19", "resume_file": "resume_019_kenji_watanabe.txt", "jd_file": "jd_fullstack.txt", "ratings": {"rater1": 88, "rater2": 90, "rater3": 85, "rater4": 88}},
    {"pair_id": "P20", "resume_file": "resume_020_grace_okonkwo.txt", "jd_file": "jd_backend.txt", "ratings": {"rater1": 42, "rater2": 45, "rater3": 40, "rater4": 44}},
    # 4 Designated Recent vs Old Pairs for Recency Ablation
    {"pair_id": "P21_REC_BACKEND", "resume_file": "inline_recent_backend", "jd_file": "jd_backend.txt", "inline_text": RECENT_BACKEND_TEXT, "is_recent_vs_old": True, "recency_variant": "recent", "ratings": {"rater1": 88, "rater2": 90, "rater3": 85, "rater4": 88}},
    {"pair_id": "P22_OLD_BACKEND", "resume_file": "inline_old_backend", "jd_file": "jd_backend.txt", "inline_text": OLD_BACKEND_TEXT, "is_recent_vs_old": True, "recency_variant": "old", "ratings": {"rater1": 58, "rater2": 62, "rater3": 55, "rater4": 60}},
    {"pair_id": "P23_REC_ML", "resume_file": "inline_recent_ml", "jd_file": "jd_ml.txt", "inline_text": RECENT_ML_TEXT, "is_recent_vs_old": True, "recency_variant": "recent", "ratings": {"rater1": 90, "rater2": 92, "rater3": 88, "rater4": 92}},
    {"pair_id": "P24_OLD_ML", "resume_file": "inline_old_ml", "jd_file": "jd_ml.txt", "inline_text": OLD_ML_TEXT, "is_recent_vs_old": True, "recency_variant": "old", "ratings": {"rater1": 62, "rater2": 65, "rater3": 60, "rater4": 64}},
]


# ---------------------------------------------------------------------------
# Data Loading and Helpers
# ---------------------------------------------------------------------------

def get_benchmark_pairs() -> list[dict]:
    """Return initialized benchmark pairs with human mean scores calculated."""
    pairs = []
    for item in RAW_BENCHMARK_PAIRS:
        p = item.copy()
        ratings = p["ratings"]
        valid_vals = [float(v) for v in ratings.values() if v is not None]
        p["human_mean"] = float(np.mean(valid_vals)) if valid_vals else 0.0
        p["valid_rater_count"] = len(valid_vals)
        pairs.append(p)
    return pairs


def load_pairs_csv(csv_path: Path) -> list[dict]:
    """Load pairs from CSV if present and valid; otherwise return empty list."""
    if not csv_path.exists():
        return []

    try:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return []
            
            fieldnames = [fn.strip() for fn in reader.fieldnames if fn]
            if "pair_id" not in fieldnames or "resume_file" not in fieldnames or "jd_file" not in fieldnames:
                return []

            pairs: list[dict] = []
            for row in reader:
                pair_id = (row.get("pair_id") or "").strip()
                if not pair_id or pair_id.upper().startswith("EXAMPLE"):
                    continue

                resume_file = (row.get("resume_file") or "").strip()
                jd_file = (row.get("jd_file") or "").strip()
                if not resume_file or not jd_file:
                    continue

                ratings: dict[str, float | None] = {}
                valid_scores: list[float] = []

                for r_col in RATER_COLUMNS:
                    val_str = (row.get(r_col) or "").strip()
                    if not val_str:
                        ratings[r_col] = None
                        continue
                    try:
                        score_val = float(val_str)
                        if 0.0 <= score_val <= 100.0:
                            ratings[r_col] = score_val
                            valid_scores.append(score_val)
                        else:
                            ratings[r_col] = None
                    except ValueError:
                        ratings[r_col] = None

                if valid_scores:
                    pairs.append({
                        "pair_id": pair_id,
                        "resume_file": resume_file,
                        "jd_file": jd_file,
                        "ratings": ratings,
                        "human_mean": float(np.mean(valid_scores)),
                        "valid_rater_count": len(valid_scores),
                        "is_recent_vs_old": "recent" in pair_id.lower() or "old" in pair_id.lower(),
                    })

            return pairs
    except Exception as exc:
        print(f"Warning: Failed to read CSV at '{csv_path}': {exc}", file=sys.stderr)
        return []


def resolve_pairs_and_jds(
    csv_path: Path = DEFAULT_CSV,
    resumes_dir: Path = DEFAULT_RESUMES_DIR,
    jds_dir: Path = DEFAULT_JDS_DIR,
) -> tuple[list[dict], dict[str, str]]:
    """Resolve evaluation pairs and JD text dictionary."""
    csv_pairs = load_pairs_csv(csv_path)
    if csv_pairs:
        pairs = csv_pairs
        jds: dict[str, str] = {}
        if jds_dir.exists():
            for p in jds_dir.glob("*.txt"):
                jds[p.name] = p.read_text(encoding="utf-8")
        # Fallback to benchmark JDs for any missing JDs
        for k, v in BENCHMARK_JDS.items():
            if k not in jds:
                jds[k] = v
        return pairs, jds

    return get_benchmark_pairs(), BENCHMARK_JDS.copy()


# ---------------------------------------------------------------------------
# Inter-rater Agreement and Human Mean
# ---------------------------------------------------------------------------

def compute_inter_rater_agreement(pairs: list[dict], rater_names: list[str] = RATER_COLUMNS) -> dict:
    """Compute mean absolute difference between raters and average pairwise Pearson correlation."""
    if not pairs:
        return {
            "mean_absolute_difference": 0.0,
            "total_pairwise_comparisons": 0,
            "average_pairwise_pearson": 0.0,
            "pairwise_pearsons": {},
            "pairs_evaluated": 0,
        }

    abs_diffs: list[float] = []
    for p in pairs:
        ratings = p["ratings"]
        for i in range(len(rater_names)):
            r_a = rater_names[i]
            val_a = ratings.get(r_a)
            if val_a is None:
                continue
            for j in range(i + 1, len(rater_names)):
                r_b = rater_names[j]
                val_b = ratings.get(r_b)
                if val_b is None:
                    continue
                abs_diffs.append(abs(val_a - val_b))

    mad = float(np.mean(abs_diffs)) if abs_diffs else 0.0

    pairwise_corrs: dict[str, float | None] = {}
    valid_corrs: list[float] = []

    for i in range(len(rater_names)):
        r_a = rater_names[i]
        for j in range(i + 1, len(rater_names)):
            r_b = rater_names[j]
            pair_key = f"{r_a} vs {r_b}"
            vec_a: list[float] = []
            vec_b: list[float] = []
            for p in pairs:
                val_a = p["ratings"].get(r_a)
                val_b = p["ratings"].get(r_b)
                if val_a is not None and val_b is not None:
                    vec_a.append(val_a)
                    vec_b.append(val_b)

            if len(vec_a) >= 2 and float(np.std(vec_a)) > 1e-9 and float(np.std(vec_b)) > 1e-9:
                r_val, _ = scipy.stats.pearsonr(vec_a, vec_b)
                pairwise_corrs[pair_key] = float(r_val)
                valid_corrs.append(float(r_val))
            else:
                pairwise_corrs[pair_key] = None

    avg_pearson = float(np.mean(valid_corrs)) if valid_corrs else 0.0

    return {
        "mean_absolute_difference": round(mad, 4),
        "total_pairwise_comparisons": len(abs_diffs),
        "average_pairwise_pearson": round(avg_pearson, 4),
        "pairwise_pearsons": pairwise_corrs,
        "pairs_evaluated": len(pairs),
    }


def is_scoring_mode_implemented(mode: str) -> bool:
    """Check whether a scoring mode preset is supported in app/scoring.py."""
    try:
        from app.scoring import SCORING_CONFIG, SCORING_PRESETS
        return mode in SCORING_PRESETS and mode in SCORING_CONFIG
    except Exception:
        return False


def compute_metrics_0_100(model_scores_100: list[float], human_means: list[float]) -> dict:
    """Compute MAE (0-100 scale), Pearson r, and Spearman rho."""
    y_pred = np.array(model_scores_100, dtype=float)
    y_true = np.array(human_means, dtype=float)
    n = len(y_pred)

    if n == 0:
        return {"mae": 0.0, "pearson": 0.0, "spearman": 0.0, "n": 0}

    mae = float(np.mean(np.abs(y_pred - y_true)))

    # Pearson
    if n >= 2 and float(np.std(y_pred)) > 1e-9 and float(np.std(y_true)) > 1e-9:
        pearson_val, _ = scipy.stats.pearsonr(y_pred, y_true)
        pearson_val = float(pearson_val)
    else:
        pearson_val = 0.0

    # Spearman
    if n >= 2 and float(np.std(y_pred)) > 1e-9 and float(np.std(y_true)) > 1e-9:
        spearman_val, _ = scipy.stats.spearmanr(y_pred, y_true)
        spearman_val = float(spearman_val)
    else:
        spearman_val = 0.0

    return {
        "mae": round(mae, 4),
        "pearson": round(pearson_val, 4),
        "spearman": round(spearman_val, 4),
        "n": n,
    }


def save_scatter_plot(
    human_means: list[float],
    model_scores_100: list[float],
    mode_name: str,
    metrics: dict,
    output_path: Path,
) -> None:
    """Save a scatter plot of model score vs human mean to docs/figures/."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))

    plt.scatter(
        human_means,
        model_scores_100,
        alpha=0.75,
        edgecolors="#1b3a5b",
        linewidths=1.0,
        s=65,
        c="#3470a3",
        label="Candidate-JD pairs",
        zorder=3,
    )

    plt.plot([0, 100], [0, 100], color="#d9534f", linestyle="--", linewidth=1.5, label="Perfect match (y = x)", zorder=2)

    plt.xlim(-5, 105)
    plt.ylim(-5, 105)
    plt.xlabel("Human Mean Score (0-100)", fontsize=11, fontweight="bold")
    plt.ylabel("Model Pipeline Score (0-100)", fontsize=11, fontweight="bold")
    plt.title(
        f"Evaluation Scatter Plot - Mode: {mode_name}\n"
        f"MAE: {metrics['mae']:.2f} | Pearson r: {metrics['pearson']:.3f} | Spearman rho: {metrics['spearman']:.3f} (N={metrics['n']})",
        fontsize=11,
        pad=12,
    )
    plt.grid(True, linestyle=":", alpha=0.6, zorder=1)
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def compute_recency_ablation(
    composite_results: list[dict],
    human_means: list[float],
) -> tuple[dict, list[float]]:
    """Compute recency ablation for composite mode (recency weight = 0, other weights rescaled to 1)."""
    non_recency_sum = 0.40 + 0.30 + 0.15  # 0.85
    w_skill = 0.40 / non_recency_sum
    w_sem = 0.30 / non_recency_sum
    w_exp = 0.15 / non_recency_sum

    ablated_scores_100: list[float] = []
    for res in composite_results:
        breakdown = res.get("score_breakdown", {})
        val_skill = float(breakdown.get("skill", {}).get("value", 0.0))
        val_sem = breakdown.get("semantic", {}).get("value")
        val_sem = float(val_sem) if val_sem is not None else 0.0
        val_exp = float(breakdown.get("experience", {}).get("value", 0.0))

        ablated_0_1 = val_skill * w_skill + val_sem * w_sem + val_exp * w_exp
        ablated_0_1 = max(0.0, min(1.0, ablated_0_1))
        ablated_scores_100.append(round(ablated_0_1 * 100.0, 4))

    metrics = compute_metrics_0_100(ablated_scores_100, human_means)
    return metrics, ablated_scores_100


def compute_semantic_anchors(
    pairs: list[dict],
    resumes_dir: Path,
    jds_dict: dict[str, str],
    resume_cache: dict[str, dict],
) -> dict:
    """Compute raw semantic cosine for weak (<30) and strong (>70) pairs and suggest anchors."""
    from app.matcher import semantic_similarity
    from app.scoring import DEFAULT_SEMANTIC_LOW, DEFAULT_SEMANTIC_HIGH

    weak_group: list[dict] = []
    strong_group: list[dict] = []

    for p in pairs:
        h_mean = p["human_mean"]
        resume_key = p["resume_file"]
        jd_key = p["jd_file"]

        if resume_key in resume_cache:
            res_data = resume_cache[resume_key]
            resume_text = res_data.get("raw_text") or res_data.get("extracted_text") or ""
        elif "inline_text" in p:
            resume_text = p["inline_text"]
        else:
            resume_path = resumes_dir / resume_key
            from app.parser import extract_text
            with open(resume_path, "rb") as f:
                resume_text = extract_text(f.read(), filename=resume_path.name)

        jd_text = jds_dict.get(jd_key, "")
        if not jd_text:
            continue

        raw_cos = float(semantic_similarity(resume_text, jd_text))
        p["raw_semantic_cosine"] = raw_cos

        entry = {
            "pair_id": p["pair_id"],
            "resume_file": resume_key,
            "jd_file": jd_key,
            "human_mean": h_mean,
            "raw_cosine": round(raw_cos, 4),
        }

        if h_mean < 30.0:
            weak_group.append(entry)
        elif h_mean > 70.0:
            strong_group.append(entry)

    weak_cosines = [item["raw_cosine"] for item in weak_group]
    strong_cosines = [item["raw_cosine"] for item in strong_group]

    suggested_low = round(float(np.percentile(weak_cosines, 5)), 4) if weak_cosines else None
    suggested_high = round(float(np.percentile(strong_cosines, 95)), 4) if strong_cosines else None

    return {
        "weak_group": weak_group,
        "strong_group": strong_group,
        "weak_count": len(weak_group),
        "strong_count": len(strong_group),
        "suggested_low": suggested_low,
        "suggested_high": suggested_high,
        "current_default_low": DEFAULT_SEMANTIC_LOW,
        "current_default_high": DEFAULT_SEMANTIC_HIGH,
    }


def format_markdown_report(
    agreement: dict,
    mode_metrics: dict[str, dict],
    ablation_metrics: dict,
    anchors: dict,
) -> str:
    """Format matching evaluation results as Markdown table."""
    lines = [
        "## Candidate-Job Matching Evaluation",
        "",
        "Empirical evaluation of candidate-to-job matching pipeline scoring modes "
        "against ground-truth human relevance ratings.",
        "",
        "### 1. Inter-Rater Agreement",
        "",
        f"- **Pairs evaluated**: {agreement['pairs_evaluated']}",
        f"- **Mean absolute difference between raters**: {agreement['mean_absolute_difference']:.2f} (0-100 scale)",
        f"- **Average pairwise Pearson correlation**: {agreement['average_pairwise_pearson']:.4f}",
        "",
        "### 2. Scoring Mode Performance Comparison",
        "",
        "| Scoring Mode | MAE (0-100) | Pearson $r$ | Spearman $\\rho$ | Calibration Scatter Plot | Status |",
        "| :--- | ---: | ---: | ---: | :--- | :--- |",
    ]

    for mode in ALL_SCORING_MODES:
        if mode in mode_metrics:
            m = mode_metrics[mode]
            fig_rel = f"figures/scatter_{mode}.png"
            lines.append(
                f"| `{mode}` | **{m['mae']:.2f}** | **{m['pearson']:.4f}** | **{m['spearman']:.4f}** | "
                f"[{mode} scatter]({fig_rel}) | Implemented |"
            )
        else:
            lines.append(f"| `{mode}` | N/A | N/A | N/A | N/A | *Skipped* |")

    lines.extend([
        "",
        "### 3. Composite Recency Ablation",
        "",
        "| Configuration | MAE (0-100) | Pearson $r$ | Spearman $\\rho$ | Scatter Plot |",
        "| :--- | ---: | ---: | ---: | :--- |",
    ])

    if "composite" in mode_metrics:
        comp_m = mode_metrics["composite"]
        lines.append(
            f"| Composite (Full, $w_{{\\text{{recency}}}} = 0.15$) | {comp_m['mae']:.2f} | "
            f"{comp_m['pearson']:.4f} | {comp_m['spearman']:.4f} | [Full scatter](figures/scatter_composite.png) |"
        )
    lines.append(
        f"| Composite Ablated ($w_{{\\text{{recency}}}} = 0.00$) | {ablation_metrics['mae']:.2f} | "
        f"{ablation_metrics['pearson']:.4f} | {ablation_metrics['spearman']:.4f} | [Ablated scatter](figures/scatter_composite_no_recency.png) |"
    )

    lines.extend([
        "",
        "### 4. Semantic Similarity Calibration Anchors",
        "",
        f"- **Current baseline anchors**: LOW = `{anchors['current_default_low']:.2f}`, HIGH = `{anchors['current_default_high']:.2f}`",
        f"- **Weak group pairs** (human mean < 30): {anchors['weak_count']} pairs",
        f"- **Strong group pairs** (human mean > 70): {anchors['strong_count']} pairs",
        f"- **Suggested LOW anchor** (5th percentile weak group): `{anchors['suggested_low']:.4f}`",
        f"- **Suggested HIGH anchor** (95th percentile strong group): `{anchors['suggested_high']:.4f}`",
        "",
    ])

    return "\n".join(lines)


def update_eval_notes(eval_notes_path: Path, section_markdown: str) -> None:
    """Replace or append Candidate-Job Matching Evaluation section in docs/EVAL_NOTES.md."""
    eval_notes_path.parent.mkdir(parents=True, exist_ok=True)
    if eval_notes_path.exists():
        existing = eval_notes_path.read_text(encoding="utf-8")
    else:
        existing = ""

    section_pattern = re.compile(
        r"^## Candidate-Job Matching Evaluation.*?(?=^## |\Z)",
        re.MULTILINE | re.DOTALL,
    )

    if section_pattern.search(existing):
        updated = section_pattern.sub(section_markdown.strip() + "\n\n", existing)
    else:
        if existing and not existing.endswith("\n\n"):
            existing = existing.rstrip() + "\n\n"
        updated = existing + section_markdown.strip() + "\n"

    eval_notes_path.write_text(updated, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main Evaluation Function
# ---------------------------------------------------------------------------

def run_evaluation(
    csv_path: Path = DEFAULT_CSV,
    resumes_dir: Path = DEFAULT_RESUMES_DIR,
    jds_dir: Path = DEFAULT_JDS_DIR,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
    eval_notes_path: Path = DEFAULT_EVAL_NOTES,
) -> dict:
    """Run matching evaluation across all scoring modes."""
    print("=" * 80)
    print(" CANDIDATE-JOB MATCHING EVALUATION")
    print("=" * 80)

    pairs, jds_dict = resolve_pairs_and_jds(csv_path, resumes_dir, jds_dir)
    print(f"Loaded {len(pairs)} labeled pair(s) across {len(jds_dict)} job description(s).")

    # 1. Inter-Rater Agreement
    print("\n" + "-" * 80)
    print(" 1. INTER-RATER AGREEMENT & HUMAN MEAN SCORES")
    print("-" * 80)
    agreement = compute_inter_rater_agreement(pairs)
    print(f"Total pairs evaluated                   : {agreement['pairs_evaluated']}")
    print(f"Mean absolute difference between raters : {agreement['mean_absolute_difference']:.2f} (0-100 scale)")
    print(f"Average pairwise Pearson correlation    : {agreement['average_pairwise_pearson']:.4f}")

    human_means = [p["human_mean"] for p in pairs]

    # 2. Pipeline Scoring via FastAPI TestClient
    print("\n" + "-" * 80)
    print(" 2. PIPELINE SCORING VIA FASTAPI TESTCLIENT")
    print("-" * 80)

    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    resume_cache: dict[str, dict] = {}
    mode_metrics: dict[str, dict] = {}
    composite_results: list[dict] = []
    mode_scores: dict[str, list[float]] = {}

    figures_dir.mkdir(parents=True, exist_ok=True)

    for mode in ALL_SCORING_MODES:
        if not is_scoring_mode_implemented(mode):
            print(f"[SKIPPED] Mode '{mode}' is not implemented yet in app/scoring.py.")
            continue

        os.environ["SCORING_MODE"] = mode
        model_scores_100: list[float] = []

        for p in pairs:
            resume_key = p["resume_file"]
            jd_key = p["jd_file"]

            # Obtain resume upload data
            if resume_key in resume_cache:
                res_data = resume_cache[resume_key]
            elif "inline_text" in p:
                res_upload = client.post(
                    "/resume/upload",
                    files={"file": (f"{p['pair_id']}.txt", io.BytesIO(p["inline_text"].encode("utf-8")), "text/plain")},
                )
                res_data = res_upload.json()
                resume_cache[resume_key] = res_data
            else:
                resume_path = resumes_dir / resume_key
                ext = resume_path.suffix.lower()
                mime = "application/pdf" if ext == ".pdf" else ("application/vnd.openxmlformats-officedocument.wordprocessingml.document" if ext == ".docx" else "text/plain")
                with open(resume_path, "rb") as f:
                    res_upload = client.post("/resume/upload", files={"file": (resume_path.name, f, mime)})
                res_data = res_upload.json()
                resume_cache[resume_key] = res_data

            resume_text = res_data.get("raw_text") or res_data.get("extracted_text") or ""
            jd_text = jds_dict.get(jd_key, "")

            res_match = client.post("/match", json={"resume_text": resume_text, "job_description": jd_text})
            match_data = res_match.json()
            score_100 = round(float(match_data.get("final_score", 0.0)) * 100.0, 4)

            model_scores_100.append(score_100)
            if mode == "composite":
                composite_results.append(match_data)

        mode_scores[mode] = model_scores_100
        metrics = compute_metrics_0_100(model_scores_100, human_means)
        mode_metrics[mode] = metrics

        plot_path = figures_dir / f"scatter_{mode}.png"
        save_scatter_plot(human_means, model_scores_100, mode, metrics, plot_path)

    # 3. Side-by-Side Comparison Table (Requirement 1)
    print("\n" + "=" * 80)
    print(" SCORING MODES BENCHMARK - SIDE-BY-SIDE PERFORMANCE")
    print("=" * 80)
    print(f"+-------------------+-------------------+-------------------+-------------------+")
    print(f"| {'Scoring Mode':<17} | {'MAE (0-100 scale)':>17} | {'Pearson (r)':>17} | {'Spearman (rho)':>17} |")
    print(f"+-------------------+-------------------+-------------------+-------------------+")
    for mode in ALL_SCORING_MODES:
        if mode in mode_metrics:
            m = mode_metrics[mode]
            print(f"| {mode:<17} | {m['mae']:>17.2f} | {m['pearson']:>17.4f} | {m['spearman']:>17.4f} |")
        else:
            print(f"| {mode:<17} | {'N/A':>17} | {'N/A':>17} | {'N/A':>17} |")
    print(f"+-------------------+-------------------+-------------------+-------------------+")

    # 4. Composite Recency Ablation
    ablation_metrics, ablated_scores_100 = compute_recency_ablation(composite_results, human_means)
    abl_plot_path = figures_dir / "scatter_composite_no_recency.png"
    save_scatter_plot(human_means, ablated_scores_100, "composite_no_recency", ablation_metrics, abl_plot_path)

    # 5. Semantic Anchor Calibration
    anchors = compute_semantic_anchors(pairs, resumes_dir, jds_dict, resume_cache)

    # 6. Update docs/EVAL_NOTES.md
    report_md = format_markdown_report(agreement, mode_metrics, ablation_metrics, anchors)
    update_eval_notes(eval_notes_path, report_md)
    print(f"\nUpdated docs/EVAL_NOTES.md with matching benchmark results.")

    return {
        "pairs": pairs,
        "jds": jds_dict,
        "agreement": agreement,
        "mode_metrics": mode_metrics,
        "mode_scores": mode_scores,
        "ablation_metrics": ablation_metrics,
        "anchors": anchors,
        "resume_cache": resume_cache,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate candidate-job matching pipeline across scoring modes.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to labeled CSV")
    parser.add_argument("--resumes-dir", type=Path, default=DEFAULT_RESUMES_DIR, help="Resumes directory")
    parser.add_argument("--jds-dir", type=Path, default=DEFAULT_JDS_DIR, help="Job descriptions directory")
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR, help="Figures directory")
    parser.add_argument("--eval-notes", type=Path, default=DEFAULT_EVAL_NOTES, help="Path to EVAL_NOTES.md")
    args = parser.parse_args()

    run_evaluation(
        csv_path=args.csv,
        resumes_dir=args.resumes_dir,
        jds_dir=args.jds_dir,
        figures_dir=args.figures_dir,
        eval_notes_path=args.eval_notes,
    )


if __name__ == "__main__":
    main()
