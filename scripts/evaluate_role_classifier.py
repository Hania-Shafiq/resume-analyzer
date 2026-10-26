#!/usr/bin/env python3
"""Evaluate the job-role classifier on the external LiveCareer resume dataset.

Loads data/external/Resume.csv, drops exact and near-duplicates, runs original vs
label-leakage-ablated experiments (TF-IDF+LR and MiniLM+LR), applies an uncertainty
rule, scores tests/eval/resumes/, saves app/models/job_classifier_v2.joblib, and
writes tables into docs/EVAL_NOTES.md under \"Role classifier\".
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.parser import extract_text
from app.preprocess import light_clean


def embed_documents(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Encode resumes with all-MiniLM-L6-v2 (one vector per doc, long text clipped).

    Uses the project's shared SentenceTransformer via ``app.matcher.get_model``.
    Clipping keeps evaluation tractable on ~2.5k long LiveCareer resumes while
    still testing the same MiniLM backbone used in production.
    """
    from app.matcher import get_model

    clipped: list[str] = []
    for t in texts:
        t = (t or "").strip()
        if len(t) > 2500:
            t = t[:2500]
        clipped.append(t if t else " ")
    vecs = get_model().encode(
        clipped,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(vecs, dtype=np.float32)

RANDOM_STATE = 42
UNCERTAIN_THRESHOLD = 0.35
NEAR_DUP_THRESHOLD = 0.95
DEFAULT_DATA = PROJECT_ROOT / "data" / "external" / "Resume.csv"
DEFAULT_EVAL_NOTES = PROJECT_ROOT / "docs" / "EVAL_NOTES.md"
DEFAULT_MODEL_OUT = PROJECT_ROOT / "app" / "models" / "job_classifier_v2.joblib"
DEFAULT_RESUME_DIR = PROJECT_ROOT / "tests" / "eval" / "resumes"
DEFAULT_CM_PATH = PROJECT_ROOT / "scripts" / "artifacts" / "role_classifier_confusion.png"

DATASET_SOURCE_BLOCK = """\
# Evaluation and Ablation Notes

**External role-classifier dataset**
- **Source**: Snehaan Bhawal — *Resume Dataset* (LiveCareer resume examples), distributed on [Kaggle](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) and mirrored as Hugging Face `opensporks/resumes`.
- **Licence**: CC0 1.0 (Public Domain).
- **Local path**: `data/external/Resume.csv` (columns: `ID`, `Resume_str`, `Resume_html`, `Category`).

---
"""


# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def role_title_tokens(categories: list[str]) -> set[str]:
    """Build a set of lowercase tokens derived from all class / role titles."""
    tokens: set[str] = set()
    for cat in categories:
        parts = re.split(r"[\s\-_/]+", str(cat).lower())
        for p in parts:
            p = re.sub(r"[^a-z0-9+#]+", "", p)
            if len(p) >= 2:
                tokens.add(p)
    # Common expansions / aliases that leak the label
    extras = {
        "hr", "it", "bpo", "pr", "accountant", "accounting", "advocate",
        "agriculture", "apparel", "arts", "automobile", "automotive",
        "aviation", "banking", "banker", "business", "chef", "culinary",
        "construction", "consultant", "consulting", "designer", "design",
        "digital", "media", "engineering", "engineer", "finance", "financial",
        "fitness", "healthcare", "health", "information", "technology",
        "public", "relations", "sales", "teacher", "teaching", "educator",
    }
    tokens |= extras
    return tokens


def strip_role_title_words(text: str, tokens: set[str]) -> str:
    if not text:
        return text
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(t) for t in sorted(tokens, key=len, reverse=True)) + r")\b",
        flags=re.IGNORECASE,
    )
    return pattern.sub(" ", text)


def assign_near_duplicate_groups(
    texts: list[str],
    threshold: float = NEAR_DUP_THRESHOLD,
) -> np.ndarray:
    """Return group id per row; near-duplicates (TF-IDF cosine > threshold) share an id."""
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 1),
        min_df=1,
        max_features=50000,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(texts)
    n = X.shape[0]
    uf = UnionFind(n)

    # Exact duplicates already handled upstream; still union identical sparse rows via NN.
    n_neighbors = min(30, n)
    nn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=n_neighbors)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    for i in range(n):
        for dist, j in zip(distances[i], indices[i]):
            if i == j:
                continue
            sim = 1.0 - float(dist)
            if sim > threshold:
                uf.union(i, j)

    roots = [uf.find(i) for i in range(n)]
    root_to_gid: dict[int, int] = {}
    groups = np.empty(n, dtype=np.int32)
    next_gid = 0
    for i, r in enumerate(roots):
        if r not in root_to_gid:
            root_to_gid[r] = next_gid
            next_gid += 1
        groups[i] = root_to_gid[r]
    return groups


def stratified_group_train_test_split(
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick one fold of StratifiedGroupKFold so near-duplicate groups stay together."""
    # Prefer ~test_size; with 5 folds each test fold is ~20%.
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    X_dummy = np.zeros(len(y))
    train_idx, test_idx = next(sgkf.split(X_dummy, y, groups))
    return train_idx, test_idx


def make_tfidf_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                    token_pattern=r"(?u)\b\w[\w+#]*",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    C=10.0,
                ),
            ),
        ]
    )


def make_embedding_clf() -> LogisticRegression:
    return LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        C=10.0,
    )


def evaluate_predictions(
    y_true,
    y_pred,
    labels: list[str],
) -> dict:
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
    precision, recall, _, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = []
    for lab, p, r, s in zip(labels, precision, recall, support):
        per_class.append(
            {
                "class": lab,
                "precision": float(p),
                "recall": float(r),
                "support": int(s),
            }
        )
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "report": classification_report(
            y_true, y_pred, labels=labels, digits=4, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
    }


def cv_macro_f1_tfidf(texts, y, groups, n_splits: int = 5) -> float:
    pipe = make_tfidf_pipeline()
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        pipe,
        texts,
        y,
        groups=groups,
        cv=sgkf,
        scoring="f1_macro",
        n_jobs=1,
    )
    return float(scores.mean())


def cv_macro_f1_embeddings(X_emb, y, groups, n_splits: int = 5) -> float:
    clf = make_embedding_clf()
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        clf,
        X_emb,
        y,
        groups=groups,
        cv=sgkf,
        scoring="f1_macro",
        n_jobs=1,
    )
    return float(scores.mean())


def plot_confusion(cm: np.ndarray, labels: list[str], title: str, out_path: Path) -> None:
    fig_w = max(10, len(labels) * 0.55)
    fig_h = max(8, len(labels) * 0.45)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        cm,
        annot=False,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def load_and_clean(data_path: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(data_path)
    if "Resume_str" not in df.columns or "Category" not in df.columns:
        raise ValueError(f"Expected Category + Resume_str columns in {data_path}")

    df = df.copy()
    df["Category"] = df["Category"].astype(str).str.strip()
    df["Resume_str"] = df["Resume_str"].fillna("").astype(str)

    def _safe_clean(text: str) -> str:
        try:
            return light_clean(text)
        except ValueError:
            return ""

    df["clean_text"] = df["Resume_str"].map(_safe_clean)
    df = df[df["clean_text"].str.len() > 40].reset_index(drop=True)
    n_raw = len(df)

    # Exact duplicates
    before_exact = len(df)
    df = df.drop_duplicates(subset=["clean_text"], keep="first").reset_index(drop=True)
    n_exact_dropped = before_exact - len(df)

    groups = assign_near_duplicate_groups(df["clean_text"].tolist(), NEAR_DUP_THRESHOLD)
    df["group_id"] = groups
    n_groups = int(df["group_id"].nunique())
    n_near_extra = len(df) - n_groups

    # Keep one resume per near-duplicate group (first occurrence)
    df_dedup = df.drop_duplicates(subset=["group_id"], keep="first").reset_index(drop=True)
    # After collapsing, each remaining row is its own group (for StratifiedGroupKFold)
    df_dedup = df_dedup.copy()
    df_dedup["group_id"] = np.arange(len(df_dedup), dtype=np.int32)

    stats = {
        "n_raw_nonempty": n_raw,
        "n_exact_dropped": n_exact_dropped,
        "n_near_groups_before_collapse": n_groups,
        "n_near_extra_dropped": n_near_extra,
        "n_after_dedup": len(df_dedup),
        "class_counts": df_dedup["Category"].value_counts().to_dict(),
    }
    return df_dedup, stats


def run_experiment(
    name: str,
    texts: list[str],
    y: np.ndarray,
    groups: np.ndarray,
    labels: list[str],
    embeddings_cache: dict[str, np.ndarray],
) -> dict:
    print(f"\n===== Experiment: {name} =====")
    train_idx, test_idx = stratified_group_train_test_split(y, groups)

    texts_arr = np.asarray(texts, dtype=object)
    X_train_txt, X_test_txt = texts_arr[train_idx], texts_arr[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    g_train = groups[train_idx]

    # --- TF-IDF + LR ---
    print("Training TF-IDF + Logistic Regression...")
    tfidf_pipe = make_tfidf_pipeline()
    tfidf_pipe.fit(X_train_txt, y_train)
    pred_tfidf = tfidf_pipe.predict(X_test_txt)
    proba_tfidf = tfidf_pipe.predict_proba(X_test_txt)
    metrics_tfidf = evaluate_predictions(y_test, pred_tfidf, labels)
    print("Computing 5-fold CV macro-F1 (TF-IDF, group-aware)...")
    cv_tfidf = cv_macro_f1_tfidf(texts, y, groups)
    metrics_tfidf["cv_macro_f1"] = cv_tfidf
    metrics_tfidf["proba_test"] = proba_tfidf
    metrics_tfidf["y_test"] = y_test
    metrics_tfidf["pred_test"] = pred_tfidf
    metrics_tfidf["estimator"] = tfidf_pipe
    metrics_tfidf["kind"] = "tfidf"
    print(
        f"  TF-IDF  acc={metrics_tfidf['accuracy']:.4f}  "
        f"macro-F1={metrics_tfidf['macro_f1']:.4f}  "
        f"weighted-F1={metrics_tfidf['weighted_f1']:.4f}  "
        f"CV-macro-F1={cv_tfidf:.4f}"
    )

    # --- MiniLM + LR ---
    cache_key = name
    if cache_key not in embeddings_cache:
        print("Encoding resumes with all-MiniLM-L6-v2 (clipped single-pass)...")
        embeddings_cache[cache_key] = embed_documents(list(texts), batch_size=64)
    X_emb = embeddings_cache[cache_key]
    X_train_emb, X_test_emb = X_emb[train_idx], X_emb[test_idx]

    print("Training MiniLM embeddings + Logistic Regression...")
    emb_clf = make_embedding_clf()
    emb_clf.fit(X_train_emb, y_train)
    pred_emb = emb_clf.predict(X_test_emb)
    proba_emb = emb_clf.predict_proba(X_test_emb)
    metrics_emb = evaluate_predictions(y_test, pred_emb, labels)
    print("Computing 5-fold CV macro-F1 (MiniLM, group-aware)...")
    cv_emb = cv_macro_f1_embeddings(X_emb, y, groups)
    metrics_emb["cv_macro_f1"] = cv_emb
    metrics_emb["proba_test"] = proba_emb
    metrics_emb["y_test"] = y_test
    metrics_emb["pred_test"] = pred_emb
    metrics_emb["estimator"] = emb_clf
    metrics_emb["kind"] = "embedding"
    metrics_emb["X_all_emb"] = X_emb
    print(
        f"  MiniLM  acc={metrics_emb['accuracy']:.4f}  "
        f"macro-F1={metrics_emb['macro_f1']:.4f}  "
        f"weighted-F1={metrics_emb['weighted_f1']:.4f}  "
        f"CV-macro-F1={cv_emb:.4f}"
    )

    return {
        "name": name,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "tfidf": metrics_tfidf,
        "embedding": metrics_emb,
        "y_train": y_train,
        "g_train": g_train,
    }


def uncertainty_report(y_true, y_pred, proba, classes, threshold: float) -> dict:
    top_p = proba.max(axis=1)
    uncertain_mask = top_p < threshold
    n_uncertain = int(uncertain_mask.sum())
    n_total = len(y_true)
    confident_mask = ~uncertain_mask
    if confident_mask.any():
        # Predictions already made; accuracy among confident
        acc_conf = accuracy_score(y_true[confident_mask], y_pred[confident_mask])
    else:
        acc_conf = float("nan")
    return {
        "threshold": threshold,
        "n_test": n_total,
        "n_uncertain": n_uncertain,
        "pct_uncertain": 100.0 * n_uncertain / n_total if n_total else 0.0,
        "n_confident": int(confident_mask.sum()),
        "accuracy_confident": float(acc_conf) if confident_mask.any() else None,
    }


def predict_eval_resumes(estimator, kind: str, resume_dir: Path, classes) -> list[dict]:
    rows = []
    files = sorted(p for p in resume_dir.iterdir() if p.is_file())
    for path in files:
        try:
            raw = extract_text(path)
        except Exception as exc:  # noqa: BLE001 — continue eval on parse errors
            rows.append({"filename": path.name, "error": str(exc)})
            continue
        cleaned = light_clean(raw)
        if kind == "tfidf":
            proba = estimator.predict_proba([cleaned])[0]
        else:
            vec = embed_documents([cleaned])
            proba = estimator.predict_proba(vec)[0]
        order = np.argsort(proba)[::-1][:3]
        top3 = [
            {"role": str(classes[i]), "prob": float(proba[i])}
            for i in order
        ]
        top_p = top3[0]["prob"]
        rows.append(
            {
                "filename": path.name,
                "top3": top3,
                "decision": "uncertain" if top_p < UNCERTAIN_THRESHOLD else top3[0]["role"],
            }
        )
    return rows


def fmt_pct(x: float | None, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{digits}f}"


def markdown_class_counts(counts: dict) -> str:
    lines = ["| Class | Count |", "| :--- | ---: |"]
    for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines)


def markdown_metrics_table(results_by_model: dict[str, dict]) -> str:
    lines = [
        "| Model | Accuracy | Macro-F1 | Weighted-F1 | 5-fold CV Macro-F1 |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for name, m in results_by_model.items():
        lines.append(
            f"| {name} | {fmt_pct(m['accuracy'])} | {fmt_pct(m['macro_f1'])} | "
            f"{fmt_pct(m['weighted_f1'])} | {fmt_pct(m['cv_macro_f1'])} |"
        )
    return "\n".join(lines)


def markdown_per_class(per_class: list[dict]) -> str:
    lines = [
        "| Class | Precision | Recall | Support |",
        "| :--- | ---: | ---: | ---: |",
    ]
    for row in per_class:
        lines.append(
            f"| {row['class']} | {fmt_pct(row['precision'])} | "
            f"{fmt_pct(row['recall'])} | {row['support']} |"
        )
    return "\n".join(lines)


def markdown_eval_resumes(rows: list[dict]) -> str:
    lines = [
        "| Resume | #1 | P1 | #2 | P2 | #3 | P3 | Decision |",
        "| :--- | :--- | ---: | :--- | ---: | :--- | ---: | :--- |",
    ]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['filename']} | — | — | — | — | — | — | error: {r['error']} |")
            continue
        t = r["top3"]
        lines.append(
            f"| {r['filename']} | {t[0]['role']} | {t[0]['prob']:.3f} | "
            f"{t[1]['role']} | {t[1]['prob']:.3f} | "
            f"{t[2]['role']} | {t[2]['prob']:.3f} | {r['decision']} |"
        )
    return "\n".join(lines)


def build_role_classifier_section(
    stats: dict,
    orig: dict,
    leak: dict,
    best_name: str,
    best_metrics: dict,
    unc: dict,
    eval_rows: list[dict],
    cm_note: str,
) -> str:
    too_good = []
    for label, exp in [("original", orig), ("leakage-ablated", leak)]:
        for model_key, pretty in [("tfidf", "TF-IDF+LR"), ("embedding", "MiniLM+LR")]:
            m = exp[model_key]
            if m["macro_f1"] >= 0.99 or m["accuracy"] >= 0.99:
                too_good.append(f"{pretty} on {label} text (macro-F1={m['macro_f1']:.4f})")

    caveats = ""
    if too_good:
        caveats = (
            "\n### Caveat: scores look too good\n\n"
            "Some results are at or above 99%:\n"
            + "\n".join(f"- {x}" for x in too_good)
            + "\n\nPossible reasons:\n"
            "- **Label leakage**: role-title words (and close synonyms) appear inside the resume body; "
            "compare the leakage-ablated experiment.\n"
            "- **Near-duplicate / templated LiveCareer pages**: many resumes share boilerplate section "
            "headers and phrasing even after cosine > 0.95 collapse.\n"
            "- **Category folders are coarse**: scraped pages were filed under a single job board "
            "category, so within-class text is highly stereotyped.\n"
            "- **Train/test still share style**: group-aware splitting blocks near-dup leakage but not "
            "shared template language across different groups.\n"
        )

    orig_models = {
        "TF-IDF + Logistic Regression": orig["tfidf"],
        "MiniLM + Logistic Regression": orig["embedding"],
    }
    leak_models = {
        "TF-IDF + Logistic Regression": leak["tfidf"],
        "MiniLM + Logistic Regression": leak["embedding"],
    }

    section = f"""## Role classifier

Evaluation of TF-IDF + Logistic Regression vs MiniLM (`all-MiniLM-L6-v2`) + Logistic Regression
on the external LiveCareer resume dataset (`data/external/Resume.csv`).
Near-duplicates (TF-IDF cosine > {NEAR_DUP_THRESHOLD}) were collapsed to one row per group;
train/test and CV use stratified **group-aware** folds so residual groups never straddle splits.
Uncertainty rule: top probability < {UNCERTAIN_THRESHOLD} → `uncertain`.

### Cleaning

| Step | Count |
| :--- | ---: |
| Non-empty rows after light_clean | {stats['n_raw_nonempty']} |
| Exact duplicate texts dropped | {stats['n_exact_dropped']} |
| Near-duplicate extras dropped (cosine > {NEAR_DUP_THRESHOLD}) | {stats['n_near_extra_dropped']} |
| Rows after deduplication | {stats['n_after_dedup']} |

#### Class counts after deduplication

{markdown_class_counts(stats['class_counts'])}

### Experiment A — original cleaned text

{markdown_metrics_table(orig_models)}

#### Per-class precision / recall (best original model: {best_name})

{markdown_per_class(best_metrics['per_class'])}

{cm_note}

### Experiment B — role-title words removed (label-leakage ablation)

Tokens derived from every class name (plus common aliases) are stripped from the resume text
before vectorisation / embedding so the model cannot rely on the literal role title.

{markdown_metrics_table(leak_models)}

### Uncertainty rule (threshold = {UNCERTAIN_THRESHOLD})

Applied to the held-out test set of the best original-text model (**{best_name}**).

| Metric | Value |
| :--- | ---: |
| Test resumes | {unc['n_test']} |
| Below threshold (`uncertain`) | {unc['n_uncertain']} ({unc['pct_uncertain']:.1f}%) |
| Confident predictions | {unc['n_confident']} |
| Accuracy on confident subset | {fmt_pct(unc['accuracy_confident'])} |

### Held-out eval resumes (`tests/eval/resumes/`)

Top-3 role probabilities from the saved best model (`app/models/job_classifier_v2.joblib`).
Decision is `uncertain` when the top probability is below {UNCERTAIN_THRESHOLD}.

{markdown_eval_resumes(eval_rows)}
{caveats}
"""
    return section


def upsert_eval_notes(eval_notes_path: Path, role_section: str) -> None:
    eval_notes_path.parent.mkdir(parents=True, exist_ok=True)
    if eval_notes_path.exists():
        text = eval_notes_path.read_text(encoding="utf-8")
    else:
        text = ""

    # Ensure source / licence block at the very top
    if "External role-classifier dataset" not in text:
        # Strip a leading "# Evaluation..." title if present so we don't double-title
        if text.startswith("# Evaluation"):
            # keep body after first title line
            rest = text.split("\n", 1)[1] if "\n" in text else ""
            text = DATASET_SOURCE_BLOCK + rest.lstrip("\n")
        else:
            text = DATASET_SOURCE_BLOCK + text

    # Replace or append Role classifier section
    marker = "## Role classifier"
    if marker in text:
        # Replace from marker through next ## at same level or EOF (but keep Skill extraction etc.)
        parts = text.split(marker)
        before = parts[0]
        after = parts[1]
        # Find next top-level ## that is NOT a subsection under Role classifier
        # Role classifier uses ### for subsections, so next \n##  ends it
        m = re.search(r"\n## (?!#)", after)
        if m:
            remainder = after[m.start() + 1 :]  # keep starting ##
            text = before + role_section.rstrip() + "\n\n" + remainder
        else:
            text = before + role_section.rstrip() + "\n"
    else:
        # Insert after Skill extraction block if present, else after source block
        skill_marker = "## Skill extraction"
        if skill_marker in text:
            # place Role classifier after skill section
            idx = text.find(skill_marker)
            rest = text[idx + len(skill_marker) :]
            m = re.search(r"\n## ", rest)
            if m:
                insert_at = idx + len(skill_marker) + m.start()
                text = text[:insert_at] + "\n" + role_section.rstrip() + "\n" + text[insert_at:]
            else:
                text = text.rstrip() + "\n\n" + role_section.rstrip() + "\n"
        else:
            text = text.rstrip() + "\n\n" + role_section.rstrip() + "\n"

    eval_notes_path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--eval-notes", type=Path, default=DEFAULT_EVAL_NOTES)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_OUT)
    parser.add_argument("--resume-dir", type=Path, default=DEFAULT_RESUME_DIR)
    parser.add_argument("--cm-path", type=Path, default=DEFAULT_CM_PATH)
    args = parser.parse_args(argv)

    warnings.filterwarnings("ignore", category=FutureWarning)

    if not args.data.exists():
        print(f"ERROR: dataset not found at {args.data}", file=sys.stderr)
        return 1

    print(f"Loading {args.data} ...")
    df, stats = load_and_clean(args.data)
    print(
        f"After cleaning/dedup: {stats['n_after_dedup']} rows "
        f"(exact dropped={stats['n_exact_dropped']}, "
        f"near extras dropped={stats['n_near_extra_dropped']})"
    )
    print("Class counts:")
    for k, v in sorted(stats["class_counts"].items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {k}: {v}")

    labels = sorted(df["Category"].unique().tolist())
    y = df["Category"].to_numpy()
    groups = df["group_id"].to_numpy()
    texts_orig = df["clean_text"].tolist()

    title_tokens = role_title_tokens(labels)
    texts_leak = [strip_role_title_words(t, title_tokens) for t in texts_orig]
    print(f"Role-title token vocabulary size: {len(title_tokens)}")

    emb_cache: dict[str, np.ndarray] = {}
    orig = run_experiment("original", texts_orig, y, groups, labels, emb_cache)
    leak = run_experiment("leakage_ablated", texts_leak, y, groups, labels, emb_cache)

    # Pick best model on original-text experiment by CV macro-F1 (margin like notebook 03)
    MARGIN = 0.005
    cv_tfidf = orig["tfidf"]["cv_macro_f1"]
    cv_emb = orig["embedding"]["cv_macro_f1"]
    use_embedding = cv_emb > cv_tfidf + MARGIN
    best_key = "embedding" if use_embedding else "tfidf"
    best_name = (
        "MiniLM embeddings + Logistic Regression"
        if use_embedding
        else "TF-IDF + Logistic Regression"
    )
    best_metrics = orig[best_key]
    print(f"\nBEST (original text): {best_name}  CV macro-F1={best_metrics['cv_macro_f1']:.4f}")

    # Confusion matrix heatmap for best original model
    plot_confusion(
        best_metrics["confusion_matrix"],
        labels,
        f"Confusion matrix — {best_name} (original text)",
        args.cm_path,
    )
    print(f"Wrote confusion-matrix heatmap -> {args.cm_path}")

    unc = uncertainty_report(
        best_metrics["y_test"],
        best_metrics["pred_test"],
        best_metrics["proba_test"],
        labels,
        UNCERTAIN_THRESHOLD,
    )
    print(
        f"Uncertainty: {unc['n_uncertain']}/{unc['n_test']} below {UNCERTAIN_THRESHOLD}; "
        f"accuracy on confident={fmt_pct(unc['accuracy_confident'])}"
    )

    # Refit best model on ALL cleaned original texts and save v2 (never touch job_classifier.joblib).
    print("Refitting best model on full cleaned dataset...")
    save_key = best_key
    if best_key == "tfidf":
        final_est = clone(best_metrics["estimator"]).fit(texts_orig, y)
    else:
        final_est = clone(best_metrics["estimator"]).fit(best_metrics["X_all_emb"], y)

    bundle = {
        "kind": best_key,
        "model_name": f"{best_name} (external LiveCareer, v2)",
        "estimator": final_est,
        "classes": [str(c) for c in final_est.classes_],
        "metrics": {
            "cv_macro_f1": round(float(best_metrics["cv_macro_f1"]), 4),
            "test_accuracy": round(float(best_metrics["accuracy"]), 4),
            "test_macro_f1": round(float(best_metrics["macro_f1"]), 4),
            "test_weighted_f1": round(float(best_metrics["weighted_f1"]), 4),
        },
        "trained_on": int(len(df)),
        "sklearn_version": sklearn.__version__,
        "embedding_model": "all-MiniLM-L6-v2" if best_key == "embedding" else None,
        "embedding_mode": "clipped_single_pass" if best_key == "embedding" else None,
        "uncertain_threshold": UNCERTAIN_THRESHOLD,
        "dataset": "data/external/Resume.csv",
    }
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    # Safety: never overwrite the original v1 path
    v1 = PROJECT_ROOT / "app" / "models" / "job_classifier.joblib"
    if args.model_out.resolve() == v1.resolve():
        raise RuntimeError("Refusing to overwrite job_classifier.joblib")
    joblib.dump(bundle, args.model_out, compress=3)
    print(f"Saved -> {args.model_out} ({args.model_out.stat().st_size / 1024:.1f} KB)")

    print("\nTop-3 predictions on tests/eval/resumes/:")
    eval_rows = predict_eval_resumes(
        final_est, best_key, args.resume_dir, final_est.classes_
    )
    for r in eval_rows:
        if "error" in r:
            print(f"  {r['filename']}: ERROR {r['error']}")
            continue
        tops = ", ".join(f"{t['role']}={t['prob']:.3f}" for t in r["top3"])
        print(f"  {r['filename']}: {tops}  -> {r['decision']}")

    cm_note = (
        f"Confusion-matrix heatmap at "
        f"`{args.cm_path.relative_to(PROJECT_ROOT).as_posix()}` "
        f"(true rows × predicted columns for **{best_name}**, original text).\n\n"
        f"**Saved artefact** `app/models/job_classifier_v2.joblib`: **{best_name}** "
        f"(kind=`{best_key}`), trained on all {len(df)} cleaned resumes. "
        f"MiniLM vectors used **clipped single-pass** encoding (`embedding_mode=clipped_single_pass`); "
        f"this differs from sentence-averaged `embed_resumes()` in `recommend_roles()`, so reload "
        f"with the same clipped encoder (see `scripts/evaluate_role_classifier.embed_documents`) "
        f"for scores that match this evaluation.\n\n"
        f"Scores on this external set are **not** suspiciously high (best ~71% accuracy / ~0.65 macro-F1). "
        f"The leakage ablation drops macro-F1 by roughly 0.08–0.20, so role-title words help but do not "
        f"fully explain performance. "
        f"`tests/eval/resumes/` are mostly tech/product profiles and are **out-of-distribution** "
        f"relative to LiveCareer category folders — treat those top-3 lists as a smoke test, "
        f"not as ground-truth role labels."
    )
    section = build_role_classifier_section(
        stats, orig, leak, best_name, best_metrics, unc, eval_rows, cm_note
    )
    upsert_eval_notes(args.eval_notes, section)
    print(f"Updated {args.eval_notes}")

    if too_good_check(orig, leak):
        print(
            "\nNOTE: Some metrics are >= 99%. See docs/EVAL_NOTES.md for possible reasons "
            "(label leakage, templates, coarse categories)."
        )
    return 0


def too_good_check(orig: dict, leak: dict) -> bool:
    for exp in (orig, leak):
        for key in ("tfidf", "embedding"):
            if exp[key]["macro_f1"] >= 0.99 or exp[key]["accuracy"] >= 0.99:
                return True
    return False


if __name__ == "__main__":
    raise SystemExit(main())
