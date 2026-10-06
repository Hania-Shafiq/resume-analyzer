"""Recommender module: job-role prediction from resume text.

The classifier is trained in ``notebooks/03_job_classifier.ipynb`` and saved to
``app/models/job_classifier.joblib``. This module loads that file and exposes
:func:`recommend_roles`.

Two kinds of saved model are supported, and the notebook keeps whichever one
scored better:

* ``"tfidf"``     - a scikit-learn Pipeline (TF-IDF -> Logistic Regression)
                    that takes cleaned text directly.
* ``"embedding"`` - a Logistic Regression that takes sentence-transformer
                    vectors. The text -> vector step lives in
                    :func:`embed_resumes` below, so training (notebook) and
                    prediction (API) use exactly the same code.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from app.preprocess import light_clean

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "job_classifier.joblib"


# ===========================================================================
# Embedding helper (used by the notebook AND by recommend_roles)
# ===========================================================================

def embed_resumes(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Turn resumes into one 384-dim vector each (all-MiniLM-L6-v2).

    A MiniLM model only reads ~256 word-pieces, which is shorter than some
    resumes. So each resume is split into sentence chunks (reusing
    ``matcher.split_sentences``), every chunk is embedded, and the chunk
    vectors are averaged and re-normalised to unit length.

    Parameters
    ----------
    texts : list[str]
        Cleaned resume texts.
    batch_size : int
        Chunks encoded per forward pass.

    Returns
    -------
    np.ndarray
        Array of shape ``(len(texts), 384)``.
    """
    # Imported here so TF-IDF-only usage never needs sentence-transformers.
    from app.matcher import get_model, split_sentences

    all_chunks: list[str] = []
    owners: list[int] = []  # which resume each chunk belongs to
    for i, text in enumerate(texts):
        chunks = split_sentences(text) or [text]
        all_chunks.extend(chunks)
        owners.extend([i] * len(chunks))

    chunk_vecs = get_model().encode(
        all_chunks,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    owners_arr = np.asarray(owners)
    out = np.zeros((len(texts), chunk_vecs.shape[1]), dtype=np.float32)
    for i in range(len(texts)):
        mean_vec = chunk_vecs[owners_arr == i].mean(axis=0)
        norm = np.linalg.norm(mean_vec)
        out[i] = mean_vec / norm if norm > 0 else mean_vec
    return out


def embed_resume(text: str) -> np.ndarray:
    """Embed a single cleaned resume. Returns shape ``(384,)``."""
    return embed_resumes([text])[0]


# ===========================================================================
# Model loading
# ===========================================================================

@lru_cache(maxsize=2)
def load_model(path: str = str(MODEL_PATH)) -> dict[str, Any]:
    """Load (and cache) the saved classifier bundle.

    The bundle is a dict with keys: ``kind``, ``estimator``, ``classes``,
    ``metrics``, ``trained_on``.

    Raises
    ------
    FileNotFoundError
        If the model file has not been created yet.
    """
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. "
            "Run notebooks/03_job_classifier.ipynb (all cells) to create it."
        )
    bundle = joblib.load(model_path)
    if bundle.get("kind") not in {"tfidf", "embedding"}:
        raise ValueError(f"Unrecognised model kind in {model_path}: {bundle.get('kind')!r}")
    return bundle


# ===========================================================================
# Public API
# ===========================================================================

def recommend_roles(resume_text: str, top_k: int = 4) -> list[dict[str, Any]]:
    """Suggest the job roles that best fit a resume.

    Parameters
    ----------
    resume_text : str
        Raw or cleaned resume text (it is cleaned again here, which is safe).
    top_k : int, optional
        How many roles to return (default 4). Capped at the number of roles
        the model knows.

    Returns
    -------
    list[dict]
        Highest probability first, e.g.::

            [{"role": "Data Science",              "probability_percent": 61.4},
             {"role": "Machine Learning Engineer", "probability_percent": 22.8},
             ...]

        ``probability_percent`` is in the range 0-100.

    Raises
    ------
    TypeError  : if *resume_text* is not a string.
    ValueError : if *resume_text* is empty, or *top_k* < 1.
    FileNotFoundError : if the model file has not been trained yet.
    """
    if not isinstance(resume_text, str):
        raise TypeError(f"Expected str, got {type(resume_text).__name__}.")
    if not resume_text.strip():
        raise ValueError("resume_text must not be empty.")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    cleaned = light_clean(resume_text)  # same cleaning as training
    bundle = load_model()
    estimator = bundle["estimator"]

    if bundle["kind"] == "tfidf":
        probs = estimator.predict_proba([cleaned])[0]
    else:  # "embedding"
        probs = estimator.predict_proba(embed_resume(cleaned).reshape(1, -1))[0]

    classes = estimator.classes_
    top_idx = np.argsort(probs)[::-1][: min(top_k, len(classes))]
    return [
        {"role": str(classes[i]), "probability_percent": round(float(probs[i]) * 100, 2)}
        for i in top_idx
    ]
