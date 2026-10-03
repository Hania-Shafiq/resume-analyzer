"""Semantic matching engine using Sentence Transformers.

Loads the `all-MiniLM-L6-v2` embedding model once (singleton via @lru_cache)
and calculates chunk-level semantic similarity between resumes and job descriptions.
"""

from functools import lru_cache
import re
from typing import Any, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from app.preprocess import light_clean

# Default embedding model
MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """Load and cache the sentence-transformers model as a singleton.

    Parameters
    ----------
    model_name : str, optional
        HuggingFace model ID, defaults to 'all-MiniLM-L6-v2'.

    Returns
    -------
    SentenceTransformer
        Cached model instance ready for encoding.
    """
    return SentenceTransformer(model_name)


def split_sentences(text: str, min_length: int = 15) -> list[str]:
    """Split text (resume or JD) into clean, meaningful sentences and bullet chunks.

    Resumes and job descriptions frequently use bullet lists, linebreaks, and
    semicolons rather than standard punctuation-delimited paragraphs. This function:
      1. Normalizes characters and whitespace with `light_clean()`.
      2. Splits lines across linebreaks and bullet points.
      3. Splits compound sentences on terminal punctuation (. ! ?).
      4. Strips bullet prefixes (•, -, *, 1., etc.).
      5. Filters out trivial fragments shorter than `min_length`.

    Parameters
    ----------
    text : str
        Input text to segment.
    min_length : int, optional
        Minimum character length for a chunk to be retained (default: 15).

    Returns
    -------
    list[str]
        List of segmented sentence/bullet chunks.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")

    text = text.strip()
    if not text:
        return []

    try:
        cleaned = light_clean(text)
    except ValueError:
        return []

    chunks: list[str] = []

    # Process line-by-line first so bullet points and list entries stay isolated
    for line in cleaned.splitlines():
        line = line.strip()
        if not line:
            continue

        # Strip standard bullet characters and list numbering like "1. ", "a) "
        line = re.sub(r"^([•\-\*•\u2022\u25CF\u25A0]+|\d+[\.\)]|[a-zA-Z][\.\)])\s*", "", line).strip()
        if not line:
            continue

        # Split multiple sentences on punctuation followed by whitespace or uppercase
        sub_sents = re.split(r"(?<=[.!?])\s+", line)
        for s in sub_sents:
            s_clean = s.strip(" \t\r\n-•*")
            if len(s_clean) >= min_length:
                chunks.append(s_clean)
            elif s_clean and chunks and len(chunks[-1]) < 35:
                # Merge very short trailing fragments with previous chunk
                chunks[-1] = f"{chunks[-1]} {s_clean}"
            elif s_clean and not chunks:
                chunks.append(s_clean)

    # Fallback: if min_length filtered everything out, take non-empty lines
    if not chunks:
        chunks = [line.strip() for line in cleaned.splitlines() if line.strip()]

    # Ultimate fallback: entire cleaned text
    if not chunks and cleaned:
        chunks = [cleaned]

    return chunks


def compute_similarity_matrix(sentences_a: list[str], sentences_b: list[str]) -> np.ndarray:
    """Compute the cosine similarity matrix between two lists of sentences.

    Parameters
    ----------
    sentences_a : list[str]
        First list of sentences (rows), e.g. JD sentences.
    sentences_b : list[str]
        Second list of sentences (columns), e.g. Resume sentences.

    Returns
    -------
    np.ndarray
        Cosine similarity matrix of shape (len(sentences_a), len(sentences_b)),
        with values in [-1.0, 1.0].
    """
    if not sentences_a or not sentences_b:
        return np.zeros((len(sentences_a), len(sentences_b)), dtype=np.float32)

    model = get_model()
    # normalize_embeddings=True produces unit vectors, so cosine similarity is the dot product
    embeddings_a = model.encode(sentences_a, normalize_embeddings=True, show_progress_bar=False)
    embeddings_b = model.encode(sentences_b, normalize_embeddings=True, show_progress_bar=False)

    return np.dot(embeddings_a, embeddings_b.T)


def semantic_similarity(
    resume_text: str,
    jd_text: str,
    return_details: bool = False,
) -> Union[float, dict[str, Any]]:
    """Compute semantic match score between a resume and a job description.

    Rather than compressing the entire resume and entire job description into single
    vectors (which causes specific requirements to be diluted by document length
    imbalance), this algorithm:
      1. Segments the JD into requirement/responsibility sentences.
      2. Segments the resume into experience/skill sentences.
      3. Embeds each sentence with `all-MiniLM-L6-v2`.
      4. For each JD sentence, finds the best cosine similarity match across all
         resume sentences.
      5. Takes the mean of those best-match similarities as the overall semantic score.

    Parameters
    ----------
    resume_text : str
        Extracted text of the candidate's resume.
    jd_text : str
        Full text of the target job description.
    return_details : bool, optional
        If True, returns a dictionary containing detailed sentence breakdown,
        similarity matrix, and best-match alignments. If False (default), returns
        a single float in [0.0, 1.0].

    Returns
    -------
    float or dict
        If return_details is False:
            Float score between 0.0 and 1.0 (rounded to 4 decimal places).
        If return_details is True:
            Dictionary with keys:
            - 'score': float (0.0 to 1.0)
            - 'jd_sentences': list[str]
            - 'resume_sentences': list[str]
            - 'similarity_matrix': np.ndarray (shape len(jd) x len(resume))
            - 'best_matches': list[dict] with JD sentence, best matching resume sentence,
                              and similarity score.
    """
    if not isinstance(resume_text, str) or not isinstance(jd_text, str):
        raise TypeError("Both resume_text and jd_text must be strings.")

    resume_chunks = split_sentences(resume_text)
    jd_chunks = split_sentences(jd_text)

    if not resume_chunks or not jd_chunks:
        if return_details:
            return {
                "score": 0.0,
                "jd_sentences": jd_chunks,
                "resume_sentences": resume_chunks,
                "similarity_matrix": np.empty((0, 0)),
                "best_matches": [],
            }
        return 0.0

    # Compute similarity matrix: shape (N_jd, M_resume)
    sim_matrix = compute_similarity_matrix(jd_chunks, resume_chunks)

    # For each JD sentence, find its best match in the resume
    best_match_indices = np.argmax(sim_matrix, axis=1)
    best_match_scores = np.max(sim_matrix, axis=1)

    # Cosine similarities can theoretically dip below 0, clip to [0.0, 1.0]
    clipped_scores = np.clip(best_match_scores, 0.0, 1.0)
    mean_score = float(np.mean(clipped_scores))
    final_score = round(max(0.0, min(1.0, mean_score)), 4)

    if not return_details:
        return final_score

    best_matches = []
    for jd_idx, (r_idx, sim) in enumerate(zip(best_match_indices, best_match_scores)):
        best_matches.append({
            "jd_sentence": jd_chunks[jd_idx],
            "best_resume_sentence": resume_chunks[r_idx],
            "similarity": round(float(sim), 4),
        })

    return {
        "score": final_score,
        "jd_sentences": jd_chunks,
        "resume_sentences": resume_chunks,
        "similarity_matrix": sim_matrix,
        "best_matches": best_matches,
    }
