"""Text preprocessing module for cleaning, normalizing, and tokenizing resume content.

Two-level strategy:
  - light_clean : surface-level cleaning only, safe for embedding models.
  - nlp_process : deep linguistic processing via spaCy for downstream analysis.
"""

import re
import unicodedata
from typing import TypedDict

import spacy

# ---------------------------------------------------------------------------
# Load spaCy model once at module level (avoids repeated disk I/O).
# The small English model is lightweight and sufficient for NER + lemmatisation.
# ---------------------------------------------------------------------------
_NLP = None  # lazy-loaded on first call to nlp_process()


def _load_nlp() -> spacy.language.Language:
    """Return a cached spaCy language model, loading it on first access."""
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load("en_core_web_sm")
        except OSError as e:
            raise OSError(
                "spaCy model 'en_core_web_sm' not found. "
                "Run:  python -m spacy download en_core_web_sm"
            ) from e
    return _NLP


# ---------------------------------------------------------------------------
# Type alias for the structured output of nlp_process()
# ---------------------------------------------------------------------------
class NLPResult(TypedDict):
    """Structured result returned by nlp_process()."""

    tokens: list[str]       # raw word-form tokens (no punctuation / spaces)
    lemmas: list[str]       # base/dictionary form of each token
    sentences: list[str]    # complete sentences detected in the text
    filtered_tokens: list[str]  # tokens with stop words and punctuation removed
    filtered_lemmas: list[str]  # lemmas with stop words and punctuation removed


# ===========================================================================
# 1. LIGHT CLEAN — suitable for sentence-transformer / embedding input
# ===========================================================================

def light_clean(text: str) -> str:
    """Normalise whitespace and remove non-standard characters; keep sentences intact.

    This is the *minimum* amount of cleaning needed before feeding text to an
    embedding model (e.g. sentence-transformers). Aggressive cleaning such as
    stop-word removal or lemmatisation would hurt semantic quality, so we skip it.

    Steps
    -----
    1. Unicode normalisation (NFKC) — convert ligatures, full-width chars, etc.
    2. Remove control characters (NULL, BEL, form-feed, …) except \\n and \\t.
    3. Replace common non-ASCII symbols (bullets, curly quotes) with ASCII.
    4. Strip lines that are purely decorative (e.g. "------").
    5. Collapse multiple blank lines into a single blank line.
    6. Normalise horizontal whitespace on each line (tabs → spaces, multi-space → one space).
    7. Strip leading / trailing whitespace from each line and the whole document.

    Parameters
    ----------
    text : str
        Raw extracted text from a resume document.

    Returns
    -------
    str
        Lightly cleaned text, suitable for embedding.

    Raises
    ------
    TypeError
        If *text* is not a string.
    ValueError
        If *text* is empty after cleaning.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")

    # Step 1 — Unicode NFKC normalisation
    # Converts ligatures (ﬁ → fi), fullwidth letters (Ａ → A), etc.
    text = unicodedata.normalize("NFKC", text)

    # Step 2 — Drop ASCII control characters (0x00–0x1F) except \n (0x0A) and \t (0x09)
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

    # Step 3 — Replace common non-ASCII punctuation with plain ASCII equivalents
    replacements = {
        "\u2018": "'",  # left single quotation mark
        "\u2019": "'",  # right single quotation mark
        "\u201C": '"',  # left double quotation mark
        "\u201D": '"',  # right double quotation mark
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2022": "-",  # bullet •
        "\u2023": "-",  # triangular bullet
        "\u25A0": "-",  # black square ■
        "\u25CF": "-",  # black circle ●
        "\u2026": "...",  # ellipsis …
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)

    # Step 4 — Remove purely decorative separator lines (e.g. "---", "===", "***")
    text = re.sub(r"^[\-=*_~#]{3,}\s*$", "", text, flags=re.MULTILINE)

    # Step 5 — Normalise horizontal whitespace within each line
    lines = []
    for line in text.splitlines():
        # Convert tabs to single spaces, collapse multiple spaces to one
        line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(line)

    # Step 6 — Collapse runs of more than one blank line into exactly one blank line
    joined = "\n".join(lines)
    joined = re.sub(r"\n{3,}", "\n\n", joined)

    # Step 7 — Final strip
    result = joined.strip()

    if not result:
        raise ValueError("Text is empty after cleaning.")

    return result


# ===========================================================================
# 2. NLP PROCESS — deep linguistic analysis via spaCy
# ===========================================================================

def nlp_process(text: str) -> NLPResult:
    """Run full spaCy NLP pipeline and return tokens, lemmas, and sentences.

    Applies the following linguistic transformations:
    1. Tokenisation — split text into individual tokens (words, punctuation, …).
    2. Sentence detection — identify sentence boundaries.
    3. Stop-word & punctuation filtering — remove low-information-content tokens.
    4. Lemmatisation — reduce each token to its base dictionary form.

    Parameters
    ----------
    text : str
        Text to process. Run *light_clean()* on raw document text first.

    Returns
    -------
    NLPResult
        A typed dict with five keys:

        - ``tokens``         — all word-form tokens (punctuation excluded).
        - ``lemmas``         — base form of every token.
        - ``sentences``      — full sentence strings detected in the text.
        - ``filtered_tokens``— tokens without stop words and punctuation.
        - ``filtered_lemmas``— lemmas without stop words and punctuation.

    Raises
    ------
    TypeError
        If *text* is not a string.
    ValueError
        If *text* is empty or blank.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")
    if not text.strip():
        raise ValueError("Text must not be empty.")

    # Load spaCy pipeline (en_core_web_sm includes tok2vec, tagger, lemmatizer, NER)
    nlp = _load_nlp()

    # Run the full pipeline on the input text
    doc = nlp(text)

    # ------------------------------------------------------------------
    # Step 1 — Collect all word-form tokens (skip whitespace-only tokens)
    # ------------------------------------------------------------------
    tokens = [tok.text for tok in doc if not tok.is_space]

    # ------------------------------------------------------------------
    # Step 2 — Collect the base/lemma form of every non-space token
    # spaCy lowercases lemmas by default for most token types.
    # ------------------------------------------------------------------
    lemmas = [tok.lemma_ for tok in doc if not tok.is_space]

    # ------------------------------------------------------------------
    # Step 3 — Collect sentence strings detected by spaCy's sentenciser
    # ------------------------------------------------------------------
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    # ------------------------------------------------------------------
    # Step 4 — Filter: keep only meaningful tokens
    # Discard if: is_stop (common English stop word), is_punct (. , ; etc.),
    # is_space (whitespace), or the token's text is empty after stripping.
    # ------------------------------------------------------------------
    def _is_meaningful(tok: spacy.tokens.Token) -> bool:
        """Return True if the token should be kept for analysis."""
        return (
            not tok.is_stop     # drop: the, a, is, to, …
            and not tok.is_punct  # drop: . , ; ( ) …
            and not tok.is_space  # drop: whitespace tokens
            and tok.text.strip()  # drop: anything that becomes empty when stripped
            and not tok.like_num  # drop: standalone numbers like 2019, 42
        )

    filtered_tokens = [tok.text for tok in doc if _is_meaningful(tok)]
    filtered_lemmas = [tok.lemma_.lower() for tok in doc if _is_meaningful(tok)]

    return NLPResult(
        tokens=tokens,
        lemmas=lemmas,
        sentences=sentences,
        filtered_tokens=filtered_tokens,
        filtered_lemmas=filtered_lemmas,
    )
