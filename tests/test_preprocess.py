"""Comprehensive unit tests for the text preprocessing module (app/preprocess.py).

Covers:
- light_clean():
  - NFKC unicode normalization (ligatures, full-width characters)
  - ASCII control character stripping (retaining \\n and \\t)
  - Non-ASCII punctuation replacement (bullets, curly quotes, dashes, ellipsis)
  - Decorative line stripping (---, ===, ***)
  - Horizontal whitespace normalization
  - Collapsing runs of blank lines
  - Input validation (TypeError on non-str, ValueError on empty/whitespace)
- nlp_process():
  - Tokenization, lemmatization, and sentence detection
  - Filtering of stop words, punctuation, and standalone numbers
  - Output schema verification (NLPResult typed dict)
  - Input validation (TypeError on non-str, ValueError on empty/blank)
"""

import pytest
from app.preprocess import light_clean, nlp_process


class TestLightClean:
    """Tests for light_clean() function."""

    def test_unicode_nfkc_normalization(self):
        """Ensure ligatures and full-width characters are normalized."""
        text = "O\uFB03ce and \uFB01le in \uFF21\uFF22\uFF23."  # Office and file in ABC.
        cleaned = light_clean(text)
        assert "Office" in cleaned
        assert "file" in cleaned
        assert "ABC" in cleaned

    def test_strip_control_characters(self):
        """Ensure control characters like NULL and BELL are removed while preserving newlines and tabs."""
        text = "Hello\x00World\x07!\nLine\x0B2\thas\x1Fcontent."
        cleaned = light_clean(text)
        assert cleaned == "HelloWorld!\nLine2 hascontent."
        assert "\x00" not in cleaned
        assert "\x07" not in cleaned

    def test_replace_non_ascii_punctuation(self):
        """Ensure smart quotes, dashes, bullets, and ellipses are replaced by ASCII."""
        text = (
            "\u2022 Item 1 \u2013 First\n"
            "\u201CQuoted text\u201D with \u2018single quotes\u2019\u2026\n"
            "\u25A0 Square bullet and \u25CF circle bullet \u2014 em dash"
        )
        cleaned = light_clean(text)
        assert "- Item 1 - First" in cleaned
        assert '"Quoted text" with \'single quotes\'...' in cleaned
        assert "- Square bullet and - circle bullet - em dash" in cleaned

    def test_decorative_separators_removed(self):
        """Ensure lines of decorative dividers (---, ===, ***, etc.) are stripped."""
        text = "Heading\n------------------\nSection Content\n==================\nFooter\n***"
        cleaned = light_clean(text)
        assert "Heading" in cleaned
        assert "Section Content" in cleaned
        assert "Footer" in cleaned
        assert "---" not in cleaned
        assert "===" not in cleaned
        assert "***" not in cleaned

    def test_whitespace_and_blank_lines_collapsed(self):
        """Ensure tabs and multiple spaces are normalized, and runs of blank lines collapsed."""
        text = "Word1 \t   Word2\n\n\n\n\nWord3"
        cleaned = light_clean(text)
        assert cleaned == "Word1 Word2\n\nWord3"

    def test_non_string_input_raises_type_error(self):
        """Ensure non-string input raises TypeError."""
        with pytest.raises(TypeError, match="Expected str"):
            light_clean(12345)
        with pytest.raises(TypeError, match="Expected str"):
            light_clean(None)

    def test_empty_and_whitespace_input_raises_value_error(self):
        """Ensure empty or purely whitespace string raises ValueError."""
        with pytest.raises(ValueError, match="empty after cleaning"):
            light_clean("")
        with pytest.raises(ValueError, match="empty after cleaning"):
            light_clean("   \n\t   \n  ")


class TestNLPProcess:
    """Tests for nlp_process() function."""

    @classmethod
    def setup_class(cls):
        """Warm up NLP pipeline for test methods."""
        cls.sample = (
            "Jane Doe is a software engineer with 5 years of experience in 2024. "
            "She designs REST APIs using FastAPI and Python."
        )
        cls.result = nlp_process(cls.sample)

    def test_result_structure_keys(self):
        """Ensure nlp_process returns all expected typed dict keys."""
        assert isinstance(self.result, dict)
        expected_keys = {"tokens", "lemmas", "sentences", "filtered_tokens", "filtered_lemmas"}
        assert expected_keys.issubset(self.result.keys())

    def test_sentence_segmentation(self):
        """Ensure sentences are properly segmented."""
        assert len(self.result["sentences"]) == 2
        assert self.result["sentences"][0].startswith("Jane Doe")
        assert self.result["sentences"][1].startswith("She designs")

    def test_tokens_and_lemmas_extraction(self):
        """Ensure tokens and lemmas are non-empty lists of strings."""
        assert len(self.result["tokens"]) > 0
        assert len(self.result["lemmas"]) > 0
        assert "software" in self.result["tokens"]
        assert "engineer" in self.result["lemmas"]

    def test_filtered_tokens_removes_stopwords_and_numbers(self):
        """Ensure filtered_tokens removes stop words, punctuation, and standalone numbers."""
        filtered = self.result["filtered_tokens"]
        # Stop words like 'is', 'a', 'with', 'of', 'in' should be removed
        assert "is" not in filtered
        assert "a" not in filtered
        assert "with" not in filtered
        # Punctuation like '.' should be removed
        assert "." not in filtered
        # Standalone numbers like '5', '2024' should be removed
        assert "5" not in filtered
        assert "2024" not in filtered
        # Meaningful tokens should be retained
        assert "Jane" in filtered
        assert "software" in filtered
        assert "engineer" in filtered
        assert "FastAPI" in filtered
        assert "Python" in filtered

    def test_filtered_lemmas_are_lowercased(self):
        """Ensure filtered lemmas are normalized to lower case."""
        lemmas = self.result["filtered_lemmas"]
        for lemma in lemmas:
            assert lemma == lemma.lower()
        assert "engineer" in lemmas
        assert "fastapi" in lemmas
        assert "python" in lemmas

    def test_non_string_nlp_raises_type_error(self):
        """Ensure non-string input raises TypeError."""
        with pytest.raises(TypeError, match="Expected str"):
            nlp_process(42)

    def test_empty_string_nlp_raises_value_error(self):
        """Ensure empty or blank string raises ValueError."""
        with pytest.raises(ValueError, match="Text must not be empty"):
            nlp_process("")
        with pytest.raises(ValueError, match="Text must not be empty"):
            nlp_process("   \n\t  ")
