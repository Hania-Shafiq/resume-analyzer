"""Unit tests for app/matcher.py (semantic matching engine)."""

import pytest
import numpy as np

from app.matcher import (
    get_model,
    split_sentences,
    compute_similarity_matrix,
    semantic_similarity,
)


class TestSentenceSplitter:
    """Tests for split_sentences()."""

    def test_bullet_points_split(self):
        text = """
        • Developed web APIs using FastAPI and Python.
        - Deployed microservices into Docker containers.
        * Configured PostgreSQL databases for high performance.
        """
        chunks = split_sentences(text)
        assert len(chunks) == 3
        assert any("FastAPI" in c for c in chunks)
        assert any("Docker" in c for c in chunks)
        assert any("PostgreSQL" in c for c in chunks)
        # Bullet symbols should be stripped
        for c in chunks:
            assert not c.startswith("•")
            assert not c.startswith("-")
            assert not c.startswith("*")

    def test_multiple_sentences_in_single_line(self):
        text = "Built backend services with Python. Designed clean REST APIs. Optimized database queries."
        chunks = split_sentences(text)
        assert len(chunks) == 3
        assert chunks[0].startswith("Built backend")
        assert chunks[1].startswith("Designed clean")
        assert chunks[2].startswith("Optimized database")

    def test_empty_and_whitespace_text(self):
        assert split_sentences("") == []
        assert split_sentences("   \n\t  ") == []

    def test_non_string_raises(self):
        with pytest.raises(TypeError):
            split_sentences(12345)


class TestMatcherModel:
    """Tests for model loading and singleton behavior."""

    def test_model_singleton(self):
        model1 = get_model()
        model2 = get_model()
        assert model1 is model2


class TestSimilarityMatrix:
    """Tests for compute_similarity_matrix()."""

    def test_shape_and_range(self):
        sents_a = ["Python web developer", "Data scientist with machine learning"]
        sents_b = ["FastAPI backend engineer", "Accountant doing taxes", "Machine learning researcher"]
        matrix = compute_similarity_matrix(sents_a, sents_b)

        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (2, 3)
        assert np.all(matrix >= -1.0) and np.all(matrix <= 1.0)

    def test_identical_sentence_high_similarity(self):
        sents = ["Experienced Python developer with FastAPI and Docker."]
        matrix = compute_similarity_matrix(sents, sents)
        assert matrix[0, 0] >= 0.99


class TestSemanticSimilarity:
    """Tests for semantic_similarity()."""

    def test_strong_matching_resume(self):
        jd = """
        Required Qualifications:
        - 3+ years experience with Python and FastAPI backend development.
        - Solid experience with PostgreSQL database optimization.
        - Hands-on knowledge of Docker containerization.
        """
        resume = """
        Software Engineer with 4 years building APIs in Python and FastAPI.
        Managed PostgreSQL databases, tuned SQL queries, and maintained schemas.
        Containerized internal applications with Docker.
        """
        score = semantic_similarity(resume, jd)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score >= 0.70  # strong match

    def test_unrelated_resume_low_similarity(self):
        jd = """
        Looking for a Senior Python Developer with Kubernetes and FastAPI experience.
        Must know Docker and cloud microservice deployment.
        """
        resume = """
        Experienced Head Chef and culinary manager with 10 years leading restaurant kitchens.
        Expert in pastry baking, inventory management, food safety, and French cuisine.
        """
        score = semantic_similarity(resume, jd)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score < 0.35  # should be low similarity

    def test_return_details_format(self):
        jd = "Required: Python and Docker experience."
        resume = "I have 3 years of Python and Docker experience."
        details = semantic_similarity(resume, jd, return_details=True)

        assert isinstance(details, dict)
        assert "score" in details
        assert "jd_sentences" in details
        assert "resume_sentences" in details
        assert "similarity_matrix" in details
        assert "best_matches" in details
        assert len(details["best_matches"]) == len(details["jd_sentences"])
        assert details["score"] >= 0.75

    def test_empty_inputs(self):
        assert semantic_similarity("", "Python developer") == 0.0
        assert semantic_similarity("Python developer", "") == 0.0

    def test_non_string_inputs_raise(self):
        with pytest.raises(TypeError):
            semantic_similarity(None, "Python developer")
        with pytest.raises(TypeError):
            semantic_similarity("Python developer", 123)
