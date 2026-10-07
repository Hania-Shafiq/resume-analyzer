"""Unit tests for recommend_roles() in app/recommender.py.

These tests need the trained model file, which is created by running
notebooks/03_job_classifier.ipynb. If it does not exist yet, the model-dependent
tests are skipped (the input-validation tests still run).
"""

import pytest

from app.recommender import MODEL_PATH, recommend_roles

needs_model = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Run notebooks/03_job_classifier.ipynb first to create app/models/job_classifier.joblib",
)

DATA_RESUME = (
    "Data analyst with 4 years of experience building Power BI and Tableau dashboards. "
    "Wrote SQL queries and automated weekly reports with Python and pandas."
)


class TestInputValidation:
    def test_non_string_raises(self):
        with pytest.raises(TypeError):
            recommend_roles(12345)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            recommend_roles("   ")

    @pytest.mark.parametrize("bad_k", [0, -1, 2.5, "3", True])
    def test_bad_top_k_raises(self, bad_k):
        with pytest.raises(ValueError):
            recommend_roles(DATA_RESUME, top_k=bad_k)


@needs_model
class TestRecommendRoles:
    def test_default_returns_four_sorted_results(self):
        out = recommend_roles(DATA_RESUME)
        assert len(out) == 4
        probs = [r["probability_percent"] for r in out]
        assert probs == sorted(probs, reverse=True)

    def test_result_format(self):
        for r in recommend_roles(DATA_RESUME, top_k=3):
            assert set(r) == {"role", "probability_percent"}
            assert isinstance(r["role"], str)
            assert 0.0 <= r["probability_percent"] <= 100.0

    def test_top_k_is_respected_and_capped(self):
        assert len(recommend_roles(DATA_RESUME, top_k=2)) == 2
        everything = recommend_roles(DATA_RESUME, top_k=1000)
        assert len(everything) < 1000
        assert sum(r["probability_percent"] for r in everything) == pytest.approx(100.0, abs=0.5)

    def test_probabilities_sum_sensibly(self):
        """Ensure probabilities sum sensibly (approx 100% when all classes returned)."""
        all_roles = recommend_roles(DATA_RESUME, top_k=1000)
        total_prob = sum(r["probability_percent"] for r in all_roles)
        assert total_prob == pytest.approx(100.0, abs=0.5)
        # For default top_k=4, sum of top probabilities should be > 0 and <= 100.0
        top4 = recommend_roles(DATA_RESUME, top_k=4)
        top4_sum = sum(r["probability_percent"] for r in top4)
        assert 0.0 < top4_sum <= 100.0

    def test_obvious_resume_gets_sensible_top_role(self):
        top = recommend_roles(DATA_RESUME, top_k=1)[0]["role"]
        assert top in {"Data Analyst", "Data Science", "Business Analyst"}
