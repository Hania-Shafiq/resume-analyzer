"""Tests for the POST /recommend-roles endpoint.

Tests that need the real trained model are skipped until
``notebooks/03_job_classifier.ipynb`` has created app/models/job_classifier.joblib.
Validation and error-path tests always run (the model is mocked where needed).
"""

import unittest
from unittest import mock

from fastapi.testclient import TestClient

from app.main import app
from app.recommender import MODEL_PATH

DATA_RESUME = (
    "Data analyst with 4 years of experience building Power BI and Tableau dashboards. "
    "Wrote SQL queries and automated weekly reports with Python and pandas."
)


class TestRecommendRolesValidation(unittest.TestCase):
    """Input validation and error mapping (no trained model required)."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_empty_text_returns_400(self):
        r = self.client.post("/recommend-roles", json={"resume_text": "   "})
        self.assertEqual(r.status_code, 400)
        self.assertIn("must not be empty", r.json()["detail"])

    def test_missing_field_returns_422(self):
        r = self.client.post("/recommend-roles", json={})
        self.assertEqual(r.status_code, 422)

    def test_invalid_top_k_returns_422(self):
        for bad in (0, -3, 51, "many"):
            r = self.client.post("/recommend-roles", json={"resume_text": DATA_RESUME, "top_k": bad})
            self.assertEqual(r.status_code, 422, msg=f"top_k={bad!r}")

    def test_missing_model_returns_503_with_helpful_message(self):
        with mock.patch("app.main.recommend_roles", side_effect=FileNotFoundError("Trained model not found ...")):
            r = self.client.post("/recommend-roles", json={"resume_text": DATA_RESUME})
        self.assertEqual(r.status_code, 503)
        self.assertIn("Trained model not found", r.json()["detail"])

    def test_unusable_text_returns_422(self):
        with mock.patch("app.main.recommend_roles", side_effect=ValueError("Text is empty after cleaning.")):
            r = self.client.post("/recommend-roles", json={"resume_text": DATA_RESUME})
        self.assertEqual(r.status_code, 422)


@unittest.skipUnless(MODEL_PATH.exists(), "Run notebooks/03_job_classifier.ipynb first to create the model.")
class TestRecommendRolesWithModel(unittest.TestCase):
    """End-to-end with the real saved model."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_default_returns_four_sorted_roles(self):
        r = self.client.post("/recommend-roles", json={"resume_text": DATA_RESUME})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(len(body["recommendations"]), 4)
        probs = [x["probability_percent"] for x in body["recommendations"]]
        self.assertEqual(probs, sorted(probs, reverse=True))
        for item in body["recommendations"]:
            self.assertEqual(set(item), {"role", "probability_percent"})
            self.assertGreaterEqual(item["probability_percent"], 0.0)
            self.assertLessEqual(item["probability_percent"], 100.0)
        self.assertIn("model", body)

    def test_top_k_is_respected(self):
        r = self.client.post("/recommend-roles", json={"resume_text": DATA_RESUME, "top_k": 2})
        self.assertEqual(len(r.json()["recommendations"]), 2)

    def test_top_k_above_number_of_roles_is_capped(self):
        r = self.client.post("/recommend-roles", json={"resume_text": DATA_RESUME, "top_k": 50})
        self.assertEqual(r.status_code, 200)
        recs = r.json()["recommendations"]
        self.assertLess(len(recs), 50)
        self.assertAlmostEqual(sum(x["probability_percent"] for x in recs), 100.0, delta=0.5)

    def test_obvious_resume_gets_sensible_top_role(self):
        top = self.client.post("/recommend-roles", json={"resume_text": DATA_RESUME, "top_k": 1}).json()
        self.assertIn(top["recommendations"][0]["role"], {"Data Analyst", "Data Science", "Business Analyst"})


if __name__ == "__main__":
    unittest.main()
