"""Integration tests for the /resume/star-rewrite API endpoint."""

import json
import re
import unittest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestStarRewriteEndpoint(unittest.TestCase):
    """Test suite for POST /resume/star-rewrite."""

    # 1. Valid request without a JD
    def test_valid_request_without_jd(self):
        """A valid resume text without JD returns suggestions, counts, and disclaimer."""
        payload = {
            "resume_text": """
            EXPERIENCE
            • Responsible for deployment of the API.
            • Worked on backend microservices with Python.
            """,
            "max_suggestions": 5,
        }
        response = client.post("/resume/star-rewrite", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("suggestions", data)
        self.assertIn("counts", data)
        self.assertIn("disclaimer", data)

        counts = data["counts"]
        self.assertGreaterEqual(counts["weak_bullets"], 1)
        self.assertGreaterEqual(counts["suggestions"], 1)
        self.assertIsInstance(counts["missing_skill_suggestions"], int)

        # Without a JD, no missing-skill suggestions should be flagged
        for s in data["suggestions"]:
            self.assertEqual(s["skill_status"], "present")
            self.assertFalse(s["requires_real_experience"])

    # 2. Valid request with a JD flagging missing-skill suggestions
    def test_valid_request_with_jd_missing_skill_flagged(self):
        """When JD has missing skills, matching category bullet suggests them with requires_real_experience=True."""
        payload = {
            "resume_text": """
            EXPERIENCE
            • Responsible for deployment of the API.
            """,
            "jd_text": "We require Docker for containerized API deployment and microservices.",
            "max_suggestions": 5,
        }
        response = client.post("/resume/star-rewrite", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        suggestions = data["suggestions"]
        self.assertGreaterEqual(len(suggestions), 1)

        first = suggestions[0]
        self.assertEqual(first["target_skill"], "docker")
        self.assertEqual(first["skill_status"], "missing")
        self.assertTrue(first["requires_real_experience"])
        self.assertTrue(first["rewritten_bullet"].startswith("If you have used Docker on this work:"))
        self.assertIn("Only include docker if you have genuine hands-on experience", first["honesty_note"])
        self.assertGreaterEqual(data["counts"]["missing_skill_suggestions"], 1)

    # 3. Empty text returns 400
    def test_empty_resume_text_returns_400(self):
        """Empty or whitespace-only resume_text returns HTTP 400 Bad Request."""
        res1 = client.post("/resume/star-rewrite", json={"resume_text": ""})
        self.assertEqual(res1.status_code, 400)
        self.assertIn("resume_text must not be empty", res1.json()["detail"])

        res2 = client.post("/resume/star-rewrite", json={"resume_text": "   \n\t  "})
        self.assertEqual(res2.status_code, 400)

    # 4. Invalid body returns 422
    def test_invalid_body_returns_422(self):
        """Missing required fields or invalid types returns HTTP 422 Unprocessable Entity."""
        # Missing resume_text
        res1 = client.post("/resume/star-rewrite", json={"jd_text": "some jd"})
        self.assertEqual(res1.status_code, 422)

        # Empty body
        res2 = client.post("/resume/star-rewrite", json={})
        self.assertEqual(res2.status_code, 422)

        # Invalid max_suggestions type
        res3 = client.post("/resume/star-rewrite", json={"resume_text": "valid text", "max_suggestions": "invalid"})
        self.assertEqual(res3.status_code, 422)

        # Malformed raw payload
        res4 = client.post(
            "/resume/star-rewrite",
            content="not json",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(res4.status_code, 422)

    # 5. Disclaimer is always present
    def test_disclaimer_is_always_present(self):
        """Response always includes the standardized disclaimer regarding true facts and structure."""
        # Test without JD
        res1 = client.post(
            "/resume/star-rewrite",
            json={"resume_text": "EXPERIENCE\n• Worked on web apps with Python."},
        )
        self.assertEqual(res1.status_code, 200)
        disc1 = res1.json().get("disclaimer", "")
        self.assertIn("structure", disc1.lower())
        self.assertIn("true facts", disc1.lower())

        # Test with JD
        res2 = client.post(
            "/resume/star-rewrite",
            json={
                "resume_text": "EXPERIENCE\n• Worked on web apps with Python.",
                "jd_text": "Required skills: Python, Django, PostgreSQL",
            },
        )
        self.assertEqual(res2.status_code, 200)
        disc2 = res2.json().get("disclaimer", "")
        self.assertEqual(disc1, disc2)

    # 6. Property test on 3 sample resumes: never contains a digit not in the input text
    def test_property_never_contains_digit_not_in_input_text(self):
        """Anti-hallucination property: Generated suggestions never contain digits not in input text."""
        samples = [
            # Sample 1: Zero digits in the resume
            """
            EXPERIENCE
            • Responsible for deployment of the API and web services.
            • Worked on backend microservices with Python and FastAPI.
            • Helped with database queries and schema design.
            """,
            # Sample 2: Contains specific years and a count digit (2, 0, 1, 8, 4, 3)
            """
            EXPERIENCE
            Software Engineer | 2018 - 2024
            • Responsible for deployment across 3 environments.
            • Worked on refactoring monolithic modules.
            """,
            # Sample 3: Contains metric percentage and hours (7, 5, 4, 0)
            """
            EXPERIENCE
            Systems Engineer
            • Responsible for server maintenance and database uptime of 75%.
            • Assisted in automated deployment scripts taking 40 minutes.
            • Involved in customer support escalations.
            """,
        ]

        jd_text = "Required skills: Docker, Kubernetes, PostgreSQL."

        for idx, sample in enumerate(samples, start=1):
            input_digits = set(re.findall(r"\d", sample))

            # Test both with and without JD
            for payload in [
                {"resume_text": sample},
                {"resume_text": sample, "jd_text": jd_text},
            ]:
                resp = client.post("/resume/star-rewrite", json=payload)
                self.assertEqual(resp.status_code, 200)
                data = resp.json()

                for s in data["suggestions"]:
                    # Verify rewritten bullet
                    bullet_digits = set(re.findall(r"\d", s["rewritten_bullet"]))
                    self.assertTrue(
                        bullet_digits.issubset(input_digits),
                        f"Sample {idx}: Rewritten bullet contained invented digits: {bullet_digits - input_digits}",
                    )

                    # Verify STAR scaffold parts
                    for part_name, part_val in s["star"].items():
                        part_digits = set(re.findall(r"\d", part_val))
                        self.assertTrue(
                            part_digits.issubset(input_digits),
                            f"Sample {idx}: STAR {part_name} contained invented digits: {part_digits - input_digits}",
                        )

    # 7. Max suggestions is respected
    def test_max_suggestions_limits_results(self):
        """Parameter max_suggestions properly truncates the returned suggestion list."""
        payload = {
            "resume_text": """
            EXPERIENCE
            • Responsible for deployment of the API.
            • Worked on backend microservices with Python.
            • Helped with database queries and schema design.
            • Involved in frontend web development.
            • Duties included server maintenance.
            """,
            "max_suggestions": 2,
        }
        response = client.post("/resume/star-rewrite", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertLessEqual(len(data["suggestions"]), 2)
        self.assertEqual(data["counts"]["suggestions"], len(data["suggestions"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
