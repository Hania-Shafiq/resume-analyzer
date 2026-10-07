"""Unit tests for analyze_job_description() in app/skills.py.

Two realistic sample JDs are used:
  JD_1 — Backend Engineer: has explicit Required / Preferred / Benefits headings.
  JD_2 — Data Scientist:   no section headings; uses inline signals only
                            ("must have", "nice to have", "familiarity with").
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.skills import analyze_job_description, JDAnalysis


# ---------------------------------------------------------------------------
# Sample JD 1 — Backend Engineer (explicit section headings)
# ---------------------------------------------------------------------------
JD_BACKEND = """
Senior Backend Engineer

We are looking for an experienced backend engineer to join our platform team.

About the Role
Design and build highly scalable microservices for our e-commerce platform.
Work closely with frontend and DevOps teams to deliver reliable APIs.

Requirements
- 5+ years of experience in backend development
- Proficiency in Python and Golang
- Experience with REST API design and implementation
- PostgreSQL and Redis for data storage and caching
- Docker and Kubernetes for container orchestration
- Git for version control
- Bachelor's degree in Computer Science or related field

Preferred
- Experience with Apache Kafka for event streaming
- Familiarity with AWS (EC2, S3, RDS)
- Knowledge of Terraform for infrastructure as code
- GraphQL API design experience

Benefits
- Competitive salary
- Remote-first culture
- Health insurance
"""

# ---------------------------------------------------------------------------
# Sample JD 2 — Data Scientist (NO section headings — inline signals only)
# ---------------------------------------------------------------------------
JD_DATA_SCIENCE = """
Data Scientist — NLP & ML

You will work on building recommendation systems and NLP pipelines.

You must have strong Python skills and experience with scikit-learn and PyTorch.
Machine learning and natural language processing are required for this role.
A minimum of 3 years of experience in a data science role is expected.
Pandas and NumPy are required for data manipulation tasks.

Familiarity with Hugging Face Transformers is a plus.
Knowledge of MLflow for experiment tracking is preferred.
Experience with Spark or Hadoop is nice to have.
A Master's degree in Data Science, Statistics, or Computer Science is required.
"""


class TestAnalyzeJDBackend(unittest.TestCase):
    """Tests for JD_BACKEND — section-heading-based splitting."""

    @classmethod
    def setUpClass(cls):
        """Run analysis once; share across all test methods in this class."""
        cls.result = analyze_job_description(JD_BACKEND)

    # ── Return type ──────────────────────────────────────────────────────────

    def test_returns_jd_analysis_instance(self):
        """analyze_job_description should return a JDAnalysis object."""
        self.assertIsInstance(self.result, JDAnalysis)

    def test_to_dict_has_all_keys(self):
        """to_dict() should include all six required keys."""
        d = self.result.to_dict()
        for key in ("required_skills", "preferred_skills", "min_experience_years",
                    "education_requirement", "job_title", "job_category"):
            self.assertIn(key, d, msg=f"Missing key: {key}")

    # ── Required skills ──────────────────────────────────────────────────────

    def test_python_is_required(self):
        """'python' should be in required_skills (listed under Requirements)."""
        self.assertIn("python", self.result.required_skills)

    def test_postgresql_is_required(self):
        """'postgresql' should be required (alias 'PostgreSQL' under Requirements)."""
        self.assertIn("postgresql", self.result.required_skills)

    def test_docker_is_required(self):
        """'docker' should appear in required_skills."""
        self.assertIn("docker", self.result.required_skills)

    def test_kubernetes_is_required(self):
        """'kubernetes' should appear in required_skills (k8s alias)."""
        self.assertIn("kubernetes", self.result.required_skills)

    def test_rest_api_is_required(self):
        """'rest api' (multi-word) should be required."""
        self.assertIn("rest api", self.result.required_skills)

    # ── Preferred skills ─────────────────────────────────────────────────────

    def test_kafka_is_preferred(self):
        """'kafka' should be preferred (listed under Preferred section)."""
        self.assertIn("kafka", self.result.preferred_skills)

    def test_terraform_is_preferred(self):
        """'terraform' should be preferred."""
        self.assertIn("terraform", self.result.preferred_skills)

    def test_graphql_is_preferred(self):
        """'graphql' should be preferred."""
        self.assertIn("graphql", self.result.preferred_skills)

    # ── No overlap rule ──────────────────────────────────────────────────────

    def test_no_skill_in_both_lists(self):
        """No canonical skill should appear in both required and preferred."""
        overlap = set(self.result.required_skills) & set(self.result.preferred_skills)
        self.assertEqual(overlap, set(), msg=f"Overlap found: {overlap}")

    # ── Experience ───────────────────────────────────────────────────────────

    def test_min_experience_is_5(self):
        """'5+ years of experience' should be parsed as 5."""
        self.assertEqual(self.result.min_experience_years, 5)

    # ── Education ────────────────────────────────────────────────────────────

    def test_education_detects_bachelors(self):
        """'Bachelor's degree in Computer Science' should be detected."""
        degrees = [e.degree for e in self.result.education_requirement]
        self.assertIn("Bachelor's", degrees)

    # ── Title / category ─────────────────────────────────────────────────────

    def test_job_category_is_backend(self):
        """JD_BACKEND should be classified as 'backend'."""
        self.assertEqual(self.result.job_category, "backend")

    def test_job_title_is_not_none(self):
        """job_title should be set (first non-empty line)."""
        self.assertIsNotNone(self.result.job_title)
        self.assertIn("Backend", self.result.job_title)


class TestAnalyzeJDDataScience(unittest.TestCase):
    """Tests for JD_DATA_SCIENCE — inline-signal fallback (no headings)."""

    @classmethod
    def setUpClass(cls):
        cls.result = analyze_job_description(JD_DATA_SCIENCE)

    # ── Inline required signals ───────────────────────────────────────────────

    def test_python_required_via_inline_signal(self):
        """'python' mentioned with 'must have' → required."""
        self.assertIn("python", self.result.required_skills)

    def test_sklearn_required_via_inline_signal(self):
        """'scikit-learn' (alias 'sklearn') → required (inline 'must have')."""
        self.assertIn("scikit-learn", self.result.required_skills)

    def test_pytorch_required(self):
        """'pytorch' should appear in required_skills."""
        self.assertIn("pytorch", self.result.required_skills)

    def test_nlp_required(self):
        """'natural language processing' (alias 'nlp') should be required."""
        self.assertIn("natural language processing", self.result.required_skills)

    def test_pandas_required(self):
        """'pandas' explicitly required for data manipulation."""
        self.assertIn("pandas", self.result.required_skills)

    # ── Inline preferred signals ──────────────────────────────────────────────

    def test_huggingface_preferred(self):
        """'hugging face' (alias) marked as 'a plus' → preferred."""
        self.assertIn("hugging face", self.result.preferred_skills)

    def test_mlflow_preferred(self):
        """'mlflow' marked as 'preferred' inline → preferred."""
        self.assertIn("mlflow", self.result.preferred_skills)

    def test_spark_preferred(self):
        """'spark' marked as 'nice to have' → preferred."""
        self.assertIn("spark", self.result.preferred_skills)

    # ── No overlap ────────────────────────────────────────────────────────────

    def test_no_skill_in_both_lists(self):
        """No canonical skill should appear in both required and preferred."""
        overlap = set(self.result.required_skills) & set(self.result.preferred_skills)
        self.assertEqual(overlap, set(), msg=f"Overlap found: {overlap}")

    # ── Experience ────────────────────────────────────────────────────────────

    def test_min_experience_is_3(self):
        """'minimum of 3 years' → 3."""
        self.assertEqual(self.result.min_experience_years, 3)

    # ── Education ─────────────────────────────────────────────────────────────

    def test_masters_required(self):
        """'Master's degree in Data Science' should be detected."""
        degrees = [e.degree for e in self.result.education_requirement]
        self.assertIn("Master's", degrees)

    # ── Category ──────────────────────────────────────────────────────────────

    def test_job_category_is_machine_learning(self):
        """JD_DATA_SCIENCE title contains 'Data Scientist' → machine_learning."""
        self.assertEqual(self.result.job_category, "machine_learning")


class TestAnalyzeJDEdgeCases(unittest.TestCase):
    """Guard-rail tests for error handling and degenerate inputs."""

    def test_empty_string_raises_value_error(self):
        """Blank input should raise ValueError."""
        with self.assertRaises(ValueError):
            analyze_job_description("   ")

    def test_non_string_raises_type_error(self):
        """Non-string input should raise TypeError."""
        with self.assertRaises(TypeError):
            analyze_job_description(42)

    def test_no_skills_jd_returns_empty_lists(self):
        """A JD with no recognisable skills should return empty skill lists."""
        result = analyze_job_description(
            "We are a great company. Apply now and enjoy free lunches every Friday."
        )
        self.assertIsInstance(result.required_skills, list)
        self.assertIsInstance(result.preferred_skills, list)

    def test_no_experience_mentioned_returns_none(self):
        """JD with no experience mention should return None."""
        result = analyze_job_description(
            "Software Engineer\nRequirements\nKnowledge of Python and Docker required."
        )
        self.assertIsNone(result.min_experience_years)

    def test_all_required_when_no_preferred_section(self):
        """JD with only a Required section should have an empty preferred list."""
        jd = "Backend Developer\nRequirements\nPython, FastAPI, PostgreSQL required."
        result = analyze_job_description(jd)
        self.assertEqual(result.preferred_skills, [])

    def test_skill_in_both_sections_resolved_to_required(self):
        """If a skill appears in both Required and Preferred, it must be only in required_skills."""
        jd = """
        Software Engineer
        Requirements:
        - Python
        - Docker
        Preferred:
        - Python
        - Kubernetes
        """
        result = analyze_job_description(jd)
        self.assertIn("python", result.required_skills)
        self.assertNotIn("python", result.preferred_skills)
        self.assertIn("kubernetes", result.preferred_skills)

    def test_no_headings_no_signals_defaults_to_required(self):
        """When JD has neither section headings nor inline signals, detected skills default to required."""
        jd = """
        Full Stack Developer
        We use Python, React, and PostgreSQL in our daily workflow.
        """
        result = analyze_job_description(jd)
        self.assertIn("python", result.required_skills)
        self.assertIn("react", result.required_skills)
        self.assertEqual(result.preferred_skills, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
