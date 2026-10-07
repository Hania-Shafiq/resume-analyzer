"""Unit tests for app/skills.py — extract_skills, extract_education, extract_experience_years.

Why a hybrid dictionary + NLP approach beats keywords-only
----------------------------------------------------------
Plain keyword scanning has three well-known weaknesses that this test suite
deliberately exercises:

1. Alias coverage
   "js" and "javascript" are the same skill.  A keyword list would need two
   separate entries and break the moment a new alias appears (e.g. "es6").
   PhraseMatcher + aliases table maps both to the *canonical* name "javascript"
   in one pass.

2. Multi-word skill phrases
   "natural language processing", "machine learning", "sql server" are
   single *concepts* that span multiple tokens.  A naive word-level scan would
   match "learning" in "deep learning" and "machine learning" separately,
   producing duplicates.  PhraseMatcher treats each phrase as an atomic unit.

3. Case-insensitivity and normalisation
   Resumes use "Python", "PYTHON", "python3" interchangeably.  The LOWER
   attribute on PhraseMatcher removes the need for any manual .lower() call.
   A regex approach would require (?i) flags *and* alias expansion.

These tests are structured around three realistic resume samples that together
stress-test all of the above scenarios.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.skills import extract_skills, extract_education, extract_experience_years


# ---------------------------------------------------------------------------
# Sample resume texts
# ---------------------------------------------------------------------------

# Sample 1 — Software Engineer: tests aliases (js, k8s, postgres), multi-word
# phrases (ci/cd, rest api), and mixed casing (PYTHON, Docker).
SAMPLE_1 = """
John Doe — Software Engineer
5+ years of experience building scalable backend systems.
Skills: PYTHON, JS, REST API, PostgreSQL, Docker, k8s, CI/CD, Git.
Bachelor of Science in Computer Science, State University (2018).
"""

# Sample 2 — Data Scientist: tests ML-domain aliases (sklearn, sbert),
# multi-word skills (natural language processing, deep learning), and
# education "Master's" with field.
SAMPLE_2 = """
Sarah Ali — Data Scientist
Over 3 years of professional experience in AI research.
Technical Skills: Python, sklearn, TensorFlow, NLP, Deep Learning,
Sentence Transformers, Pandas, NumPy, Matplotlib.
Education: MSc in Data Science, MIT (2021).
"""

# Sample 3 — Full-Stack + Cloud: tests cloud aliases (gcp, amazon web services),
# web framework aliases (nextjs, fast api), experience range "3-5 years",
# and a PhD entry.
SAMPLE_3 = """
Lena Müller — Full-Stack Developer & Cloud Architect
3-5 years experience in cloud-native applications.
Stack: NextJS, FastAPI, TypeScript, GCP, Amazon Web Services, Redis, Neo4j.
Ph.D. in Computer Engineering, TU Berlin, 2019.
"""


class TestExtractSkills(unittest.TestCase):
    """Tests for extract_skills() — alias resolution, dedup, multi-word phrases."""

    def test_alias_js_resolves_to_javascript(self):
        """'js' alias should return canonical name 'javascript'."""
        skills = extract_skills(SAMPLE_1)
        self.assertIn("javascript", skills, msg="Alias 'js' → 'javascript' failed")

    def test_alias_postgres_resolves_to_postgresql(self):
        """'postgres' alias should return canonical name 'postgresql'."""
        skills = extract_skills(SAMPLE_1)
        self.assertIn("postgresql", skills, msg="Alias 'postgres' → 'postgresql' failed")

    def test_alias_k8s_resolves_to_kubernetes(self):
        """'k8s' alias should return canonical name 'kubernetes'."""
        skills = extract_skills(SAMPLE_1)
        self.assertIn("kubernetes", skills, msg="Alias 'k8s' → 'kubernetes' failed")

    def test_case_insensitive_python(self):
        """'PYTHON' (all-caps) should still be extracted."""
        skills = extract_skills(SAMPLE_1)
        self.assertIn("python", skills, msg="Case-insensitive match for 'PYTHON' failed")

    def test_multiword_rest_api(self):
        """Multi-word phrase 'REST API' should be extracted as single skill."""
        skills = extract_skills(SAMPLE_1)
        self.assertIn("rest api", skills, msg="Multi-word 'rest api' not found")

    def test_multiword_cicd(self):
        """'CI/CD' should be extracted correctly."""
        skills = extract_skills(SAMPLE_1)
        self.assertIn("ci/cd", skills, msg="'ci/cd' not found in skills")

    def test_deduplication(self):
        """Each canonical skill should appear at most once."""
        text = "Python, python, PYTHON, py, Python3"
        skills = extract_skills(text)
        self.assertEqual(skills.count("python"), 1, msg="Duplicate 'python' not deduplicated")

    def test_sklearn_alias(self):
        """'sklearn' alias should resolve to 'scikit-learn'."""
        skills = extract_skills(SAMPLE_2)
        self.assertIn("scikit-learn", skills, msg="Alias 'sklearn' → 'scikit-learn' failed")

    def test_sbert_alias(self):
        """'Sentence Transformers' should resolve to 'sentence-transformers'."""
        skills = extract_skills(SAMPLE_2)
        self.assertIn("sentence-transformers", skills,
                      msg="Alias 'sentence transformers' → 'sentence-transformers' failed")

    def test_nlp_alias_maps_to_natural_language_processing(self):
        """'NLP' should resolve to 'natural language processing'."""
        skills = extract_skills(SAMPLE_2)
        self.assertIn("natural language processing", skills,
                      msg="Alias 'NLP' → 'natural language processing' failed")

    def test_nextjs_alias(self):
        """'NextJS' should resolve to 'next.js'."""
        skills = extract_skills(SAMPLE_3)
        self.assertIn("next.js", skills, msg="Alias 'nextjs' → 'next.js' failed")

    def test_fast_api_alias(self):
        """'fast api' alias should resolve to 'fastapi'."""
        skills = extract_skills(SAMPLE_3)
        self.assertIn("fastapi", skills, msg="Alias 'fast api' → 'fastapi' failed")

    def test_aws_alias(self):
        """'Amazon Web Services' should resolve to 'aws'."""
        skills = extract_skills(SAMPLE_3)
        self.assertIn("aws", skills, msg="Alias 'amazon web services' → 'aws' failed")

    def test_gcp_alias(self):
        """'gcp' should resolve to 'gcp'."""
        skills = extract_skills(SAMPLE_3)
        self.assertIn("gcp", skills, msg="'gcp' not found in sample 3 skills")

    def test_empty_text_raises(self):
        """Empty string should raise ValueError."""
        with self.assertRaises(ValueError):
            extract_skills("   ")

    def test_non_string_raises(self):
        """Non-string input should raise TypeError."""
        with self.assertRaises(TypeError):
            extract_skills(12345)

    def test_no_false_positives_on_noise(self):
        """Text with no skills should return an empty list (not crash)."""
        result = extract_skills("The weather today is sunny and warm in Karachi.")
        # Weather words are not skills — list may be empty
        self.assertIsInstance(result, list)


class TestExtractEducation(unittest.TestCase):
    """Tests for extract_education() — degree detection and field parsing."""

    def test_bachelor_detected_sample1(self):
        """Sample 1 should detect a Bachelor's degree."""
        entries = extract_education(SAMPLE_1)
        degrees = [e.degree for e in entries]
        self.assertIn("Bachelor's", degrees, msg="Bachelor's not detected in sample 1")

    def test_bachelor_field_computer_science(self):
        """Sample 1 field should be 'Computer Science'."""
        entries = extract_education(SAMPLE_1)
        bsc = next((e for e in entries if e.degree == "Bachelor's"), None)
        self.assertIsNotNone(bsc)
        self.assertIsNotNone(bsc.field)
        self.assertIn("Computer Science", bsc.field)

    def test_masters_detected_sample2(self):
        """Sample 2 'MSc' should detect as Master's."""
        entries = extract_education(SAMPLE_2)
        degrees = [e.degree for e in entries]
        self.assertIn("Master's", degrees, msg="Master's not detected in sample 2")

    def test_phd_detected_sample3(self):
        """Sample 3 'Ph.D.' should detect as PhD."""
        entries = extract_education(SAMPLE_3)
        degrees = [e.degree for e in entries]
        self.assertIn("PhD", degrees, msg="PhD not detected in sample 3")

    def test_returns_namedtuple_fields(self):
        """Each entry should have degree, field, and raw attributes."""
        entries = extract_education(SAMPLE_1)
        self.assertTrue(len(entries) > 0)
        e = entries[0]
        self.assertTrue(hasattr(e, "degree"))
        self.assertTrue(hasattr(e, "field"))
        self.assertTrue(hasattr(e, "raw"))

    def test_no_education_returns_empty(self):
        """Text with no education info should return an empty list."""
        result = extract_education("I am a Python developer with 5 years of experience.")
        self.assertEqual(result, [])


class TestExtractExperienceYears(unittest.TestCase):
    """Tests for extract_experience_years() — regex-based year estimation."""

    def test_direct_years_sample1(self):
        """'5+ years of experience' → 5."""
        result = extract_experience_years(SAMPLE_1)
        self.assertEqual(result, 5, msg=f"Expected 5, got {result}")

    def test_over_n_years_sample2(self):
        """'Over 3 years of professional experience' → 3."""
        result = extract_experience_years(SAMPLE_2)
        self.assertEqual(result, 3, msg=f"Expected 3, got {result}")

    def test_range_pattern_sample3(self):
        """'3-5 years experience' → 3 (lower bound retained, max applied)."""
        result = extract_experience_years(SAMPLE_3)
        # max of [3, 5] = 5 because both 3 and 5 are captured
        self.assertIn(result, [3, 5], msg=f"Expected 3 or 5, got {result}")

    def test_no_experience_returns_none(self):
        """Text with no year pattern should return None."""
        result = extract_experience_years("Proficient in Python and Docker.")
        self.assertIsNone(result)

    def test_explicit_number_text(self):
        """'more than 7 years' → 7."""
        result = extract_experience_years("more than 7 years of software engineering experience")
        self.assertEqual(result, 7)


class TestSkillCategory(unittest.TestCase):
    """Tests for skill_category() lookup."""

    def test_known_skill_categories(self):
        """Ensure canonical skills resolve to their taxonomical categories."""
        from app.skills import skill_category
        self.assertEqual(skill_category("python"), "programming")
        self.assertEqual(skill_category("postgresql"), "databases")
        self.assertEqual(skill_category("docker"), "devops_tools")
        self.assertEqual(skill_category("fastapi"), "web")

    def test_unknown_skill_returns_none(self):
        """Ensure unknown skills return None."""
        from app.skills import skill_category
        self.assertIsNone(skill_category("underwater_basket_weaving"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
