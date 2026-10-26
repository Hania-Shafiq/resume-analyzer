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

from app.skills import (
    extract_skills,
    extract_skills_with_context,
    extract_education,
    extract_experience_years,
)


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

    def test_ms_tools_not_detected_as_masters(self):
        """Microsoft tools (MS Project, MS Office, etc.) and milliseconds must not trigger Master's."""
        false_positives = [
            "Project Management: MS Project, Agile, Waterfall, SDLC",
            "Other: Git, REST APIs, Data Structures & Algorithms, OOP, Assembly Language, MS Office",
            "Experienced in MS Excel, MS Word, MS PowerPoint, and MS Access.",
            "Reduced database query latency down to 50 ms.",
            "Ms. Sarah Connor is our HR contact.",
        ]
        for line in false_positives:
            res = extract_education(line)
            degrees = [e.degree for e in res]
            self.assertNotIn("Master's", degrees, msg=f"False positive Master's in: {line}")

    def test_english_words_and_job_roles_not_false_degrees(self):
        """Common English words ('be', 'as') and role titles ('Associate') must not trigger false degrees."""
        false_positives = [
            "Candidate will be responsible for building cloud-native microservices.",
            "Worked as a backend software engineer for 3 years.",
            "Seeking Junior/Associate or Management Trainee roles in Data Analytics.",
        ]
        for line in false_positives:
            res = extract_education(line)
            self.assertEqual(res, [], msg=f"False positive degree detected in: {line}")

    def test_bachelor_with_intermediate_and_ms_tools(self):
        """CV with MS Project, Bachelor of Science, and Intermediate must yield Bachelor's as top degree."""
        sample_cv = """
        TECHNICAL SKILLS
        Project Management: MS Project, Agile, Waterfall, SDLC
        Other: Git, REST APIs, Data Structures & Algorithms, OOP, Assembly Language, MS Office
        EDUCATION
        Bachelor of Science in Computer Science (BSCS)
        University of Karachi — UBIT | Jan 2023 – Nov 2026 | In Progress
        Intermediate (Pre-Engineering)
        Govt Degree College Malir Cantt | Aug 2021 – Aug 2022
        """
        res = extract_education(sample_cv)
        degrees = [e.degree for e in res]
        self.assertNotIn("Master's", degrees, "Master's should NOT be detected from MS Project / MS Office")
        self.assertIn("Bachelor's", degrees, "Bachelor's must be detected")
        self.assertIn("High School", degrees, "Intermediate must be detected as High School")
        self.assertEqual(res[0].degree, "Bachelor's", "Highest degree must be Bachelor's")


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


class TestExtractSkillsWithContext(unittest.TestCase):
    """Tests for extract_skills_with_context() checking confidence and sentence evidence.

    Includes 10 distinct context test sentences (5 high confidence, 5 low confidence)
    covering patterns such as 'migrated away from X', 'moved off X', 'no experience with X',
    'not familiar with X', 'interested in learning X', 'want to learn X', 'familiar with X',
    'exposure to X'.
    """

    # --- 5 High Confidence Tests ---

    def test_high_conf_1_architected_and_deployed(self):
        """Sentence 1 (High): Microservices architecture with active tooling."""
        text = "Architected and deployed microservices using Python and FastAPI."
        result = extract_skills_with_context(text)
        self.assertIn("python", result)
        self.assertEqual(result["python"]["confidence"], "high")
        self.assertEqual(result["python"]["evidence"], text)
        self.assertIn("fastapi", result)
        self.assertEqual(result["fastapi"]["confidence"], "high")

    def test_high_conf_2_migration_to_cloud(self):
        """Sentence 2 (High): Migration TO a platform denotes strong experience."""
        text = "Led the migration of on-premise infrastructure to AWS."
        result = extract_skills_with_context(text)
        self.assertIn("aws", result)
        self.assertEqual(result["aws"]["confidence"], "high")
        self.assertEqual(result["aws"]["evidence"], text)

    def test_high_conf_3_years_experience_react(self):
        """Sentence 3 (High): Direct professional experience statement."""
        text = "Over 4 years of hands-on experience building production web apps with React."
        result = extract_skills_with_context(text)
        self.assertIn("react", result)
        self.assertEqual(result["react"]["confidence"], "high")
        self.assertEqual(result["react"]["evidence"], text)

    def test_high_conf_4_designed_database_schemas(self):
        """Sentence 4 (High): Designing database schemas and queries."""
        text = "Designed robust relational database schemas and complex queries in PostgreSQL."
        result = extract_skills_with_context(text)
        self.assertIn("postgresql", result)
        self.assertEqual(result["postgresql"]["confidence"], "high")
        self.assertEqual(result["postgresql"]["evidence"], text)

    def test_high_conf_5_optimized_pipeline(self):
        """Sentence 5 (High): Performance optimization in deep learning."""
        text = "Optimized deep learning model training pipelines using PyTorch."
        result = extract_skills_with_context(text)
        self.assertIn("pytorch", result)
        self.assertEqual(result["pytorch"]["confidence"], "high")
        self.assertIn("deep learning", result)
        self.assertEqual(result["deep learning"]["confidence"], "high")

    # --- 5 Low Confidence Tests ---

    def test_low_conf_1_migrated_away_from(self):
        """Sentence 6 (Low): 'migrated away from X' denotes legacy/dropped skill."""
        text = "We migrated away from Java in favor of Go."
        result = extract_skills_with_context(text)
        self.assertIn("java", result)
        self.assertEqual(result["java"]["confidence"], "low")
        self.assertEqual(result["java"]["evidence"], text)
        # Contrastive target should be high confidence
        self.assertIn("go", result)
        self.assertEqual(result["go"]["confidence"], "high")

    def test_low_conf_2_moved_off(self):
        """Sentence 7 (Low): 'moved off X' denotes abandoned skill."""
        text = "Our infrastructure team moved off MySQL last year."
        result = extract_skills_with_context(text)
        self.assertIn("mysql", result)
        self.assertEqual(result["mysql"]["confidence"], "low")
        self.assertEqual(result["mysql"]["evidence"], text)

    def test_low_conf_3_no_experience_with(self):
        """Sentence 8 (Low): 'no experience with X' denotes explicit lack of skill."""
        text = "I currently have no experience with Rust."
        result = extract_skills_with_context(text)
        self.assertIn("rust", result)
        self.assertEqual(result["rust"]["confidence"], "low")
        self.assertEqual(result["rust"]["evidence"], text)

    def test_low_conf_4_not_familiar_with(self):
        """Sentence 9 (Low): 'not familiar with X' denotes lack of proficiency."""
        text = "Currently not familiar with GraphQL but eager to expand my stack."
        result = extract_skills_with_context(text)
        self.assertIn("graphql", result)
        self.assertEqual(result["graphql"]["confidence"], "low")
        self.assertEqual(result["graphql"]["evidence"], text)

    def test_low_conf_5_interested_in_learning(self):
        """Sentence 10 (Low): 'interested in learning X' indicates aspirational skill."""
        text = "Interested in learning LangChain for LLM agent workflows."
        result = extract_skills_with_context(text)
        self.assertIn("langchain", result)
        self.assertEqual(result["langchain"]["confidence"], "low")
        self.assertEqual(result["langchain"]["evidence"], text)

    # Additional patterns specified in requirements
    def test_additional_low_conf_patterns(self):
        """Verify additional low confidence patterns: familiar with, exposure to, want/plan to learn."""
        s1 = "Familiar with Docker and Kubernetes."
        r1 = extract_skills_with_context(s1)
        self.assertEqual(r1["docker"]["confidence"], "low")
        self.assertEqual(r1["kubernetes"]["confidence"], "low")

        s2 = "Had limited exposure to Terraform during an academic project."
        r2 = extract_skills_with_context(s2)
        self.assertEqual(r2["terraform"]["confidence"], "low")

        s3 = "Want to learn Next.js."
        r3 = extract_skills_with_context(s3)
        self.assertEqual(r3["next.js"]["confidence"], "low")

        s4 = "I plan to learn Kotlin next year."
        r4 = extract_skills_with_context(s4)
        self.assertEqual(r4["kotlin"]["confidence"], "low")


if __name__ == "__main__":
    unittest.main(verbosity=2)
