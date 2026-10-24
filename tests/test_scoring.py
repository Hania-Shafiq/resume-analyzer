"""Unit tests for the scoring module (app/scoring.py).

Verifies:
1. Preset weights sum to 1.0.
2. skills_only preset matches today's existing calculation on stored examples.
3. Determinism across repeated runs.
4. Ordering: strong > medium > weak in both hybrid and proposal modes.
5. Graceful fallback to skills_only when the semantic model fails.
6. All scores and components stay bounded in [0.0, 1.0].
7. Environment variable configuration (SCORING_MODE).
"""

from __future__ import annotations

import os
import pytest

from app.matcher import normalize_similarity
from app.scoring import (
    DEFAULT_SEMANTIC_HIGH,
    DEFAULT_SEMANTIC_LOW,
    SCORING_CONFIG,
    compute_scores,
    extract_candidate_degree_level,
    extract_candidate_years,
    extract_jd_required_degree_level,
    extract_jd_required_years,
)
from app.skills import EducationEntry, JDAnalysis, analyze_job_description


# ===========================================================================
# 1. Preset Weights Sum to 1.0
# ===========================================================================

class TestScoringConfigWeights:
    """Verify SCORING_CONFIG presets and weight invariants."""

    def test_presets_exist(self):
        assert "skills_only" in SCORING_CONFIG
        assert "hybrid" in SCORING_CONFIG
        assert "proposal" in SCORING_CONFIG

    @pytest.mark.parametrize("preset", ["skills_only", "hybrid", "proposal", "composite"])
    def test_weights_sum_to_one(self, preset: str):
        weights = SCORING_CONFIG[preset]
        assert sum(weights.values()) == pytest.approx(1.0, rel=1e-5)

    def test_tuned_default_anchors(self):
        assert DEFAULT_SEMANTIC_LOW == 0.20
        assert DEFAULT_SEMANTIC_HIGH == 0.65

    def test_preset_exact_ratios(self):
        assert SCORING_CONFIG["skills_only"]["skill"] == 1.0
        assert SCORING_CONFIG["skills_only"]["semantic"] == 0.0

        assert SCORING_CONFIG["hybrid"]["skill"] == 0.70
        assert SCORING_CONFIG["hybrid"]["semantic"] == 0.30

        assert SCORING_CONFIG["proposal"]["skill"] == 0.40
        assert SCORING_CONFIG["proposal"]["semantic"] == 0.40
        assert SCORING_CONFIG["proposal"]["experience"] == 0.10
        assert SCORING_CONFIG["proposal"]["education"] == 0.10


# ===========================================================================
# 2. skills_only Equals Today's Results on Stored Examples
# ===========================================================================

class TestSkillsOnlyEquivalence:
    """Verify skills_only yields identical results to today's scoring logic."""

    @pytest.mark.parametrize(
        "res_skills,req,pref,expected_score",
        [
            # Example 1: Full match on both required & preferred
            (
                ["python", "fastapi", "docker", "aws"],
                ["python", "fastapi"],
                ["docker", "aws"],
                1.0,
            ),
            # Example 2: Partial match (5 of 6 required, 2 of 4 preferred)
            # 0.80 * (5/6) + 0.20 * (2/4) = 0.666667 + 0.1000 = 0.7667
            (
                ["python", "fastapi", "docker", "git", "postgresql", "redis", "aws"],
                ["python", "fastapi", "postgresql", "docker", "git", "kubernetes"],
                ["redis", "aws", "kafka", "terraform"],
                0.7667,
            ),
            # Example 3: Required only JD (2 of 3 required matched, no preferred)
            # 2/3 = 0.6667
            (
                ["python", "sql"],
                ["python", "sql", "linux"],
                [],
                0.6667,
            ),
            # Example 4: Preferred only JD (1 of 2 preferred matched, no required)
            # 1/2 = 0.5000
            (
                ["docker"],
                [],
                ["docker", "kubernetes"],
                0.5,
            ),
            # Example 5: Zero matched skills
            (
                ["html", "css"],
                ["python", "django"],
                ["docker"],
                0.0,
            ),
            # Example 6: Empty JD requirements
            (
                ["python"],
                [],
                [],
                0.0,
            ),
        ],
    )
    def test_stored_examples_match_today_logic(
        self,
        res_skills: list[str],
        req: list[str],
        pref: list[str],
        expected_score: float,
    ):
        jd = {
            "required_skills": req,
            "preferred_skills": pref,
            "min_experience_years": None,
            "education_requirement": [],
        }
        res = compute_scores(
            resume_text="Sample text",
            resume_skills=res_skills,
            jd_analysis=jd,
            resume_profile=None,
            mode="skills_only",
        )

        assert res["scoring_mode"] == "skills_only"
        assert res["semantic_similarity"] is None
        assert res["skill_overlap_score"] == pytest.approx(expected_score, abs=1e-4)
        assert res["final_score"] == pytest.approx(expected_score, abs=1e-4)
        assert res["breakdown"]["skill"]["value"] == pytest.approx(expected_score, abs=1e-4)
        assert res["breakdown"]["skill"]["weight"] == 1.0
        assert res["breakdown"]["semantic"]["value"] is None

    def test_skills_only_with_real_jdanalysis_object(self):
        jd_text = (
            "Job Title: Senior Backend Developer\n"
            "Requirements:\n"
            "- Python\n"
            "- FastAPI\n"
            "- PostgreSQL\n"
            "Nice to have:\n"
            "- Docker\n"
            "- Redis\n"
            "Minimum 3 years of experience. Bachelor's degree required."
        )
        jd_analysis = analyze_job_description(jd_text)
        candidate_skills = ["python", "fastapi", "docker"]

        res = compute_scores(
            resume_text="I build backends with Python, FastAPI, and Docker.",
            resume_skills=candidate_skills,
            jd_analysis=jd_analysis,
            resume_profile={"experience_years": 4, "education": "Bachelor's"},
            mode="skills_only",
        )

        # 2 of 3 required matched (python, fastapi) -> 2/3
        # 1 of 2 preferred matched (docker) -> 1/2
        expected_skill = round(0.80 * (2 / 3) + 0.20 * (1 / 2), 4)  # 0.5333 + 0.10 = 0.6333
        assert res["skill_overlap_score"] == pytest.approx(expected_skill, abs=1e-4)
        assert res["final_score"] == pytest.approx(expected_skill, abs=1e-4)
        assert res["scoring_mode"] == "skills_only"
        assert res["semantic_similarity"] is None


# ===========================================================================
# 3. Determinism
# ===========================================================================

class TestScoringDeterminism:
    """Verify that identical inputs produce identical scores across modes."""

    @pytest.mark.parametrize("mode", ["skills_only", "hybrid", "proposal"])
    def test_same_input_twice_gives_identical_score(self, mode: str):
        resume_text = (
            "Senior Backend Engineer with 5 years experience in Python, FastAPI, "
            "Docker, PostgreSQL, and AWS cloud microservices."
        )
        skills = ["python", "fastapi", "docker", "postgresql", "aws"]
        jd_text = (
            "Backend Engineer needed with strong Python, FastAPI, and PostgreSQL skills. "
            "Experience with Docker and AWS preferred. Minimum 3 years of experience. "
            "Bachelor's degree in Computer Science."
        )
        jd_analysis = analyze_job_description(jd_text)
        profile = {
            "experience_years": 5,
            "education": [{"degree": "Bachelor's", "field": "Computer Science"}],
        }

        res_1 = compute_scores(resume_text, skills, jd_analysis, profile, mode=mode)
        res_2 = compute_scores(resume_text, skills, jd_analysis, profile, mode=mode)

        assert res_1["final_score"] == res_2["final_score"]
        assert res_1["skill_overlap_score"] == res_2["skill_overlap_score"]
        assert res_1["semantic_similarity"] == res_2["semantic_similarity"]
        assert res_1["experience_fit"] == res_2["experience_fit"]
        assert res_1["education_fit"] == res_2["education_fit"]
        assert res_1["breakdown"] == res_2["breakdown"]


# ===========================================================================
# 4. Ordering: Strong > Medium > Weak in Hybrid and Proposal Modes
# ===========================================================================

class TestScoringOrdering:
    """Verify candidate ranking hierarchy: strong > medium > weak."""

    JD_TEXT = (
        "Job Title: Lead Python Backend Architect\n"
        "Requirements:\n"
        "- Python\n"
        "- FastAPI\n"
        "- PostgreSQL\n"
        "- Microservices\n"
        "Preferred:\n"
        "- Docker\n"
        "- Kubernetes\n"
        "- AWS\n"
        "Requires 5+ years of software engineering experience. Master's degree preferred."
    )

    STRONG_RESUME = (
        "Lead Backend Engineer with 7 years of deep experience architecting high-throughput "
        "Python microservices, FastAPI REST APIs, and PostgreSQL database optimization. "
        "Extensive container orchestration using Docker and Kubernetes deployed on AWS. "
        "Master of Science in Computer Science."
    )
    STRONG_SKILLS = ["python", "fastapi", "postgresql", "docker", "kubernetes", "aws"]
    STRONG_PROFILE = {"experience_years": 7, "education": "Master's"}

    MEDIUM_RESUME = (
        "Junior Developer with 2 years of experience building basic Python scripts and small "
        "Flask endpoints. Familiar with relational databases and fundamental Docker usage. "
        "Bachelor of Science in Information Technology."
    )
    MEDIUM_SKILLS = ["python", "docker"]
    MEDIUM_PROFILE = {"experience_years": 2, "education": "Bachelor's"}

    WEAK_RESUME = (
        "Creative Graphic Designer with 1 year of experience in brand identity, typography, "
        "Adobe Photoshop, and marketing flyers. High School Diploma."
    )
    WEAK_SKILLS = ["photoshop", "figma"]
    WEAK_PROFILE = {"experience_years": 1, "education": "High School"}

    @pytest.mark.parametrize("mode", ["hybrid", "proposal"])
    def test_strong_medium_weak_ordering(self, mode: str):
        jd = analyze_job_description(self.JD_TEXT)

        score_strong = compute_scores(
            self.STRONG_RESUME, self.STRONG_SKILLS, jd, self.STRONG_PROFILE, mode=mode
        )
        score_medium = compute_scores(
            self.MEDIUM_RESUME, self.MEDIUM_SKILLS, jd, self.MEDIUM_PROFILE, mode=mode
        )
        score_weak = compute_scores(
            self.WEAK_RESUME, self.WEAK_SKILLS, jd, self.WEAK_PROFILE, mode=mode
        )

        assert score_strong["final_score"] > score_medium["final_score"], (
            f"Expected strong ({score_strong['final_score']}) > medium ({score_medium['final_score']}) in {mode}"
        )
        assert score_medium["final_score"] > score_weak["final_score"], (
            f"Expected medium ({score_medium['final_score']}) > weak ({score_weak['final_score']}) in {mode}"
        )

        # Semantic component must also respect strong > medium > weak
        assert score_strong["semantic_similarity"] is not None
        assert score_medium["semantic_similarity"] is not None
        assert score_weak["semantic_similarity"] is not None
        assert score_strong["semantic_similarity"] > score_medium["semantic_similarity"]
        assert score_medium["semantic_similarity"] > score_weak["semantic_similarity"]


# ===========================================================================
# 5. Graceful Fallback When Model Loader Raises
# ===========================================================================

class TestSemanticModelFallback:
    """Verify fallback to skills_only when semantic encoder/loader raises."""

    def test_fallback_when_get_model_fails(self, monkeypatch):
        def _mock_failing_model(*args, **kwargs):
            raise OSError("Model checkpoint not found or out of memory")

        # Monkeypatch the model loader in app.matcher
        monkeypatch.setattr("app.matcher.get_model", _mock_failing_model)

        skills = ["python", "fastapi"]
        jd = {"required_skills": ["python", "fastapi"], "preferred_skills": []}

        # Request hybrid mode
        res = compute_scores(
            resume_text="Python developer",
            resume_skills=skills,
            jd_analysis=jd,
            resume_profile=None,
            mode="hybrid",
        )

        # Must gracefully degrade to skills_only without throwing
        assert res["scoring_mode"] == "skills_only"
        assert res["semantic_similarity"] is None
        assert res["weights_used"] == SCORING_CONFIG["skills_only"]
        assert res["final_score"] == res["skill_overlap_score"]
        assert res["final_score"] == 1.0

    def test_fallback_when_semantic_similarity_fails(self, monkeypatch):
        def _mock_failing_similarity(*args, **kwargs):
            raise RuntimeError("CUDA device error")

        monkeypatch.setattr("app.matcher.semantic_similarity", _mock_failing_similarity)

        res = compute_scores(
            resume_text="Python developer",
            resume_skills=["python"],
            jd_analysis={"required_skills": ["python"], "preferred_skills": []},
            mode="proposal",
        )

        assert res["scoring_mode"] == "skills_only"
        assert res["semantic_similarity"] is None
        assert res["final_score"] == 1.0


# ===========================================================================
# 6. Score Bounds [0.0, 1.0] Under Extreme / Edge Inputs
# ===========================================================================

class TestScoreBounds:
    """Verify every score and component is strictly bounded in [0.0, 1.0]."""

    @pytest.mark.parametrize("mode", ["skills_only", "hybrid", "proposal"])
    def test_scores_within_zero_and_one(self, mode: str):
        # Extreme profile values
        edge_profiles = [
            # 1. Extreme experience (100 years vs 2 years requirement)
            {"experience_years": 100, "education": "PhD"},
            # 2. Zero experience vs 10 years requirement
            {"experience_years": 0, "education": None},
            # 3. Negative experience (defensive)
            {"experience_years": -5, "education": "High School"},
            # 4. Empty profile
            {},
            # 5. None profile
            None,
        ]

        edge_jds = [
            # Fully specified JD
            {
                "required_skills": ["python", "fastapi"],
                "preferred_skills": ["docker"],
                "min_experience_years": 3,
                "education_requirement": [{"degree": "Bachelor's"}],
            },
            # JD with no requirements at all
            {
                "required_skills": [],
                "preferred_skills": [],
                "min_experience_years": None,
                "education_requirement": [],
            },
            # JD with very high requirements
            {
                "required_skills": ["c++", "rust", "cuda"],
                "preferred_skills": ["distributed systems"],
                "min_experience_years": 15,
                "education_requirement": [{"degree": "PhD"}],
            },
        ]

        for prof in edge_profiles:
            for jd in edge_jds:
                res = compute_scores(
                    resume_text="Python, Docker developer",
                    resume_skills=["python", "docker"],
                    jd_analysis=jd,
                    resume_profile=prof,
                    mode=mode,
                )

                assert 0.0 <= res["final_score"] <= 1.0
                assert 0.0 <= res["skill_overlap_score"] <= 1.0
                assert 0.0 <= res["experience_fit"] <= 1.0
                assert 0.0 <= res["education_fit"] <= 1.0
                if res["semantic_similarity"] is not None:
                    assert 0.0 <= res["semantic_similarity"] <= 1.0

    def test_normalize_similarity_helper_bounds(self):
        # Test values below, within, and above [LOW, HIGH]
        assert normalize_similarity(0.00, low=0.15, high=0.65) == 0.0
        assert normalize_similarity(0.10, low=0.15, high=0.65) == 0.0
        assert normalize_similarity(0.15, low=0.15, high=0.65) == 0.0
        assert normalize_similarity(0.40, low=0.15, high=0.65) == pytest.approx(0.50, abs=1e-3)
        assert normalize_similarity(0.65, low=0.15, high=0.65) == 1.0
        assert normalize_similarity(0.95, low=0.15, high=0.65) == 1.0
        assert normalize_similarity(-0.50, low=0.15, high=0.65) == 0.0


# ===========================================================================
# 7. Environment Variable and Mode Configuration
# ===========================================================================

class TestScoringModeEnvVar:
    """Verify SCORING_MODE environment variable configuration."""

    def test_default_env_var_is_skills_only(self, monkeypatch):
        monkeypatch.delenv("SCORING_MODE", raising=False)
        res = compute_scores(
            resume_text="test",
            resume_skills=["python"],
            jd_analysis={"required_skills": ["python"], "preferred_skills": []},
            mode=None,
        )
        assert res["scoring_mode"] == "skills_only"

    def test_env_var_sets_hybrid_mode(self, monkeypatch):
        monkeypatch.setenv("SCORING_MODE", "hybrid")
        res = compute_scores(
            resume_text="Senior Python Backend Engineer building FastAPI apps",
            resume_skills=["python"],
            jd_analysis={
                "required_skills": ["python"],
                "preferred_skills": [],
                "description": "Python backend developer needed",
            },
            mode=None,
        )
        assert res["scoring_mode"] == "hybrid"
        assert res["semantic_similarity"] is not None

    def test_unknown_mode_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown scoring mode"):
            compute_scores(
                resume_text="test",
                resume_skills=[],
                jd_analysis={},
                mode="non_existent_mode",
            )


# ===========================================================================
# 8. Composite Scoring Mode (Feature 6)
# ===========================================================================

class TestCompositeScoringMode:
    """Verify composite scoring preset (semantic, skill, experience, recency)."""

    def test_composite_weights_sum_to_one(self):
        """Composite weights must sum to 1.0 and match defined ratios."""
        assert "composite" in SCORING_CONFIG
        weights = SCORING_CONFIG["composite"]
        assert sum(weights.values()) == pytest.approx(1.0, rel=1e-5)
        assert weights["skill"] == 0.40
        assert weights["semantic"] == 0.30
        assert weights["experience"] == 0.15
        assert weights["recency"] == 0.15
        assert "education" not in weights or weights.get("education") == 0.0

    def test_recent_role_scores_higher_than_legacy_role_in_composite_mode(self):
        """Two resumes with identical skills where one used them in 2025-Present and the other in 2018."""
        jd_text = (
            "Senior Backend Engineer\n"
            "Requirements:\n"
            "- Python\n"
            "- FastAPI\n"
            "- Docker\n"
            "Minimum 3 years of experience."
        )
        jd_analysis = analyze_job_description(jd_text)

        resume_recent = (
            "Alex Smith\n"
            "EXPERIENCE\n"
            "Lead Backend Engineer | TechCorp\n"
            "Jan 2025 - Present\n"
            "• Architected scalable microservices using Python, FastAPI, and Docker.\n"
            "• Optimized database performance.\n"
            "Software Developer | PriorCo\n"
            "2022 - 2024\n"
            "• Built web services.\n"
        )

        resume_legacy = (
            "Alex Smith\n"
            "EXPERIENCE\n"
            "Software Developer | LegacySystems\n"
            "2016 - 2018\n"
            "• Built backend tools with Python, FastAPI, and Docker.\n"
            "Engineering Manager | MgmtCorp\n"
            "2019 - Present\n"
            "• Managed engineering operations and hiring.\n"
        )

        skills = ["python", "fastapi", "docker"]

        # In skills_only mode: both have identical skills, so final scores are equal
        res_recent_skills_only = compute_scores(
            resume_text=resume_recent,
            resume_skills=skills,
            jd_analysis=jd_analysis,
            mode="skills_only",
        )
        res_legacy_skills_only = compute_scores(
            resume_text=resume_legacy,
            resume_skills=skills,
            jd_analysis=jd_analysis,
            mode="skills_only",
        )
        assert res_recent_skills_only["final_score"] == pytest.approx(
            res_legacy_skills_only["final_score"], abs=1e-4
        )

        # In composite mode: recent role has recency weight 1.0, legacy has decaying weight
        res_recent_comp = compute_scores(
            resume_text=resume_recent,
            resume_skills=skills,
            jd_analysis=jd_analysis,
            mode="composite",
        )
        res_legacy_comp = compute_scores(
            resume_text=resume_legacy,
            resume_skills=skills,
            jd_analysis=jd_analysis,
            mode="composite",
        )

        assert res_recent_comp["scoring_mode"] == "composite"
        assert res_legacy_comp["scoring_mode"] == "composite"
        assert res_recent_comp["final_score"] > res_legacy_comp["final_score"]
        assert res_recent_comp["skill_recency"] is not None
        assert res_recent_comp["skill_recency"]["python"]["source"] == "present_role"
        assert res_legacy_comp["skill_recency"]["python"]["source"] == "dated_role"

    def test_keyword_stuffed_scores_lower_than_accomplishment_sentences(self, monkeypatch):
        """A keyword-stuffed resume scores lower than accomplishment sentences via evidence multipliers."""
        import sys
        import types

        # Create mock app.evidence module with skill_multipliers
        mock_evidence = types.ModuleType("app.evidence")

        def mock_skill_multipliers(text, skills):
            if "Accomplishment" in text:
                return {s: "high_depth" for s in skills}
            return {s: "listed" for s in skills}

        mock_evidence.skill_multipliers = mock_skill_multipliers
        monkeypatch.setitem(sys.modules, "app.evidence", mock_evidence)

        jd_text = "Job Description\nRequirements:\n- Python\n- SQL\n"
        jd_analysis = analyze_job_description(jd_text)

        resume_accomplishment = (
            "Accomplishment Resume\n"
            "EXPERIENCE\n"
            "Senior Engineer | Acme\n"
            "2023 - Present\n"
            "• Spearheaded Python backend service reducing latency by 40% and optimized SQL queries.\n"
        )

        resume_keyword_stuffed = (
            "Keyword Stuffed Resume\n"
            "SKILLS\n"
            "Python, SQL, Java, C++, Ruby, PHP\n"
            "EXPERIENCE\n"
            "Engineer | Beta\n"
            "2023 - Present\n"
            "• General development work.\n"
        )

        skills = ["python", "sql"]

        res_acc = compute_scores(
            resume_text=resume_accomplishment,
            resume_skills=skills,
            jd_analysis=jd_analysis,
            mode="composite",
        )
        res_stuff = compute_scores(
            resume_text=resume_keyword_stuffed,
            resume_skills=skills,
            jd_analysis=jd_analysis,
            mode="composite",
        )

        # Both have same plain coverage
        assert res_acc["breakdown"]["skill"]["plain_coverage"] == res_stuff["breakdown"]["skill"]["plain_coverage"]
        # Accomplishment resume has higher adjusted coverage and higher final score
        assert res_acc["breakdown"]["skill"]["adjusted_coverage"] > res_stuff["breakdown"]["skill"]["adjusted_coverage"]
        assert res_acc["final_score"] > res_stuff["final_score"]

    def test_breakdown_contributions_add_up_to_final_score(self):
        """Breakdown contributions must sum to final_score with tolerance 1e-9."""
        jd_text = (
            "Software Engineer\n"
            "Requirements:\n"
            "- Python\n"
            "- Docker\n"
            "Preferred:\n"
            "- Redis\n"
            "Experience: 2 years."
        )
        jd_analysis = analyze_job_description(jd_text)
        resume_text = (
            "Jane Doe\n"
            "EXPERIENCE\n"
            "Software Engineer | Acme\n"
            "2023 - Present\n"
            "• Built microservices using Python, Docker, and Redis.\n"
        )

        res = compute_scores(
            resume_text=resume_text,
            resume_skills=["python", "docker", "redis"],
            jd_analysis=jd_analysis,
            resume_profile={"experience_years": 3},
            mode="composite",
        )

        breakdown = res["breakdown"]
        total_contribution = sum(
            item["weighted_contribution"]
            for item in breakdown.values()
            if isinstance(item, dict) and "weighted_contribution" in item
        )

        assert total_contribution == pytest.approx(res["final_score"], abs=1e-9)

    def test_composite_mode_determinism(self):
        """Running composite scoring twice with identical inputs yields identical scores."""
        jd_text = "Requirements:\n- Python\n- FastAPI\n"
        jd_analysis = analyze_job_description(jd_text)
        resume_text = (
            "EXPERIENCE\n"
            "Developer | Corp\n"
            "2024 - Present\n"
            "• Developed with Python and FastAPI.\n"
        )
        skills = ["python", "fastapi"]

        run1 = compute_scores(
            resume_text=resume_text,
            resume_skills=skills,
            jd_analysis=jd_analysis,
            mode="composite",
        )
        run2 = compute_scores(
            resume_text=resume_text,
            resume_skills=skills,
            jd_analysis=jd_analysis,
            mode="composite",
        )

        assert run1["final_score"] == run2["final_score"]
        assert run1["breakdown"] == run2["breakdown"]
        assert run1["skill_recency"] == run2["skill_recency"]

    def test_composite_all_scores_within_zero_one(self):
        """Every score and breakdown value in composite mode stays bounded in [0.0, 1.0]."""
        jd_text = "Requirements:\n- Python\nPreferred:\n- Docker\n"
        jd_analysis = analyze_job_description(jd_text)
        resume_text = (
            "EXPERIENCE\n"
            "Developer | Startup\n"
            "2022 - 2024\n"
            "• Used Python.\n"
        )

        res = compute_scores(
            resume_text=resume_text,
            resume_skills=["python"],
            jd_analysis=jd_analysis,
            mode="composite",
        )

        assert 0.0 <= res["final_score"] <= 1.0
        for name, item in res["breakdown"].items():
            if isinstance(item, dict) and "value" in item and item["value"] is not None:
                assert 0.0 <= item["value"] <= 1.0
            if isinstance(item, dict) and "weighted_contribution" in item:
                assert 0.0 <= item["weighted_contribution"] <= 1.0

    def test_composite_fallback_when_model_loader_raises(self, monkeypatch):
        """Fallback to skills_only when the semantic model fails in composite mode."""
        from app import matcher

        def failing_get_model():
            raise RuntimeError("SentenceTransformer weights corrupted")

        monkeypatch.setattr(matcher, "get_model", failing_get_model)

        jd_text = "Requirements:\n- Python\n"
        jd_analysis = analyze_job_description(jd_text)
        resume_text = "EXPERIENCE\nDeveloper 2024 - Present\n• Python\n"

        res = compute_scores(
            resume_text=resume_text,
            resume_skills=["python"],
            jd_analysis=jd_analysis,
            mode="composite",
        )

        # Must gracefully degrade to skills_only
        assert res["scoring_mode"] == "skills_only"
        assert res["semantic_similarity"] is None
        assert res["final_score"] == pytest.approx(1.0, abs=1e-4)

    def test_composite_evidence_unavailable_note_added(self, monkeypatch):
        """When app.evidence is not available, notes indicate evidence_unavailable."""
        import sys
        monkeypatch.setitem(sys.modules, "app.evidence", None)
        jd_text = "Requirements:\n- Python\n"
        jd_analysis = analyze_job_description(jd_text)
        resume_text = "Python engineer"

        res = compute_scores(
            resume_text=resume_text,
            resume_skills=["python"],
            jd_analysis=jd_analysis,
            mode="composite",
        )

        assert "plain_coverage" in res["breakdown"]["skill"]
        assert "adjusted_coverage" in res["breakdown"]["skill"]
        assert "evidence_unavailable" in res["breakdown"]["skill"].get("notes", [])
        assert "evidence_unavailable" in res["breakdown"].get("notes", [])

    def test_composite_skill_recency_table_present_only_in_composite(self):
        """skill_recency table is populated in composite mode and None in all other modes."""
        jd = {"required_skills": ["python"], "preferred_skills": []}
        resume = "EXPERIENCE\nDev | 2024 - Present\n• Python\n"

        res_comp = compute_scores(resume_text=resume, resume_skills=["python"], jd_analysis=jd, mode="composite")
        assert res_comp["skill_recency"] is not None
        assert "python" in res_comp["skill_recency"]

        res_skills = compute_scores(resume_text=resume, resume_skills=["python"], jd_analysis=jd, mode="skills_only")
        assert res_skills["skill_recency"] is None

        res_hybrid = compute_scores(resume_text=resume, resume_skills=["python"], jd_analysis=jd, mode="hybrid")
        assert res_hybrid["skill_recency"] is None

        res_prop = compute_scores(resume_text=resume, resume_skills=["python"], jd_analysis=jd, mode="proposal")
        assert res_prop["skill_recency"] is None

