"""Unit tests for the skill recency module (app/recency.py).

Verifies Feature 6 specifications:
1. 'Jan 2025 - Present' role yields weight 1.0.
2. Skill last used in 2025 gets weight above 0.75 (with today=2026).
3. Skill last used in 2018 gets weight below 0.20 (with today=2026).
4. Multiple roles mentioning the same skill resolve to the latest end year.
5. Undated skills get neutral weight 0.5.
6. Education sections and degree dates are strictly ignored.
7. Two-line date layout (role on line 1, date on line 2) is correctly parsed.
8. Empty resume text handles gracefully.
9. Unicode text, non-ASCII bullets, and dashes are handled seamlessly.
10. All recency weights remain bounded in [0.0, 1.0].
11. RECENCY_CONFIG presets (half_life, floor, undated) are configurable.
12. skill_recency_score obeys 0.80 required / 0.20 preferred weighting and edge cases.
"""

from __future__ import annotations

import datetime
import pytest

from app.recency import (
    RECENCY_CONFIG,
    RECENCY_UNDATED,
    recency_weight,
    skill_last_used,
    skill_recency_score,
)

FIXED_TODAY = datetime.date(2026, 1, 15)


# ===========================================================================
# 1. Individual Weighting Tests (Jan 2025 - Present, 2025, 2018, Undated)
# ===========================================================================

class TestRecencyWeightCalculations:
    """Verify recency_weight formula and boundary values."""

    def test_jan_2025_present_gets_weight_1_0(self):
        """Jan 2025 - Present role yields age 0 and weight 1.0."""
        text = """
        EXPERIENCE
        Senior Software Engineer | TechCorp
        Jan 2025 - Present
        • Architected microservices with Python and FastAPI.
        """
        last_used = skill_last_used(text, skills=["python"], today=FIXED_TODAY)
        assert last_used["python"]["last_used_year"] == 2026
        assert last_used["python"]["source"] == "present_role"

        weight = recency_weight(last_used["python"]["last_used_year"], today=FIXED_TODAY)
        assert weight == pytest.approx(1.0, abs=1e-4)

    def test_skill_last_used_in_2025_above_0_75(self):
        """Skill last used in 2025 gets weight above 0.75 with today fixed to 2026."""
        # age = 2026 - 2025 = 1 year; 0.5 ** (1/3) = 0.7937 > 0.75
        text = """
        EXPERIENCE
        Backend Developer | Acme Corp | 2023 - 2025
        • Built backend services in Go and PostgreSQL.
        """
        last_used = skill_last_used(text, skills=["go"], today=FIXED_TODAY)
        assert last_used["go"]["last_used_year"] == 2025
        assert last_used["go"]["source"] == "dated_role"

        weight = recency_weight(last_used["go"]["last_used_year"], today=FIXED_TODAY)
        assert weight > 0.75
        assert weight == pytest.approx(0.7937, abs=1e-3)

    def test_skill_last_used_in_2018_below_0_20(self):
        """Skill last used in 2018 gets weight below 0.20 with today fixed to 2026."""
        # age = 2026 - 2018 = 8 years; 0.5 ** (8/3) = 0.1575 < 0.20
        text = """
        EXPERIENCE
        Junior Developer | OldCo | 2016 - 2018
        • Wrote batch processing scripts in Ruby.
        """
        last_used = skill_last_used(text, skills=["ruby"], today=FIXED_TODAY)
        assert last_used["ruby"]["last_used_year"] == 2018
        assert last_used["ruby"]["source"] == "dated_role"

        weight = recency_weight(last_used["ruby"]["last_used_year"], today=FIXED_TODAY)
        assert weight < 0.20
        assert weight == pytest.approx(0.1575, abs=1e-3)

    def test_undated_skill_gets_0_5(self):
        """Undated skill (e.g. only in Skills section or undated entry) gets 0.5."""
        text = """
        SKILLS
        Languages: Rust, Scala, Kotlin
        """
        last_used = skill_last_used(text, skills=["rust"], today=FIXED_TODAY)
        assert last_used["rust"]["last_used_year"] is None
        assert last_used["rust"]["source"] == "undated"

        weight = recency_weight(last_used["rust"]["last_used_year"], today=FIXED_TODAY)
        assert weight == pytest.approx(0.5, abs=1e-4)


# ===========================================================================
# 2. Multi-Role Linking Rule (Latest End Year Precedence)
# ===========================================================================

class TestMultiRoleLinkingRule:
    """Verify that when a skill appears in multiple roles, the latest end year wins."""

    def test_two_roles_use_latest_end_year(self):
        text = """
        EXPERIENCE
        Lead Backend Engineer | CloudScale
        2021 - 2024
        • Architected Python microservices and Kafka pipelines.

        Software Developer | StartupCo
        2016 - 2018
        • Built ETL pipelines with Python and MySQL.
        """
        last_used = skill_last_used(text, skills=["python"], today=FIXED_TODAY)
        assert last_used["python"]["last_used_year"] == 2024
        assert last_used["python"]["source"] == "dated_role"
        assert "CloudScale" in last_used["python"]["evidence"] or "Python microservices" in last_used["python"]["evidence"]

        # Recency weight reflects 2024 (age 2), NOT 2018 (age 8)
        w = recency_weight(last_used["python"]["last_used_year"], today=FIXED_TODAY)
        assert w == pytest.approx(0.6300, abs=1e-3)

    def test_past_role_and_present_role_uses_present_year(self):
        text = """
        EXPERIENCE
        Senior Engineer | CurrentCo
        Jan 2025 - Present
        • Full stack development with React and TypeScript.

        Junior Developer | PastCo
        2019 - 2022
        • Maintained React UI dashboard.
        """
        last_used = skill_last_used(text, skills=["react"], today=FIXED_TODAY)
        assert last_used["react"]["last_used_year"] == 2026
        assert last_used["react"]["source"] == "present_role"
        w = recency_weight(last_used["react"]["last_used_year"], today=FIXED_TODAY)
        assert w == 1.0


# ===========================================================================
# 3. Layouts & Section Filtering (Education, 2-line layout, Undated)
# ===========================================================================

class TestLayoutsAndSectionFiltering:
    """Verify Education exclusion, two-line date layouts, and undated entries."""

    def test_education_years_are_ignored(self):
        """Dates in education section must never set a skill's last-used year."""
        text = """
        EDUCATION
        Stanford University | 2014 - 2018
        B.S. in Computer Science
        Relevant Coursework: C++, Algorithms, Compilers

        EXPERIENCE
        Data Analyst | Analytics Corp | 2022 - 2025
        • Built dashboards with SQL and Tableau.
        """
        last_used = skill_last_used(text, skills=["c++", "sql"], today=FIXED_TODAY)

        # SQL is from dated experience (2025)
        assert last_used["sql"]["last_used_year"] == 2025
        assert last_used["sql"]["source"] == "dated_role"

        # C++ is only in Education; 2018 must NOT be used
        assert last_used["c++"]["last_used_year"] is None
        assert last_used["c++"]["source"] == "undated"
        assert recency_weight(last_used["c++"]["last_used_year"], today=FIXED_TODAY) == 0.5

    def test_two_line_date_layout_works(self):
        """Date range on its own line beneath title/company is linked correctly."""
        text = """
        EXPERIENCE
        Software Engineer | Acme Corp
        Jan 2020 - Dec 2022
        • Developed REST APIs with Django and PostgreSQL.
        • Dockerized microservices.

        Staff Engineer
        Tech Solutions LLC
        2023 - Present
        • Led migration to Kubernetes and Go.
        """
        last_used = skill_last_used(text, skills=["django", "kubernetes"], today=FIXED_TODAY)

        # 1st entry (2-line layout): 2020-2022
        assert last_used["django"]["last_used_year"] == 2022
        assert last_used["django"]["source"] == "dated_role"
        assert "Django" in last_used["django"]["evidence"]

        # 2nd entry (3-line layout): 2023-Present
        assert last_used["kubernetes"]["last_used_year"] == 2026
        assert last_used["kubernetes"]["source"] == "present_role"
        assert "Kubernetes" in last_used["kubernetes"]["evidence"]

    def test_undated_entry_block(self):
        """Skills in an entry without dates receive undated status."""
        text = """
        EXPERIENCE
        Personal Projects
        • Built mobile application using Flutter and Firebase.
        """
        last_used = skill_last_used(text, skills=["flutter"], today=FIXED_TODAY)
        assert last_used["flutter"]["last_used_year"] is None
        assert last_used["flutter"]["source"] == "undated"
        assert "Flutter" in last_used["flutter"]["evidence"]


# ===========================================================================
# 4. Robustness: Empty, Unicode, and Invariant Checks
# ===========================================================================

class TestRobustnessAndInvariants:
    """Verify empty text, unicode, and weight boundaries."""

    def test_empty_text(self):
        res = skill_last_used("", skills=["python", "docker"], today=FIXED_TODAY)
        assert res["python"]["last_used_year"] is None
        assert res["python"]["source"] == "undated"
        assert res["python"]["evidence"] is None
        assert res["docker"]["last_used_year"] is None

        # Automatic extraction on empty text
        auto_res = skill_last_used("", skills=None, today=FIXED_TODAY)
        assert auto_res == {}

    def test_unicode_text(self):
        """Accented text, em-dash, and bullet glyphs are parsed seamlessly."""
        text = """
        EXPÉRIENCE PROFESSIONNELLE
        Ingénieur Logiciel — Société Générale — 2021 – 2024
        • Développeur Backend en Python et Docker.
        """
        last_used = skill_last_used(text, skills=["python", "docker"], today=FIXED_TODAY)
        assert last_used["python"]["last_used_year"] == 2024
        assert last_used["python"]["source"] == "dated_role"
        assert last_used["docker"]["last_used_year"] == 2024

    def test_all_weights_stay_within_0_1(self):
        """Check weight boundaries under extreme ages and floor conditions."""
        # Age 0 (today)
        assert 0.0 <= recency_weight(2026, today=FIXED_TODAY) <= 1.0
        # Future year clamp
        assert 0.0 <= recency_weight(2030, today=FIXED_TODAY) <= 1.0
        # Very old year (floor 0.05 applied)
        w_old = recency_weight(1980, today=FIXED_TODAY)
        assert w_old == pytest.approx(0.05, abs=1e-4)
        assert 0.0 <= w_old <= 1.0
        # None (undated)
        assert 0.0 <= recency_weight(None, today=FIXED_TODAY) <= 1.0

    def test_recency_config_customizable(self):
        """Verify that half_life and floor overrides are respected."""
        # Fast decay: half_life=1.0 year, age=2 -> 0.5 ** 2 = 0.25
        w_fast = recency_weight(2024, today=FIXED_TODAY, half_life=1.0)
        assert w_fast == pytest.approx(0.25, abs=1e-3)

        # Higher floor: floor=0.20
        w_floored = recency_weight(1990, today=FIXED_TODAY, floor=0.20)
        assert w_floored == pytest.approx(0.20, abs=1e-4)


# ===========================================================================
# 5. Composite Scoring: skill_recency_score
# ===========================================================================

class TestSkillRecencyScore:
    """Verify skill_recency_score 0.80 / 0.20 weighting and edge cases."""

    def test_recency_score_both_required_and_preferred(self):
        # python in present role (weight 1.0)
        # docker in 2022 role (weight 0.3969)
        last_used = {
            "python": {"last_used_year": 2026, "source": "present_role", "evidence": "..."},
            "docker": {"last_used_year": 2022, "source": "dated_role", "evidence": "..."},
        }
        res = skill_recency_score(
            matched_required=["python"],
            matched_preferred=["docker"],
            last_used=last_used,
            today=FIXED_TODAY,
        )
        assert "score" in res
        assert "per_skill" in res
        # 0.80 * 1.0 + 0.20 * 0.3969 = 0.80 + 0.0794 = 0.8794
        assert res["score"] == pytest.approx(0.8794, abs=1e-3)
        assert res["per_skill"]["python"]["weight"] == 1.0
        assert res["per_skill"]["docker"]["weight"] == pytest.approx(0.3969, abs=1e-3)

    def test_recency_score_required_only(self):
        last_used = {
            "python": {"last_used_year": 2026, "source": "present_role", "evidence": "..."},
        }
        res = skill_recency_score(
            matched_required=["python"],
            matched_preferred=[],
            last_used=last_used,
            today=FIXED_TODAY,
        )
        assert res["score"] == 1.0

    def test_recency_score_preferred_only(self):
        last_used = {
            "docker": {"last_used_year": 2025, "source": "dated_role", "evidence": "..."},
        }
        res = skill_recency_score(
            matched_required=[],
            matched_preferred=["docker"],
            last_used=last_used,
            today=FIXED_TODAY,
        )
        # 2025 has weight 0.7937
        assert res["score"] == pytest.approx(0.7937, abs=1e-3)

    def test_recency_score_nothing_matched(self):
        res = skill_recency_score(
            matched_required=[],
            matched_preferred=[],
            last_used={},
            today=FIXED_TODAY,
        )
        assert res["score"] == 0.0
        assert res["per_skill"] == {}


# ===========================================================================
# 6. Before / After Note Example Requirement
# ===========================================================================

class TestBeforeAfterRecencyComparison:
    """Demonstrate contrasting weights for active role vs dated legacy role."""

    def test_present_vs_legacy_role_weights(self):
        """The same skill in a 2025-Present role vs 2016-2018 role produces distinct weights."""
        candidate_active = """
        EXPERIENCE
        Senior Engineer | InnovateTech
        2025 - Present
        • Built cloud pipelines using Python.
        """

        candidate_legacy = """
        EXPERIENCE
        Software Developer | LegacySystems
        2016 - 2018
        • Built batch scripts using Python.
        """

        res_active = skill_last_used(candidate_active, skills=["python"], today=FIXED_TODAY)
        res_legacy = skill_last_used(candidate_legacy, skills=["python"], today=FIXED_TODAY)

        weight_active = recency_weight(res_active["python"]["last_used_year"], today=FIXED_TODAY)
        weight_legacy = recency_weight(res_legacy["python"]["last_used_year"], today=FIXED_TODAY)

        # Active role: weight 1.0
        assert weight_active == 1.0
        # Legacy role: weight 0.1575
        assert weight_legacy == pytest.approx(0.1575, abs=1e-3)

        # Clear contrast: active weight is > 6x legacy weight
        assert weight_active > 6.0 * weight_legacy
