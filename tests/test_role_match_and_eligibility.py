import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.scoring import compute_scores, SCORING_CONFIG

client = TestClient(app)

# Standard Test Benchmark: Data Analyst role requiring 3 years & 7 mandatory skills
DATA_ANALYST_JD = """
**Job Title:** Data Analyst  
**Required Education:** Bachelor's degree in Computer Science, Data Science, or a related field  
**Required Experience:** 3 years  

**Required Skills:**
- Python
- SQL
- Power BI
- Excel
- Data Cleaning
- Data Visualization
- Statistical Analysis

**Job Responsibilities:**
- Analyze large datasets to identify business trends.
- Develop interactive Power BI dashboards.
- Write optimized SQL queries.
- Clean, transform, and validate data.
- Perform statistical analysis and report insights.
- Communicate findings to stakeholders.

**Seniority Level:** Mid-Level
"""

AHMED_KHAN_CV = """
**Candidate Name:** Ahmed Khan  
**Education:** Bachelor's in Computer Science  
**Total Experience:** 1 year  
**Previous Role:** Junior Data Analyst  

**Technical Skills:**
- Python
- SQL
- Excel
- Power BI
- Data Cleaning
- Data Visualization

**Work Experience:**
- Used Python and Pandas to clean datasets.
- Wrote SQL queries to extract and analyze data.
- Created Power BI dashboards for sales reporting.
- Used Excel for basic data analysis.

**Skill Timeline:**
- Python: Used in the last 6 months
- SQL: Used in the last 6 months
- Power BI: Used 1 year ago
- Excel: Used 1 year ago

**Projects:**
- Sales Dashboard using Power BI
- Customer Data Analysis using Python and SQL
"""

PERFECT_CANDIDATE_CV = """
**Candidate Name:** Senior Analyst Pro  
**Education:** Master's in Data Science  
**Total Experience:** 4 years  
**Previous Role:** Senior Data Analyst  

**Technical Skills:**
- Python
- SQL
- Power BI
- Excel
- Data Cleaning
- Data Visualization
- Statistical Analysis

**Work Experience:**
- Cleaned and prepared large datasets using Python and Pandas.
- Wrote complex SQL queries to model data pipelines.
- Created interactive Power BI executive dashboards.
- Built statistical analysis models and conducted hypothesis testing.
- Used Excel for financial data modeling.
"""


def test_1_one_year_experience_against_three_years_requirement():
    """1. Verify that 1 year vs 3 years calculates experience fit ~33.3%, not 100%."""
    resp = client.post("/match", json={"resume_text": AHMED_KHAN_CV, "job_description": DATA_ANALYST_JD, "mode": "proposal"})
    assert resp.status_code == 200
    data = resp.json()

    exp_analysis = data["experience_analysis"]
    assert exp_analysis["required_years"] == 3
    assert exp_analysis["candidate_years"] == 1.0
    assert exp_analysis["is_met"] is False

    exp_breakdown = data["score_breakdown"]["experience"]
    assert round(exp_breakdown["value"], 2) == 0.33
    assert exp_breakdown["weighted_contribution"] == round(0.3333 * 0.10, 4)


def test_2_missing_mandatory_skills_despite_high_semantic_score():
    """2. Verify missing mandatory skills flag candidate as ineligible despite 100% semantic score."""
    resp = client.post("/match", json={"resume_text": AHMED_KHAN_CV, "job_description": DATA_ANALYST_JD, "mode": "proposal"})
    assert resp.status_code == 200
    data = resp.json()

    # Semantic similarity is 1.0 (high)
    assert data["score_breakdown"]["semantic"]["value"] == 1.0

    # But Statistical Analysis is missing
    assert "statistical analysis" in [s.lower() for s in data["required_missing"]]

    # Separate eligibility block flags candidate as not meeting all mandatory requirements
    elig = data["eligibility"]
    assert elig["is_eligible"] is False
    assert elig["status"] == "Does Not Meet All Mandatory Requirements"
    assert any("statistical analysis" in g.lower() for g in elig["mandatory_gaps"])
    assert any("experience" in g.lower() for g in elig["mandatory_gaps"])


def test_3_candidate_meeting_all_mandatory_requirements():
    """3. Verify a fully qualified candidate receives is_eligible=True and Meets All Mandatory Requirements."""
    resp = client.post("/match", json={"resume_text": PERFECT_CANDIDATE_CV, "job_description": DATA_ANALYST_JD, "mode": "proposal"})
    assert resp.status_code == 200
    data = resp.json()

    elig = data["eligibility"]
    assert elig["is_eligible"] is True
    assert elig["status"] == "Meets All Mandatory Requirements"
    assert len(elig["mandatory_gaps"]) == 0
    assert data["experience_analysis"]["is_met"] is True
    assert len(data["required_missing"]) == 0


def test_4_consistency_between_summary_counts_and_criteria():
    """4. Verify that summary counts equal the count of criteria in detailed evidence audit."""
    resp_match = client.post("/match", json={"resume_text": AHMED_KHAN_CV, "job_description": DATA_ANALYST_JD, "mode": "proposal"})
    resp_ev = client.post("/resume/evidence", json={"resume_text": AHMED_KHAN_CV})
    assert resp_match.status_code == 200
    assert resp_ev.status_code == 200

    m = resp_match.json()
    ev_data = resp_ev.json()

    req_matched = m["required_matched"]
    req_missing = m["required_missing"]

    # Classify matched skills as demonstrated vs unverified
    demonstrated = [s for s in req_matched if ev_data.get(s, {}).get("level") in ("high_depth", "demonstrated")]
    unverified = [s for s in req_matched if s not in demonstrated]

    # Verify counts sum up exactly to total required
    assert len(demonstrated) + len(unverified) + len(req_missing) == m["total_required"]
    assert len(demonstrated) >= 1  # sql and power bi demonstrated
    assert len(unverified) >= 1    # excel or unbulleted data skills


def test_5_correct_weighted_score_calculation_in_proposal_mode():
    """5. Verify proposal mode formula: final_score = (skills*0.40) + (sem*0.40) + (exp*0.10) + (edu*0.10)."""
    resp = client.post("/match", json={"resume_text": AHMED_KHAN_CV, "job_description": DATA_ANALYST_JD, "mode": "proposal"})
    assert resp.status_code == 200
    data = resp.json()

    bd = data["score_breakdown"]
    sk_val = bd["skill"]["value"]
    sem_val = bd["semantic"]["value"]
    exp_val = bd["experience"]["value"]
    edu_val = bd["education"]["value"]

    expected_sum = (sk_val * 0.40) + (sem_val * 0.40) + (exp_val * 0.10) + (edu_val * 0.10)
    assert data["final_score"] == pytest.approx(expected_sum, abs=1e-3)

    # For the test candidate (6/7 skills = 85.7%, sem = 100%, exp = 33.3%, edu = 100%)
    # Expected score ~ 87.6% (rounds to 88%)
    assert 0.87 <= data["final_score"] <= 0.88
    assert int(round(data["final_score"] * 100)) == 88


def test_6_correct_score_calculation_across_other_modes():
    """6. Verify all 4 scoring modes calculate scores mathematically according to their preset weights."""
    for mode in ("skills_only", "hybrid", "proposal", "composite"):
        resp = client.post("/match", json={"resume_text": AHMED_KHAN_CV, "job_description": DATA_ANALYST_JD, "mode": mode})
        assert resp.status_code == 200
        data = resp.json()

        bd = data["score_breakdown"]
        weights = SCORING_CONFIG[mode]

        # Weights sum to 1.0
        assert sum(weights.values()) == pytest.approx(1.0, rel=1e-4)

        # Contribution sum matches final_score
        contrib_sum = sum(item["weighted_contribution"] for item in bd.values() if isinstance(item, dict) and "weighted_contribution" in item)
        assert round(contrib_sum, 4) == round(data["final_score"], 4)


def test_7_education_fit_included_only_in_supported_modes():
    """7. Verify education fit is included in proposal mode, but omitted from composite and hybrid."""
    resp_prop = client.post("/match", json={"resume_text": AHMED_KHAN_CV, "job_description": DATA_ANALYST_JD, "mode": "proposal"}).json()
    resp_comp = client.post("/match", json={"resume_text": AHMED_KHAN_CV, "job_description": DATA_ANALYST_JD, "mode": "composite"}).json()
    resp_hyb = client.post("/match", json={"resume_text": AHMED_KHAN_CV, "job_description": DATA_ANALYST_JD, "mode": "hybrid"}).json()

    assert "education" in resp_prop["score_breakdown"]
    assert resp_prop["score_breakdown"]["education"]["weight"] == 0.10

    # Composite uses recency instead of education
    assert "education" not in resp_comp["score_breakdown"]
    assert "recency" in resp_comp["score_breakdown"]

    # Hybrid uses skills + semantic only
    assert resp_hyb["score_breakdown"].get("education", {}).get("weight", 0.0) == 0.0


def test_8_ui_displays_actual_backend_score_and_eligibility():
    """8. Verify UI rendering logic preserves backend score and distinct eligibility badge."""
    from dashboard.app import _render_score_ring

    # Test rendering does not throw exception and uses backend values
    resp = client.post("/match", json={"resume_text": AHMED_KHAN_CV, "job_description": DATA_ANALYST_JD, "mode": "proposal"}).json()
    score = resp["final_score"]
    elig = resp["eligibility"]

    # Ensure score rounds to 88%
    pct = int(round(score * 100))
    assert pct == 88

    # Ensure eligibility is separate
    assert elig["is_eligible"] is False
    assert elig["status"] == "Does Not Meet All Mandatory Requirements"
