"""Tests for app/evidence.py and POST /resume/evidence (require en_core_web_sm)."""

import pytest
from fastapi.testclient import TestClient

from app.evidence import (
    EVIDENCE_MULTIPLIERS,
    evidence_counts,
    skill_evidence,
    skill_multipliers,
)
from app.main import app

client = TestClient(app)


def test_empty_text_returns_empty():
    assert skill_evidence("") == {}
    assert skill_evidence("   \n ") == {}
    assert skill_multipliers("") == {}


def test_no_skills_in_text():
    assert skill_evidence("The weather today is sunny and warm.") == {}


def test_multiplier_values_exact():
    assert EVIDENCE_MULTIPLIERS == {"high_depth": 1.0, "demonstrated": 0.85, "listed": 0.5}


def test_keyword_stuffed_skills_section_is_listed():
    text = "SKILLS\nPython, Docker, Kubernetes, Redis, AWS\n"
    ev = skill_evidence(text)
    assert ev
    assert all(v["level"] == "listed" for v in ev.values())
    assert all(v["multiplier"] == 0.5 and v["evidence"] is None for v in ev.values())


def test_plain_comma_list_is_listed():
    ev = skill_evidence("Docker, Kubernetes, Redis", ["docker", "kubernetes", "redis"])
    assert {v["level"] for v in ev.values()} == {"listed"}


def test_pipe_list_is_listed():
    ev = skill_evidence("Python | Docker | AWS", ["python", "docker", "aws"])
    assert {v["level"] for v in ev.values()} == {"listed"}


def test_dependency_case_deployed_service_on_docker():
    ev = skill_evidence("Deployed a service on Docker.", ["docker"])
    assert ev["docker"]["level"] in ("demonstrated", "high_depth")
    assert any(s.startswith("action_verb:deploy") for s in ev["docker"]["signals"])
    assert ev["docker"]["evidence"] == "Deployed a service on Docker."


def test_metric_only_sentence_is_demonstrated():
    ev = skill_evidence("Redis cluster serving 5000 requests per second.", ["redis"])
    assert ev["redis"]["level"] == "demonstrated"
    assert any(s.startswith("metric:") for s in ev["redis"]["signals"])


def test_verb_and_metric_is_high_depth():
    ev = skill_evidence("Reduced API latency by 40% using Redis.", ["redis"])
    assert ev["redis"]["level"] == "high_depth"
    assert ev["redis"]["multiplier"] == 1.0
    assert "metric:40%" in ev["redis"]["signals"]


def test_years_of_experience_is_not_a_metric():
    ev = skill_evidence("Python, 5 years of experience.", ["python"])
    assert ev["python"]["level"] == "listed"


def test_list_and_experience_sentence_uses_strongest_evidence():
    text = (
        "Skills: Python, Docker\n"
        "EXPERIENCE\n"
        "Deployed Docker containers to production, cutting release time by 30%.\n"
    )
    ev = skill_evidence(text, ["docker", "python"])
    assert ev["docker"]["level"] == "high_depth"
    assert ev["python"]["level"] == "listed"


def test_unicode_text():
    text = "Résumé – Société Générale\nDeployed Docker services for 2M users, reducing costs by 25%.\n"
    ev = skill_evidence(text, ["docker"])
    assert ev["docker"]["level"] == "high_depth"


def test_alias_resolves_to_canonical_key():
    ev = skill_evidence("Deployed services on k8s.", ["K8s"])
    assert "kubernetes" in ev


def test_skills_absent_from_text_are_omitted():
    assert skill_evidence("Deployed Docker images.", ["docker", "rust"]).keys() == {"docker"}


def test_multipliers_match_evidence():
    text = "Skills: Python\nBuilt Docker pipelines, saving 20% of build time.\n"
    ev = skill_evidence(text, ["python", "docker"])
    mult = skill_multipliers(text, ["python", "docker"])
    assert mult == {k: v["multiplier"] for k, v in ev.items()}
    assert mult["python"] == 0.5 and mult["docker"] == 1.0


def test_idempotent():
    text = "Skills: Python, Docker\nDeployed Docker containers, reducing build time by 30%.\n"
    assert skill_evidence(text) == skill_evidence(text)


def test_counts_helper():
    text = "Skills: Python\nBuilt Docker pipelines, saving 20% of build time.\n"
    counts = evidence_counts(skill_evidence(text, ["python", "docker"]))
    assert counts == {"high_depth": 1, "demonstrated": 0, "listed": 1}


def test_non_string_raises():
    with pytest.raises(TypeError):
        skill_evidence(123)


# ---------------------------- endpoint -----------------------------------
def test_endpoint_success_with_counts():
    resp = client.post("/resume/evidence", json={
        "resume_text": "Skills: Python\nDeployed Docker containers, reducing build time by 30%.\n",
        "skills": ["Python", "Docker"],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["docker"]["level"] == "high_depth"
    assert data["python"]["level"] == "listed"
    assert data["counts"] == {"high_depth": 1, "demonstrated": 0, "listed": 1}


def test_endpoint_skills_optional():
    resp = client.post("/resume/evidence", json={"resume_text": "Deployed Docker images, cutting time by 10%."})
    assert resp.status_code == 200
    assert "docker" in resp.json()


def test_endpoint_empty_text_400():
    assert client.post("/resume/evidence", json={"resume_text": "  \n"}).status_code == 400


def test_endpoint_missing_text_422():
    assert client.post("/resume/evidence", json={}).status_code == 422
