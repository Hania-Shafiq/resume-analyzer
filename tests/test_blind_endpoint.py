"""Tests for the Blind Screening API endpoints (/resume/anonymize and /resume/dossier)."""

from __future__ import annotations

import hashlib
import json
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


SAMPLE_RESUME = (
    "Muhammad Usman\n"
    "Full Stack Software Engineer\n"
    "Email: usman@example.com\n"
    "Phone: +92 300 1234567\n"
    "Portfolio: https://linkedin.com/in/usman\n"
    "Location: Karachi, Pakistan\n\n"
    "Skills: Python, FastAPI, Docker, PostgreSQL, React, Git\n\n"
    "Experience:\n"
    "Senior Developer | Tech Innovators | Jan 2021 - Present\n"
    "Built cloud APIs with FastAPI and Docker. His technical leadership improved system uptime.\n"
    "Junior Engineer | Web Solutions | Jan 2019 - Dec 2020\n"
    "Maintained database queries and developed internal tools.\n\n"
    "Education:\n"
    "Bachelor of Science in Computer Science, 2018"
)


def test_anonymize_endpoint_success(client):
    """POST /resume/anonymize with valid resume text returns 200 and anonymized structure."""
    payload = {"resume_text": SAMPLE_RESUME}
    response = client.post("/resume/anonymize", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "text" in data
    assert "redactions" in data
    assert "audit" in data
    assert "residual_flags" in data

    # Check that PII has been replaced
    assert "Muhammad Usman" not in data["text"]
    assert "usman@example.com" not in data["text"]
    assert "+92 300 1234567" not in data["text"]
    assert "https://linkedin.com/in/usman" not in data["text"]
    assert "Bachelor of Science in Computer Science, [YEAR]" in data["text"]
    assert "Jan 2021 - Present" in data["text"]

    # Check audit structure
    for entry in data["audit"]:
        assert "type" in entry
        assert "placeholder" in entry
        assert "line" in entry
        assert "rule" in entry


def test_anonymize_endpoint_empty_text_400(client):
    """POST /resume/anonymize returns 400 when resume_text is empty or blank."""
    resp1 = client.post("/resume/anonymize", json={"resume_text": ""})
    assert resp1.status_code == 400

    resp2 = client.post("/resume/anonymize", json={"resume_text": "   \n\t  "})
    assert resp2.status_code == 400


def test_anonymize_endpoint_with_options(client):
    """POST /resume/anonymize respects optional switches."""
    payload = {
        "resume_text": SAMPLE_RESUME,
        "options": {"phone": False, "location": False},
    }
    response = client.post("/resume/anonymize", json=payload)
    assert response.status_code == 200

    data = response.json()
    # Phone and location are preserved because their options were False
    assert "+92 300 1234567" in data["text"]
    assert "Karachi" in data["text"]

    # Email and Name were still redacted
    assert "usman@example.com" not in data["text"]
    assert "Muhammad Usman" not in data["text"]


def test_dossier_endpoint_success(client):
    """POST /resume/dossier returns structured blind candidate dossier."""
    payload = {"resume_text": SAMPLE_RESUME}
    response = client.post("/resume/dossier", json=payload)
    assert response.status_code == 200

    data = response.json()
    # Required keys according to specification
    assert "candidate_ref" in data
    assert "anonymized_text" in data
    assert "skills" in data
    assert "experience" in data
    assert "education_level" in data
    assert "redaction_summary" in data
    assert "audit" in data
    assert "residual_flags" in data
    assert "disclaimer" in data

    # Check candidate ref format
    expected_ref = "C-" + hashlib.sha256(SAMPLE_RESUME.encode("utf-8")).hexdigest()[:8]
    assert data["candidate_ref"] == expected_ref

    # Skills extraction
    skills = data["skills"]
    assert isinstance(skills, list)
    assert "python" in skills
    assert "fastapi" in skills

    # Experience years
    assert isinstance(data["experience"], (int, float))
    assert data["experience"] >= 2

    # Education level
    assert data["education_level"] == "Bachelor's"

    # Redaction summary
    assert isinstance(data["redaction_summary"], list)
    assert len(data["redaction_summary"]) > 0

    # Disclaimer
    assert "disclaimer" in data
    assert "heuristic" in data["disclaimer"].lower()
    assert "right-to-work" in data["disclaimer"].lower()


def test_dossier_endpoint_empty_text_400(client):
    """POST /resume/dossier returns 400 for empty or whitespace-only resume text."""
    resp1 = client.post("/resume/dossier", json={"resume_text": ""})
    assert resp1.status_code == 400

    resp2 = client.post("/resume/dossier", json={"resume_text": "   "})
    assert resp2.status_code == 400


def test_dossier_contains_no_pii(client):
    """The entire dossier payload must contain no candidate name, email, phone, link, or grad year."""
    payload = {"resume_text": SAMPLE_RESUME}
    response = client.post("/resume/dossier", json=payload)
    assert response.status_code == 200

    raw_json = json.dumps(response.json())

    # Candidate name
    assert "Muhammad Usman" not in raw_json
    # Candidate email
    assert "usman@example.com" not in raw_json
    # Candidate phone
    assert "+92 300 1234567" not in raw_json
    assert "300 1234567" not in raw_json
    # Candidate portfolio link
    assert "linkedin.com/in/usman" not in raw_json
    # Graduation year
    assert "2018" not in raw_json
