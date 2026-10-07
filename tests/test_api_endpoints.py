"""FastAPI TestClient integration tests for all Resume Analyzer API endpoints.

Covers:
- POST /resume/upload
  - Success cases (PDF, DOCX, TXT)
  - Failure cases (unsupported format, empty file, corrupted file, missing payload, oversized file limit)
- POST /job/analyze
  - Success cases (structured output)
  - Failure cases (empty JD text, missing description field, invalid JSON)
- POST /match
  - Success cases (match score, skill breakdown)
  - Failure cases (empty resume_text, empty job_description, missing resume_id / resume_text)
- GET /recommendations
  - Success cases (with JD, without JD)
  - Failure cases (empty resume_text, missing resume_text param)
- GET /skill-gap
  - Success cases (required & preferred gaps)
  - Failure cases (empty parameters, missing query parameters)
"""

import io
import pytest
from fastapi.testclient import TestClient
import docx

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from app.main import app, MAX_FILE_SIZE


@pytest.fixture(scope="module")
def client():
    """Shared FastAPI TestClient fixture."""
    return TestClient(app)


def make_pdf_bytes(text: str) -> bytes:
    """Helper to generate in-memory PDF bytes."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), text)
    data = doc.write()
    doc.close()
    return data


def make_docx_bytes(text: str) -> bytes:
    """Helper to generate in-memory DOCX bytes."""
    doc = docx.Document()
    for line in text.splitlines():
        if line.strip():
            doc.add_paragraph(line.strip())
    stream = io.BytesIO()
    doc.save(stream)
    return stream.getvalue()


# ===========================================================================
# 1. POST /resume/upload Tests
# ===========================================================================

class TestResumeUploadEndpoint:
    """Test suite for POST /resume/upload."""

    def test_upload_pdf_success(self, client):
        """Upload a valid PDF resume and verify extracted metadata and text."""
        pdf_bytes = make_pdf_bytes("Jane Doe\nSenior Python Engineer\nSkills: Python, FastAPI, Docker\n5 years experience")
        response = client.post(
            "/resume/upload",
            files={"file": ("resume.pdf", pdf_bytes, "application/pdf")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "resume.pdf"
        assert data["char_count"] > 0
        assert "python" in data["skills"]
        assert "fastapi" in data["skills"]
        assert "extracted_text" in data
        assert "text_preview" in data
        assert data["experience_years"] == 5

    def test_upload_docx_success(self, client):
        """Upload a valid DOCX resume and verify extraction."""
        docx_bytes = make_docx_bytes("Hania Shafiq\nData Scientist\nSkills: Python, PyTorch, Pandas\nMaster of Science in Data Science")
        response = client.post(
            "/resume/upload",
            files={"file": ("resume.docx", docx_bytes, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "resume.docx"
        assert "python" in data["skills"]
        assert any(e["degree"] == "Master's" for e in data["education"])

    def test_upload_txt_success(self, client):
        """Upload a valid TXT resume and verify extraction."""
        txt_bytes = b"Alex Murphy\nDevOps Engineer\nSkills: Kubernetes, Docker, Terraform\n4 years of experience"
        response = client.post(
            "/resume/upload",
            files={"file": ("resume.txt", txt_bytes, "text/plain")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "kubernetes" in data["skills"]
        assert data["experience_years"] == 4

    def test_upload_wrong_file_type_fails(self, client):
        """Upload an unsupported file format (.png / .xyz) should fail with 422."""
        response = client.post(
            "/resume/upload",
            files={"file": ("avatar.png", b"\x89PNG\r\n\x1a\nfake png data", "image/png")},
        )
        assert response.status_code == 422
        assert "Unsupported file format" in response.json()["detail"]

    def test_upload_empty_file_fails(self, client):
        """Upload an empty file (0 bytes) should fail with 400."""
        response = client.post(
            "/resume/upload",
            files={"file": ("empty.pdf", b"", "application/pdf")},
        )
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_upload_corrupted_file_fails(self, client):
        """Upload corrupted PDF bytes should fail gracefully with 422."""
        corrupted_bytes = b"%PDF-corrupted-random-binary-junk-data"
        response = client.post(
            "/resume/upload",
            files={"file": ("corrupted.pdf", corrupted_bytes, "application/pdf")},
        )
        assert response.status_code == 422
        assert "Failed to parse" in response.json()["detail"]

    def test_upload_missing_file_payload_fails(self, client):
        """POST without file field should return 422 unprocessable entity."""
        response = client.post("/resume/upload")
        assert response.status_code == 422

    def test_upload_very_large_file_size_limit(self, client):
        """POST /resume/upload should reject files exceeding MAX_FILE_SIZE (10 MB)."""
        # Create a payload exceeding MAX_FILE_SIZE (10 MB + 100 KB)
        oversized_bytes = b"Python developer resume line.\n" * ((MAX_FILE_SIZE // 30) + 1000)
        assert len(oversized_bytes) > MAX_FILE_SIZE

        response = client.post(
            "/resume/upload",
            files={"file": ("oversized.txt", oversized_bytes, "text/plain")},
        )
        # Expected behavior: endpoint rejects oversized payload with 400 or 413
        # Actual behavior: upload_resume lacks MAX_FILE_SIZE check, processing the entire file
        assert response.status_code in [400, 413], (
            f"Expected file size limit rejection (400 or 413), but got {response.status_code}. "
            "MAX_FILE_SIZE is not enforced on /resume/upload."
        )


# ===========================================================================
# 2. POST /job/analyze Tests
# ===========================================================================

class TestJobAnalyzeEndpoint:
    """Test suite for POST /job/analyze."""

    def test_analyze_jd_success(self, client):
        """Analyze a valid job description with title and description."""
        payload = {
            "job_title": "Senior Backend Developer",
            "description": (
                "Requirements:\n"
                "- Strong experience in Python, FastAPI, and PostgreSQL\n"
                "- 4+ years of professional experience\n"
                "Preferred:\n"
                "- Experience with Docker and Redis\n"
            ),
        }
        response = client.post("/job/analyze", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "python" in data["required_skills"]
        assert "fastapi" in data["required_skills"]
        assert "docker" in data["preferred_skills"]
        assert data["min_experience_years"] == 4
        assert data["job_category"] == "backend"

    def test_analyze_jd_empty_text_fails(self, client):
        """Analyze with blank description should return 400."""
        payload = {"job_title": "Software Engineer", "description": "   \n  "}
        response = client.post("/job/analyze", json=payload)
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_analyze_jd_missing_description_field_fails(self, client):
        """Analyze without required 'description' field in body should return 422."""
        payload = {"job_title": "Lead Architect"}
        response = client.post("/job/analyze", json=payload)
        assert response.status_code == 422


# ===========================================================================
# 3. POST /match Tests
# ===========================================================================

class TestMatchEndpoint:
    """Test suite for POST /match."""

    def test_match_multiline_success(self, client):
        """Match valid resume text against a well-structured multiline job description."""
        payload = {
            "resume_text": "Experienced Software Engineer with Python, FastAPI, and PostgreSQL.",
            "job_description": "Requirements:\n- Python\n- FastAPI\n- Docker\nPreferred:\n- Redis\n",
        }
        response = client.post("/match", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "skill_overlap_score" in data
        assert 0.0 <= data["skill_overlap_score"] <= 1.0
        assert "python" in data["required_matched"]
        assert "fastapi" in data["required_matched"]
        assert "docker" in data["required_missing"]

    def test_match_single_line_inline_headings(self, client):
        """Single-line JD with 'Requirements: ... Preferred: ...' inline signals."""
        payload = {
            "resume_text": "Experienced Software Engineer with Python, FastAPI, and PostgreSQL.",
            "job_description": "Requirements: Python, FastAPI, and Docker. Preferred: Redis.",
        }
        response = client.post("/match", json=payload)
        assert response.status_code == 200
        data = response.json()
        # Due to a bug in _split_jd_into_sections (checking _PREFERRED_INLINE before _REQUIRED_INLINE on single lines),
        # all skills end up in preferred, and required_matched is empty!
        assert "python" in data["required_matched"], (
            "Bug: Single-line JD with inline 'Requirements:' and 'Preferred:' classifies all skills into preferred."
        )

    def test_match_empty_resume_text_fails(self, client):
        """Match with empty resume_text should return 400."""
        payload = {"resume_text": "  ", "job_description": "Requirements: Python."}
        response = client.post("/match", json=payload)
        assert response.status_code == 400
        assert "resume_text must not be empty" in response.json()["detail"]

    def test_match_empty_job_description_fails(self, client):
        """Match with empty job_description should return 400."""
        payload = {"resume_text": "Python developer", "job_description": "  "}
        response = client.post("/match", json=payload)
        assert response.status_code == 400
        assert "job_description must not be empty" in response.json()["detail"]

    def test_match_missing_resume_id_or_resume_text(self, client):
        """If caller attempts ID-based matching or omits resume_text, API must return 422."""
        payload = {"resume_id": "resume_abc_123", "job_description": "Requirements: Python"}
        response = client.post("/match", json=payload)
        assert response.status_code == 422


# ===========================================================================
# 4. GET /recommendations Tests
# ===========================================================================

class TestRecommendationsEndpoint:
    """Test suite for GET /recommendations."""

    def test_recommendations_with_jd_success(self, client):
        """Get skill recommendations given resume and target JD."""
        response = client.get(
            "/recommendations",
            params={
                "resume_text": "Python developer with Django.",
                "job_description": "Requirements:\n- Python\n- Docker\n- Kubernetes\n- AWS",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "recommended_to_learn" in data
        assert "docker" in data["recommended_to_learn"]
        assert "kubernetes" in data["recommended_to_learn"]

    def test_recommendations_without_jd_success(self, client):
        """Get generic recommendations when no JD is supplied."""
        response = client.get(
            "/recommendations",
            params={"resume_text": "Python developer with SQL."},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["recommended_to_learn"]) > 0

    def test_recommendations_empty_resume_text_fails(self, client):
        """Empty resume_text query param should return 400."""
        response = client.get("/recommendations", params={"resume_text": "   "})
        assert response.status_code == 400
        assert "resume_text must not be empty" in response.json()["detail"]

    def test_recommendations_missing_resume_text_fails(self, client):
        """Omitted resume_text query param should return 422."""
        response = client.get("/recommendations")
        assert response.status_code == 422


# ===========================================================================
# 5. GET /skill-gap Tests
# ===========================================================================

class TestSkillGapEndpoint:
    """Test suite for GET /skill-gap."""

    def test_skill_gap_multiline_success(self, client):
        """Compute skill gap between resume and multiline JD."""
        response = client.get(
            "/skill-gap",
            params={
                "resume_text": "Proficient in Python and Git.",
                "job_description": "Requirements:\n- Python\n- Docker\n- PostgreSQL\nPreferred:\n- Redis\n",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "required_gap" in data
        assert "docker" in data["required_gap"]
        assert "postgresql" in data["required_gap"]
        assert "redis" in data["preferred_gap"]
        assert data["gap_count"] >= 3

    def test_skill_gap_single_line_inline_headings(self, client):
        """Single-line JD with inline 'Requirements:' and 'Preferred:' signals."""
        response = client.get(
            "/skill-gap",
            params={
                "resume_text": "Proficient in Python and Git.",
                "job_description": "Requirements: Python, Docker, PostgreSQL. Preferred: Redis.",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "docker" in data["required_gap"], (
            "Bug: Single-line JD with inline 'Requirements:' and 'Preferred:' classifies all skills into preferred."
        )

    def test_skill_gap_empty_resume_text_fails(self, client):
        """Empty resume_text in skill-gap query should return 400."""
        response = client.get(
            "/skill-gap",
            params={"resume_text": "  ", "job_description": "Requirements: Python."},
        )
        assert response.status_code == 400
        assert "Both resume_text and job_description are required" in response.json()["detail"]

    def test_skill_gap_empty_job_description_fails(self, client):
        """Empty job_description in skill-gap query should return 400."""
        response = client.get(
            "/skill-gap",
            params={"resume_text": "Python developer", "job_description": "  "},
        )
        assert response.status_code == 400
        assert "Both resume_text and job_description are required" in response.json()["detail"]

    def test_skill_gap_missing_params_fails(self, client):
        """Missing required query params should return 422."""
        response = client.get("/skill-gap", params={"resume_text": "Python"})
        assert response.status_code == 422
