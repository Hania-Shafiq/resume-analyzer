"""Comprehensive unit and integration tests for bulk upload, ranking, and edge cases."""

import hashlib
import io
import pytest
from fastapi.testclient import TestClient
import docx

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from app.main import app
from app.parser import extract_text
from app.ranker import rank_candidates


client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers to generate in-memory files for tests
# ---------------------------------------------------------------------------

def make_pdf_bytes(text: str) -> bytes:
    """Create in-memory PDF bytes with text."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), text)
    data = doc.write()
    doc.close()
    return data


def make_docx_bytes(text: str) -> bytes:
    """Create in-memory DOCX bytes with text."""
    doc = docx.Document()
    for line in text.split("\n"):
        if line.strip():
            doc.add_paragraph(line.strip())
    stream = io.BytesIO()
    doc.save(stream)
    return stream.getvalue()


def make_txt_bytes(text: str) -> bytes:
    """Create in-memory UTF-8 bytes."""
    return text.encode("utf-8")


# ---------------------------------------------------------------------------
# Unit Tests — Parser
# ---------------------------------------------------------------------------

class TestParserFormats:
    """Test PDF, DOCX, and TXT parsing in app/parser.py."""

    def test_parse_txt_success(self):
        content = "Jane Doe\nSenior Python Engineer\nSkills: Python, FastAPI, Docker\n5 years experience"
        parsed = extract_text(make_txt_bytes(content), filename="resume.txt")
        assert "Jane Doe" in parsed
        assert "FastAPI" in parsed

    def test_parse_pdf_success(self):
        content = "John Smith - Data Scientist\nSkills: Machine Learning, PyTorch"
        parsed = extract_text(make_pdf_bytes(content), filename="resume.pdf")
        assert "John Smith" in parsed
        assert "PyTorch" in parsed

    def test_parse_docx_success(self):
        content = "Alice Wong\nFull Stack Developer\nSkills: React, Node.js, TypeScript"
        parsed = extract_text(make_docx_bytes(content), filename="resume.docx")
        assert "Alice Wong" in parsed
        assert "TypeScript" in parsed

    def test_parse_unsupported_format_raises(self):
        with pytest.raises(ValueError) as exc:
            extract_text(b"some content", filename="resume.jpg")
        assert "Unsupported file format" in str(exc.value)

    def test_parse_empty_raises(self):
        with pytest.raises(ValueError) as exc:
            extract_text(b"", filename="resume.txt")
        assert "empty" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# Unit Tests — Ranker & Tie-Breaking
# ---------------------------------------------------------------------------

class TestRankerLogic:
    """Test deterministic tie-breaking and Top-N slicing in app/ranker.py."""

    def test_basic_ranking_order(self):
        results = [
            {"status": "analyzed", "candidate_name": "Low Match", "match_score": 0.30, "skills_match_count": 1, "experience_years": 2, "required_matched": ["python"], "total_required": 4},
            {"status": "analyzed", "candidate_name": "High Match", "match_score": 0.90, "skills_match_count": 4, "experience_years": 5, "required_matched": ["python", "fastapi", "docker", "aws"], "total_required": 4},
            {"status": "analyzed", "candidate_name": "Mid Match", "match_score": 0.60, "skills_match_count": 2, "experience_years": 3, "required_matched": ["python", "docker"], "total_required": 4},
        ]
        ranked = rank_candidates(results)
        names = [c["candidate_name"] for c in ranked["ranked"]]
        assert names == ["High Match", "Mid Match", "Low Match"]
        assert ranked["ranked"][0]["rank"] == 1
        assert ranked["ranked"][1]["rank"] == 2
        assert ranked["ranked"][2]["rank"] == 3

    def test_tie_breaking_by_skills_count(self):
        # Same score (0.80), but candidate A has 4 matched skills vs candidate B's 3
        results = [
            {"status": "analyzed", "candidate_name": "Candidate B", "match_score": 0.80, "skills_match_count": 3, "experience_years": 5},
            {"status": "analyzed", "candidate_name": "Candidate A", "match_score": 0.80, "skills_match_count": 4, "experience_years": 5},
        ]
        ranked = rank_candidates(results)
        assert ranked["ranked"][0]["candidate_name"] == "Candidate A"
        assert ranked["ranked"][1]["candidate_name"] == "Candidate B"

    def test_tie_breaking_by_experience(self):
        # Same score and skill count, candidate with higher experience wins
        results = [
            {"status": "analyzed", "candidate_name": "Junior", "match_score": 0.80, "skills_match_count": 3, "experience_years": 2},
            {"status": "analyzed", "candidate_name": "Senior", "match_score": 0.80, "skills_match_count": 3, "experience_years": 8},
        ]
        ranked = rank_candidates(results)
        assert ranked["ranked"][0]["candidate_name"] == "Senior"
        assert ranked["ranked"][1]["candidate_name"] == "Junior"

    def test_tie_breaking_by_name_alphabetical(self):
        # Identical scores, skills, and experience: tie-break alphabetically
        results = [
            {"status": "analyzed", "candidate_name": "Zack", "match_score": 0.80, "skills_match_count": 3, "experience_years": 5},
            {"status": "analyzed", "candidate_name": "Aaron", "match_score": 0.80, "skills_match_count": 3, "experience_years": 5},
        ]
        ranked = rank_candidates(results)
        assert ranked["ranked"][0]["candidate_name"] == "Aaron"
        assert ranked["ranked"][1]["candidate_name"] == "Zack"

    def test_top_n_slicing(self):
        results = [
            {"status": "analyzed", "candidate_name": f"Cand {i}", "match_score": i * 0.1, "skills_match_count": i, "experience_years": i}
            for i in range(1, 11)
        ]
        res_top3 = rank_candidates(results, top_n=3)
        assert len(res_top3["ranked"]) == 3
        assert res_top3["top_n_returned"] == 3
        assert res_top3["ranked"][0]["candidate_name"] == "Cand 10"  # highest score first

    def test_top_n_greater_than_total_shows_notice(self):
        results = [
            {"status": "analyzed", "candidate_name": "Only One", "match_score": 0.85, "skills_match_count": 3, "experience_years": 4}
        ]
        res = rank_candidates(results, top_n=5)
        assert len(res["ranked"]) == 1
        assert res["notice"] is not None
        assert "Showing all" in res["notice"]

    def test_top_n_zero_or_negative_raises(self):
        with pytest.raises(ValueError):
            rank_candidates([], top_n=0)
        with pytest.raises(ValueError):
            rank_candidates([], top_n=-5)


# ---------------------------------------------------------------------------
# Integration Tests — Bulk Analyze API
# ---------------------------------------------------------------------------

class TestBulkAnalyzeAPI:
    """Test /resume/bulk-analyze endpoint end-to-end."""

    JD = (
        "Role: Senior Backend Engineer\n"
        "Requirements: Python, FastAPI, Docker, PostgreSQL\n"
        "Preferred: Kubernetes, AWS\n"
        "Experience: 5+ years"
    )

    def test_bulk_analyze_end_to_end(self):
        r1_text = "Alice Smith\nPython, FastAPI, Docker, PostgreSQL\n7 years experience\nMaster of Science in Computer Science"
        r2_text = "Bob Johnson\nPython, Flask, Git\n2 years experience\nBachelor of Science in Computer Science"

        files = [
            ("files", ("alice.txt", make_txt_bytes(r1_text), "text/plain")),
            ("files", ("bob.txt", make_txt_bytes(r2_text), "text/plain")),
        ]
        response = client.post(
            "/resume/bulk-analyze",
            files=files,
            data={"job_description": self.JD},
        )
        assert response.status_code == 200
        data = response.json()

        assert data["summary"]["total_uploaded"] == 2
        assert data["summary"]["total_analyzed"] == 2
        assert data["summary"]["total_failed"] == 0

        # Alice should have higher match score than Bob
        results = data["results"]
        alice = next(r for r in results if "alice" in r["filename"])
        bob = next(r for r in results if "bob" in r["filename"])
        assert alice["match_score"] > bob["match_score"]
        assert "fastapi" in alice["required_matched"]
        assert "docker" in alice["required_matched"]

    def test_duplicate_file_detection(self):
        content = "Charlie Brown\nPython, Docker\n3 years experience"
        b = make_txt_bytes(content)

        files = [
            ("files", ("charlie_1.txt", b, "text/plain")),
            ("files", ("charlie_copy.txt", b, "text/plain")),
        ]
        response = client.post(
            "/resume/bulk-analyze",
            files=files,
            data={"job_description": self.JD},
        )
        assert response.status_code == 200
        data = response.json()

        assert data["summary"]["total_uploaded"] == 2
        assert data["summary"]["total_analyzed"] == 1
        assert data["summary"]["total_duplicates"] == 1

        copy_entry = next(r for r in data["results"] if r["filename"] == "charlie_copy.txt")
        assert copy_entry["status"] == "duplicate"

    def test_failed_file_does_not_stop_others(self):
        good = "Valid Candidate\nPython, FastAPI\n4 years experience"
        corrupt = b"\x00\x01\x02\x03not a real pdf"

        files = [
            ("files", ("good.txt", make_txt_bytes(good), "text/plain")),
            ("files", ("corrupted.pdf", corrupt, "application/pdf")),
        ]
        response = client.post(
            "/resume/bulk-analyze",
            files=files,
            data={"job_description": self.JD},
        )
        assert response.status_code == 200
        data = response.json()

        assert data["summary"]["total_uploaded"] == 2
        assert data["summary"]["total_analyzed"] == 1
        assert data["summary"]["total_failed"] == 1

        good_entry = next(r for r in data["results"] if r["filename"] == "good.txt")
        corrupt_entry = next(r for r in data["results"] if r["filename"] == "corrupted.pdf")
        assert good_entry["status"] == "analyzed"
        assert corrupt_entry["status"] == "failed"
        assert corrupt_entry["error"] is not None

    def test_unsupported_file_extension_rejected(self):
        files = [
            ("files", ("image.png", b"fake image bytes", "image/png")),
        ]
        response = client.post(
            "/resume/bulk-analyze",
            files=files,
            data={"job_description": self.JD},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["summary"]["total_failed"] == 1
        assert "Unsupported format" in data["results"][0]["error"]

    def test_empty_job_description_returns_400(self):
        files = [
            ("files", ("cand.txt", make_txt_bytes("Python dev"), "text/plain")),
        ]
        response = client.post(
            "/resume/bulk-analyze",
            files=files,
            data={"job_description": "   "},
        )
        assert response.status_code == 400


# ---------------------------------------------------------------------------
# Integration Tests — Rank & CSV Export Endpoints
# ---------------------------------------------------------------------------

class TestRankAndExportAPI:
    """Test /rank and /export/csv endpoints."""

    def test_rank_endpoint(self):
        payload = {
            "results": [
                {"status": "analyzed", "candidate_name": "Cand A", "match_score": 0.85, "skills_match_count": 3, "experience_years": 5},
                {"status": "analyzed", "candidate_name": "Cand B", "match_score": 0.60, "skills_match_count": 2, "experience_years": 2},
            ],
            "top_n": 1,
        }
        response = client.post("/rank", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert len(data["ranked"]) == 1
        assert data["ranked"][0]["candidate_name"] == "Cand A"

    def test_export_csv_endpoint(self):
        ranked = [
            {
                "rank": 1,
                "candidate_name": "Top Candidate",
                "match_score": 0.95,
                "required_matched": ["python", "fastapi"],
                "required_missing": [],
                "preferred_matched": ["docker"],
                "preferred_missing": [],
                "experience_years": 6,
                "education": [{"degree": "Master's", "field": "CS"}],
            }
        ]
        response = client.post("/export/csv", json={"ranked": ranked})
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/csv")
        csv_text = response.text
        assert "Top Candidate" in csv_text
        assert "95" in csv_text
        assert "Professional Experience" in csv_text
        assert "Freelance Experience" in csv_text
        assert "python, fastapi" in csv_text
