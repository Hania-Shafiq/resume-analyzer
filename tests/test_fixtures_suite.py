"""End-to-end tests running the actual 27 fixture files through the bulk pipeline."""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.ranker import rank_candidates

client = TestClient(app)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "resumes"


class TestFixturesBatch:
    """Run all 27 generated fixture resumes through bulk-analyze."""

    JD = (
        "Position: Senior Full-Stack Engineer\n"
        "Requirements:\n"
        "- Python, FastAPI, Docker, PostgreSQL\n"
        "- Experience with React and TypeScript\n"
        "Preferred:\n"
        "- Kubernetes, AWS, Redis, CI/CD\n"
        "Experience: 4+ years of professional experience."
    )

    def test_all_fixtures_batch_upload(self):
        """Upload all 27 fixture files simultaneously in one batch."""
        fixture_files = list(FIXTURES_DIR.glob("*"))
        assert len(fixture_files) >= 27

        files_payload = []
        for fp in fixture_files:
            mime = "application/pdf" if fp.suffix == ".pdf" else (
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document" if fp.suffix == ".docx"
                else "text/plain"
            )
            files_payload.append(
                ("files", (fp.name, fp.read_bytes(), mime))
            )

        resp = client.post(
            "/resume/bulk-analyze",
            files=files_payload,
            data={"job_description": self.JD},
        )
        assert resp.status_code == 200
        data = resp.json()

        summary = data["summary"]
        results = data["results"]

        # Verification of summary counts
        assert summary["total_uploaded"] == len(fixture_files)
        # Should have successfully analyzed most valid resumes
        assert summary["total_analyzed"] >= 19
        # Corrupted / empty / wrong_type must fail gracefully
        assert summary["total_failed"] >= 3
        # Duplicate must be detected
        assert summary["total_duplicates"] >= 1

        # Check edge case files individually:
        empty_entry = next((r for r in results if r["filename"] == "empty.pdf"), None)
        assert empty_entry is not None
        assert empty_entry["status"] == "failed"
        assert "no extractable text" in empty_entry["error"].lower() or "empty" in empty_entry["error"].lower()

        corrupt_entry = next((r for r in results if r["filename"] == "corrupted.pdf"), None)
        assert corrupt_entry is not None
        assert corrupt_entry["status"] == "failed"

        wrong_type_entry = next((r for r in results if r["filename"] == "wrong_type.jpg"), None)
        assert wrong_type_entry is not None
        assert wrong_type_entry["status"] == "failed"
        assert "unsupported" in wrong_type_entry["error"].lower()

        # One of duplicate_of_001.pdf or resume_001_aisha_patel.pdf must be marked duplicate
        dup_entry = next((r for r in results if r["filename"] == "duplicate_of_001.pdf"), None)
        orig_entry = next((r for r in results if r["filename"] == "resume_001_aisha_patel.pdf"), None)
        assert dup_entry is not None and orig_entry is not None
        assert dup_entry["status"] == "duplicate" or orig_entry["status"] == "duplicate"

        # Check ranking Top 5 vs Top 10
        ranked_top5 = rank_candidates(results, top_n=5)
        assert len(ranked_top5["ranked"]) == 5
        assert ranked_top5["ranked"][0]["rank"] == 1
        assert ranked_top5["ranked"][0]["match_score"] >= ranked_top5["ranked"][1]["match_score"]

        ranked_top10 = rank_candidates(results, top_n=10)
        assert len(ranked_top10["ranked"]) == 10
        # Order of top 5 must match between top 5 and top 10 slices
        for i in range(5):
            assert ranked_top5["ranked"][i]["candidate_name"] == ranked_top10["ranked"][i]["candidate_name"]

    def test_edge_case_100_files_scale(self):
        """Simulate 100 files by duplicating valid resumes."""
        sample_file = FIXTURES_DIR / "resume_017_alex_murphy.txt"
        content = sample_file.read_text(encoding="utf-8")

        # 100 unique candidate texts
        files_payload = [
            ("files", (f"applicant_{i:03d}.txt", f"Candidate {i}\nSkills: Python, FastAPI\n{content}".encode("utf-8"), "text/plain"))
            for i in range(100)
        ]

        resp = client.post(
            "/resume/bulk-analyze",
            files=files_payload,
            data={"job_description": self.JD},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["summary"]["total_uploaded"] == 100
        assert data["summary"]["total_analyzed"] == 100

        # Shortlist Top 20
        ranked = rank_candidates(data["results"], top_n=20)
        assert len(ranked["ranked"]) == 20
        assert ranked["top_n_returned"] == 20
