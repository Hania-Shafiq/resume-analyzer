"""Unit tests for candidate identity extraction (name, phone, links) and endpoints."""

import pytest
from app.parser import (
    extract_candidate_name,
    extract_phone,
    extract_links,
)


class TestExtractCandidateName:
    """Tests for extract_candidate_name(text, filename) with 8 diverse samples."""

    def test_sample_1_title_case(self):
        text = "Ayesha Khan\nPython Developer\nayesha@example.com\nSkills: Python, FastAPI"
        assert extract_candidate_name(text, "ayesha_resume.pdf") == "Ayesha Khan"

    def test_sample_2_all_caps(self):
        text = "MUHAMMAD USMAN\nSenior DevOps Engineer\nusman@example.com"
        assert extract_candidate_name(text, "candidate.pdf") == "MUHAMMAD USMAN"

    def test_sample_3_first_line_job_title(self):
        text = (
            "Senior Full Stack Software Engineer & Cloud Architect\n"
            "Fatima Noor\n"
            "fatima@example.com\n"
            "Summary: 7 years building scalable systems."
        )
        assert extract_candidate_name(text, "resume.pdf") == "Fatima Noor"

    def test_sample_4_cv_pdf_generic_filename(self):
        text = "Bilal Ahmed\nSoftware Engineer\nbilal@example.com"
        assert extract_candidate_name(text, "cv.pdf") == "Bilal Ahmed"

    def test_sample_5_resume_software_dev_2026_filename(self):
        text = "Zainab Tariq\nData Engineer\nzainab@example.com"
        assert extract_candidate_name(text, "Resume_Software_Dev_2026.pdf") == "Zainab Tariq"

    def test_sample_6_multi_word_name(self):
        text = "Syed Ali Raza Rizvi\nBackend Developer\nali.raza@example.com"
        assert extract_candidate_name(text, "profile.docx") == "Syed Ali Raza Rizvi"

    def test_sample_7_spacy_person_fallback(self):
        # First 5 lines disqualified by section keywords, email, and phone
        text = (
            "RESUME\n"
            "PROFILE\n"
            "contact@example.com\n"
            "+92 300 1234567\n"
            "SUMMARY\n"
            "Dedicated engineer Johnathan Taylor with over 6 years of industry experience."
        )
        assert extract_candidate_name(text, "cv.pdf") == "Johnathan Taylor"

    def test_sample_8_infer_filename_fallback(self):
        # No name in first 5 lines (all disqualified) and no person entity
        text = (
            "RESUME\n"
            "PROFILE\n"
            "EDUCATION\n"
            "SKILLS\n"
            "python, docker, kubernetes, postgresql, aws, linux\n"
            "experience with distributed databases and cloud services."
        )
        assert extract_candidate_name(text, "sarah_connor_resume_2026.pdf") == "Sarah Connor"


class TestExtractPhone:
    """Tests for extract_phone(text) covering Pakistani & intl formats and negatives."""

    def test_pakistani_plus_92_formats(self):
        assert extract_phone("Contact: +92 300 1234567") == "+92 300 1234567"
        assert extract_phone("Phone: +92-321-7654321") == "+92-321-7654321"
        assert extract_phone("Mobile: +923331234567") == "+923331234567"
        assert extract_phone("Cell: +92 (300) 1234567") == "+92 (300) 1234567"

    def test_pakistani_03xx_formats(self):
        assert extract_phone("Contact: 0300-1234567") == "0300-1234567"
        assert extract_phone("Phone: 0321 7654321") == "0321 7654321"
        assert extract_phone("Tel: 03451234567") == "03451234567"
        assert extract_phone("Cell: (0300) 1234567") == "(0300) 1234567"

    def test_international_formats(self):
        assert extract_phone("US Office: +1 (555) 123-4567") == "+1 (555) 123-4567"
        assert extract_phone("UK Cell: +44 20 7946 0958") == "+44 20 7946 0958"
        assert extract_phone("Mobile: +44 7911 123456") == "+44 7911 123456"

    def test_never_return_years(self):
        text = "Graduated high school in 2016 and university in 2020. 2021 - 2024 work."
        assert extract_phone(text) is None

    def test_never_return_dates(self):
        text = "DOB: 1995-10-25. Start date: 15/08/2021. End date: 2023-01-01."
        assert extract_phone(text) is None

    def test_never_return_ids(self):
        text = "National ID / CNIC: 35201-1234567-1. Roll Number: 12345. Pass: 98765432."
        assert extract_phone(text) is None

    def test_empty_and_no_phone(self):
        assert extract_phone("") is None
        assert extract_phone("No contact numbers here, just email@test.com") is None
        assert extract_phone(None) is None

    def test_returns_first_valid_phone(self):
        text = "Call primary +92 300 1234567 or backup 0321-7654321"
        assert extract_phone(text) == "+92 300 1234567"


class TestExtractLinks:
    """Tests for extract_links(text) returning {linkedin, github, other}."""

    def test_links_with_https(self):
        text = (
            "Profile: https://www.linkedin.com/in/hania-shafiq\n"
            "Code: https://github.com/Hania-Shafiq\n"
            "Portfolio: https://haniashafiq.dev"
        )
        links = extract_links(text)
        assert links == {
            "linkedin": "https://www.linkedin.com/in/hania-shafiq",
            "github": "https://github.com/Hania-Shafiq",
            "other": "https://haniashafiq.dev",
        }

    def test_links_with_www(self):
        text = (
            "LinkedIn: www.linkedin.com/in/hania\n"
            "GitHub: www.github.com/hania\n"
            "Blog: www.haniashafiq.com"
        )
        links = extract_links(text)
        assert links == {
            "linkedin": "www.linkedin.com/in/hania",
            "github": "www.github.com/hania",
            "other": "www.haniashafiq.com",
        }

    def test_links_without_https_and_without_www(self):
        text = (
            "Connect: linkedin.com/in/johndoe\n"
            "Projects: github.com/johndoe\n"
            "Site: johndoe.me"
        )
        links = extract_links(text)
        assert links == {
            "linkedin": "linkedin.com/in/johndoe",
            "github": "github.com/johndoe",
            "other": "johndoe.me",
        }

    def test_absent_links(self):
        text = "Only github here: https://github.com/developer"
        links = extract_links(text)
        assert links == {
            "linkedin": None,
            "github": "https://github.com/developer",
            "other": None,
        }

    def test_empty_and_no_links(self):
        assert extract_links("") == {"linkedin": None, "github": None, "other": None}
        assert extract_links(None) == {"linkedin": None, "github": None, "other": None}
        assert extract_links("No links, just pure text") == {
            "linkedin": None,
            "github": None,
            "other": None,
        }

    def test_emails_are_not_links(self):
        text = "Emails: john@linkedin.com and alice@github.com and user@example.com"
        links = extract_links(text)
        assert links == {"linkedin": None, "github": None, "other": None}

    def test_trailing_punctuation_stripped(self):
        text = "Check out (https://github.com/sample)."
        links = extract_links(text)
        assert links["github"] == "https://github.com/sample"


class TestEndpointsIdentity:
    """Tests that /resume/upload and /resume/bulk-analyze integrate identity extraction."""

    @pytest.fixture(autouse=True)
    def setup_client(self):
        from fastapi.testclient import TestClient
        from app.main import app

        self.client = TestClient(app)

    def test_upload_resume_includes_identity_fields(self):
        resume_content = (
            "Hania Shafiq\n"
            "Data Scientist & AI Engineer\n"
            "Email: hania@example.com\n"
            "Phone: +92 300 1234567\n"
            "LinkedIn: https://linkedin.com/in/hania\n"
            "GitHub: https://github.com/hania\n"
            "Education: BS in Computer Science\n"
            "Experience: 5 years in machine learning\n"
            "Skills: Python, FastAPI, Docker, PyTorch"
        ).encode("utf-8")

        response = self.client.post(
            "/resume/upload",
            files={"file": ("resume.txt", resume_content, "text/plain")},
        )
        assert response.status_code == 200
        data = response.json()

        # New fields required by prompt: candidate_name, phone, links
        assert "candidate_name" in data
        assert data["candidate_name"] == "Hania Shafiq"
        assert "phone" in data
        assert data["phone"] == "+92 300 1234567"
        assert "links" in data
        assert data["links"]["linkedin"] == "https://linkedin.com/in/hania"
        assert data["links"]["github"] == "https://github.com/hania"

        # All existing fields stay unchanged
        assert "filename" in data
        assert data["filename"] == "resume.txt"
        assert "email" in data
        assert data["email"] == "hania@example.com"
        assert "char_count" in data
        assert "skills" in data
        assert "education" in data
        assert "experience_years" in data
        assert "text_preview" in data
        assert "extracted_text" in data
        assert "raw_text" in data

    def test_bulk_analyze_uses_extract_candidate_name(self):
        # File is named generic 'cv.pdf' or 'cv.txt', but text contains candidate name
        resume_content = (
            "Usman Tariq\n"
            "Software Developer\n"
            "Email: usman@example.com\n"
            "Skills: Python, Docker, FastAPI\n"
            "Experience: 3 years building web APIs."
        ).encode("utf-8")

        response = self.client.post(
            "/resume/bulk-analyze",
            files=[("files", ("cv.txt", resume_content, "text/plain"))],
            data={"job_description": "Looking for Python, Docker, FastAPI developer."},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["summary"]["total_analyzed"] == 1
        result = data["results"][0]
        assert result["status"] == "analyzed"
        # Previously infer_candidate_name("cv.txt") would have yielded "cv"
        # With extract_candidate_name, it extracts "Usman Tariq"
        assert result["candidate_name"] == "Usman Tariq"





