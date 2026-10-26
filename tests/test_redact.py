"""Unit tests for the blind-screening anonymizer (app/redact.py)."""

from __future__ import annotations

import json
import pytest

from app.redact import anonymize, find_residual_flags
from app.skills import extract_skills


def test_empty_text():
    """Anonymizing empty text or None returns empty text with 0 redactions."""
    res_empty = anonymize("")
    assert res_empty["text"] == ""
    assert res_empty["redactions"] == []
    assert res_empty["audit"] == []
    assert res_empty["residual_flags"] == []

    res_spaces = anonymize("    \n   ")
    assert res_spaces["text"].strip() == ""
    assert res_spaces["redactions"] == []


def test_nothing_to_redact():
    """Text with purely technical descriptions and no PII remains unchanged."""
    text = (
        "backend software developer\n"
        "skills: python, fastapi, docker, kubernetes, postgresql\n"
        "responsible for designing distributed systems and optimizing queries"
    )
    result = anonymize(text)
    assert result["text"] == text
    assert len(result["redactions"]) == 0
    assert len(result["audit"]) == 0
    assert len(result["residual_flags"]) == 0


def test_idempotence():
    """Running anonymize() a second time on already-anonymized text changes nothing."""
    sample = (
        "Farhan Ahmed\n"
        "Backend Developer\n"
        "Email: farhan.ahmed@example.com\n"
        "Phone: +92 300 1234567\n"
        "GitHub: https://github.com/farhan-ahmed\n"
        "Education: Bachelor of Science in Computer Science, 2020\n"
        "Experience: Jan 2021 - Present: Software Engineer\n"
        "He maintained the microservices architecture using Python and Docker. His work was exemplary."
    )
    r1 = anonymize(sample)
    r2 = anonymize(r1["text"])

    assert r2["text"] == r1["text"]
    assert len(r2["redactions"]) == 0
    assert len(r2["audit"]) == 0


def test_skills_preserved():
    """Technical skills remain untouched in the anonymized text."""
    sample = (
        "Johnathan Vance\n"
        "Senior Cloud Engineer\n"
        "Skills: Python, FastAPI, Docker, Kubernetes, PostgreSQL, AWS, Git, CI/CD\n"
        "Built serverless pipelines with Python and deployed onto AWS with Docker."
    )
    res = anonymize(sample)
    output = res["text"]

    extracted = extract_skills(output)
    assert "python" in extracted
    assert "fastapi" in extracted
    assert "docker" in extracted
    assert "kubernetes" in extracted
    assert "postgresql" in extracted
    assert "aws" in extracted


def test_work_dates_preserved():
    """Professional work history dates are kept untouched."""
    sample = (
        "Aisha Patel\n"
        "Full Stack Developer\n"
        "Experience:\n"
        "Software Engineer | ABC Tech | Jan 2021 - Present\n"
        "Junior Developer | XYZ Corp | March 2018 - December 2020\n"
        "Freelance Developer | 2017 - 2018\n"
        "Education:\n"
        "BS Computer Science, 2017"
    )
    res = anonymize(sample)
    output = res["text"]

    assert "Jan 2021 - Present" in output
    assert "March 2018 - December 2020" in output
    assert "2017 - 2018" in output
    # But graduation year is redacted
    assert "BS Computer Science, [YEAR]" in output


def test_unicode_names():
    """Unicode and accented candidate names are redacted."""
    sample1 = (
        "José García\n"
        "Software Engineer\n"
        "Email: jose@example.com"
    )
    r1 = anonymize(sample1)
    assert "José García" not in r1["text"]
    assert "[NAME]" in r1["text"]

    sample2 = (
        "Aïsha Patel\n"
        "Data Scientist\n"
        "Email: aisha@example.com"
    )
    r2 = anonymize(sample2)
    assert "Aïsha Patel" not in r2["text"]
    assert "[NAME]" in r2["text"]

    sample3 = (
        "Müller Schmidt\n"
        "DevOps Engineer"
    )
    r3 = anonymize(sample3)
    assert "Müller Schmidt" not in r3["text"]
    assert "[NAME]" in r3["text"]


def test_name_also_skill_word():
    """A candidate name that is also a skill word (Jade or Python) is redacted in header but preserved as skill."""
    # Example 1: Jade
    jade_sample = (
        "Jade\n"
        "Frontend Engineer\n"
        "Skills: Jade, Node.js, Express\n"
        "Experience:\n"
        "Jan 2020 - Present: Full stack development with Jade template engine"
    )
    res_jade = anonymize(jade_sample)
    out_jade = res_jade["text"]

    lines = out_jade.splitlines()
    assert lines[0] == "[NAME]"
    assert "Frontend Engineer" in out_jade
    assert "Skills: Jade, Node.js, Express" in out_jade
    assert "development with Jade" in out_jade

    # Example 2: Python
    python_sample = (
        "Python\n"
        "Backend Developer\n"
        "Skills: Python, FastAPI\n"
        "Experience:\n"
        "Jan 2021 - Present: Built microservices using Python"
    )
    res_python = anonymize(python_sample)
    out_python = res_python["text"]

    py_lines = out_python.splitlines()
    assert py_lines[0] == "[NAME]"
    assert "Skills: Python, FastAPI" in out_python
    assert "using Python" in out_python

    # Verify skill extraction still extracts python
    skills = extract_skills(out_python)
    assert "python" in skills
    assert "fastapi" in skills


def test_pronoun_grammar_his_becomes_their():
    """Pronoun grammar replaces 'his' with 'their', 'he' with 'they', 'him' with 'them', etc."""
    sample = (
        "Summary:\n"
        "His leadership was instrumental to his team. He is an agile practitioner.\n"
        "Her technical guidance inspired her colleagues. We promoted him last year.\n"
        "He was responsible for deployment, and he has managed three releases himself."
    )
    res = anonymize(sample)
    output = res["text"]

    # 'His' -> 'Their'
    assert "Their leadership" in output
    assert "their team" in output

    # 'He is' -> 'They are'
    assert "They are an agile" in output

    # 'Her technical guidance' -> 'Their technical guidance'
    assert "Their technical guidance" in output
    assert "their colleagues" in output

    # 'promoted him' -> 'promoted them'
    assert "promoted them" in output

    # 'He was' -> 'They were', 'he has' -> 'they have'
    assert "They were responsible" in output
    assert "they have managed" in output

    # 'himself' -> 'themselves'
    assert "themselves" in output

    # No gendered pronouns remaining
    residual_pronouns = find_residual_flags(output)
    assert not any(f["type"] == "pronoun" for f in residual_pronouns)


def test_graduation_year_redacted_work_date_kept():
    """Education graduation years are redacted while 'Jan 2021 - Present' is preserved."""
    sample = (
        "Alex Morgan\n"
        "Experience:\n"
        "Senior Developer, Jan 2021 - Present\n"
        "Software Engineer, 2018 - 2020\n"
        "Education:\n"
        "Bachelor of Science in Software Engineering, 2018\n"
        "Class of 2014 - High School Diploma"
    )
    res = anonymize(sample)
    output = res["text"]

    assert "Jan 2021 - Present" in output
    assert "2018 - 2020" in output
    assert "Bachelor of Science in Software Engineering, [YEAR]" in output
    assert "Class of [YEAR] - High School Diploma" in output


def test_audit_never_contains_original_value():
    """The audit trail must NEVER store or log original PII values."""
    secret_name = "Khadija Tariq"
    secret_email = "khadija.secret99@testcorp.org"
    secret_phone = "+92 333 7654321"
    secret_link = "https://linkedin.com/in/khadija-secret-profile"
    secret_year = "2016"

    sample = (
        f"{secret_name}\n"
        f"Email: {secret_email}\n"
        f"Phone: {secret_phone}\n"
        f"LinkedIn: {secret_link}\n"
        f"Education: Master of Science, {secret_year}"
    )
    res = anonymize(sample)
    audit = res["audit"]

    # Verify audit is non-empty
    assert len(audit) > 0

    # Ensure every audit entry has required keys
    for entry in audit:
        assert "type" in entry
        assert "placeholder" in entry
        assert "line" in entry
        assert "rule" in entry

        # Verify no secret value leaked into any audit field
        entry_str = json.dumps(entry)
        assert secret_name not in entry_str
        assert secret_email not in entry_str
        assert secret_phone not in entry_str
        assert "khadija" not in entry_str.lower()
        assert secret_link not in entry_str
        assert secret_year not in entry_str


def test_pakistani_cv_fields():
    """Pakistani CV specific personal cue fields are completely redacted."""
    sample = (
        "Muhammad Usman\n"
        "Father's Name: Muhammad Taj\n"
        "CNIC: 42101-9876543-1\n"
        "Domicile: Sindh (Urban)\n"
        "Nationality: Pakistani\n"
        "Religion: Islam\n"
        "Date of Birth: 12-10-1994\n"
        "Marital Status: Married\n"
        "Gender: Male\n"
        "[Passport size photograph attached]"
    )
    res = anonymize(sample)
    output = res["text"]

    assert "Muhammad Taj" not in output
    assert "42101-9876543-1" not in output
    assert "Sindh (Urban)" not in output
    assert "Pakistani" not in output
    assert "Islam" not in output
    assert "12-10-1994" not in output
    assert "Married" not in output
    assert "[PERSONAL]" in output


def test_options_switching():
    """Every redaction type can be individually switched off via options."""
    sample = (
        "Jane Doe\n"
        "Email: jane@example.com\n"
        "Phone: +92 300 1234567\n"
        "Location: Karachi, Pakistan"
    )
    # Turn off email and location redaction
    opts = {"email": False, "location": False}
    res = anonymize(sample, options=opts)
    output = res["text"]

    # Email and location are NOT redacted
    assert "jane@example.com" in output
    assert "Karachi" in output
    assert "Pakistan" in output

    # Name and phone ARE redacted
    assert "Jane Doe" not in output
    assert "+92 300 1234567" not in output
    assert "[NAME]" in output
    assert "[PHONE]" in output


def test_residual_flags():
    """Second pass residual_flags identifies unredacted PII or title case in top lines."""
    # Text with unredacted phone and email
    text = (
        "Unknown Candidate\n"
        "Contact: user@sample.com\n"
        "Call: +92 300 9999999\n"
        "Education:\n"
        "BS CS 2019"
    )
    flags = find_residual_flags(text)
    flag_types = {f["type"] for f in flags}

    assert "email" in flag_types
    assert "phone" in flag_types
    assert "year" in flag_types
    assert "name" in flag_types
