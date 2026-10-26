"""Unit tests for the experience extraction and classification module (app/experience.py)."""

import datetime
import pytest

from app.experience import (
    DateInterval,
    ExperienceSummary,
    classify_bucket,
    extract_experience,
    format_duration,
    merge_intervals,
    parse_date_endpoints,
)

FIXED_NOW = datetime.date(2026, 10, 6)


class TestDateParsingAndIntervals:
    """Test date parsing and date endpoint conversion."""

    def test_month_year_to_month_year(self):
        iv = parse_date_endpoints("Jan 2021", "Mar 2023", now=FIXED_NOW)
        assert iv is not None
        assert iv.start_year == 2021
        assert iv.start_month == 1
        assert iv.end_year == 2023
        assert iv.end_month == 3

    def test_mm_yyyy_to_mm_yyyy(self):
        iv = parse_date_endpoints("01/2021", "03/2023", now=FIXED_NOW)
        assert iv is not None
        assert iv.start_year == 2021
        assert iv.start_month == 1
        assert iv.end_year == 2023
        assert iv.end_month == 3

    def test_year_to_year(self):
        iv = parse_date_endpoints("2021", "2023", now=FIXED_NOW)
        assert iv is not None
        assert iv.start_year == 2021
        assert iv.end_year == 2023

    def test_present_handling(self):
        iv = parse_date_endpoints("Jan 2024", "Present", now=FIXED_NOW)
        assert iv is not None
        assert iv.start_year == 2024
        assert iv.start_month == 1
        assert iv.end_year == 2026
        assert iv.end_month == 10

    def test_current_now_synonyms(self):
        for term in ["Current", "now", "till date", "ongoing"]:
            iv = parse_date_endpoints("2025", term, now=FIXED_NOW)
            assert iv is not None
            assert iv.end_year == 2026
            assert iv.end_month == 10


class TestIntervalMergingAndOverlap:
    """Test interval merging to avoid double-counting overlapping dates."""

    def test_non_overlapping_intervals(self):
        iv1 = DateInterval(2020, 1, 2020, 12)  # 12 mos
        iv2 = DateInterval(2022, 1, 2022, 12)  # 12 mos
        total = merge_intervals([iv1, iv2])
        assert total == 24

    def test_fully_overlapping_intervals(self):
        iv1 = DateInterval(2020, 1, 2022, 12)  # 36 mos
        iv2 = DateInterval(2020, 6, 2021, 6)   # subset
        total = merge_intervals([iv1, iv2])
        assert total == 36

    def test_partially_overlapping_intervals(self):
        iv1 = DateInterval(2020, 1, 2021, 6)   # Jan 2020 to Jun 2021 (18 mos)
        iv2 = DateInterval(2021, 1, 2021, 12)  # Jan 2021 to Dec 2021
        # Combined: Jan 2020 to Dec 2021 = 24 mos
        total = merge_intervals([iv1, iv2])
        assert total == 24


class TestClassification:
    """Test professional vs freelance classification."""

    def test_freelance_keywords(self):
        cases = [
            "Freelance Python Developer | Upwork",
            "Freelancer on Fiverr",
            "Self-employed consultant",
            "Independent Contractor for Client X",
        ]
        for c in cases:
            bucket, is_intern, conf = classify_bucket(c)
            assert bucket == "freelance"

    def test_professional_roles(self):
        cases = [
            "Senior Backend Engineer | TechCorp Inc.",
            "Software Engineer at Acme LLC",
            "Principal Developer, CloudScale Technologies",
        ]
        for c in cases:
            bucket, is_intern, conf = classify_bucket(c)
            assert bucket == "professional"
            assert is_intern is False

    def test_internship_classification(self):
        bucket, is_intern, conf = classify_bucket("Software Engineering Intern | Google")
        assert bucket == "professional"
        assert is_intern is True


class TestDurationFormatting:
    """Test format_duration string representation."""

    def test_zero_months(self):
        assert format_duration(0) == "None"

    def test_only_months(self):
        assert format_duration(6) == "6 mos"
        assert format_duration(1) == "1 mo"

    def test_exact_years(self):
        assert format_duration(12) == "1 yr"
        assert format_duration(24) == "2 yrs"

    def test_years_and_months(self):
        assert format_duration(28) == "2 yrs 4 mos"
        assert format_duration(13) == "1 yr 1 mo"


class TestFullResumeExperienceExtraction:
    """Test realistic scenarios end-to-end."""

    def test_resume_with_professional_and_freelance(self):
        text = """
        John Doe
        WORK EXPERIENCE
        Senior Engineer | TechCorp Inc. | Jan 2022 - Dec 2023
        Freelance Web Developer | Upwork | Jan 2021 - Dec 2021
        """
        summary = extract_experience(text, now=FIXED_NOW)
        assert summary.professional_str == "2 yrs"
        assert summary.freelance_str == "1 yr"
        assert summary.combined_str == "3 yrs"
        assert summary.has_any_experience is True

    def test_resume_with_freelance_only(self):
        text = """
        Jane Smith
        EXPERIENCE
        Freelancer Full-Stack Developer | Fiverr | Jan 2021 - Dec 2023
        """
        summary = extract_experience(text, now=FIXED_NOW)
        assert summary.professional_str == "None"
        assert summary.freelance_str == "3 yrs"
        assert summary.has_any_experience is True

    def test_fresher_with_internships_only(self):
        text = """
        Fresh Graduate
        EXPERIENCE
        Software Engineering Intern | TechStart | Jun 2023 - Aug 2023
        """
        summary = extract_experience(text, now=FIXED_NOW)
        assert summary.professional_str == "3 mos"
        assert summary.freelance_str == "None"
        assert summary.has_internships is True
        assert summary.internship_note == "Includes internship experience"

    def test_resume_with_no_experience_at_all(self):
        text = """
        Alex Johnson
        SUMMARY
        Passionate coder looking for first job.
        SKILLS
        Python, Git
        EDUCATION
        B.S. in Computer Science
        """
        summary = extract_experience(text, now=FIXED_NOW)
        assert summary.professional_str == "Not indicated"
        assert summary.freelance_str == "Not indicated"
        assert summary.has_any_experience is False

    def test_fallback_explicit_statement_only(self):
        text = """
        Bob Wilson
        SUMMARY
        Over 5 years of professional experience in distributed systems.
        SKILLS
        Python, Docker
        """
        summary = extract_experience(text, now=FIXED_NOW)
        assert summary.professional_str == "5 yrs"
        assert summary.freelance_str == "None"
        assert summary.has_any_experience is True


class TestExperienceWeaknessFixes:
    """Tests reproducing and verifying fixes for experience extraction weaknesses."""

    def test_date_range_on_own_line_takes_title_and_company_from_previous_lines(self):
        """PyMuPDF splits title, company, and date across separate lines.
        Before fix:
            title = 'Jan 2021 - Dec 2023', company = '' (date line itself used as title)
        After fix:
            title = 'Senior Software Engineer', company = 'Google'
        Months before fix: 36 mos, Months after fix: 36 mos.
        """
        text = """
        Hania Shafiq
        EXPERIENCE
        Senior Software Engineer
        Google
        Jan 2021 - Dec 2023
        """
        summary = extract_experience(text, now=FIXED_NOW)
        assert len(summary.entries) == 1
        assert summary.entries[0].title == "Senior Software Engineer"
        assert summary.entries[0].company == "Google"
        assert summary.professional_months == 36

    def test_date_range_on_own_line_freelance_bucket(self):
        """When date is on its own line below freelance role and platform, bucket must be freelance.
        Before fix:
            professional_months = 24, freelance_months = 0 (classified as professional because line only had dates)
        After fix:
            professional_months = 0, freelance_months = 24
        """
        text = """
        EXPERIENCE
        Freelance Full-Stack Developer
        Upwork
        Jan 2021 - Dec 2022
        """
        summary = extract_experience(text, now=FIXED_NOW)
        assert summary.professional_months == 0
        assert summary.freelance_months == 24
        assert summary.entries[0].title == "Freelance Full-Stack Developer"
        assert summary.entries[0].company == "Upwork"

    def test_education_dates_same_line_degree_ignored(self):
        """Dates next to degree or university words must not count as professional experience.
        Before fix:
            professional_months = 48 (2018 - 2022 counted as professional experience)
        After fix:
            professional_months = 0
        """
        text = """
        EDUCATION
        B.S. in Computer Science | 2018 - 2022
        FAST National University
        """
        summary = extract_experience(text, now=FIXED_NOW)
        assert summary.professional_months == 0
        assert summary.has_any_experience is False

    def test_education_section_dates_ignored(self):
        """Date ranges inside an Education section must not count as professional experience.
        Before fix:
            professional_months = 60 (36 education months + 24 work months merged)
        After fix:
            professional_months = 24 (only professional work experience counted)
        """
        text = """
        EDUCATION
        Master of Science in Software Engineering
        Stanford University
        2018 - 2020

        WORK EXPERIENCE
        Software Engineer
        Acme Corp
        2022 - 2023
        """
        summary = extract_experience(text, now=FIXED_NOW)
        assert summary.professional_months == 24
        assert summary.professional_str == "2 yrs"

    def test_since_year_format(self):
        """'Since 2022' format should resolve from Jan 2022 to the current month (Oct 2026).
        Before fix:
            professional_months = 0 ('Since 2022' was not matched)
        After fix:
            professional_months = 58 (Jan 2022 through Oct 2026 inclusive = 4 yrs 10 mos)
        """
        text = """
        EXPERIENCE
        Lead Architect | Cloud Systems | Since 2022
        """
        summary = extract_experience(text, now=FIXED_NOW)
        assert summary.professional_months == 58
        assert summary.professional_str == "4 yrs 10 mos"

    def test_date_formats_and_synonyms(self):
        """Test 'Jan 2021 - Present', '01/2021 - 03/2023', '2021 - 2023', and 'Current', 'Now'.
        Before fix:
            'Since 2022' failed; 'Now' and 'Current' with single years were incomplete.
        After fix:
            All formats properly resolve to appropriate month counts.
        """
        # 01/2021 - 03/2023: Jan 2021 to Mar 2023 = 27 months
        # Before fix: 27, After fix: 27
        t1 = "EXPERIENCE\nDev | TechCorp | 01/2021 - 03/2023"
        s1 = extract_experience(t1, now=FIXED_NOW)
        assert s1.professional_months == 27

        # Jan 2021 - Present: Jan 2021 to Oct 2026 = 70 months
        # Before fix: 70, After fix: 70
        t2 = "EXPERIENCE\nDev | TechCorp | Jan 2021 - Present"
        s2 = extract_experience(t2, now=FIXED_NOW)
        assert s2.professional_months == 70

        # 2021 - 2023: Jan 2021 to Dec 2023 = 36 months
        # Before fix: 36, After fix: 36
        t3 = "EXPERIENCE\nDev | TechCorp | 2021 - 2023"
        s3 = extract_experience(t3, now=FIXED_NOW)
        assert s3.professional_months == 36

        # Jan 2024 - Now: Jan 2024 to Oct 2026 = 34 months
        # Before fix: 34, After fix: 34
        t4 = "EXPERIENCE\nDev | TechCorp | Jan 2024 - Now"
        s4 = extract_experience(t4, now=FIXED_NOW)
        assert s4.professional_months == 34

        # 2024 - Current: Jan 2024 to Oct 2026 = 34 months
        # Before fix: 34, After fix: 34
        t5 = "EXPERIENCE\nDev | TechCorp | 2024 - Current"
        s5 = extract_experience(t5, now=FIXED_NOW)
        assert s5.professional_months == 34

    def test_mixed_heterogeneous_date_formats(self):
        """Test heterogeneous date range formats (Month-Year to Year, MM/YYYY to Month-Year, Year to Month-Year)."""
        # MM/YYYY to Month-Year: 01/2021 to Dec 2023 = 36 months
        t1 = "EXPERIENCE\nSoftware Engineer at TechCorp\n01/2021 - Dec 2023"
        s1 = extract_experience(t1, now=FIXED_NOW)
        assert s1.professional_months == 36

        # Year to Month-Year: 2020 to Dec 2022 = 36 months
        t2 = "EXPERIENCE\nSoftware Engineer at TechCorp\n2020 - Dec 2022"
        s2 = extract_experience(t2, now=FIXED_NOW)
        assert s2.professional_months == 36

        # Month-Year to Year: Jan 2021 to 2023 = 36 months
        t3 = "EXPERIENCE\nSoftware Engineer at TechCorp\nJan 2021 - 2023"
        s3 = extract_experience(t3, now=FIXED_NOW)
        assert s3.professional_months == 36


