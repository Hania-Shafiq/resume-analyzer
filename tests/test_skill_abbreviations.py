"""Regression test suite for comprehensive skill abbreviation normalization,
contextual disambiguation, and bidirectional matching (resumes <-> job descriptions).
"""

import pytest
from app.skills import (
    extract_skills,
    extract_skills_with_context,
    normalize_skill,
    analyze_job_description,
)
from app.scoring import compute_scores


class TestSkillNormalization:
    """Category 1: Canonical skill alias normalization."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("FS", "full stack"),
            ("full-stack", "full stack"),
            ("Full Stack", "full stack"),
            ("FE", "frontend"),
            ("front-end", "frontend"),
            ("BE", "backend"),
            ("back-end", "backend"),
            ("JS", "javascript"),
            ("JavaScript", "javascript"),
            ("TS", "typescript"),
            ("TypeScript", "typescript"),
            ("React", "react"),
            ("React.js", "react"),
            ("reactjs", "react"),
            ("Node", "node.js"),
            ("Node.js", "node.js"),
            ("nodejs", "node.js"),
            ("DB", "database"),
            ("Database", "database"),
            ("SQL", "sql"),
            ("Structured Query Language", "sql"),
            ("API", "api"),
            ("Application Programming Interface", "api"),
            ("REST", "rest api"),
            ("Representational State Transfer", "rest api"),
            ("UI", "ui"),
            ("User Interface", "ui"),
            ("UX", "ux"),
            ("User Experience", "ux"),
            ("AI", "artificial intelligence"),
            ("Artificial Intelligence", "artificial intelligence"),
            ("ML", "machine learning"),
            ("Machine Learning", "machine learning"),
            ("DL", "deep learning"),
            ("Deep Learning", "deep learning"),
            ("NLP", "natural language processing"),
            ("Natural Language Processing", "natural language processing"),
            ("Git", "git"),
            ("Git (version control)", "git"),
            ("CI/CD", "ci/cd"),
            ("Continuous Integration", "ci/cd"),
            ("Continuous Deployment", "ci/cd"),
            ("K8s", "kubernetes"),
            ("Kubernetes", "kubernetes"),
            ("AWS", "aws"),
            ("Amazon Web Services", "aws"),
            ("Docker", "docker"),
            ("Postgres", "postgresql"),
            ("PostgreSQL", "postgresql"),
        ],
    )
    def test_canonical_alias_mapping(self, raw, expected):
        assert normalize_skill(raw) == expected


class TestBidirectionalMatching:
    """Category 2 & 3: Bidirectional JD <-> Resume matching with case & punctuation variations."""

    def test_abbreviations_in_jd_and_expanded_in_resume(self):
        """JD uses short abbreviations; resume uses fully expanded terms."""
        jd_text = """
        Mid-Level Developer
        Requirements:
        Must have experience with JS, TS, React, Node, Postgres, Docker, K8s, CI/CD, AWS.
        """
        resume_text = """
        Senior Software Engineer
        Experience:
        Architected platforms using JavaScript, TypeScript, React.js, and Node.js.
        Managed PostgreSQL databases, automated deployments with Docker and Kubernetes.
        Implemented continuous integration and continuous deployment pipelines on Amazon Web Services.
        """
        jd_analysis = analyze_job_description(jd_text)
        resume_skills = extract_skills(resume_text)

        scores = compute_scores(
            resume_text=resume_text,
            resume_skills=resume_skills,
            jd_analysis=jd_analysis.to_dict(),
            mode="skills_only",
        )
        assert scores["final_score"] == 1.0
        assert scores["breakdown"]["skill"]["value"] == 1.0

    def test_expanded_in_jd_and_abbreviations_in_resume(self):
        """JD uses expanded formal terms; resume uses industry abbreviations."""
        jd_text = """
        Full Stack Engineer Position
        Required Skills:
        - JavaScript and TypeScript
        - React.js and Node.js
        - PostgreSQL database
        - Docker and Kubernetes orchestration
        - Continuous Integration / Continuous Deployment
        - Amazon Web Services
        """
        resume_text = """
        Full Stack Developer
        Skills: JS, TS, React, Node, Postgres, DB, Docker, K8s, CI/CD, AWS.
        """
        jd_analysis = analyze_job_description(jd_text)
        resume_skills = extract_skills(resume_text)

        scores = compute_scores(
            resume_text=resume_text,
            resume_skills=resume_skills,
            jd_analysis=jd_analysis.to_dict(),
            mode="skills_only",
        )
        req_set = set(jd_analysis.required_skills)
        matched = set(resume_skills) & req_set
        assert req_set.issubset(matched)


class TestDuplicateAndDeduplication:
    """Category 4: Duplicate aliases without double-counting."""

    def test_deduplication_of_abbreviation_and_expanded_term(self):
        """Mentioning both 'JS' and 'JavaScript' counts as 1 skill, not 2."""
        text = "Skilled in JS, JavaScript, javascript es6, and ecmascript."
        skills = extract_skills(text)
        assert skills == ["javascript"]
        assert len(skills) == 1

    def test_deduplication_across_tech_stack(self):
        text = """
        Full Stack (FS) Developer specializing in Frontend (FE) and Backend (BE).
        Extensive background in Postgres, PostgreSQL, K8s, Kubernetes, Docker, Node, and Node.js.
        """
        skills = extract_skills(text)
        assert len(skills) == len(set(skills))
        assert skills.count("full stack") == 1
        assert skills.count("frontend") == 1
        assert skills.count("backend") == 1
        assert skills.count("postgresql") == 1
        assert skills.count("kubernetes") == 1
        assert skills.count("node.js") == 1


class TestFalsePositivePrevention:
    """Category 5: Contextual disambiguation to prevent false positives."""

    def test_auxiliary_verb_be_rejected(self):
        """The English verb 'be' ('must be', 'will be', 'to be') must NOT match Backend."""
        sentences = [
            "The candidate must be able to communicate effectively.",
            "He will be responsible for leading sprints.",
            "You should be comfortable with agile processes.",
            "To be considered, apply online.",
        ]
        for s in sentences:
            skills = extract_skills(s)
            assert "backend" not in skills, f"False positive 'backend' extracted from '{s}'"

    def test_academic_degree_be_rejected(self):
        """B.E. degree must NOT match Backend."""
        text = "Graduated with a B.E. in Electrical Engineering from NED University."
        skills = extract_skills(text)
        assert "backend" not in skills

    def test_be_developer_accepted(self):
        """'BE developer' or 'FE/BE' must correctly match backend."""
        text = "Hiring a senior BE Developer with strong Python skills."
        skills = extract_skills(text)
        assert "backend" in skills

    def test_milliliters_rejected_for_ml(self):
        """'500 ml' volume measurement must NOT match Machine Learning."""
        text = "Added 500 ml of buffer solution and 250ml of reagent to the vessel."
        skills = extract_skills(text)
        assert "machine learning" not in skills

    def test_machine_learning_ml_accepted(self):
        """'ML Engineer' or 'Python, ML' must match Machine Learning."""
        text = "Senior ML Engineer with experience building PyTorch models."
        skills = extract_skills(text)
        assert "machine learning" in skills

    def test_decibels_rejected_for_db(self):
        """'40 dB' sound level must NOT match Database."""
        text = "Equipment acoustic noise emission was certified below 40 dB at 1 meter."
        skills = extract_skills(text)
        assert "database" not in skills

    def test_database_db_accepted(self):
        """'DB Administrator' or 'SQL DB' must match Database."""
        text = "Managed relational DB systems with PostgreSQL and MySQL."
        skills = extract_skills(text)
        assert "database" in skills

    def test_node_fs_module_rejected(self):
        """'fs module' in Node.js must NOT match Full Stack."""
        text = "Used the node fs module to asynchronously stream data."
        skills = extract_skills(text)
        assert "full stack" not in skills

    def test_graph_node_rejected(self):
        """'node in the tree' or 'DOM node' must NOT match Node.js."""
        text = "Iterated over each DOM node and traversed the binary tree node structure."
        skills = extract_skills(text)
        assert "node.js" not in skills

    def test_node_backend_accepted(self):
        """'Node developer' or 'Node, Express' must match Node.js."""
        text = "Full Stack Engineer with 3 years of Node, Express, and React experience."
        skills = extract_skills(text)
        assert "node.js" in skills


class TestCompoundSkillsAndSeparation:
    """Category 6: Compound skills and technology boundary separation."""

    def test_rest_api_combination(self):
        """REST API matches 'rest api' and recognizes 'api', but 'API' alone never matches 'rest api'."""
        rest_api_text = "Designed and maintained REST APIs for high-throughput microservices."
        rest_skills = extract_skills(rest_api_text)
        assert "rest api" in rest_skills
        assert "api" in rest_skills

        generic_api_text = "Integrated third-party payment APIs and webhook consumers."
        generic_skills = extract_skills(generic_api_text)
        assert "api" in generic_skills
        assert "rest api" not in generic_skills

    def test_ai_does_not_award_ml_dl_nlp(self):
        """Mentioning AI alone grants 'artificial intelligence' but NOT ML, DL, or NLP."""
        text = "Passionate about the future of Artificial Intelligence (AI) and prompt design."
        skills = extract_skills(text)
        assert "artificial intelligence" in skills
        assert "machine learning" not in skills
        assert "deep learning" not in skills
        assert "natural language processing" not in skills

    def test_git_distinct_from_other_vcs(self):
        """Git is recognized distinctly without conflating with other version control tools."""
        text = "Version control with Git and GitHub."
        skills = extract_skills(text)
        assert "git" in skills

    def test_postgresql_distinct_from_sql(self):
        """PostgreSQL does NOT automatically award SQL unless SQL is explicitly stated."""
        text = "Maintained PostgreSQL database cluster with replication."
        skills = extract_skills(text)
        assert "postgresql" in skills
        assert "sql" not in skills

        text_with_both = "Wrote complex SQL queries for PostgreSQL and MySQL databases."
        skills_both = extract_skills(text_with_both)
        assert "postgresql" in skills_both
        assert "sql" in skills_both


class TestScoringBackwardCompatibility:
    """Category 7: Scoring behavior and backward compatibility."""

    def test_scoring_weights_preserved(self):
        """Verify that SCORING_CONFIG weights and compute_scores API contracts remain intact."""
        from app.scoring import SCORING_CONFIG
        assert SCORING_CONFIG["skills_only"]["skill"] == 1.0
        assert SCORING_CONFIG["composite"]["skill"] == 0.40
        assert SCORING_CONFIG["composite"]["semantic"] == 0.30
        assert SCORING_CONFIG["composite"]["experience"] == 0.15
        assert SCORING_CONFIG["composite"]["recency"] == 0.15

    def test_evidence_preservation_with_context(self):
        """Original sentence context is preserved as evidence for candidate skills."""
        text = "Experienced with React and Node.js in high-load production environments."
        res = extract_skills_with_context(text)
        assert "react" in res
        assert "node.js" in res
        assert "React" in res["react"]["evidence"]
        assert res["react"]["confidence"] == "high"
