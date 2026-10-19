"""Unit tests for scripts/suggest_skills.py candidate skill suggestion tool."""

import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.suggest_skills import (
    clean_chunk_text,
    is_valid_candidate,
    load_taxonomy_terms,
    suggest_skills,
)

DEFAULT_SKILLS_JSON = Path(__file__).resolve().parent.parent / "app" / "data" / "skills.json"


class TestSuggestSkills(unittest.TestCase):
    """Tests for scripts/suggest_skills.py."""

    def test_suggest_skills_from_folder_creates_csv(self):
        """Test scanning small sample files extracts candidates and writes candidates.csv."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # File 1
            f1 = temp_path / "resume1.txt"
            f1.write_text(
                "Experience building observability pipelines with OpenTelemetry and Jaeger.\n"
                "Used Python for backend services.",
                encoding="utf-8",
            )

            # File 2
            f2 = temp_path / "resume2.txt"
            f2.write_text(
                "Implemented distributed tracing using OpenTelemetry across Kubernetes clusters.\n"
                "Proficient in Docker and FastAPI.",
                encoding="utf-8",
            )

            candidates = suggest_skills(temp_path)
            csv_path = temp_path / "candidates.csv"

            # 1. Assert candidates.csv was created
            self.assertTrue(csv_path.exists(), "candidates.csv was not created.")

            # 2. Assert CSV format: term, count, example_sentence
            with open(csv_path, "r", encoding="utf-8") as fh:
                reader = csv.reader(fh)
                header = next(reader)
                self.assertEqual(header, ["term", "count", "example_sentence"])
                rows = list(reader)
                self.assertTrue(len(rows) > 0, "candidates.csv has no rows.")

            # 3. Assert OpenTelemetry was identified as a candidate
            candidate_terms = [c["term"].lower() for c in candidates]
            self.assertIn("opentelemetry", candidate_terms)

            # 4. Check example sentence
            otel_entry = next(c for c in candidates if c["term"].lower() == "opentelemetry")
            self.assertTrue(len(otel_entry["example_sentence"]) > 0)
            self.assertIn("OpenTelemetry", otel_entry["example_sentence"])

    def test_ranking_by_frequency_across_files(self):
        """Test candidate ranking prioritizes terms appearing across more files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Term A ("ApacheFlink") appears in 3 files
            # Term B ("TemporalWorkflow") appears in 1 file
            (temp_path / "doc1.txt").write_text(
                "Engineered streaming pipelines using ApacheFlink and Kafka.\n"
                "Evaluated TemporalWorkflow for orchestrating tasks.",
                encoding="utf-8",
            )
            (temp_path / "doc2.txt").write_text(
                "Scaled ApacheFlink stateful stream processing clusters on AWS.\n"
                "Python and SQL experience.",
                encoding="utf-8",
            )
            (temp_path / "doc3.txt").write_text(
                "Maintained low-latency ApacheFlink real-time analytics.\n"
                "Designed PostgreSQL schemas.",
                encoding="utf-8",
            )

            candidates = suggest_skills(temp_path)

            terms_by_rank = [c["term"].lower() for c in candidates]
            self.assertIn("apacheflink", terms_by_rank)
            self.assertIn("temporalworkflow", terms_by_rank)

            # ApacheFlink must rank before TemporalWorkflow due to frequency across 3 files vs 1
            idx_flink = terms_by_rank.index("apacheflink")
            idx_temporal = terms_by_rank.index("temporalworkflow")
            self.assertLess(
                idx_flink,
                idx_temporal,
                msg="ApacheFlink (3 files) should rank higher than TemporalWorkflow (1 file).",
            )

            flink_entry = next(c for c in candidates if c["term"].lower() == "apacheflink")
            self.assertEqual(flink_entry["count"], 3)

    def test_excludes_existing_taxonomy_and_never_modifies_skills_json(self):
        """Test existing taxonomy skills are excluded and skills.json is NEVER edited."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            sample_text = (
                "Skills: Python, Docker, FastAPI, PostgreSQL, Kubernetes, React, Redis.\n"
                "Also developed game prototypes using UnrealEngine."
            )
            (temp_path / "resume.txt").write_text(sample_text, encoding="utf-8")

            # Check skills.json state before execution
            self.assertTrue(DEFAULT_SKILLS_JSON.exists())
            before_bytes = DEFAULT_SKILLS_JSON.read_bytes()
            before_hash = hashlib.sha256(before_bytes).hexdigest()
            before_mtime = DEFAULT_SKILLS_JSON.stat().st_mtime_ns

            candidates = suggest_skills(temp_path)

            # Check skills.json state after execution
            after_bytes = DEFAULT_SKILLS_JSON.read_bytes()
            after_hash = hashlib.sha256(after_bytes).hexdigest()
            after_mtime = DEFAULT_SKILLS_JSON.stat().st_mtime_ns

            # 1. Assert skills.json was never modified
            self.assertEqual(
                before_hash,
                after_hash,
                "CRITICAL: skills.json was modified! The script must NEVER edit skills.json.",
            )
            self.assertEqual(
                before_mtime,
                after_mtime,
                "skills.json file modification timestamp changed.",
            )

            # 2. Assert known taxonomy skills are NOT present in suggestions
            candidate_terms = [c["term"].lower() for c in candidates]
            for known_skill in ["python", "docker", "fastapi", "postgresql", "kubernetes", "react", "redis"]:
                self.assertNotIn(
                    known_skill,
                    candidate_terms,
                    f"Known taxonomy skill '{known_skill}' should be excluded from candidates.",
                )

            # 3. Assert untracked term "UnrealEngine" was extracted
            self.assertIn("unrealengine", candidate_terms)


if __name__ == "__main__":
    unittest.main(verbosity=2)
