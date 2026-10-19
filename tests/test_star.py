"""Unit tests for the rule-based STAR bullet optimizer (app/star.py)."""

import unittest
from app.star import (
    STRONG_ACTION_VERBS,
    split_bullets,
    suggest_star,
    weak_bullet_reasons,
)
from app.skills import skill_category


class TestStarBulletOptimizer(unittest.TestCase):
    """Test suite for STAR bullet optimization, bullet splitting, and weakness detection."""

    # 1. Bullets are split correctly
    def test_bullets_split_correctly(self):
        """Extract bullets from experience section with various bullet markers."""
        text = """
        EXPERIENCE
        • Developed REST APIs using FastAPI and PostgreSQL.
        - Deployed containerized applications to AWS.
        * Maintained CI/CD pipelines across multiple environments.
        """
        bullets = split_bullets(text)
        self.assertEqual(len(bullets), 3)
        self.assertIn("Developed REST APIs using FastAPI and PostgreSQL.", bullets)
        self.assertIn("Deployed containerized applications to AWS.", bullets)
        self.assertIn("Maintained CI/CD pipelines across multiple environments.", bullets)

    # 2. Education lines and contact lines are ignored
    def test_education_and_contact_lines_ignored(self):
        """Ensure Education section, degree lines, and contact headers are excluded."""
        text = """
        John Doe
        john.doe@example.com
        +1 (555) 123-4567

        WORK EXPERIENCE
        • Responsible for deployment of the API.

        EDUCATION
        B.S. in Computer Science | Stanford University
        2018 - 2022
        """
        bullets = split_bullets(text)
        self.assertEqual(len(bullets), 1)
        self.assertEqual(bullets[0], "Responsible for deployment of the API.")

    # 3. 'Responsible for deployment of the API' is flagged
    def test_responsible_for_deployment_flagged(self):
        """'Responsible for deployment of the API' triggers weak opener and missing verb/metric."""
        bullet = "Responsible for deployment of the API."
        reasons = weak_bullet_reasons(bullet)
        self.assertIn("weak_opener:responsible for", reasons)
        self.assertIn("no_action_verb", reasons)
        self.assertIn("no_metric", reasons)
        self.assertIn("no_skill_mentioned", reasons)

    # 4. Weak openers lexicon coverage
    def test_all_weak_openers_detected(self):
        """Verify all required weak openers are correctly identified."""
        openers = [
            ("responsible for", "Responsible for system backups."),
            ("worked on", "Worked on database migrations."),
            ("helped with", "Helped with frontend bug fixes."),
            ("involved in", "Involved in sprint planning meetings."),
            ("duties included", "Duties included writing unit tests."),
            ("assisted in", "Assisted in onboarding new team members."),
            ("participated in", "Participated in daily standups and retrospectives."),
        ]
        for name, bullet in openers:
            reasons = weak_bullet_reasons(bullet)
            self.assertIn(f"weak_opener:{name}", reasons, msg=f"Failed to flag '{name}' in '{bullet}'")

    # 5. A strong bullet with a verb and a metric is NOT suggested
    def test_strong_bullet_with_verb_and_metric_not_suggested(self):
        """A well-crafted bullet starting with a strong action verb and metric is not suggested."""
        text = """
        EXPERIENCE
        • Architected and deployed microservices in Python, reducing latency by 40%.
        • Responsible for deployment of the API.
        """
        suggestions = suggest_star(text)
        self.assertEqual(len(suggestions), 1)
        self.assertEqual(suggestions[0]["original"], "Responsible for deployment of the API.")

    # 6. No digit appears in any suggestion unless it is in the original bullet
    def test_no_digit_unless_in_original(self):
        """Ensure no digit 0-9 appears in generated suggestions when original has no digits."""
        text = """
        EXPERIENCE
        • Responsible for deployment of the API.
        • Worked on backend microservices.
        """
        suggestions = suggest_star(text, missing_skills=["docker"])
        for s in suggestions:
            # Check all fields for prohibited digits
            self.assertFalse(any(c.isdigit() for c in s["star"]["situation"]), "Digit found in situation")
            self.assertFalse(any(c.isdigit() for c in s["star"]["task"]), "Digit found in task")
            self.assertFalse(any(c.isdigit() for c in s["star"]["action"]), "Digit found in action")
            self.assertFalse(any(c.isdigit() for c in s["star"]["result"]), "Digit found in result")
            self.assertFalse(any(c.isdigit() for c in s["rewritten_bullet"]), "Digit found in rewritten_bullet")
            for p in s["placeholders"]:
                self.assertFalse(any(c.isdigit() for c in p), "Digit found in placeholder")

    # 7. Missing-skill suggestions have requires_real_experience=true
    def test_missing_skill_sets_requires_real_experience_true(self):
        """Missing skill suggestions set requires_real_experience=True and conditional template."""
        text = """
        EXPERIENCE
        • Responsible for deployment of the API.
        """
        suggestions = suggest_star(text, missing_skills=["docker"])
        self.assertEqual(len(suggestions), 1)
        s = suggestions[0]
        self.assertEqual(s["skill_status"], "missing")
        self.assertTrue(s["requires_real_experience"])
        self.assertTrue(s["rewritten_bullet"].startswith("If you have used Docker on this work:"))
        self.assertIn("Only include docker if you have genuine hands-on experience", s["honesty_note"])

    # 8. Present skill sets requires_real_experience=false
    def test_present_skill_sets_requires_real_experience_false(self):
        """Present skill suggestions set requires_real_experience=False."""
        text = """
        EXPERIENCE
        • Worked on microservices with Python and FastAPI.
        """
        suggestions = suggest_star(text)
        self.assertEqual(len(suggestions), 1)
        s = suggestions[0]
        self.assertEqual(s["skill_status"], "present")
        self.assertFalse(s["requires_real_experience"])
        self.assertFalse(s["rewritten_bullet"].startswith("If you have used"))
        self.assertEqual(s["target_skill"], "python")

    # 9. No suggestion names a skill outside the taxonomy
    def test_no_suggestion_names_skill_outside_taxonomy(self):
        """Any suggested target_skill must be in skills.json taxonomy."""
        text = """
        EXPERIENCE
        • Responsible for deployment of the API.
        """
        # Pass both a valid taxonomy skill and an invalid fake skill
        suggestions = suggest_star(text, missing_skills=["fake_skill_xyz", "docker"])
        self.assertEqual(len(suggestions), 1)
        target = suggestions[0]["target_skill"]
        self.assertIsNotNone(target)
        self.assertIsNotNone(skill_category(target), f"Skill '{target}' is not in skills.json taxonomy.")
        self.assertEqual(target, "docker")

    # 10. Category matching never suggests an unrelated skill
    def test_category_matching_never_suggests_unrelated_skill(self):
        """Never suggest a missing skill that has no category match with the bullet."""
        text = """
        EXPERIENCE
        • Responsible for deployment of the container infrastructure.
        """
        # Missing skills include an unrelated database skill and an unrelated web skill
        suggestions = suggest_star(text, missing_skills=["react", "sqlite"])
        self.assertEqual(len(suggestions), 1)
        # Bullet is devops/cloud, so react/sqlite should NOT be suggested
        self.assertIsNone(
            suggestions[0]["target_skill"],
            "Unrelated category skill was improperly suggested for a deployment bullet.",
        )
        self.assertEqual(suggestions[0]["skill_status"], "present")

    # 11. Empty and whitespace text
    def test_empty_and_whitespace_text(self):
        """Empty or whitespace-only inputs return empty list."""
        self.assertEqual(suggest_star(""), [])
        self.assertEqual(suggest_star("   \n\t  "), [])
        self.assertEqual(split_bullets(""), [])
        self.assertEqual(split_bullets("   "), [])

    # 12. Unicode text handling
    def test_unicode_text(self):
        """Ensure bullets with unicode bullets and characters are handled cleanly."""
        text = """
        EXPERIENCE
        \u2022 Responsible for internationalization and l10n support.
        \u25aa Worked on backend caching for Zürich and München users.
        """
        bullets = split_bullets(text)
        self.assertEqual(len(bullets), 2)
        suggestions = suggest_star(text)
        self.assertEqual(len(suggestions), 2)
        self.assertIn("Owned", suggestions[0]["rewritten_bullet"])
        self.assertIn("Built", suggestions[1]["rewritten_bullet"])
        self.assertIn("Zürich", suggestions[1]["rewritten_bullet"])

    # 13. Determinism
    def test_determinism(self):
        """Multiple runs with identical inputs produce identical output."""
        text = """
        EXPERIENCE
        • Responsible for deployment of the API.
        • Worked on backend microservices with Python.
        • Assisted in database schema design.
        """
        run1 = suggest_star(text, missing_skills=["docker"])
        run2 = suggest_star(text, missing_skills=["docker"])
        self.assertEqual(run1, run2)

    # 14. Max suggestions respected
    def test_max_suggestions_respected(self):
        """max_suggestions limits the number of returned recommendations."""
        text = """
        EXPERIENCE
        • Responsible for deployment of the API.
        • Worked on backend microservices.
        • Helped with frontend bug fixes.
        • Involved in sprint planning.
        • Duties included server maintenance.
        """
        suggestions = suggest_star(text, max_suggestions=2)
        self.assertEqual(len(suggestions), 2)

    # 15. Sorting by weakness then position
    def test_sorting_by_weakness_then_position(self):
        """Bullets with more weakness reasons rank before bullets with fewer reasons."""
        text = """
        EXPERIENCE
        • Worked on Python microservices for backend systems.
        • Responsible for deployment of the API.
        """
        # Bullet 2 has weak opener + no action verb + no metric + no skill (4 reasons)
        # Bullet 1 has weak opener + no action verb + no metric (3 reasons, has Python skill and >= 6 words)
        suggestions = suggest_star(text)
        self.assertEqual(len(suggestions), 2)
        self.assertEqual(suggestions[0]["original"], "Responsible for deployment of the API.")
        self.assertEqual(suggestions[1]["original"], "Worked on Python microservices for backend systems.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
