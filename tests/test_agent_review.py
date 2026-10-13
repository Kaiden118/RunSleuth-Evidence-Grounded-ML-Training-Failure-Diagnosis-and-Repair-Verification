"""Tests for reviewed correctness labels of agent scripts; no model and no training."""

import json
import tempfile
import unittest
from pathlib import Path

from runsleuth import agent_review, agent_tasks
from runsleuth.agent_tasks import HEALTHY

REPOSITORY = Path(__file__).resolve().parents[1]
HEALTHY_SOURCE = agent_tasks.task_source(HEALTHY)
REWRITTEN = HEALTHY_SOURCE.replace(
    "hospital not in (VALIDATION_HOSPITAL, TEST_HOSPITAL)",
    "hospital != VALIDATION_HOSPITAL and hospital != TEST_HOSPITAL",
)
SLOWER = HEALTHY_SOURCE.replace("LEARNING_RATE = 1e-4", "LEARNING_RATE = 5e-5")


def episode(final, correct=None, basis="needs_review", **fields):
    """The part of an episode that labelling reads, with what a reviewer must not see."""
    return {
        "final_script": final,
        "semantically_correct": correct,
        "semantic_basis": basis,
        "repair_task": "validation_hospital_in_training",
        "claimed_correct": True,
        "verification": {"decision": "accepted"},
        **fields,
    }


def review(source, correct, **fields):
    return {
        "program_sha256": agent_review.program_sha256(source),
        "correct": correct,
        "reason": "Read against the task statement.",
        "reviewer": "a reviewer",
        "blind_to_verifier": True,
        **fields,
    }


class AgentReviewTests(unittest.TestCase):
    def reviews(self, *items):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviews.json"
            path.write_text(json.dumps({"reviews": list(items)}), encoding="utf-8")
            return agent_review.load_reviews(path)

    def test_a_program_hash_ignores_comments_and_layout_but_not_the_code(self):
        commented = REWRITTEN.replace(
            "    return hospital", "    # both are held out\n\n    return hospital"
        )
        self.assertNotEqual(commented, REWRITTEN)
        self.assertEqual(
            agent_review.program_sha256(commented), agent_review.program_sha256(REWRITTEN)
        )
        self.assertNotEqual(
            agent_review.program_sha256(REWRITTEN), agent_review.program_sha256(HEALTHY_SOURCE)
        )
        self.assertIsNone(agent_review.program_sha256("def broken(:"))

    def test_a_review_labels_only_what_the_source_left_open(self):
        reviews = self.reviews(review(REWRITTEN, True), review(HEALTHY_SOURCE, False))
        self.assertEqual(agent_review.label(episode(REWRITTEN), reviews), (True, "reviewed"))
        self.assertEqual(agent_review.label(episode(SLOWER), reviews), (None, "needs_review"))
        decided = episode(HEALTHY_SOURCE, True, "same_program_as_healthy_script")
        self.assertEqual(
            agent_review.label(decided, reviews), (True, "same_program_as_healthy_script")
        )
        broken = episode("def broken(:")
        self.assertEqual(agent_review.label(broken, reviews), (None, "needs_review"))

    def test_a_reviewer_sees_each_program_once_and_only_its_difference(self):
        episodes = [
            episode(REWRITTEN),
            episode(REWRITTEN, repair_task=HEALTHY, claimed_correct=False),
            episode(SLOWER),
            episode(HEALTHY_SOURCE, True, "same_program_as_healthy_script"),
        ]
        programs = agent_review.pending(episodes, {})
        self.assertEqual([entry["episodes"] for entry in programs], [2, 1])
        self.assertEqual(set(programs[0]), {"program_sha256", "episodes", "diff"})
        self.assertEqual(
            [line for line in programs[1]["diff"] if line[0] in "+-"],
            ["-LEARNING_RATE = 1e-4", "+LEARNING_RATE = 5e-5"],
        )
        text = agent_review.pending_text(programs)
        self.assertIn(agent_review.program_sha256(SLOWER), text)
        for hidden in ("validation_hospital_in_training", "accepted", "claimed"):
            self.assertNotIn(hidden, text)
        left = agent_review.pending(episodes, self.reviews(review(REWRITTEN, True)))
        self.assertEqual([entry["episodes"] for entry in left], [1])
        self.assertEqual(agent_review.pending_text([]), "No script needs a review.")

    def test_a_malformed_review_is_refused(self):
        with self.assertRaisesRegex(ValueError, "reviewed twice"):
            self.reviews(review(REWRITTEN, True), review(REWRITTEN, False))
        with self.assertRaisesRegex(ValueError, "correct .* must be true or false"):
            self.reviews(review(REWRITTEN, "yes"))
        with self.assertRaisesRegex(ValueError, "blind_to_verifier .* must be true or false"):
            self.reviews(review(REWRITTEN, True, blind_to_verifier=None))
        with self.assertRaisesRegex(ValueError, "has no reason"):
            self.reviews(review(REWRITTEN, True, reason=" "))

    def test_the_reviews_of_the_repository_load_and_explain_themselves(self):
        # The study's reviews are made blind, after its scripts exist; the pilots' are apart.
        self.assertEqual(agent_review.load_reviews(REPOSITORY / agent_review.REVIEWS_PATH), {})
        pilot = REPOSITORY / "evaluations" / "pilot_script_reviews.json"
        reviews = agent_review.load_reviews(pilot)
        self.assertEqual(len(reviews), 4)
        self.assertEqual({item["blind_to_verifier"] for item in reviews.values()}, {False})
        self.assertIn("development pilots", json.loads(pilot.read_text("utf-8"))["scope"])
        for program, item in reviews.items():
            self.assertRegex(program, r"^[0-9a-f]{64}$")
            self.assertTrue(any(line.startswith("+") for line in item["diff"]))
            self.assertGreater(len(item["reason"]), 40)


if __name__ == "__main__":
    unittest.main()
