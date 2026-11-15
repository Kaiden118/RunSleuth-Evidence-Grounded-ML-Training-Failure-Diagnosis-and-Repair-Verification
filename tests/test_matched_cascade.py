"""Tests for v2 then an LLM reviewer: how the decisions combine and how they are counted."""

import unittest

from runsleuth import matched_cascade

V2 = {"w1": "reject", "w2": "review", "c1": "accept", "c2": "review", "c3": "review"}
LABELS = {"w1": "wrong", "w2": "wrong", "c1": "correct", "c2": "correct", "c3": "correct"}


class CascadeTests(unittest.TestCase):
    def test_the_reviewer_decides_only_what_v2_sends_to_review(self):
        reviewer = {"w2": "reject", "c2": "accept", "c1": "reject"}
        together = matched_cascade.combined(V2, reviewer)
        self.assertEqual(
            together,
            {"w1": "reject", "w2": "reject", "c1": "accept", "c2": "accept", "c3": "review"},
        )

    def test_a_failed_call_decides_nothing(self):
        records = [
            {"item": "c2", "status": "completed", "decision": "accept"},
            {"item": "c3", "status": "api_error", "decision": None},
            {"item": "w2", "status": "invalid_output", "decision": "review"},
        ]
        self.assertEqual(matched_cascade.decisions(records), {"c2": "accept", "w2": "review"})

    def test_the_score_sets_three_verifiers_side_by_side(self):
        table = matched_cascade.score(
            V2, {"w2": "accept", "c2": "accept", "c3": "reject"}, LABELS, {"w2", "c2", "c3"}
        )
        self.assertEqual(
            table["v2 alone"]["all"],
            {
                "wrong_accepted": (0, 2),
                "correct_accepted": (1, 3),
                "correct_rejected": (0, 3),
                "review": (3, 5),
            },
        )
        self.assertEqual(
            table["reviewer on v2's review set"]["all"],
            {
                "wrong_accepted": (1, 1),
                "correct_accepted": (1, 2),
                "correct_rejected": (1, 2),
                "review": (0, 3),
            },
        )
        self.assertEqual(
            table["v2 then the reviewer"]["agents"],
            {
                "wrong_accepted": (1, 1),
                "correct_accepted": (1, 2),
                "correct_rejected": (1, 2),
                "review": (0, 3),
            },
        )
        text = matched_cascade.score_text(table, [])
        rows = {" ".join(line.split()[:-4]): line.split()[-4:] for line in text.splitlines()[1:-1]}
        self.assertEqual(rows["v2 alone all"], ["0/2", "1/3", "0/3", "3/5"])
        self.assertEqual(rows["v2 then the reviewer all"], ["1/2", "2/3", "1/3", "0/5"])
        self.assertIn("0 reviews, 0 calls: 0 input and 0 output tokens", text)


if __name__ == "__main__":
    unittest.main()
