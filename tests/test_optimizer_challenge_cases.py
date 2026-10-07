"""Challenge separation, fixed semantics and independent real-CPU label checks."""

import ast
import hashlib
import importlib.util
import unittest
from collections import Counter
from copy import deepcopy

from runsleuth.optimizer_challenge_cases import challenge_cases, challenge_sha256, verify_known_case
from runsleuth.optimizer_factory_analysis import analyze_optimizer_factory
from runsleuth.optimizer_factory_cases import factory_cases


class ChallengeCaseTests(unittest.TestCase):
    def test_twelve_balanced_unique_cases_have_stable_hashes_and_parse_in_py311(self):
        cases = challenge_cases()
        self.assertEqual(
            Counter(row["expected"] for row in cases), {"stale": 4, "current": 4, "inconclusive": 4}
        )
        self.assertEqual(len({case["id"] for case in cases}), 12)
        self.assertEqual(len({case["source"] for case in cases}), 12)
        for case in cases:
            with self.subTest(case=case["id"]):
                ast.parse(case["source"], feature_version=(3, 11))
                self.assertEqual(
                    case["source_sha256"], hashlib.sha256(case["source"].encode()).hexdigest()
                )
        self.assertEqual(challenge_sha256(), challenge_sha256())

    def test_not_a_reused_development_source_or_exact_ast(self):
        prior = {
            ast.dump(ast.parse(case["source"]), include_attributes=False)
            for case in factory_cases()
        }
        for case in challenge_cases():
            with self.subTest(case=case["id"]):
                self.assertNotIn(
                    ast.dump(ast.parse(case["source"]), include_attributes=False), prior
                )

    def test_labels_do_not_come_from_the_current_analyzer(self):
        rows = challenge_cases()
        refused_known = [
            row
            for row in rows
            if row["expected"] != "inconclusive"
            and analyze_optimizer_factory(row["source"])["decision"] == "unsupported"
        ]
        self.assertEqual(len(refused_known), 4)
        self.assertEqual({row["expected"] for row in refused_known}, {"stale", "current"})

    def test_caller_mutations_do_not_change_canonical_sources_or_labels(self):
        before = challenge_sha256()
        case = challenge_cases()[0]
        case["source"] = "raise RuntimeError()"
        self.assertEqual(challenge_sha256(), before)
        with self.assertRaises(ValueError):
            verify_known_case(case, {})

    def test_underspecified_cases_are_refused_before_any_torch_import_or_execution(self):
        for case in challenge_cases():
            if case["expected"] == "inconclusive":
                with self.subTest(case=case["id"]), self.assertRaises(ValueError):
                    verify_known_case(case, {})

    def test_expected_label_cannot_be_changed_to_authorize_execution(self):
        case = deepcopy(challenge_cases()[2])
        case["expected"] = "current"
        with self.assertRaises(ValueError):
            verify_known_case(case, {})

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Real CPU oracle requires torch")
    def test_all_eight_known_labels_match_real_membership_gradients_and_updates(self):
        total = 0
        for case in challenge_cases():
            if case["expected"] == "inconclusive":
                continue
            with self.subTest(case=case["id"]):
                receipt = {}
                verify_known_case(case, receipt)
                self.assertEqual(receipt["status"], "verified")
                self.assertTrue(all(receipt["checks"].values()))
                self.assertTrue(receipt["step_accounting_complete"])
                self.assertEqual(receipt["device"], "cpu")
                total += receipt["optimizer_steps_recorded"]
        self.assertEqual(total, 8)


if __name__ == "__main__":
    unittest.main()
