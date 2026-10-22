"""CPU tests for the verification challenge set: its plan, splits, labels and partial repairs."""

import json
import tempfile
import unittest
import warnings
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from test_pipeline_tasks import tiny

from runsleuth import pipeline_tasks
from runsleuth.pipeline_tasks import HEALTHY, SPLITS

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

if torch is not None:
    from runsleuth import challenge_set, pipeline_contract
    from runsleuth.challenge_set import FAMILIES, PARTIAL_REPAIRS, plan, split_of

SEEDS = (11111, 22222, 33333)


@unittest.skipIf(torch is None, "Install torch and torchvision for the challenge set tests")
class PlanTests(unittest.TestCase):
    def test_the_splits_have_the_agreed_candidates(self):
        test = Counter(candidate.kind for candidate in plan("test", SEEDS))
        self.assertEqual(
            test, {"repair": 216, "version": 13, "partial_repair": 6, "evidence_removed": 26}
        )
        development = Counter(candidate.kind for candidate in plan("development"))
        self.assertEqual(development, {"repair": 36, "version": 12})
        text = challenge_set.plan_text(plan("test", SEEDS))
        self.assertTrue(text.startswith("261 candidates"))

    def test_agents_start_from_every_fault_and_variant_but_the_control(self):
        for split in SPLITS:
            points = challenge_set.starting_points(split)
            kinds = Counter()
            for family, name in points:
                item = pipeline_tasks.version(name, family)
                self.assertEqual(item.split, split)
                self.assertFalse(item.control)
                kinds[(family, item.task, item.correct)] += 1
            # One fault and one correct variant of each of the six tasks.
            self.assertEqual(len(kinds), 12)
            self.assertEqual(set(kinds.values()), {1})
        with self.assertRaises(ValueError):
            challenge_set.starting_points("validation")

    def test_everything_follows_the_split_of_its_starting_version(self):
        for split in SPLITS:
            candidates = plan(split, SEEDS if split == "test" else None)
            by_id = {candidate.id: candidate for candidate in candidates}
            self.assertEqual(len(by_id), len(candidates))
            for candidate in candidates:
                self.assertEqual(split_of(candidate.start, candidate.family), split)
                self.assertEqual(candidate.split, split)
                self.assertIn(candidate.origin, challenge_set.ORIGINS)
                if candidate.parent:
                    self.assertEqual(by_id[candidate.parent].kind, "version")
                    self.assertEqual(by_id[candidate.parent].start, candidate.start)

    def test_development_runs_on_seed_7_and_test_runs_on_one_drawn_seed_per_repetition(self):
        development = plan("development")
        self.assertEqual({candidate.seed for candidate in development}, {7})
        self.assertEqual(
            {candidate.model for candidate in development if candidate.origin == "agent"},
            {"gemini-3.5-flash-lite"},
        )
        test = plan("test", SEEDS)
        agents = [candidate for candidate in test if candidate.origin == "agent"]
        self.assertEqual({candidate.model for candidate in agents}, set(challenge_set.MODELS))
        seeds = {(candidate.repetition, candidate.seed) for candidate in agents}
        self.assertEqual(seeds, {(1, SEEDS[0]), (2, SEEDS[1]), (3, SEEDS[2])})
        others = {candidate.seed for candidate in test if candidate.origin != "agent"}
        self.assertEqual(others, {SEEDS[0]})
        self.assertTrue(all(candidate.seed is None for candidate in plan("test")))
        with self.assertRaises(ValueError):
            plan("test", SEEDS[:2])

    def test_the_healthy_project_is_development_and_contract_cases_are_in_no_split(self):
        for family in FAMILIES:
            self.assertEqual(split_of(HEALTHY, family), "development")
            for name in pipeline_contract.CASES[family]:
                with self.assertRaises(ValueError):
                    split_of(name, family)

    def test_test_seeds_are_drawn_once_from_seeds_never_used(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seeds.json"
            draws = [7, 42, 5, 5, 6, 8]
            with patch.object(challenge_set.secrets, "randbelow", side_effect=draws):
                seeds = challenge_set.draw_test_seeds(path)
            self.assertEqual(seeds, [5, 6, 8])
            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(record["seeds"], seeds)
            self.assertIn(7, record["excluded_seeds"])
            with self.assertRaises(FileExistsError):
                challenge_set.draw_test_seeds(path)


@unittest.skipIf(torch is None, "Install torch and torchvision for the challenge set tests")
class LabelTests(unittest.TestCase):
    FAMILY = "camelyon_pipeline"

    def result(self, status, failed=(), clauses=None, **extra):
        return {
            "family": self.FAMILY,
            "contract_version": pipeline_contract.CONTRACT_VERSION,
            "family_sha256": pipeline_tasks.family_sha256(self.FAMILY),
            "status": status,
            "failed": list(failed),
            "clauses": clauses or {},
            **extra,
        }

    def test_the_hidden_tests_decide_the_label(self):
        quiet = {"delivery": {"passed": True, "detail": ""}}
        passed = challenge_set.label(self.result("passed", clauses=quiet), self.FAMILY)
        self.assertEqual((passed["label"], passed["outcome"]), ("correct", "passed"))
        self.assertFalse(passed["partially_tested"])
        noted = {
            "delivery": {"passed": True, "detail": "a best epoch after a restart was not tested"}
        }
        partial = challenge_set.label(self.result("passed", clauses=noted), self.FAMILY)
        self.assertEqual(partial["label"], "correct")
        self.assertTrue(partial["partially_tested"])
        restart = {"restart": {"passed": False, "detail": "trains epoch 2 on other batches"}}
        failed = challenge_set.label(self.result("failed", ["restart"], restart), self.FAMILY)
        self.assertEqual((failed["label"], failed["outcome"]), ("wrong", "failed"))
        self.assertEqual(failed["failed_clauses"], ["restart"])

    def test_a_candidate_that_cannot_run_is_wrong_and_not_runnable(self):
        for status in challenge_set.NOT_RUNNABLE:
            result = self.result(status, ["training_samples"], problems=["import of 'os'"])
            labelled = challenge_set.label(result, self.FAMILY)
            self.assertEqual((labelled["label"], labelled["outcome"]), ("wrong", "not_runnable"))
            self.assertEqual(labelled["failed_clauses"], [])
            self.assertEqual(labelled["contract_status"], status)

    def test_labels_of_another_contract_or_family_state_are_refused(self):
        for changed in (
            {"contract_version": pipeline_contract.CONTRACT_VERSION - 1},
            {"family_sha256": "0" * 64},
            {"family": "ham10000_pipeline"},
        ):
            with self.assertRaises(ValueError):
                challenge_set.label({**self.result("passed"), **changed}, self.FAMILY)
        with self.assertRaises(ValueError):
            challenge_set.label(self.result("unknown"), self.FAMILY)


@unittest.skipIf(torch is None, "Install torch and torchvision for the challenge set tests")
class CandidateTests(unittest.TestCase):
    def test_removing_evidence_keeps_everything_else(self):
        pipeline = {
            "attempts": [1],
            "tracked": [2],
            "checkpoints": [3],
            "training_reads": {"read_while_training": 4},
            "epochs": [5],
            "delivery": {"attempt": 2},
        }
        report = {"status": "completed", "pipeline": pipeline}
        restart = challenge_set.remove_evidence(report, "no_restart_record")
        self.assertEqual(set(restart["pipeline"]), {"training_reads", "epochs", "delivery"})
        reads = challenge_set.remove_evidence(report, "no_data_reads")
        self.assertEqual(set(reads["pipeline"]), {"attempts", "tracked", "checkpoints", "delivery"})
        self.assertEqual(report["pipeline"], pipeline)
        self.assertEqual(restart["status"], "completed")
        with self.assertRaises(ValueError):
            challenge_set.remove_evidence(report, "no_metrics")

    def test_every_partial_repair_mends_a_test_fault_and_fails_what_it_should(self):
        for name, repair in PARTIAL_REPAIRS.items():
            with self.subTest(partial_repair=name):
                fault = pipeline_tasks.version(repair.fault, repair.family)
                self.assertEqual((fault.split, fault.correct), ("test", False))
                self.assertEqual(split_of(name, repair.family), "test")
                healthy = tiny(pipeline_contract.case_sources(HEALTHY, repair.family))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    result = pipeline_contract.contract_result(
                        tiny(challenge_set.partial_sources(name)), repair.family, healthy
                    )
                self.assertEqual(sorted(result["failed"]), sorted(repair.fails), result)

    def test_partial_repairs_are_new_projects(self):
        for family in FAMILIES:
            names = (
                HEALTHY,
                *pipeline_tasks.family(family).names,
                *pipeline_contract.CASES[family],
            )
            known = [pipeline_contract.case_sources(name, family) for name in set(names)]
            for name, repair in PARTIAL_REPAIRS.items():
                if repair.family == family:
                    self.assertNotIn(challenge_set.partial_sources(name), known, name)


if __name__ == "__main__":
    unittest.main()
