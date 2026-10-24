"""CPU tests for judging the verifiers: decisions, reviewer prompts and scores, on
candidates of the tiny network generated, run and labelled as test_candidates does."""

import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

# The modules, not their classes: a TestCase imported here would run here as well.
import test_candidates
import test_llm_reviewer
import test_pipeline_tasks
from test_pipeline_workspace import write_project

from runsleuth import pipeline_tasks
from runsleuth.pipeline_tasks import HEALTHY

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

if torch is not None:
    from runsleuth import candidates, challenge_set, pipeline_contract, verifier_evaluation

OPEN_GATE = {"id": 1.0, "ood": 1.0}


class RateTests(unittest.TestCase):
    def test_rates_have_their_denominators(self):
        rows = [("wrong", "accept")] * 2 + [("wrong", "reject"), ("wrong", "review")]
        rows += [("correct", "accept")] * 5 + [("correct", "reject")]
        rated = verifier_evaluation.rates(rows)
        self.assertEqual(rated["table"]["wrong"], {"accept": 2, "review": 1, "reject": 1})
        self.assertEqual(rated["false_acceptance"], {"count": 2, "of": 4, "rate": 0.5})
        self.assertEqual(rated["wrong_among_accepted"], {"count": 2, "of": 7, "rate": 0.2857})
        self.assertEqual(rated["correct_acceptance"]["of"], 6)
        self.assertEqual(rated["false_rejection"]["count"], 1)
        self.assertEqual(rated["review_share"], {"count": 1, "of": 10, "rate": 0.1})
        self.assertEqual(rated["automatic_coverage"]["count"], 9)
        self.assertIsNone(
            verifier_evaluation.rates([("correct", "accept")])["false_acceptance"]["rate"]
        )

    def test_a_reviewer_judging_its_makers_candidates_is_scored_apart(self):
        def candidate(model, origin="agent"):
            return challenge_set.Candidate(
                "id",
                "camelyon_pipeline",
                "restart",
                "optimizer_not_restored",
                "test",
                origin,
                "repair",
                model=model,
            )

        groups = verifier_evaluation.vendor_groups("llm_reviewer/gemini-4-argon")
        same, other = groups["reviewer_same_vendor"], groups["reviewer_other_vendor"]
        self.assertTrue(same(candidate("gemini-3.5-flash-lite"), None))
        self.assertTrue(other(candidate("gpt-6.1-sol"), None))
        self.assertFalse(
            same(candidate(None, "human"), None) or other(candidate(None, "human"), None)
        )
        combined = verifier_evaluation.vendor_groups("runsleuth_v2+llm_reviewer/claude-sonnet-5.5")
        self.assertFalse(combined["reviewer_same_vendor"](candidate("muse-spark-1.3"), None))
        self.assertEqual(verifier_evaluation.vendor_groups("runsleuth_v1"), {})
        for reviewer in challenge_set.REVIEWERS:
            self.assertNotIn(reviewer, challenge_set.MODELS)
            self.assertIn(reviewer, challenge_set.VENDORS)

    def test_v2_and_the_reviewer_combine_by_the_rule_fixed_before_the_test(self):
        v2 = {"a": "reject", "b": "accept", "c": "accept", "d": "review", "e": "accept"}
        reviewer = {"a": "accept", "b": "accept", "c": "reject", "d": "accept"}
        self.assertEqual(
            verifier_evaluation.combined(v2, reviewer),
            {"a": "reject", "b": "accept", "c": "review", "d": "review"},
        )
        cost = verifier_evaluation.v2_cost(
            [{"runs": 2, "steps": 10, "seconds": 30.0}, {"runs": 0, "steps": 0, "seconds": 0.0}]
        )
        self.assertEqual((cost["runs_per_candidate"], cost["seconds_per_candidate"]), (1.0, 15.0))

    def test_an_evidence_removed_candidate_is_judged_on_its_parent_run_without_its_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_report.json"
            path.write_text(
                '{"status": "completed", "pipeline": {"attempts": [1], "epochs": [2]}}',
                encoding="utf-8",
            )
            runs = {"runs": {"parent": {"status": "completed", "report": str(path)}}}
            candidate = challenge_set.Candidate(
                "parent/no_restart_record",
                "camelyon_pipeline",
                "restart",
                "optimizer_not_restored",
                "test",
                "program_transform",
                "evidence_removed",
                transform="no_restart_record",
                parent="parent",
            )
            record = verifier_evaluation.candidate_run(candidate, runs)
        self.assertEqual(record["pipeline"], {"epochs": [2]})
        refused = verifier_evaluation.run_record({"status": "refused", "problems": ["os"]})
        self.assertEqual(refused["status"], "refused")


@unittest.skipIf(torch is None, "Install torch and torchvision for the pipeline tests")
class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.root = Path(temporary.name)
        patchers = (
            patch("runsleuth.pipeline_tasks.version_sources", test_pipeline_tasks.tiny_version),
            patch.object(candidates, "candidates", lambda _split: test_candidates.subset()),
            # The tiny network is not the family the gate was calibrated on.
            patch.object(verifier_evaluation.metric_gate, "load_gate", lambda _f: OPEN_GATE),
        )
        for patcher in patchers:
            patcher.start()
            cls.addClassCleanup(patcher.stop)
        root = cls.root / "candidates"
        candidates.generate(
            "development",
            "gemini-3.5-flash-lite",
            test_candidates.FakeAgent(),
            "fake",
            root=root,
            workspaces=cls.root / "workspaces",
            runner=candidates.Runner(
                cls.root / "short",
                data=test_candidates.SYNTHETIC,
                run=test_candidates.run_in_process,
            ),
            progress=lambda _line: None,
        )
        runner = candidates.Runner(
            cls.root / "runs", data=test_candidates.SYNTHETIC, run=test_candidates.run_in_process
        )
        candidates.run_all("development", root=root, runner=runner, progress=lambda _: None)
        healthy = write_project(cls.root / "healthy", pipeline_tasks.version_sources(HEALTHY))

        def contract(project, family):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                return pipeline_contract.run_contract(project, family, healthy)

        candidates.label_all("development", root=root, contract=contract, progress=lambda _: None)
        cls.candidates_root = root

    def test_the_keyless_verifiers_decide_every_candidate_with_a_run(self):
        decided = verifier_evaluation.decide("development", self.candidates_root)
        for name in verifier_evaluation.KEYLESS:
            self.assertEqual(len(decided["verifiers"][name]), 6, name)
        v1 = decided["verifiers"]["runsleuth_v1"]
        for candidate_id, decision in v1.items():
            expected = "reject" if "positions_index_base" in candidate_id else "accept"
            self.assertEqual(decision, expected, candidate_id)
        self.assertTrue((self.candidates_root / "development" / "decisions.json").exists())

    def test_reviewer_prompts_hold_no_label_and_scores_count_every_verifier(self):
        items = verifier_evaluation.review_items("development", self.candidates_root)
        self.assertEqual(len(items), 6)
        for item in items:
            for word in ("failed_clauses", "partially_tested", "contract", "positions_index_base"):
                self.assertNotIn(word, item["prompt"])
        client = test_llm_reviewer.FakeClient([test_llm_reviewer.reply()] * len(items))
        llm_reviewer = verifier_evaluation.llm_reviewer
        llm_reviewer.review_all(
            items,
            verifier_evaluation.reviews_dir("development", "fake", self.candidates_root),
            client,
            "fake",
            provider="fake",
            progress=lambda _line: None,
        )
        verifier_evaluation.decide("development", self.candidates_root)
        scored = verifier_evaluation.score("development", self.candidates_root, reviewers=("fake",))
        self.assertEqual(scored["labels"], {"wrong": 3, "correct": 3})
        reviewer = scored["verifiers"]["llm_reviewer/fake"]["groups"]["all"]
        self.assertEqual(reviewer["false_acceptance"], {"count": 3, "of": 3, "rate": 1.0})
        v1 = scored["verifiers"]["runsleuth_v1"]
        self.assertEqual(v1["groups"]["all"]["false_acceptance"]["count"], 0)
        self.assertEqual(v1["groups"]["agent"]["false_acceptance"]["of"], 2)
        self.assertEqual(v1["cases"]["camelyon_pipeline/positions_index_base"], {"wrong reject": 3})
        text = verifier_evaluation.score_text(scored)
        self.assertIn("llm_reviewer/fake", text)
        self.assertIn("decisions by case", text)


if __name__ == "__main__":
    unittest.main()
