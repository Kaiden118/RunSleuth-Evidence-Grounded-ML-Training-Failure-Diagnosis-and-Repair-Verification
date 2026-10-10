"""Tests for repair verification: trajectory comparison and the acceptance criteria."""

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

from runsleuth import agent_verification as verification

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def example(name: str) -> dict:
    return json.loads((EXAMPLES / name / "run_report.json").read_text(encoding="utf-8"))


def recorded(status: str = "completed", **changes) -> dict:
    """A run report with the model states of two epochs of three steps, four hashed."""
    workspace = {
        "seed": 7,
        "epochs": 2,
        "steps_per_epoch": 3,
        "max_train_batches": None,
        "hashed_steps": 4,
        "stop_after_epochs": None,
        "initial_state_sha256": "start",
        "step_state_sha256": ["s1", "s2", "s3", "s4"],
        "epoch_state_sha256": ["s3", "s6"],
        "final_state_sha256": "s6",
    }
    return {"status": status, "workspace": {**workspace, **changes}}


class TrajectoryTests(unittest.TestCase):
    def compare(self, **changes) -> dict:
        return verification.compare_trajectories(recorded(**changes), recorded())

    def test_a_full_run_that_matches_every_hashed_state_is_identical(self):
        result = self.compare()
        self.assertEqual(result["outcome"], "identical_full_run")
        self.assertEqual(result["checked"], {"steps_hashed": 4, "epoch_ends": 2, "epochs": 2})
        self.assertEqual(
            verification.trajectory_text(result),
            "identical over the full run (4 hashed steps and 2 of 2 epoch ends, final state)",
        )

    def test_the_first_differing_state_is_named(self):
        cases = {
            "diverged at initial state": {"initial_state_sha256": "other"},
            "diverged at step 2": {"step_state_sha256": ["s1", "x", "y", "z"]},
            # The hashed steps agree; the fault acts later in the run.
            "diverged at epoch 2": {"epoch_state_sha256": ["s3", "x"], "final_state_sha256": "x"},
            "diverged at final state": {"final_state_sha256": "x"},
        }
        for text, changes in cases.items():
            with self.subTest(text=text):
                result = self.compare(**changes)
                self.assertEqual(result["outcome"], "diverged")
                self.assertEqual(verification.trajectory_text(result), text)

    def test_a_run_stopped_early_agrees_only_over_the_checked_steps(self):
        stopped = {
            "stop_after_epochs": 1,
            "step_state_sha256": ["s1", "s2", "s3"],
            "epoch_state_sha256": ["s3"],
            "final_state_sha256": None,
        }
        result = self.compare(**stopped)
        self.assertEqual(result["outcome"], "identical_over_checked_steps")
        self.assertEqual(result["checked"], {"steps_hashed": 3, "epoch_ends": 1, "epochs": 2})
        self.assertIn("later steps are unknown", verification.trajectory_text(result))
        # A fault in the second epoch is invisible to it, and visible to the full run.
        late = {"epoch_state_sha256": ["s3", "x"], "final_state_sha256": "x"}
        reference = recorded()
        self.assertEqual(
            verification.compare_trajectories(recorded(**stopped), recorded(**late))["outcome"],
            "identical_over_checked_steps",
        )
        self.assertEqual(
            verification.compare_trajectories(recorded(**late), reference)["outcome"], "diverged"
        )
        early = self.compare(**{**stopped, "step_state_sha256": ["s1", "s2", "x"]})
        self.assertEqual(early["first_difference"], {"at": "step", "index": 3})

    def test_runs_that_cannot_be_compared_say_why(self):
        cases = {
            "differ in epochs": recorded(epochs=3),
            "differ in seed, hashed_steps": recorded(seed=8, hashed_steps=2),
            "candidate run did not complete": recorded(status="failed"),
            "recorded no model states": {"status": "completed"},
        }
        for reason, candidate in cases.items():
            with self.subTest(reason=reason):
                result = verification.compare_trajectories(candidate, recorded())
                self.assertEqual(result["outcome"], "not_comparable")
                self.assertIn(reason, result["reason"])
                self.assertIn(reason, verification.trajectory_text(result))
        result = verification.compare_trajectories(recorded(), recorded(status="diverged"))
        self.assertIn("reference run did not complete", result["reason"])


class AcceptanceCriteriaTests(unittest.TestCase):
    """On the held-out example runs: a frozen head whose metrics look healthy."""

    def verify(self, candidate, faulty=None) -> dict:
        return verification.verify_repair_run(
            candidate, example("clean"), faulty, optimizer_name="AdamW"
        )

    def test_metrics_accept_an_unrepaired_silent_fault_and_the_mechanism_check_does_not(self):
        faulty = example("frozen_head")
        result = self.verify(copy.deepcopy(faulty), faulty)
        self.assertEqual(result["decision"], "rejected")
        self.assertEqual(
            result["criteria"],
            {
                "before_and_after_metrics": False,
                "reference_metrics": True,
                "mechanism_check": False,
                "mechanism_check_and_reference_metrics": False,
            },
        )

    def test_a_correct_repair_whose_metrics_fall_is_kept_by_the_reference_comparison(self):
        reference = example("clean")
        candidate = example("frozen_head_repaired")
        candidate["final_metrics"].update(
            {key: value for key, value in reference["final_metrics"].items() if "validation" in key}
        )
        # The faulty run looked better than the healthy one, as a leak makes it look.
        inflated = example("frozen_head")
        for split in ("id", "ood"):
            inflated["final_metrics"][f"{split}_validation_accuracy"] = min(
                1.0, reference["final_metrics"][f"{split}_validation_accuracy"] + 0.02
            )
        result = self.verify(candidate, inflated)
        self.assertEqual(result["decision"], "accepted")
        self.assertEqual(
            result["criteria"],
            {
                "before_and_after_metrics": False,
                "reference_metrics": True,
                "mechanism_check": True,
                "mechanism_check_and_reference_metrics": True,
            },
        )

    def test_a_repair_that_removes_the_fault_and_harms_performance_fails_on_metrics(self):
        candidate = example("frozen_head_repaired")
        candidate["final_metrics"]["ood_validation_accuracy"] -= 0.05
        result = self.verify(candidate)
        self.assertEqual(result["decision"], "rejected")
        criteria = result["criteria"]
        self.assertIsNone(criteria["before_and_after_metrics"])
        self.assertTrue(criteria["mechanism_check"])
        self.assertFalse(criteria["reference_metrics"])
        self.assertFalse(criteria["mechanism_check_and_reference_metrics"])

    def test_runs_without_recorded_states_are_verified_without_a_trajectory(self):
        result = self.verify(example("frozen_head_repaired"), example("frozen_head"))
        self.assertEqual(result["decision"], "accepted")
        self.assertEqual(result["trajectory"]["outcome"], "not_comparable")
        text = verification.repair_text(result)
        self.assertIn("Decision: accepted", text)
        self.assertIn("mechanism check and reference metrics: accepts", text)
        self.assertIn("Trajectory against the reference: not comparable", text)

    def test_command_line_exits_with_one_when_the_repair_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = {}
            for name in ("clean", "frozen_head"):
                run = example(name)
                run["optimizer"] = "AdamW"
                paths[name] = Path(temporary) / f"{name}.json"
                paths[name].write_text(json.dumps(run), encoding="utf-8")
            arguments = [
                "--candidate",
                str(paths["frozen_head"]),
                "--reference",
                str(paths["clean"]),
            ]
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = verification.main(arguments)
        self.assertEqual(code, 1)
        self.assertIn("re-diagnosis against the reference found frozen_head", output.getvalue())
        self.assertIn("before and after metrics: not available", output.getvalue())


if __name__ == "__main__":
    unittest.main()
