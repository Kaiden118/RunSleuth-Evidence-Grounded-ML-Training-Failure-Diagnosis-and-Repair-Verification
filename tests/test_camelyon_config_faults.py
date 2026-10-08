"""CPU integration tests for the Camelyon17 configuration-fault runner (tiny model, fake data)."""

import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import runsleuth.camelyon_config_faults as runner
from runsleuth.camelyon_optimizer_probe import read_json

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

if torch is not None:
    from camelyon_harness import CamelyonHarness
else:
    CamelyonHarness = unittest.TestCase


@unittest.skipIf(torch is None, "Install torch and torchvision for CPU integration tests")
class ConfigFaultTrainingTests(CamelyonHarness):
    def run_training(self):
        with self.patched_builders():
            return read_json(
                runner.run_config_fault_training(
                    self.reference, self.output, epochs=2, requested_device="cpu"
                )
            )

    def reassess(self, report, variants=None, plan=None):
        return runner.assess_config_fault_training(
            report["variants"] if variants is None else variants,
            report["reference_initial_state_sha256"],
            2,
            report["checkpoint_comparisons"],
            report["variant_plan"] if plan is None else plan,
        )

    def test_five_variants_reproduce_both_faults_and_verify_repairs(self):
        report = self.run_training()
        self.assertEqual(report["status"], "completed")
        self.assertTrue(report["mechanism_reproduced"], report["checks"])
        self.assertEqual(list(report["variants"]), list(runner.VARIANTS))
        self.assertEqual(report["budget"], {"training_epoch_upper_bound": 10, "model_calls": 0})
        self.assertEqual(report["variant_plan"]["high_learning_rate"]["learning_rate"], 3.0)
        self.assertEqual(len({str(model.training_order) for model in self.models}), 1)
        for fault in runner.FAULTS:
            with self.subTest(fault=fault):
                verification = report["repair_verification"][fault]
                self.assertTrue(verification["structure_verified"], verification)
                self.assertEqual(
                    verification["decision"],
                    "accepted" if verification["performance_nonregression"] else "rejected",
                )
        missing = report["variants"]["missing_optimizer_step"]["parameter_group_epochs"]
        self.assertEqual([row["training_forwards"] for row in missing], [3, 3])
        self.assertEqual([row["optimizer_steps"] for row in missing], [0, 0])
        predictions = report["predictions"]
        self.assertTrue(predictions["missing_step_parameters_bitwise_unchanged"])
        self.assertEqual(
            predictions["repaired_vs_clean"],
            {"high_learning_rate": "bitwise_equal", "missing_optimizer_step": "bitwise_equal"},
        )
        for ratio in predictions["high_learning_rate_first_update_ratio_to_clean"].values():
            self.assertAlmostEqual(ratio, 100, delta=1)
        for variant, learning_rate in (("clean", 0.03), ("high_learning_rate", 3.0)):
            for value in predictions["inferred_first_step_learning_rate"][variant].values():
                self.assertAlmostEqual(value, learning_rate, delta=0.1 * learning_rate)

    def test_diverged_high_learning_rate_is_evidence_and_skips_its_comparator(self):
        report = self.run_training()
        variants = copy.deepcopy(report["variants"])
        high = variants["high_learning_rate"]
        high.update(
            status="diverged",
            completed_epochs=0,
            divergence={"epoch": 1, "message": "non-finite"},
            parameter_group_epochs=[],
        )
        high.pop("final_metrics")
        result = self.reassess(report, variants)
        self.assertTrue(result["mechanism_reproduced"], result["checks"])
        verification = result["repair_verification"]["high_learning_rate"]
        self.assertEqual(verification["comparators"], ["clean"])
        self.assertEqual(verification["informational_comparators"], [])
        self.assertIn("high_learning_rate", verification["skipped_comparators"])
        self.assertIsNone(result["predictions"]["high_learning_rate_first_update_ratio_to_clean"])

    def test_assessment_rejects_broken_mechanisms_and_repairs(self):
        report = self.run_training()
        variants = copy.deepcopy(report["variants"])
        variants["missing_optimizer_step"]["parameter_group_epochs"][0]["optimizer_steps"] = 1
        result = self.reassess(report, variants)
        self.assertFalse(
            result["checks"]["missing_optimizer_step"][
                "missing_step_forwards_without_steps_every_epoch"
            ]
        )
        self.assertEqual(
            result["repair_verification"]["missing_optimizer_step"]["decision"], "rejected"
        )

        plan = copy.deepcopy(report["variant_plan"])
        plan["high_learning_rate_repaired"]["learning_rate"] *= 10
        verification = self.reassess(report, plan=plan)["repair_verification"]
        self.assertFalse(
            verification["high_learning_rate"]["structural_checks"][
                "repair_restores_reference_value"
            ]
        )
        self.assertEqual(verification["high_learning_rate"]["decision"], "rejected")

        variants = copy.deepcopy(report["variants"])
        variants["high_learning_rate_repaired"]["parameter_group_epochs"][1]["head"][
            "mean_parameter_update_l2_norm"
        ] = 0.0
        verification = self.reassess(report, variants)["repair_verification"]
        self.assertFalse(
            verification["high_learning_rate"]["structural_checks"][
                "repaired_learns_with_a_step_per_forward"
            ]
        )

    def test_cli_exits_nonzero_unless_completed_and_both_repairs_accepted(self):
        accepted = {"decision": "accepted"}
        cases = (
            ({"status": "completed", "repair_verification": {"a": accepted, "b": accepted}}, 0),
            (
                {
                    "status": "completed",
                    "repair_verification": {"a": accepted, "b": {"decision": "rejected"}},
                },
                1,
            ),
            ({"status": "inconclusive", "repair_verification": {"a": accepted}}, 1),
        )
        for report, code in cases:
            with self.subTest(report=report):
                with (
                    patch.object(sys, "argv", ["faults", "--reference-run", "reference"]),
                    patch.object(
                        runner, "run_config_fault_training", return_value=Path("report.json")
                    ) as run,
                    patch.object(runner, "read_json", return_value=report),
                ):
                    if code:
                        with self.assertRaises(SystemExit):
                            runner.main()
                    else:
                        runner.main()
                run.assert_called_once_with(
                    Path("reference"), Path("artifacts/config_fault_training"), 3, None
                )

    def test_invalid_epoch_budget_does_no_work(self):
        for epochs in (0, 4, True, 2.0):
            with self.subTest(epochs=epochs), patch.object(runner, "load_reference") as load:
                with self.assertRaises(ValueError):
                    runner.run_config_fault_training(self.reference, self.output, epochs=epochs)
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
