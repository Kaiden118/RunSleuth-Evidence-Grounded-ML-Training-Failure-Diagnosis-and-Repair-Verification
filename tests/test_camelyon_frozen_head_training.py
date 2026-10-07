"""CPU integration tests for the stage-2 frozen-head training runner (tiny model, fake data)."""

import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import runsleuth.camelyon_frozen_head_training as runner
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
class FrozenHeadTrainingTests(CamelyonHarness):
    def run_training(self):
        with self.patched_builders():
            return read_json(
                runner.run_frozen_head_training(
                    self.reference, self.output, epochs=2, requested_device="cpu"
                )
            )

    def test_four_variants_reproduce_mechanisms_and_record_predictions(self):
        report = self.run_training()
        self.assertEqual(report["status"], "completed")
        self.assertTrue(report["mechanism_reproduced"])
        self.assertTrue(all(report["checks"].values()), report["checks"])
        self.assertEqual(list(report["variants"]), list(runner.VARIANTS))
        self.assertEqual(report["budget"], {"training_epoch_upper_bound": 8, "model_calls": 0})
        self.assertEqual(len({str(model.training_order) for model in self.models}), 1)
        verification = report["repair_verification"]
        self.assertTrue(verification["structure_verified"], verification["structural_checks"])
        self.assertEqual(verification["comparators"], ["clean", "frozen_head"])
        self.assertEqual(
            verification["decision"],
            "accepted" if verification["performance_nonregression"] else "rejected",
        )
        self.assertEqual(
            report["checkpoint_comparisons"]["head_unchanged_from_initial"],
            {
                "clean": False,
                "stale_head": True,
                "frozen_head": True,
                "frozen_head_repaired": False,
            },
        )
        predictions = report["predictions"]
        self.assertTrue(predictions["P1_frozen_head_bitwise_unchanged"])
        self.assertEqual(predictions["P2_frozen_vs_stale"]["outcome"], "bitwise_equal")
        self.assertEqual(predictions["P3_repaired_vs_clean"]["outcome"], "bitwise_equal")
        self.assertEqual(len(list(self.output.glob("run-*/repair_verification_policy.json"))), 1)
        snapshot = next(self.output.glob("run-*/source_snapshot"))
        self.assertEqual(
            sorted(path.name for path in snapshot.iterdir()), sorted(runner.EXTRA_SOURCES)
        )

    def test_assessment_rejects_broken_mechanisms_and_repairs(self):
        report = self.run_training()
        variants, comparisons = report["variants"], report["checkpoint_comparisons"]
        expected = report["reference_initial_state_sha256"]
        cases = (
            (
                ("frozen_head", "parameter_group_epochs", 0, "head", "max_tensors_with_gradient"),
                2,
                "frozen_head_has_no_gradients_or_updates_every_epoch",
            ),
            (
                ("frozen_head_repaired", "repair", "timing"),
                "after_first_optimizer_step",
                "repair_before_first_step_with_empty_state",
            ),
            (
                ("frozen_head_repaired", "repair", "model_state_sha256_after"),
                "changed",
                "repair_preserves_model_state",
            ),
            (
                (
                    "frozen_head_repaired",
                    "parameter_group_epochs",
                    1,
                    "head",
                    "min_trainable_tensors",
                ),
                0,
                "repaired_all_tensors_trainable_with_gradients_and_updates_every_epoch",
            ),
        )
        for path, value, failed_check in cases:
            with self.subTest(failed_check=failed_check):
                changed = copy.deepcopy(variants)
                target = changed
                for component in path[:-1]:
                    target = target[component]
                target[path[-1]] = value
                result = runner.assess_frozen_head_training(changed, expected, 2, comparisons)
                verification = result["repair_verification"]
                self.assertIs(verification["structural_checks"][failed_check], False)
                self.assertEqual(verification["decision"], "rejected")

        changed = copy.deepcopy(variants)
        frozen_loss = variants["frozen_head"]["final_metrics"]["ood_validation_loss"]
        changed["frozen_head_repaired"]["final_metrics"]["ood_validation_loss"] = (
            10 * frozen_loss + 1
        )
        verification = runner.assess_frozen_head_training(changed, expected, 2, comparisons)[
            "repair_verification"
        ]
        self.assertTrue(verification["structure_verified"])
        self.assertFalse(verification["performance_nonregression"])
        self.assertEqual(verification["decision"], "rejected")

    def test_cli_exits_nonzero_unless_completed_and_accepted(self):
        cases = (
            ({"status": "completed", "repair_verification": {"decision": "accepted"}}, False),
            ({"status": "completed", "repair_verification": {"decision": "rejected"}}, True),
            ({"status": "inconclusive", "repair_verification": {"decision": "accepted"}}, True),
        )
        for report, exits in cases:
            with self.subTest(report=report):
                with (
                    patch.object(sys, "argv", ["training", "--reference-run", "reference"]),
                    patch.object(
                        runner, "run_frozen_head_training", return_value=Path("report.json")
                    ) as run,
                    patch.object(runner, "read_json", return_value=report),
                ):
                    if exits:
                        with self.assertRaises(SystemExit):
                            runner.main()
                    else:
                        runner.main()
                run.assert_called_once_with(
                    Path("reference"), Path("artifacts/frozen_head_training"), 3, None
                )

    def test_invalid_epoch_budget_does_no_work(self):
        for epochs in (0, 4, True, 2.0):
            with self.subTest(epochs=epochs), patch.object(runner, "load_reference") as load:
                with self.assertRaises(ValueError):
                    runner.run_frozen_head_training(self.reference, self.output, epochs=epochs)
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
