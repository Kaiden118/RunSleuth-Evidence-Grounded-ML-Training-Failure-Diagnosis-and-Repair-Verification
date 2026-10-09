"""CPU integration tests for the training-loop fault runner (tiny BatchNorm model, fake data)."""

import copy
import unittest

import runsleuth.camelyon_loop_faults as runner
import runsleuth.camelyon_variant_runner as variant_runner
from runsleuth.camelyon_optimizer_probe import read_json
from runsleuth.signature_matching import diagnose, extract_evidence, load_library

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

if torch is not None:
    from camelyon_harness import CamelyonHarness, TinyBatchNormClassifier
else:
    CamelyonHarness, TinyBatchNormClassifier = unittest.TestCase, None


@unittest.skipIf(torch is None, "Install torch and torchvision for CPU integration tests")
class LoopFaultTrainingTests(CamelyonHarness):
    model_class = TinyBatchNormClassifier
    # Twelve steps per epoch, so a two-epoch cosine stepped per batch cycles within one.
    training_samples = 24
    # A fine-tuning rate, so the clean run does not look like a high learning rate.
    learning_rate = 1e-4

    def run_training(self):
        with self.patched_builders():
            return read_json(
                runner.run_loop_fault_training(
                    self.reference, self.output, epochs=2, requested_device="cpu"
                )
            )

    def test_five_variants_reproduce_both_faults_and_verify_repairs(self):
        report = self.run_training()
        self.assertEqual(report["status"], "completed")
        self.assertTrue(report["mechanism_reproduced"], report["checks"])
        self.assertEqual(list(report["variants"]), list(runner.VARIANTS))
        variants = report["variants"]
        eval_mode = variants["train_in_eval_mode"]["parameter_group_epochs"]
        self.assertEqual([row["eval_mode_training_forwards"] for row in eval_mode], [12, 12])
        per_batch = variants["scheduler_stepped_per_batch"]["parameter_group_epochs"]
        self.assertEqual(
            [
                (row["learning_rate"]["changes"], row["learning_rate"]["rebounds"])
                for row in per_batch
            ],
            [(11, 3), (11, 3)],
        )
        predictions = report["predictions"]
        for rate, expected in zip(
            predictions["clean_epoch_learning_rates"], (1e-4, 5e-5), strict=True
        ):
            self.assertAlmostEqual(rate, expected)
        comparisons = report["checkpoint_comparisons"]
        self.assertEqual(comparisons["batchnorm_buffers"], 3)
        self.assertTrue(comparisons["buffers_unchanged_from_initial"]["train_in_eval_mode"])
        self.assertFalse(comparisons["buffers_unchanged_from_initial"]["clean"])
        self.assertFalse(comparisons["parameters_unchanged_from_initial"]["train_in_eval_mode"])
        self.assertEqual(
            predictions["repaired_vs_clean"],
            {"train_in_eval_mode": "bitwise_equal", "scheduler_stepped_per_batch": "bitwise_equal"},
        )
        for fault, verification in report["repair_verification"].items():
            with self.subTest(fault=fault):
                self.assertTrue(verification["structure_verified"], verification)
                self.assertEqual(verification["comparators"], ["clean"])
                self.assertEqual(verification["decision"], "accepted")

    def test_the_signatures_find_each_fault_and_wait_for_a_reference_without_one(self):
        report = self.run_training()
        library = load_library()
        variants = report["variants"]
        for name in runner.VARIANTS:
            fault = name if name in runner.FAULTS else None
            with self.subTest(variant=name):
                run = variants[name]
                with_reference = diagnose(
                    extract_evidence(run, variants["clean"], "AdamW"), library
                )
                alone = diagnose(extract_evidence(run, None, "AdamW"), library)
                self.assertEqual(with_reference["diagnosis"], fault or "no_known_fault")
                self.assertEqual(
                    alone["diagnosis"], "pending_reference" if fault else "no_known_fault"
                )

    def test_assessment_rejects_broken_mechanisms(self):
        report = self.run_training()
        variants = copy.deepcopy(report["variants"])
        variants["train_in_eval_mode"]["parameter_group_epochs"][1][
            "eval_mode_training_forwards"
        ] = 0
        variants["scheduler_stepped_per_batch"]["parameter_group_epochs"][0]["learning_rate"][
            "changes"
        ] = 0
        result = runner.assess_loop_fault_training(
            variants,
            report["reference_initial_state_sha256"],
            2,
            report["checkpoint_comparisons"],
            report["variant_plan"],
        )
        self.assertFalse(result["mechanism_reproduced"])
        for fault, check in (
            ("train_in_eval_mode", "eval_mode_variant_trains_every_forward_in_eval_mode"),
            ("scheduler_stepped_per_batch", "per_batch_rate_changes_at_every_step"),
        ):
            with self.subTest(fault=fault):
                self.assertFalse(result["checks"][fault][check])
                self.assertEqual(result["repair_verification"][fault]["decision"], "rejected")

    def test_unknown_schedules_and_epoch_budgets_do_no_work(self):
        with self.assertRaisesRegex(ValueError, "lr_schedule"):
            variant_runner.run_training_variant(
                "clean",
                None,
                None,
                None,
                None,
                None,
                None,
                make_optimizer=None,
                task="t",
                lr_schedule="linear",
            )
        for epochs in (0, 4, True):
            with self.subTest(epochs=epochs), self.assertRaises(ValueError):
                runner.run_loop_fault_training(self.reference, self.output, epochs=epochs)


if __name__ == "__main__":
    unittest.main()
