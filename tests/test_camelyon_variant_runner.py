"""CPU tests for the shared Camelyon17 variant loop: parity, missing steps and divergence."""

import copy
import unittest
from dataclasses import replace

import runsleuth.camelyon_optimizer_training as experiment
import runsleuth.camelyon_variant_runner as variant_runner
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

TIMING = {"epoch_seconds", "peak_cuda_memory_mb"}


def probe_optimizer(model, variant, config):
    from runsleuth.optimizer_probe import make_probe_optimizer

    optimizer = make_probe_optimizer(
        model,
        variant=variant,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    return optimizer, None


@unittest.skipIf(torch is None, "Install torch and torchvision for CPU integration tests")
class VariantRunnerTests(CamelyonHarness):
    def run_variant(self, variant, name, run=None, **options):
        config, state, expected = self.reference_state()
        root = self.output / name
        root.mkdir(parents=True, exist_ok=True)
        variant_config = replace(
            config, epochs=2, run_name=variant, output_dir=str(root), device="cpu"
        )
        if run is None:

            def run(*args):
                return variant_runner.run_training_variant(
                    *args, make_optimizer=probe_optimizer, task="test_variant", **options
                )

        with self.patched_builders():
            return run(
                variant,
                variant_config,
                self.manifest,
                state,
                expected,
                root / variant,
                torch.device("cpu"),
            )

    def test_loop_matches_stale_experiment_runner(self):
        for variant in ("clean", "stale_head"):
            with self.subTest(variant=variant):
                original = self.run_variant(variant, "original", run=experiment._run_variant)
                mirror = self.run_variant(variant, "mirror")
                for key in (
                    "initial_state_sha256",
                    "optimizer_audit",
                    "final_optimizer_audit",
                    "initialization_metrics",
                    "parameter_group_epochs",
                ):
                    self.assertEqual(mirror[key], original[key], key)
                self.assertEqual(
                    {k: v for k, v in mirror["final_metrics"].items() if k not in TIMING},
                    {k: v for k, v in original["final_metrics"].items() if k not in TIMING},
                )
                left, right = (
                    torch.load(report["model_checkpoint"], map_location="cpu", weights_only=True)
                    for report in (original, mirror)
                )
                self.assertTrue(all(torch.equal(left[key], right[key]) for key in left))

    def test_performance_gate_matches_stale_experiment_policy(self):
        with self.patched_builders():
            report = read_json(
                experiment.run_optimizer_training(
                    self.reference,
                    self.output / "stale",
                    epochs=2,
                    requested_device="cpu",
                    verify_rebind=True,
                )
            )
        variants, expected = report["variants"], report["reference_initial_state_sha256"]
        for changes in (
            {},
            {"ood_validation_loss": 10.0},
            {"id_validation_accuracy": 0.0},
            {"ood_validation_accuracy": float("nan")},
        ):
            with self.subTest(changes=changes):
                changed = copy.deepcopy(variants)
                changed["stale_head_repaired"]["final_metrics"].update(changes)
                self.assertEqual(
                    variant_runner.performance_checks(
                        changed, "stale_head_repaired", ("clean", "stale_head")
                    ),
                    experiment.assess_rebind_training(changed, expected, 2)["performance_checks"],
                )

    def test_missing_step_is_recorded_as_forwards_without_updates(self):
        report = self.run_variant(
            "clean", "missing", optimizer_step_enabled=False, allow_missing_steps=True
        )
        self.assertEqual(report["status"], "completed")
        self.assertFalse(report["optimizer_step_enabled"])
        for row in report["parameter_group_epochs"]:
            self.assertEqual(
                (row["optimizer_steps"], row["training_forwards"], row["head"], row["backbone"]),
                (0, 3, None, None),
            )
        self.assertGreater(report["final_metrics"]["mean_gradient_norm"], 0)
        self.assertEqual(report["final_metrics"]["mean_parameter_update_norm"], 0)
        final = torch.load(report["model_checkpoint"], map_location="cpu", weights_only=True)
        for key, value in self.initial.items():
            self.assertTrue(torch.equal(final[key], value), key)

    def test_missing_step_without_permission_fails_the_variant(self):
        with self.assertRaisesRegex(RuntimeError, "no completed optimizer step"):
            self.run_variant("clean", "strict", optimizer_step_enabled=False)
        report = read_json(self.output / "strict" / "clean" / "run_report.json")
        self.assertEqual(report["status"], "failed")

    def test_divergence_is_recorded_only_when_requested(self):
        self.nan_training_images = True
        report = self.run_variant("clean", "recorded", record_divergence=True)
        self.assertEqual(report["status"], "diverged")
        self.assertEqual(report["divergence"]["epoch"], 1)
        self.assertEqual(report["completed_epochs"], 0)
        self.assertNotIn("final_metrics", report)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.run_variant("clean", "unrecorded")
        failed = read_json(self.output / "unrecorded" / "clean" / "run_report.json")
        self.assertEqual(failed["status"], "failed")


def final_metrics(accuracy, loss):
    return {
        "final_metrics": {
            f"{domain}_validation_{name}": value
            for domain in ("id", "ood")
            for name, value in (("accuracy", accuracy), ("loss", loss))
        }
    }


class RepairGateTests(unittest.TestCase):
    # A silent fault: the faulty run beats clean and the repair restores clean exactly.
    variants = {
        "clean": final_metrics(0.90, 0.30),
        "faulty": final_metrics(0.93, 0.25),
        "repaired": final_metrics(0.90, 0.30),
    }

    def gate(self, variants=None, reference="clean", faulty="faulty"):
        return variant_runner.verify_repair(
            self.variants if variants is None else variants,
            "repaired",
            reference=reference,
            faulty=faulty,
        )

    def test_reference_gates_and_the_faulty_run_is_informational(self):
        gate = self.gate()
        self.assertEqual(gate["gate"]["version"], 2)
        self.assertEqual(
            (gate["comparators"], gate["informational_comparators"]), (["clean"], ["faulty"])
        )
        self.assertTrue(gate["performance_nonregression"])
        # Gate v1 rejected this repair: it trails the faulty run by 3 points.
        self.assertFalse(all(check["passed"] for check in gate["informational_checks"]))
        self.assertTrue(gate["faulty_passes_gate_vs_reference"])
        self.assertEqual(
            gate["performance_checks"],
            variant_runner.performance_checks(self.variants, "repaired", ("clean",)),
        )

    def test_regression_against_the_reference_still_rejects(self):
        gate = self.gate({**self.variants, "repaired": final_metrics(0.85, 0.30)})
        self.assertFalse(gate["performance_nonregression"])
        failed = {check["metric"] for check in gate["performance_checks"] if not check["passed"]}
        self.assertEqual(failed, {"accuracy_drop"})

    def test_a_harmful_fault_does_not_pass_the_gate_itself(self):
        gate = self.gate({**self.variants, "faulty": final_metrics(0.70, 0.60)})
        self.assertTrue(gate["performance_nonregression"])
        self.assertFalse(gate["faulty_passes_gate_vs_reference"])

    def test_without_a_reference_the_faulty_run_gates(self):
        gate = self.gate(reference=None)
        self.assertEqual((gate["comparators"], gate["informational_comparators"]), (["faulty"], []))
        self.assertFalse(gate["performance_nonregression"])
        self.assertEqual(gate["informational_checks"], [])
        self.assertIsNone(gate["faulty_passes_gate_vs_reference"])

    def test_no_valid_comparator_fails_instead_of_passing_on_zero_checks(self):
        gate = self.gate(reference=None, faulty=None)
        self.assertEqual((gate["comparators"], gate["performance_checks"]), ([], []))
        self.assertFalse(gate["performance_nonregression"])


if __name__ == "__main__":
    unittest.main()
