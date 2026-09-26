"""Artifact-only healthy-control checks; no training dependencies required."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from runsleuth.healthy_control import inspect_healthy_control


class HealthyControlMetricsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reference = self.root / "reference"
        self.candidate = self.root / "candidate"
        self.reference.mkdir()
        self.candidate.mkdir()
        self.reference_config = {
            "run_name": "reference",
            "output_dir": str(self.reference),
            "learning_rate": 0.01,
            "optimizer_step_enabled": True,
            "epochs": 2,
            "seed": 42,
            "train_fraction": 0.8,
        }
        self.candidate_config = dict(
            self.reference_config,
            run_name="candidate",
            output_dir=str(self.candidate),
            learning_rate=0.02,
        )
        self.reference_rows = [
            self.row(1, 0.65, 0.8, 0.9, 0.1),
            self.row(2, 0.80, 0.5, 0.3, 0.05),
        ]
        self.candidate_rows = [
            self.row(1, 0.67, 0.75, 0.8, 0.2),
            self.row(2, 0.79, 0.6, 0.2, 0.06),
        ]
        self.write(self.reference, self.reference_rows)
        self.write(self.candidate, self.candidate_rows)

    @staticmethod
    def row(epoch, accuracy, loss, gradient, update):
        return dict(
            epoch=epoch,
            validation_accuracy=accuracy,
            validation_loss=loss,
            mean_gradient_norm=gradient,
            mean_parameter_update_norm=update,
        )

    @staticmethod
    def write(run, rows):
        (run / "metrics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    def inspect(self, **overrides):
        arguments = dict(
            reference_run=self.reference,
            candidate_run=self.candidate,
            reference_config=self.reference_config,
            candidate_config=self.candidate_config,
        )
        arguments.update(overrides)
        return inspect_healthy_control(**arguments)

    def test_known_metrics_pass_and_report_expected_summary(self):
        result = self.inspect()
        self.assertTrue(result["passed"])
        self.assertEqual(
            result["policy"],
            {"max_validation_accuracy_shortfall": 0.03, "max_validation_loss_ratio": 1.25},
        )
        self.assertEqual(
            result["reference"],
            {
                "final_validation_accuracy": 0.8,
                "final_validation_loss": 0.5,
                "first_epoch_gradient_norm": 0.9,
                "first_epoch_parameter_update_norm": 0.1,
                "final_epoch_gradient_norm": 0.3,
                "final_epoch_parameter_update_norm": 0.05,
            },
        )
        self.assertEqual(
            result["config_changes"],
            {"learning_rate": {"reference": 0.01, "candidate": 0.02}},
        )
        self.assertAlmostEqual(result["checks"][0]["observed_value"], 0.01)
        self.assertEqual(result["checks"][1]["observed_value"], 1.2)
        self.assertTrue(all(check["passed"] for check in result["checks"]))

    def test_accuracy_or_loss_regression_fails_its_check(self):
        for metric, value, failed_index in (
            ("validation_accuracy", 0.76, 0),
            ("validation_loss", 0.63, 1),
        ):
            with self.subTest(metric=metric):
                rows = copy.deepcopy(self.candidate_rows)
                rows[-1][metric] = value
                self.write(self.candidate, rows)
                result = self.inspect()
                self.assertFalse(result["passed"])
                self.assertFalse(result["checks"][failed_index]["passed"])
                self.assertTrue(result["checks"][1 - failed_index]["passed"])

    def test_exact_decimal_policy_boundaries_pass(self):
        rows = copy.deepcopy(self.candidate_rows)
        rows[-1].update(validation_accuracy=0.77, validation_loss=0.625)
        self.write(self.candidate, rows)
        result = self.inspect()
        self.assertTrue(result["passed"])
        self.assertEqual(result["checks"][0]["observed_value"], 0.03)
        self.assertEqual(result["checks"][1]["observed_value"], 1.25)

    def test_finite_losses_with_unrepresentable_ratio_are_invalid(self):
        reference_rows = copy.deepcopy(self.reference_rows)
        candidate_rows = copy.deepcopy(self.candidate_rows)
        reference_rows[-1]["validation_loss"] = 1e-308
        candidate_rows[-1]["validation_loss"] = 1e308
        self.write(self.reference, reference_rows)
        self.write(self.candidate, candidate_rows)
        with self.assertRaisesRegex(ValueError, "validation_loss_ratio"):
            self.inspect()

    def test_accuracy_gain_and_zero_candidate_norms_are_allowed(self):
        rows = copy.deepcopy(self.candidate_rows)
        rows[-1].update(validation_accuracy=0.9, mean_gradient_norm=0, mean_parameter_update_norm=0)
        rows[0]["mean_parameter_update_norm"] = 0
        self.write(self.candidate, rows)
        self.assertTrue(self.inspect()["passed"])

    def test_missing_reordered_duplicate_and_boolean_epochs_are_invalid(self):
        variants = [
            self.candidate_rows[:1],
            self.candidate_rows + [self.candidate_rows[-1]],
            list(reversed(self.candidate_rows)),
            [self.candidate_rows[0], self.candidate_rows[0]],
            [dict(self.candidate_rows[0], epoch=True), self.candidate_rows[1]],
        ]
        for rows in variants:
            with self.subTest(rows=rows):
                self.write(self.candidate, rows)
                with self.assertRaises(ValueError):
                    self.inspect()

    def test_bad_numeric_metrics_on_any_row_are_invalid(self):
        for field in (
            "validation_accuracy",
            "validation_loss",
            "mean_gradient_norm",
            "mean_parameter_update_norm",
        ):
            for value in (True, None, "0.1", float("nan"), float("inf"), -1):
                for run, original in (
                    (self.reference, self.reference_rows),
                    (self.candidate, self.candidate_rows),
                ):
                    with self.subTest(field=field, value=value, run=run.name):
                        rows = copy.deepcopy(original)
                        rows[0][field] = value
                        self.write(run, rows)
                        with self.assertRaises(ValueError):
                            self.inspect()
                        self.write(run, original)
        rows = copy.deepcopy(self.candidate_rows)
        rows[-1]["validation_accuracy"] = 1.01
        self.write(self.candidate, rows)
        with self.assertRaises(ValueError):
            self.inspect()

    def test_reference_comparison_denominators_must_be_positive(self):
        for index, field in (
            (1, "validation_loss"),
            (0, "mean_parameter_update_norm"),
            (1, "mean_gradient_norm"),
        ):
            with self.subTest(field=field):
                rows = copy.deepcopy(self.reference_rows)
                rows[index][field] = 0
                self.write(self.reference, rows)
                with self.assertRaises(ValueError):
                    self.inspect()

    def test_alias_and_missing_directories_are_invalid(self):
        for candidate in (
            self.reference,
            self.reference / ".." / "reference",
            self.root / "missing",
        ):
            with self.subTest(candidate=candidate):
                with self.assertRaises(ValueError):
                    self.inspect(candidate_run=candidate)

    def test_symlink_alias_directory_is_invalid(self):
        alias = self.root / "alias"
        try:
            alias.symlink_to(self.reference, target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"Directory symlink creation is unavailable: {exc}")
        with self.assertRaises(ValueError):
            self.inspect(candidate_run=alias)

    def test_controlled_configs_preserve_types_and_reject_nonfinite_values(self):
        for candidate_seed in (True, 1.0):
            with self.subTest(candidate_seed=candidate_seed):
                with self.assertRaises(ValueError):
                    self.inspect(
                        reference_config=dict(self.reference_config, seed=1),
                        candidate_config=dict(self.candidate_config, seed=candidate_seed),
                    )
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite JSON"):
                    self.inspect(
                        reference_config=dict(self.reference_config, train_fraction=value),
                        candidate_config=dict(self.candidate_config, train_fraction=value),
                    )

    def test_config_drift_disabled_step_and_bad_rates_are_invalid(self):
        changes = [
            {"seed": 43},
            {"train_fraction": 0.9},
            {"epochs": 3},
            {"new_field": True},
            {"optimizer_step_enabled": False},
            {"optimizer_step_enabled": 1},
        ]
        changes += [
            {"learning_rate": rate}
            for rate in (0.01, 0, -0.01, True, None, "0.02", float("nan"), float("inf"))
        ]
        for change in changes:
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    self.inspect(candidate_config=dict(self.candidate_config, **change))
        with self.assertRaises(ValueError):
            self.inspect(reference_config=dict(self.reference_config, optimizer_step_enabled=False))

    def test_epochs_must_be_positive_integers(self):
        for epochs in (0, -1, True, 2.0, "2", None):
            with self.subTest(epochs=epochs):
                with self.assertRaises(ValueError):
                    self.inspect(
                        reference_config=dict(self.reference_config, epochs=epochs),
                        candidate_config=dict(self.candidate_config, epochs=epochs),
                    )
        with self.assertRaises(ValueError):
            self.inspect(candidate_config=dict(self.candidate_config, epochs=2.0))

    def test_missing_or_malformed_metrics_are_invalid(self):
        metrics = self.candidate / "metrics.jsonl"
        metrics.unlink()
        with self.assertRaises(ValueError):
            self.inspect()
        for content in ("{\n{}\n", "[]\n[]\n", "{}\n{}\n", "\n\n"):
            with self.subTest(content=content):
                metrics.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.inspect()


if __name__ == "__main__":
    unittest.main()
