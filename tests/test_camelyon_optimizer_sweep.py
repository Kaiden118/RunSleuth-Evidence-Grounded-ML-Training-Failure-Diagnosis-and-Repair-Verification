"""Seed-sweep orchestration with real artifact validation and simulated training."""

import contextlib
import hashlib
import io
import json
import shutil
import sys
import unittest
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import test_optimizer_evidence as evidence_fixtures

from runsleuth import camelyon_optimizer_sweep as sweep
from runsleuth.camelyon_optimizer_training import REPAIR_POLICY, assess_rebind_training


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


class RebindArtifacts:
    """A recorded baseline and stale-binding pair on disk, from the evidence fixtures."""

    def __init__(self, test: unittest.TestCase):
        fixture = evidence_fixtures.OptimizerEvidenceTests()
        fixture.setUp()
        test.addCleanup(fixture.doCleanups)
        self.root = fixture.root
        # The sweep only executes from the workspace root.
        test.enterContext(patch.object(Path, "cwd", return_value=self.root))
        self.candidate = fixture.directory
        self.reference = self.root / "clean"
        shutil.copytree(self.candidate, self.reference)
        report = read_json(self.reference / "run_report.json")
        report["variant"] = "clean"
        report["optimizer_audit"].update(missing_trainable_names=[], foreign_parameter_tensors=0)
        for row in report["parameter_group_epochs"]:
            row["head"]["mean_parameter_update_l2_norm"] = 0.001
        evidence_fixtures.write_rows(
            self.reference / "parameter_group_metrics.jsonl", report["parameter_group_epochs"]
        )
        evidence_fixtures.write_json(self.reference / "run_report.json", report)
        self.baseline = self.root / "baseline"
        self.baseline.mkdir()
        for name in ("config.json", "data_manifest.json"):
            shutil.copyfile(self.reference / name, self.baseline / name)
        (self.baseline / "initial_state_dict.pt").write_bytes(b"tensor-loader test fixture")
        evidence_fixtures.write_json(
            self.baseline / "run_report.json",
            {
                "status": "completed",
                "error": None,
                "completed_epochs": 2,
                "task": "camelyon17_resnet18_development_baseline",
                "initial_checkpoint_sha256": evidence_fixtures.digest(
                    self.baseline / "initial_state_dict.pt"
                ),
                "data_manifest_sha256": evidence_fixtures.digest(
                    self.baseline / "data_manifest.json"
                ),
            },
        )

    def training_result(self, output_dir, *, reject=False):
        """A completed stale-binding training report, rejected by the gate on request."""
        directory = output_dir / "pair-fixture"
        directory.mkdir(parents=True)
        variants = {}
        for name in ("clean", "stale_head", "stale_head_repaired"):
            source = self.candidate if name == "stale_head" else self.reference
            child = directory / name
            shutil.copytree(source, child)
            config = read_json(child / "config.json")
            config.update(run_name=name, output_dir=str(directory))
            evidence_fixtures.write_json(child / "config.json", config)
            variant = read_json(child / "run_report.json")
            variant.update(
                variant=name,
                run_directory=str(child),
                final_optimizer_audit=deepcopy(variant["optimizer_audit"]),
                initialization_metrics={
                    f"{domain}_validation_{metric}": 0.5
                    for domain in ("id", "ood")
                    for metric in ("accuracy", "loss")
                },
            )
            if name == "stale_head_repaired":
                variant["repair"] = {
                    "action": "rebuild_optimizer",
                    "timing": "before_first_optimizer_step",
                    "optimizer_state_entries_before": 0,
                    "model_state_sha256_before": "a" * 64,
                    "model_state_sha256_after": "a" * 64,
                    "optimizer_audit_before": deepcopy(variants["stale_head"]["optimizer_audit"]),
                }
                if reject:
                    domains = [
                        json.loads(line)
                        for line in (child / "domain_metrics.jsonl").read_text().splitlines()
                    ]
                    domains[-1]["ood_validation_accuracy"] = 0.7
                    evidence_fixtures.write_rows(child / "domain_metrics.jsonl", domains)
                    variant["final_metrics"]["ood_validation_accuracy"] = 0.7
            evidence_fixtures.write_json(child / "run_report.json", variant)
            variants[name] = variant
        verification = assess_rebind_training(variants, "a" * 64, 2)
        policy_path = directory / "repair_verification_policy.json"
        evidence_fixtures.write_json(
            policy_path,
            {
                "policy": dict(REPAIR_POLICY),
                "scope": "development experiment; not a held-out benchmark",
                "epochs_per_variant": 2,
                "comparators": ["clean", "stale_head"],
                "domains": ["id", "ood"],
                "checkpoint_selection": "fixed_final_epoch",
                "ood_used_for_repair_acceptance": True,
                "ood_used_for_checkpoint_selection": False,
            },
        )
        report = {
            "schema_version": 1,
            "status": "completed",
            "error": None,
            "task": "camelyon17_optimizer_rebind_verification",
            "reference_run": str(self.baseline),
            "epochs_per_variant": 2,
            "training_mode": "from_reference_initial_state",
            "checkpoint_selection": "fixed_final_epoch",
            "test_evaluated": False,
            "repair_acceptance_requested": True,
            "repair_acceptance_evaluated": True,
            "training_epoch_upper_bound": 6,
            "reference_initial_state_sha256": "a" * 64,
            "reference_config_sha256": evidence_fixtures.digest(self.baseline / "config.json"),
            "reference_initial_checkpoint_sha256": evidence_fixtures.digest(
                self.baseline / "initial_state_dict.pt"
            ),
            "reference_manifest_sha256": evidence_fixtures.digest(
                self.baseline / "data_manifest.json"
            ),
            "controls": {"performance_threshold": dict(REPAIR_POLICY)},
            "variants": variants,
            "repair_verification": verification,
            "repair_verification_policy": {
                "path": str(policy_path),
                "sha256": evidence_fixtures.digest(policy_path),
            },
        }
        path = directory / "optimizer_training_report.json"
        evidence_fixtures.write_json(path, report)
        return path


class CamelyonOptimizerSweepTests(unittest.TestCase):
    def setUp(self):
        self.fixture = RebindArtifacts(self)
        self.root = self.fixture.root
        self.source = self.fixture.baseline
        report_path = self.source / "run_report.json"
        report = read_json(report_path)
        report["test_evaluated"] = False
        evidence_fixtures.write_json(report_path, report)
        self.outcomes = {}
        self.options = {"workspace_root": self.root, "epochs": 2, "seeds": (7, 2026)}
        self.baseline_mock = self.enterContext(
            patch.object(sweep, "_run_baseline", side_effect=self.make_baseline)
        )
        self.training_mock = self.enterContext(
            patch.object(sweep, "run_optimizer_training", side_effect=self.make_training)
        )
        self.enterContext(patch.object(sweep, "_initial_state_hash", return_value="a" * 64))

    def run_sweep(self, **overrides):
        path = sweep.run_optimizer_sweep(self.source, **{**self.options, **overrides})
        return path, read_json(path)

    def make_baseline(self, config):
        output = Path(config.output_dir) / "baseline-fixture"
        output.mkdir(parents=True)
        config.save(output / "config.json")
        for name in ("data_manifest.json", "initial_state_dict.pt"):
            shutil.copyfile(self.source / name, output / name)
        report = read_json(self.source / "run_report.json")
        report.update(
            completed_epochs=config.epochs,
            initial_checkpoint_sha256=evidence_fixtures.digest(output / "initial_state_dict.pt"),
            data_manifest_sha256=evidence_fixtures.digest(output / "data_manifest.json"),
        )
        evidence_fixtures.write_json(output / "run_report.json", report)
        return output

    def make_training(self, reference_run, output_dir, epochs=3, *, verify_rebind=False):
        self.assertEqual(epochs, 2)
        self.assertTrue(verify_rebind)
        config = read_json(reference_run / "config.json")
        self.fixture.baseline = reference_run
        for directory in (self.fixture.reference, self.fixture.candidate):
            evidence_fixtures.write_json(directory / "config.json", config)
        outcome = self.outcomes.get(config["seed"], "accepted")
        if isinstance(outcome, BaseException):
            raise outcome
        return self.fixture.training_result(output_dir, reject=outcome == "rejected")

    def test_prepare_records_fixed_subset_policy_and_total_budget_without_training(self):
        before = {p.name: p.read_bytes() for p in self.source.iterdir()}
        path, result = self.run_sweep()
        self.baseline_mock.assert_not_called()
        self.training_mock.assert_not_called()
        self.assertEqual(result["status"], "prepared")
        self.assertFalse(result["execution_started"])
        self.assertEqual([row["seed"] for row in result["cases"]], [7, 2026])
        self.assertTrue(all(row["status"] == "not_run" for row in result["cases"]))
        self.assertEqual(result["budget"]["training_epoch_upper_bound"], 16)
        self.assertEqual(result["budget"]["model_calls"], 0)
        self.assertEqual(result["policy"], REPAIR_POLICY)
        self.assertEqual(result["summary"]["structural_verified"]["denominator"], 2)
        self.assertTrue(path.is_relative_to(self.root))
        self.assertEqual({p.name: p.read_bytes() for p in self.source.iterdir()}, before)

    def test_invalid_seeds_and_budgets_fail_before_writing_or_training(self):
        for seeds in ((), (7, 7), (42,), (True,), (-1,), (2**32,), (1.5,), ("7",), (1, 2, 3, 4)):
            with self.subTest(seeds=seeds), self.assertRaises((ValueError, TypeError)):
                self.run_sweep(seeds=seeds, execute=True)
        for epochs in (False, 0, 4, 1.5, "2", 3):
            with self.subTest(epochs=epochs), self.assertRaises((ValueError, TypeError)):
                self.run_sweep(epochs=epochs, execute=True)
        for execute in (0, 1, "false", None):
            with self.subTest(execute=execute), self.assertRaises((ValueError, TypeError)):
                self.run_sweep(execute=execute)
        config_path = self.source / "config.json"
        config = read_json(config_path)
        config["device"] = "auto"
        evidence_fixtures.write_json(config_path, config)
        with self.assertRaisesRegex(ValueError, "explicit cpu or cuda"):
            self.run_sweep(execute=True)
        self.baseline_mock.assert_not_called()
        self.training_mock.assert_not_called()
        self.assertFalse((self.root / "artifacts/optimizer_sweeps").exists())

    def test_rejected_seed_remains_completed_and_does_not_stop_next_seed(self):
        self.outcomes[7] = "rejected"
        path, result = self.run_sweep(execute=True)
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["execution_started"])
        self.assertEqual([c["status"] for c in result["cases"]], ["completed", "completed"])
        self.assertEqual([c["decision"] for c in result["cases"]], ["rejected", "accepted"])
        self.assertEqual(result["summary"]["completed_seeds"], 2)
        self.assertEqual(result["summary"]["rejected"], 1)
        self.assertEqual(result["summary"]["accepted"], 1)
        self.assertEqual(
            result["summary"]["structural_verified"], {"numerator": 2, "denominator": 2}
        )
        # Seeds stay out of directory names, so long random seeds fit Windows paths.
        self.assertEqual(
            [case["output_directory"].rsplit("/", 1)[-1] for case in result["cases"]],
            ["s1", "s2"],
        )
        self.assertEqual(
            {call.args[0].run_name for call in self.baseline_mock.call_args_list}, {"baseline"}
        )
        source_config = read_json(self.source / "config.json")
        for call in self.baseline_mock.call_args_list:
            actual = asdict(call.args[0])
            for key, value in source_config.items():
                if key not in ("run_name", "seed", "epochs", "output_dir"):
                    self.assertEqual(actual[key], value)
        plan = self.root / result["plan"]["path"]
        self.assertEqual(evidence_fixtures.digest(plan), result["plan"]["sha256"])
        self.assertTrue(plan.parent == path.parent)

    def test_runtime_error_counts_in_denominator_and_next_seed_still_runs(self):
        self.outcomes[7] = RuntimeError("simulated training failure")
        _, result = self.run_sweep(execute=True)
        self.assertEqual(result["status"], "completed_with_errors")
        self.assertEqual([c["status"] for c in result["cases"]], ["error", "completed"])
        self.assertIsNone(result["cases"][0]["decision"])
        self.assertIn("simulated training failure", result["cases"][0]["error"]["message"])
        self.assertEqual(result["summary"]["error_cases"], 1)
        self.assertEqual(
            result["summary"]["structural_verified"], {"numerator": 1, "denominator": 2}
        )
        self.assertEqual(self.training_mock.call_count, 2)

    def test_changed_baseline_config_or_sample_identity_blocks_paired_training(self):
        def changed_baseline(config):
            path = self.make_baseline(config)
            if config.seed == 7:
                values = asdict(config)
                values["learning_rate"] *= 2
                evidence_fixtures.write_json(path / "config.json", values)
            else:
                manifest_path = path / "data_manifest.json"
                manifest = read_json(manifest_path)
                selected = manifest["splits"]["train"]
                selected["selected_sample_ids"][0] = 99
                selected["selected_sample_ids_sha256"] = hashlib.sha256(
                    json.dumps(selected["selected_sample_ids"], separators=(",", ":")).encode()
                ).hexdigest()
                evidence_fixtures.write_json(manifest_path, manifest)
                report_path = path / "run_report.json"
                report = read_json(report_path)
                report["data_manifest_sha256"] = evidence_fixtures.digest(manifest_path)
                evidence_fixtures.write_json(report_path, report)
            return path

        self.baseline_mock.side_effect = changed_baseline
        _, result = self.run_sweep(execute=True)
        self.training_mock.assert_not_called()
        self.assertEqual(result["summary"]["error_cases"], 2)
        self.assertTrue(all(c["decision"] is None for c in result["cases"]))

    def test_old_training_report_cannot_be_attached_to_new_sweep(self):
        old_report = self.fixture.training_result(self.root / "old-training")
        self.training_mock.side_effect = None
        self.training_mock.return_value = old_report
        _, result = self.run_sweep(execute=True)
        self.assertEqual(result["summary"]["error_cases"], 2)
        self.assertEqual(result["summary"]["accepted"], 0)

    def test_outside_baseline_and_wrong_working_directory_cannot_start_pair(self):
        with (
            patch.object(Path, "cwd", return_value=self.root.parent),
            self.assertRaises(ValueError),
        ):
            self.run_sweep(execute=True)
        self.baseline_mock.assert_not_called()
        self.baseline_mock.side_effect = None
        self.baseline_mock.return_value = self.source
        _, result = self.run_sweep(execute=True)
        self.training_mock.assert_not_called()
        self.assertEqual(result["summary"]["error_cases"], 2)

    def test_tampered_acceptance_is_recomputed_from_real_training_artifacts(self):
        def tampered_training(*args, **kwargs):
            path = self.make_training(*args, **kwargs)
            report = read_json(path)
            report["repair_verification"]["decision"] = "accepted"
            evidence_fixtures.write_json(path, report)
            return path

        self.outcomes[7] = "rejected"
        self.training_mock.side_effect = tampered_training
        _, result = self.run_sweep(execute=True)
        self.assertEqual([c["status"] for c in result["cases"]], ["error", "completed"])
        self.assertIsNone(result["cases"][0]["decision"])

    def test_interruption_saves_partial_summary_and_leaves_future_seeds_unrun(self):
        self.outcomes[2026] = KeyboardInterrupt()
        path, result = self.run_sweep(seeds=(7, 2026, 19), execute=True)
        self.assertEqual(result["status"], "interrupted")
        self.assertEqual(
            [c["status"] for c in result["cases"]], ["completed", "interrupted", "not_run"]
        )
        self.assertEqual(result["summary"]["planned_seeds"], 3)
        self.assertEqual(result["summary"]["completed_seeds"], 1)
        self.assertEqual(result["summary"]["structural_verified"]["denominator"], 3)
        self.assertEqual(self.baseline_mock.call_count, 2)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["status"], "interrupted")

    def test_plan_and_progress_are_saved_before_expensive_work(self):
        def inspected_baseline(config):
            seed_dir = Path(config.output_dir).parent
            summary_paths = list(seed_dir.parent.glob("*summary.json"))
            self.assertEqual(len(summary_paths), 1)
            result = read_json(summary_paths[0])
            case = next(c for c in result["cases"] if c["seed"] == config.seed)
            self.assertEqual(case["status"], "running")
            self.assertTrue((self.root / result["plan"]["path"]).is_file())
            return self.make_baseline(config)

        self.baseline_mock.side_effect = inspected_baseline
        _, result = self.run_sweep(execute=True)
        self.assertEqual(result["summary"]["completed_seeds"], 2)

    def test_cli_defaults_to_preparation_and_rejection_is_a_completed_experiment(self):
        arguments = ["sweep", "--reference-run", "baseline", "--epochs", "2"]
        self.outcomes = {7: "rejected", 2026: "rejected"}
        with patch.object(sys, "argv", arguments), contextlib.redirect_stdout(io.StringIO()):
            sweep.main()
        self.baseline_mock.assert_not_called()
        with (
            patch.object(sys, "argv", arguments + ["--execute"]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            sweep.main()  # Rejection is a measured outcome, not a batch-execution error.
        self.assertEqual(self.training_mock.call_count, 2)
        self.outcomes = {7: RuntimeError("simulated failure")}
        with (
            patch.object(sys, "argv", arguments + ["--execute"]),
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaises(SystemExit) as caught,
        ):
            sweep.main()
        self.assertEqual(caught.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
