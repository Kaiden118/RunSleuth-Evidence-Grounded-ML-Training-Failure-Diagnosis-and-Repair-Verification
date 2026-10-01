"""Artifact-backed handoff tests; mock tensor loading and training, never validation."""

import contextlib
import io
import json
import shutil
import sys
import unittest
from copy import deepcopy
from unittest.mock import patch

import test_optimizer_agent_diagnosis as diagnosis_fixtures
import test_optimizer_evidence as evidence_fixtures

from runsleuth import optimizer_agent_repair as controller
from runsleuth.camelyon_optimizer_training import REPAIR_POLICY, assess_rebind_training
from runsleuth.optimizer_evidence import inspect_optimizer_run


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


class OptimizerAgentRepairTests(unittest.TestCase):
    def setUp(self):
        # Reuse its real artifact factory without inheriting or rediscovering its tests.
        self.fixture = evidence_fixtures.OptimizerEvidenceTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.cwd_patch = patch.object(controller.Path, "cwd", return_value=self.root)
        self.cwd_mock = self.cwd_patch.start()
        self.addCleanup(self.cwd_patch.stop)
        self.candidate = self.fixture.directory
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
        self.agent_path = self.root / "agent_report.json"
        self.options = {
            "workspace_root": self.root,
            "reference_run": self.reference,
            "candidate_run": self.candidate,
            "baseline_run": self.baseline,
            "epochs": 2,
        }
        self.refresh_agent()
        self.state_hash = patch.object(controller, "_initial_state_hash", return_value="a" * 64)
        self.state_hash_mock = self.state_hash.start()
        self.addCleanup(self.state_hash.stop)

    def refresh_agent(self, *, defect=True):
        trace = []
        for call_id, directory in (
            ("reference-call", self.reference),
            ("candidate-call", self.candidate),
        ):
            data = inspect_optimizer_run(directory, require_file=self.fixture.require_file)
            data["run_directory"] = directory.relative_to(self.root).as_posix()
            entry = diagnosis_fixtures.make_entry(call_id, data)
            entry["result"].update(error_code=None, error_message=None)
            trace.append(entry)
        diagnosis = diagnosis_fixtures.make_payload(trace, defect=defect)
        self.agent = {
            "model": "fixture-model",
            "profile": "optimizer_binding",
            "status": "completed",
            "error": None,
            "pending_evidence": [],
            "first_pass_valid": False,
            "validation_retries": 1,
            "tool_trace": trace,
            "diagnosis": diagnosis,
            "final_text": json.dumps(diagnosis),
        }
        self.save_agent()

    def save_agent(self):
        evidence_fixtures.write_json(self.agent_path, self.agent)

    def snapshot(self):
        return {
            path.relative_to(self.root).as_posix(): path.read_bytes()
            for path in self.root.rglob("*")
            if path.is_file()
        }

    def assert_blocked(self, **overrides):
        with patch.object(controller, "run_optimizer_training") as runner:
            with self.assertRaises((ValueError, TypeError, FileNotFoundError)):
                controller.run_optimizer_agent_repair(
                    self.agent_path, **{**self.options, **overrides}, execute=True
                )
            runner.assert_not_called()

    def test_preparation_replays_real_evidence_without_writing_or_training(self):
        before = self.snapshot()
        with patch.object(controller, "run_optimizer_training") as runner:
            plan = controller.prepare_optimizer_repair(self.agent_path, **self.options)
        runner.assert_not_called()
        self.assertEqual(self.snapshot(), before)
        self.state_hash_mock.assert_called_once_with(self.baseline / "initial_state_dict.pt")
        self.assertEqual(plan["status"], "prepared")
        self.assertFalse(plan["execution_started"])
        self.assertIsNone(plan["decision"])
        self.assertEqual(plan["action"]["name"], "rebuild_optimizer_before_first_step")
        self.assertEqual(plan["action"]["owner"], "controller")
        self.assertIsNone(plan["action"]["agent_proposed_patch"])
        self.assertFalse(plan["agent"]["first_pass_valid"])
        self.assertEqual(plan["agent"]["validation_retries"], 1)
        self.assertEqual(plan["agent"]["diagnosis"], self.agent["diagnosis"])
        self.assertEqual(plan["policy"], REPAIR_POLICY)
        self.assertEqual(plan["budget"]["training_epoch_upper_bound"], 6)
        self.assertEqual(plan["budget"]["model_calls"], 0)
        self.assertEqual(plan["budget"]["variants"], ["clean", "stale_head", "stale_head_repaired"])

    def test_default_run_persists_preparation_and_preserves_input_artifacts(self):
        before = self.snapshot()
        with patch.object(controller, "run_optimizer_training") as runner:
            report_path = controller.run_optimizer_agent_repair(self.agent_path, **self.options)
        runner.assert_not_called()
        report = read_json(report_path)
        self.assertEqual(report["status"], "prepared")
        self.assertFalse(report["execution_started"])
        self.assertIsNone(report["decision"])
        for name, contents in before.items():
            self.assertEqual((self.root / name).read_bytes(), contents)

    def test_invalid_report_envelopes_or_citations_cannot_start_training(self):
        original = deepcopy(self.agent)
        for field, value in (
            ("status", "invalid_diagnosis"),
            ("profile", "training"),
            ("error", "failed validation"),
            ("pending_evidence", [{"tool_name": "inspect_optimizer_run"}]),
        ):
            with self.subTest(field=field):
                self.agent = deepcopy(original)
                self.agent[field] = value
                self.save_agent()
                self.assert_blocked()
        self.agent = deepcopy(original)
        self.agent["diagnosis"]["implementation_evidence"][0]["expected_value"] = 999
        self.save_agent()
        self.assert_blocked()

    def test_trace_replay_rejects_changed_values_types_and_unscoped_paths(self):
        original = deepcopy(self.agent)
        mutations = (
            lambda entry: entry["result"]["data"]["optimizer_audit"].update(
                foreign_parameter_tensors=1
            ),
            lambda entry: entry["result"].update(ok=1),
            lambda entry: entry.update(raw_arguments=json.dumps({"run_directory": "unscoped"})),
            lambda entry: entry.update(tool_name="load_run_config"),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                self.agent = deepcopy(original)
                mutate(self.agent["tool_trace"][-1])
                self.save_agent()
                self.assert_blocked()

    def test_changed_recorded_artifact_is_detected_even_if_json_is_equivalent(self):
        path = self.candidate / "config.json"
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        self.assert_blocked()

    def test_only_the_supported_stale_head_mechanism_is_eligible(self):
        report_path = self.candidate / "run_report.json"
        original = read_json(report_path)
        for names, foreign, gradient, backbone in (
            (["layer.weight", "layer.bias"], 2, 0.25, 0.04),
            (["fc.bias", "fc.weight"], 1, 0.25, 0.04),
            (["fc.bias", "fc.weight"], 2, 0.0, 0.04),
            (["fc.bias", "fc.weight"], 2, 0.25, 0.0),
        ):
            with self.subTest(names=names, foreign=foreign, gradient=gradient, backbone=backbone):
                report = deepcopy(original)
                report["optimizer_audit"].update(
                    missing_trainable_names=names,
                    foreign_parameter_tensors=foreign,
                    optimizer_unique_parameter_tensors=60 + foreign,
                )
                for row in report["parameter_group_epochs"]:
                    row["head"]["mean_gradient_l2_norm"] = gradient
                    row["backbone"]["mean_parameter_update_l2_norm"] = backbone
                evidence_fixtures.write_json(report_path, report)
                evidence_fixtures.write_rows(
                    self.candidate / "parameter_group_metrics.jsonl",
                    report["parameter_group_epochs"],
                )
                self.refresh_agent()
                self.assert_blocked()

    def test_no_defect_diagnosis_cannot_authorize_a_repair(self):
        for source in self.reference.iterdir():
            shutil.copyfile(source, self.candidate / source.name)
        self.refresh_agent(defect=False)
        self.assert_blocked()

    def test_baseline_mismatches_and_initial_tensor_hash_block_training(self):
        for name in ("initial_state_dict.pt", "data_manifest.json"):
            with self.subTest(name=name):
                path = self.baseline / name
                old = path.read_bytes()
                path.write_bytes(old + b"\n")
                self.assert_blocked()
                path.write_bytes(old)
        config_path = self.baseline / "config.json"
        original = read_json(config_path)
        evidence_fixtures.write_json(config_path, {**original, "seed": original["seed"] + 1})
        self.assert_blocked()
        evidence_fixtures.write_json(config_path, original)
        self.state_hash_mock.return_value = "b" * 64
        self.assert_blocked()

    def test_epoch_bounds_and_missing_or_outside_inputs_block_training(self):
        for epochs in (False, 0, 4, 1.5, "2", 3):
            with self.subTest(epochs=epochs):
                self.assert_blocked(epochs=epochs)
        for key in ("reference_run", "candidate_run", "baseline_run"):
            with self.subTest(key=key):
                self.assert_blocked(**{key: self.root / "missing"})
        self.assert_blocked(candidate_run=self.root.parent / "outside")
        for missing in ("reference_run", "candidate_run", "baseline_run"):
            with (
                self.subTest(missing=missing),
                patch.object(controller, "run_optimizer_training") as runner,
            ):
                options = {key: value for key, value in self.options.items() if key != missing}
                with self.assertRaises(TypeError):
                    controller.run_optimizer_agent_repair(self.agent_path, **options, execute=True)
                runner.assert_not_called()
        before = self.snapshot()
        with patch.object(controller.Path, "cwd", return_value=self.root.parent):
            self.assert_blocked()
        self.assertEqual(self.snapshot(), before, "Wrong working directory must fail before writes")
        for execute in ("false", 0, 1, None):
            with (
                self.subTest(execute=execute),
                patch.object(controller, "run_optimizer_training") as runner,
            ):
                with self.assertRaises(ValueError):
                    controller.run_optimizer_agent_repair(
                        self.agent_path, **self.options, execute=execute
                    )
                runner.assert_not_called()

    def training_result(self, output_dir, *, reject=False):
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

    def test_execute_writes_plan_first_and_uses_canonical_bounded_training(self):
        for reject in (False, True):
            with self.subTest(reject=reject):

                def run_training(
                    reference_run,
                    output_dir,
                    epochs=3,
                    requested_device=None,
                    *,
                    verify_rebind=False,
                    reject=reject,
                ):
                    self.assertEqual(reference_run, self.baseline)
                    self.assertEqual(epochs, 2)
                    self.assertTrue(verify_rebind)
                    plans = list(output_dir.parent.glob("*plan*.json"))
                    self.assertEqual(len(plans), 1, "The plan must exist before training starts")
                    plan = read_json(plans[0])
                    self.assertEqual(plan["status"], "prepared")
                    self.assertFalse(plan["execution_started"])
                    self.assertEqual(plan["budget"]["training_epoch_upper_bound"], 6)
                    return self.training_result(output_dir, reject=reject)

                with patch.object(
                    controller, "run_optimizer_training", side_effect=run_training
                ) as runner:
                    report_path = controller.run_optimizer_agent_repair(
                        self.agent_path, **self.options, execute=True
                    )
                runner.assert_called_once()
                result = read_json(report_path)
                self.assertEqual(result["status"], "completed")
                self.assertTrue(result["execution_started"])
                self.assertEqual(result["decision"], "rejected" if reject else "accepted")
                self.assertEqual(result["repair_verification"]["decision"], result["decision"])

    def test_training_failure_is_recorded_without_a_success_decision(self):
        with patch.object(
            controller,
            "run_optimizer_training",
            side_effect=RuntimeError("fixture training failed"),
        ) as runner:
            report_path = controller.run_optimizer_agent_repair(
                self.agent_path, **self.options, execute=True
            )
        runner.assert_called_once()
        result = read_json(report_path)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["execution_started"])
        self.assertIsNone(result["decision"])
        self.assertIn("fixture training failed", result["error"]["message"])

    def test_old_or_tampered_training_result_cannot_be_accepted(self):
        old_path = self.training_result(self.root / "old-experiment")
        with patch.object(controller, "run_optimizer_training", return_value=old_path):
            result = read_json(
                controller.run_optimizer_agent_repair(self.agent_path, **self.options, execute=True)
            )
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["decision"])

        def incorrect_decision(reference_run, output_dir, *args, **kwargs):
            path = self.training_result(output_dir, reject=True)
            report = read_json(path)
            report["repair_verification"]["decision"] = "accepted"
            evidence_fixtures.write_json(path, report)
            return path

        with patch.object(controller, "run_optimizer_training", side_effect=incorrect_decision):
            result = read_json(
                controller.run_optimizer_agent_repair(self.agent_path, **self.options, execute=True)
            )
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["decision"])

        for damage in ("missing_file", "mismatched_report", "mismatched_telemetry"):
            with self.subTest(damage=damage):

                def invalid_child(reference_run, output_dir, *args, damage=damage, **kwargs):
                    path = self.training_result(output_dir)
                    child = path.parent / "stale_head_repaired"
                    if damage == "missing_file":
                        (child / "parameter_group_metrics.jsonl").unlink()
                    elif damage == "mismatched_report":
                        report = read_json(child / "run_report.json")
                        report["completed_epochs"] = 1
                        evidence_fixtures.write_json(child / "run_report.json", report)
                    else:
                        rows = [
                            json.loads(line)
                            for line in (child / "domain_metrics.jsonl").read_text().splitlines()
                        ]
                        rows[-1]["ood_validation_accuracy"] = 0.5
                        evidence_fixtures.write_rows(child / "domain_metrics.jsonl", rows)
                    return path

                with patch.object(controller, "run_optimizer_training", side_effect=invalid_child):
                    result = read_json(
                        controller.run_optimizer_agent_repair(
                            self.agent_path, **self.options, execute=True
                        )
                    )
                self.assertEqual(result["status"], "failed")
                self.assertIsNone(result["decision"])

    def test_cli_requires_scoped_inputs_and_preserves_execution_and_rejection_status(self):
        base = [
            "optimizer_agent_repair",
            "--agent-report",
            "agent_report.json",
            "--reference-run",
            "clean",
            "--candidate-run",
            "stale_head",
            "--baseline-run",
            "baseline",
            "--epochs",
            "2",
        ]
        for execute in (False, True):
            with self.subTest(execute=execute):
                outcome = self.root / "cli_outcome.json"
                evidence_fixtures.write_json(
                    outcome,
                    {
                        "status": "completed" if execute else "prepared",
                        "execution_started": execute,
                        "decision": "rejected" if execute else None,
                        "action": {},
                        "budget": {},
                        "error": None,
                    },
                )
                with (
                    patch.object(sys, "argv", base + (["--execute"] if execute else [])),
                    patch.object(controller.Path, "cwd", return_value=self.root),
                    patch.object(
                        controller, "run_optimizer_agent_repair", return_value=outcome
                    ) as run,
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    if execute:
                        with self.assertRaises(SystemExit) as caught:
                            controller.main()
                        self.assertEqual(caught.exception.code, 1)
                    else:
                        controller.main()
                run.assert_called_once()
                self.assertIs(run.call_args.kwargs["execute"], execute)
                self.assertEqual(run.call_args.kwargs["epochs"], 2)

        invalid = [base[:7] + base[9:]]  # Missing the independently required baseline.
        for value in ("../outside", "/outside", "C:\\outside", "C:outside"):
            arguments = list(base)
            arguments[6] = value
            invalid.append(arguments)
        invalid.append(base[:-1] + ["4"])
        for arguments in invalid:
            with (
                self.subTest(arguments=arguments),
                patch.object(sys, "argv", arguments),
                patch.object(controller, "run_optimizer_agent_repair") as run,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                with self.assertRaises(SystemExit) as caught:
                    controller.main()
                self.assertEqual(caught.exception.code, 2)
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
