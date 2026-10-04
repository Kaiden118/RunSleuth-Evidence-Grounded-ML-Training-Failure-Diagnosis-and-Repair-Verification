"""Replay complete artifact chains; mock only the new training execution."""

import contextlib
import io
import json
import shutil
import unittest
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import test_optimizer_agent_diagnosis as diagnosis_fixtures
import test_optimizer_evidence as evidence_fixtures
import test_optimizer_source_verify as verification_fixtures

from runsleuth.optimizer_evidence import inspect_optimizer_run
from runsleuth.optimizer_source_repair import _sha
from runsleuth.optimizer_source_runtime import _checks
from runsleuth.optimizer_source_train import (
    HELPERS,
    main,
    prepare_source_training,
    run_source_training,
)
from runsleuth.optimizer_source_verify import run_source_verification

read_json = verification_fixtures.read_json
write_json = evidence_fixtures.write_json
write_rows = evidence_fixtures.write_rows
digest = evidence_fixtures.digest


class SourceTrainControllerTests(unittest.TestCase):
    def setUp(self):
        self.fixture = verification_fixtures.OptimizerSourceVerifyTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.source = self.fixture.fixture
        self.root = self.fixture.root
        self.upgrade_pair(
            self.source.pair, self.source.case, self.source.agent, self.fixture.baseline
        )
        self.refresh_verification()

    def upgrade_pair(self, pair, case, agent, baseline=None):
        parent_path = pair / "optimizer_training_report.json"
        parent = read_json(parent_path)
        parent.update(epochs_per_variant=3, device="cpu")
        for name in ("clean", "stale_head"):
            directory = pair / name
            config = read_json(directory / "config.json")
            config.update(epochs=3, device="cpu")
            write_json(directory / "config.json", config)
            rows = {}
            for stem in ("metrics", "domain_metrics", "parameter_group_metrics"):
                rows[stem] = [
                    json.loads(line)
                    for line in (directory / f"{stem}.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ]
                rows[stem].append({**deepcopy(rows[stem][-1]), "epoch": 3})
                write_rows(directory / f"{stem}.jsonl", rows[stem])
            report = read_json(directory / "run_report.json")
            report.update(
                completed_epochs=3,
                parameter_group_epochs=rows["parameter_group_metrics"],
                final_metrics={**rows["metrics"][-1], **rows["domain_metrics"][-1]},
                final_optimizer_audit=deepcopy(report["optimizer_audit"]),
                initialization_metrics={
                    "id_validation_accuracy": 0.5,
                    "id_validation_loss": 0.7,
                    "ood_validation_accuracy": 0.5,
                    "ood_validation_loss": 0.7,
                },
            )
            write_json(directory / "run_report.json", report)
            parent["variants"][name] = report
        write_json(parent_path, parent)
        source_directory = Path(__file__).resolve().parents[1] / "src/runsleuth"
        for name in HELPERS:
            shutil.copyfile(source_directory / name, pair / "source_snapshot" / name)
        self.source.refresh_source_hashes(pair)
        environment = read_json(pair / "environment.json")
        environment.update(torch="fixture-version", cuda_runtime=None)
        write_json(pair / "environment.json", environment)
        if baseline is None:
            baseline = self.fixture.bind_baseline(pair)
        else:
            shutil.copyfile(pair / "clean" / "config.json", baseline / "config.json")
            report = read_json(baseline / "run_report.json")
            report["completed_epochs"] = 3
            write_json(baseline / "run_report.json", report)
            self.fixture.bind_parent_hashes(pair, baseline)
        trace = []
        for call_id, key in (
            ("reference-call", "reference_run"),
            ("candidate-call", "candidate_run"),
        ):
            data = inspect_optimizer_run(
                self.root / case[key], require_file=self.source.fixture.fixture.require_file
            )
            data["run_directory"] = case[key]
            entry = diagnosis_fixtures.make_entry(call_id, data)
            entry["result"].update(error_code=None, error_message=None)
            trace.append(entry)
        payload = diagnosis_fixtures.make_payload(trace, defect=True)
        agent.update(
            tool_trace=trace,
            diagnosis=payload,
            final_text=json.dumps(payload),
            validation_attempts=[
                {
                    "model_call": 2,
                    "raw_final_text": json.dumps(payload),
                    "valid": True,
                    "error": None,
                }
            ],
        )
        self.source.fixture.save_agent(agent, case)

    def probe_result(self, job, _device):
        audit = read_json(self.source.pair / "clean" / "run_report.json")["optimizer_audit"]
        variants = {}
        for name in ("clean", "stale_head", "source_patched"):
            row = {
                "constructor_variant": "clean" if name == "clean" else "stale_head",
                "optimizer_state_entries_before": 0,
                "initial_state_sha256": job["initial_state_sha256"],
                "post_step_state_sha256": ("c" if name == "stale_head" else "b") * 64,
                "pre_update_loss": 0.7,
                "pre_update_accuracy": 0.5,
                "optimizer_audit": deepcopy(audit),
                "head": {"gradient_l2_norm": 0.3, "parameter_update_l2_norm": 0.003},
                "backbone": {"gradient_l2_norm": 1.0, "parameter_update_l2_norm": 0.04},
            }
            if name == "stale_head":
                row["optimizer_audit"].update(
                    missing_trainable_names=["fc.bias", "fc.weight"], foreign_parameter_tensors=2
                )
                row["head"]["parameter_update_l2_norm"] = 0.0
            variants[name] = row
        return {
            "execution_scope": "extracted_make_probe_optimizer_only",
            "source_patch_executed": True,
            "training_mode": "from_reference_initial_state",
            "optimizer_steps": 3,
            "reference_initial_state_sha256": job["initial_state_sha256"],
            "variants": variants,
            "checks": _checks(variants, job["initial_state_sha256"]),
            "mechanism_verified": True,
            "performance_recovery_evaluated": False,
        }

    def refresh_verification(self):
        with contextlib.redirect_stdout(io.StringIO()):
            proposal = self.source.execute()
            with patch(
                "runsleuth.optimizer_source_verify._execute_case", side_effect=self.probe_result
            ):
                self.verification = run_source_verification(proposal, workspace_root=self.root)

    def prepare(self):
        return prepare_source_training(self.verification, workspace_root=self.root)

    def execute(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return run_source_training(
                self.verification, workspace_root=self.root, requested_device="cpu"
            )

    def fake_train(self, job, directory, _device):
        # Policy must already exist, and the controller must request only three epochs.
        self.assertEqual(
            read_json(directory.parents[1] / "policy.json")["thresholds"],
            {"max_accuracy_drop": 0.01, "max_loss_ratio": 1.1},
        )
        self.assertEqual(job["epochs"], 3)
        directory.mkdir(parents=True)
        report = deepcopy(job["comparators"]["clean"])
        report.update(
            schema_version=1,
            task="camelyon17_source_patch_training",
            variant="source_patched",
            run_directory=str(directory),
            source_patch_executed=True,
            training_started=True,
            execution_scope="extracted_make_probe_optimizer_only",
            training_mode="from_reference_initial_state",
            checkpoint_selection="fixed_final_epoch",
            test_evaluated=False,
            step_accounting_complete=True,
            partial_epoch_possible=False,
            optimizer_steps_recorded=6,
            optimizer_state_entries_before=0,
            model_state_sha256_before_constructor=job["initial_state_sha256"],
            model_state_sha256_after_constructor=job["initial_state_sha256"],
            final_state_sha256="d" * 64,
            environment=job["expected_environment"],
            source_execution={
                "original_source_sha256": _sha(job["original_source"].encode()),
                "proposed_source_sha256": _sha(job["proposed_source"].encode()),
                "constructor_variant": "stale_head",
            },
        )
        write_json(directory / "config.json", asdict(job["config"]))
        write_json(directory / "data_manifest.json", job["manifest"])
        report["data_manifest_sha256"] = digest(directory / "data_manifest.json")
        write_json(
            directory / "baseline_metrics.json", {"epoch": 0, **report["initialization_metrics"]}
        )
        metrics = [
            {
                key: value
                for key, value in report["final_metrics"].items()
                if key
                in (
                    "epoch",
                    "train_loss",
                    "train_accuracy",
                    "validation_accuracy",
                    "validation_loss",
                    "mean_gradient_norm",
                    "mean_parameter_update_norm",
                )
            }
        ]
        domains = [
            {
                key: value
                for key, value in report["final_metrics"].items()
                if key not in metrics[0] or key == "epoch"
            }
        ]
        for name, row in (("metrics", metrics[0]), ("domain_metrics", domains[0])):
            write_rows(directory / f"{name}.jsonl", [{**row, "epoch": i} for i in (1, 2, 3)])
        write_rows(directory / "parameter_group_metrics.jsonl", report["parameter_group_epochs"])
        checkpoint = directory / "model_state_dict.pt"
        checkpoint.write_bytes(b"mock tensor serialization; no real training in controller tests")
        report.update(model_checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint))
        write_json(directory / "run_report.json", report)
        return report

    def assert_rejected_before_training(self):
        with patch("runsleuth.optimizer_source_train.train_source_patch") as train:
            with self.assertRaises((ValueError, OSError)):
                self.execute()
        train.assert_not_called()
        self.assertFalse((self.root / "artifacts/optimizer_source_training").exists())

    def test_real_preflight_chain_is_read_only_and_budgeted(self):
        before = self.source.input_bytes()
        plan = self.prepare()
        self.assertEqual(before, self.source.input_bytes())
        self.assertEqual(
            plan["report"]["budget"],
            {
                "new_training_runs": 1,
                "epochs_per_run": 3,
                "new_training_epoch_upper_bound": 3,
                "api_calls": 0,
                "historical_control_runs_reused": 2,
            },
        )
        self.assertEqual(len(plan["jobs"]), 1)

    def test_step_success_label_cannot_hide_a_changed_mechanism(self):
        value = read_json(self.verification)
        value["cases"][0]["result"]["variants"]["source_patched"]["head"][
            "parameter_update_l2_norm"
        ] = 0.0
        write_json(self.verification, value)
        self.assert_rejected_before_training()

    def test_step_count_and_source_identity_must_match(self):
        original = read_json(self.verification)
        for field, value in (("optimizer_steps_recorded", True), ("baseline", {})):
            with self.subTest(field=field):
                changed = deepcopy(original)
                changed["cases"][0][field] = value
                write_json(self.verification, changed)
                self.assert_rejected_before_training()

    def test_changed_proposed_source_is_rejected_before_training(self):
        saved = read_json(self.verification)
        proposal = read_json(self.root / saved["source_proposal"]["path"])
        path = self.root / proposal["cases"][0]["proposed_patch"]["proposed_file"]
        path.write_bytes(path.read_bytes() + b"\n# altered\n")
        self.assert_rejected_before_training()

    def test_recorded_helper_logic_drift_is_rejected_even_with_new_hashes(self):
        path = self.source.pair / "source_snapshot/train.py"
        path.write_bytes(path.read_bytes() + b"\nchanged_training = True\n")
        environment_path = self.source.pair / "environment.json"
        environment = read_json(environment_path)
        environment["source_sha256"]["train.py"] = digest(path)
        write_json(environment_path, environment)
        self.refresh_verification()
        self.assert_rejected_before_training()

    def test_helper_formatting_only_changes_are_allowed(self):
        path = self.source.pair / "source_snapshot/train.py"
        path.write_bytes(path.read_bytes() + b"\n# formatting-only historical difference\n")
        environment_path = self.source.pair / "environment.json"
        environment = read_json(environment_path)
        environment["source_sha256"]["train.py"] = digest(path)
        write_json(environment_path, environment)
        self.refresh_verification()
        self.assertEqual(len(self.prepare()["jobs"]), 1)

    def test_success_saves_fixed_policy_reuses_controls_and_preserves_inputs(self):
        before = self.source.input_bytes()
        with patch(
            "runsleuth.optimizer_source_train.train_source_patch", side_effect=self.fake_train
        ) as train:
            path = self.execute()
        train.assert_called_once()
        report = read_json(path)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["summary"]["accepted"], 1)
        self.assertEqual(report["summary"]["completed_new_epochs"], 3)
        self.assertEqual(report["summary"]["optimizer_steps_recorded"], 6)
        self.assertEqual(report["policy_sha256"], digest(path.parent / "policy.json"))
        self.assertFalse(report["test_evaluated"])
        for original, content in before.items():
            self.assertEqual(original.read_bytes(), content, str(original))

    def test_no_change_controls_do_not_schedule_training(self):
        case, agent = self.source.fixture.add_case("control", defect=False)
        self.source.bind_pair(case, agent, defect=False)
        self.refresh_verification()
        with patch(
            "runsleuth.optimizer_source_train.train_source_patch", side_effect=self.fake_train
        ) as train:
            report = read_json(self.execute())
        train.assert_called_once()
        self.assertEqual(report["summary"]["no_change"], 1)
        self.assertFalse(report["cases"][1]["execution_started"])

    def test_bad_checkpoint_receipt_is_an_error_not_an_acceptance(self):
        def changed(job, directory, device):
            report = self.fake_train(job, directory, device)
            (directory / "model_state_dict.pt").write_bytes(b"different")
            return report

        with patch("runsleuth.optimizer_source_train.train_source_patch", side_effect=changed):
            report = read_json(self.execute())
        self.assertEqual(report["status"], "completed_with_errors")
        self.assertEqual(report["summary"]["accepted"], 0)
        self.assertIsNone(report["cases"][0]["decision"])

    def test_performance_rejection_is_recorded_without_lowering_threshold(self):
        def bad_performance(job, directory, device):
            report = self.fake_train(job, directory, device)
            report["final_metrics"]["ood_validation_accuracy"] = 0.7
            domain_path = directory / "domain_metrics.jsonl"
            rows = [
                json.loads(line) for line in domain_path.read_text(encoding="utf-8").splitlines()
            ]
            rows[-1]["ood_validation_accuracy"] = 0.7
            write_rows(domain_path, rows)
            write_json(directory / "run_report.json", report)
            return report

        with patch(
            "runsleuth.optimizer_source_train.train_source_patch", side_effect=bad_performance
        ):
            report = read_json(self.execute())
        self.assertEqual(report["status"], "completed_with_rejections")
        self.assertEqual(report["summary"]["rejected"], 1)
        self.assertTrue(report["cases"][0]["verification"]["structure_verified"])
        self.assertFalse(report["cases"][0]["verification"]["performance_nonregression"])

    def test_partial_runtime_error_keeps_accounting_and_stops_later_case(self):
        case, agent = self.source.fixture.add_case("second-defect", defect=True)
        pair = self.source.bind_pair(case, agent, defect=True)
        self.upgrade_pair(pair, case, agent)
        self.refresh_verification()

        def failed(_job, directory, _device):
            directory.mkdir(parents=True)
            write_json(
                directory / "run_report.json",
                {
                    "completed_epochs": 1,
                    "optimizer_steps_recorded": 2,
                    "step_accounting_complete": False,
                    "partial_epoch_possible": True,
                    "status": "failed",
                    "error": {"message": "interrupted during epoch 2"},
                },
            )
            raise KeyboardInterrupt("interrupted during epoch 2")

        with patch(
            "runsleuth.optimizer_source_train.train_source_patch", side_effect=failed
        ) as train:
            report = read_json(self.execute())
        train.assert_called_once()
        self.assertEqual(report["status"], "completed_with_errors")
        self.assertEqual(report["summary"]["completed_new_epochs"], 1)
        self.assertEqual(report["summary"]["optimizer_steps_recorded"], 2)
        self.assertFalse(report["summary"]["step_accounting_complete"])
        self.assertTrue(report["summary"]["partial_epoch_possible"])
        self.assertEqual(report["cases"][1]["status"], "not_run")
        self.assertFalse(report["cases"][1]["execution_started"])

    def test_wrong_device_is_rejected_before_outputs(self):
        with self.assertRaisesRegex(ValueError, "historical controls' device"):
            run_source_training(
                self.verification, workspace_root=self.root, requested_device="cuda"
            )
        self.assertFalse((self.root / "artifacts/optimizer_source_training").exists())

    def test_cli_nonzero_on_rejection_prints_report_path(self):
        path = self.root / "cli_report.json"
        write_json(path, {"status": "completed_with_rejections", "summary": {}, "cases": []})
        output = io.StringIO()
        with (
            patch(
                "sys.argv",
                [
                    "source-train",
                    "--verification-report",
                    str(self.verification),
                    "--device",
                    "cpu",
                ],
            ),
            patch("runsleuth.optimizer_source_train.run_source_training", return_value=path),
            contextlib.redirect_stdout(output),
        ):
            with self.assertRaises(SystemExit) as raised:
                main()
        self.assertEqual(raised.exception.code, 1)
        self.assertIn(f"source_training_report={path}", output.getvalue())


if __name__ == "__main__":
    unittest.main()
