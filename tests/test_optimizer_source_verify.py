"""Verify source proposals are replayed and baseline identity is checked before execution."""

import contextlib
import io
import json
import shutil
import unittest
from unittest.mock import patch

import test_optimizer_evidence as evidence_fixtures
import test_optimizer_source_repair as source_fixtures

from runsleuth.optimizer_source_verify import (
    main,
    prepare_source_verification,
    run_source_verification,
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


class OptimizerSourceVerifyTests(unittest.TestCase):
    def setUp(self):
        self.fixture = source_fixtures.OptimizerSourceRepairTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.baseline = self.bind_baseline(self.fixture.pair)
        self.proposal_path = self.fixture.execute()

    def bind_baseline(self, pair):
        baseline = self.root / "baselines" / pair.name
        baseline.mkdir(parents=True)
        for name in ("config.json", "data_manifest.json"):
            shutil.copyfile(pair / "clean" / name, baseline / name)
        checkpoint = baseline / "initial_state_dict.pt"
        checkpoint.write_bytes(b"dummy tensor checkpoint; runtime tensor load is mocked")
        config = read_json(baseline / "config.json")
        evidence_fixtures.write_json(
            baseline / "run_report.json",
            {
                "task": "camelyon17_resnet18_development_baseline",
                "status": "completed",
                "error": None,
                "completed_epochs": config["epochs"],
                "initial_checkpoint_sha256": evidence_fixtures.digest(checkpoint),
                "data_manifest_sha256": evidence_fixtures.digest(baseline / "data_manifest.json"),
            },
        )
        self.bind_parent_hashes(pair, baseline)
        return baseline

    def bind_parent_hashes(self, pair, baseline):
        path = pair / "optimizer_training_report.json"
        parent = read_json(path)
        parent["reference_run"] = baseline.relative_to(self.root).as_posix()
        for filename, key in (
            ("config.json", "reference_config_sha256"),
            ("data_manifest.json", "reference_manifest_sha256"),
            ("initial_state_dict.pt", "reference_initial_checkpoint_sha256"),
        ):
            parent[key] = evidence_fixtures.digest(baseline / filename)
        evidence_fixtures.write_json(path, parent)

    def prepare(self):
        return prepare_source_verification(self.proposal_path, workspace_root=self.root)

    def execute(self):
        return run_source_verification(
            self.proposal_path, workspace_root=self.root, requested_device="cpu"
        )

    def rewrite_proposal(self):
        self.proposal_path = self.fixture.execute()

    def assert_rejected_before_execution(self):
        with patch("runsleuth.optimizer_source_verify._execute_case") as execute:
            with self.assertRaises((ValueError, FileNotFoundError)):
                self.execute()
        execute.assert_not_called()
        self.assertFalse((self.root / "artifacts" / "optimizer_source_verifications").exists())

    def test_prepare_replays_inputs_without_writes_or_tensor_execution(self):
        before = self.fixture.input_bytes()
        with patch("runsleuth.optimizer_source_verify._execute_case") as execute:
            plan = self.prepare()
        execute.assert_not_called()
        self.assertEqual(len(plan["jobs"]), 1)
        self.assertEqual(before, self.fixture.input_bytes())

    def test_modified_exported_proposed_source_is_rejected(self):
        proposal = read_json(self.proposal_path)["cases"][0]["proposed_patch"]
        path = self.root / proposal["proposed_file"]
        path.write_bytes(path.read_bytes() + b"\nraise RuntimeError('injected')\n")
        self.assert_rejected_before_execution()

    def test_modified_exported_original_source_is_rejected(self):
        proposal = read_json(self.proposal_path)["cases"][0]["proposed_patch"]
        path = self.root / proposal["original_file"]
        path.write_bytes(path.read_bytes() + b"\n# changed\n")
        self.assert_rejected_before_execution()

    def test_modified_patch_diff_is_rejected(self):
        proposal = read_json(self.proposal_path)["cases"][0]["proposed_patch"]
        path = self.root / proposal["patch_file"]
        path.write_bytes(path.read_bytes() + b"# changed\n")
        self.assert_rejected_before_execution()

    def test_forged_proposal_metadata_is_rejected(self):
        report = read_json(self.proposal_path)
        report["cases"][0]["agent"]["labels"]["implementation_status"] = (
            "no_optimizer_binding_defect"
        )
        evidence_fixtures.write_json(self.proposal_path, report)
        self.assert_rejected_before_execution()

    def test_changed_baseline_checkpoint_bytes_are_rejected(self):
        path = self.baseline / "initial_state_dict.pt"
        path.write_bytes(b"changed checkpoint")
        self.assert_rejected_before_execution()

    def test_pair_checkpoint_hash_must_match_even_if_baseline_own_hash_matches(self):
        checkpoint = self.baseline / "initial_state_dict.pt"
        checkpoint.write_bytes(b"different but internally consistent baseline")
        path = self.baseline / "run_report.json"
        report = read_json(path)
        report["initial_checkpoint_sha256"] = evidence_fixtures.digest(checkpoint)
        evidence_fixtures.write_json(path, report)
        self.assert_rejected_before_execution()

    def test_wrong_baseline_seed_is_rejected_with_internally_consistent_hashes(self):
        path = self.baseline / "config.json"
        config = read_json(path)
        config["seed"] += 1
        evidence_fixtures.write_json(path, config)
        self.bind_parent_hashes(self.fixture.pair, self.baseline)
        self.rewrite_proposal()
        self.assert_rejected_before_execution()

    def test_wrong_baseline_data_identity_is_rejected_with_matching_file_hashes(self):
        path = self.baseline / "data_manifest.json"
        manifest = read_json(path)
        manifest["source_provenance"]["metadata_sha256"] = "c" * 64
        evidence_fixtures.write_json(path, manifest)
        report_path = self.baseline / "run_report.json"
        report = read_json(report_path)
        report["data_manifest_sha256"] = evidence_fixtures.digest(path)
        evidence_fixtures.write_json(report_path, report)
        self.bind_parent_hashes(self.fixture.pair, self.baseline)
        self.rewrite_proposal()
        self.assert_rejected_before_execution()

    def test_incomplete_baseline_is_rejected(self):
        path = self.baseline / "run_report.json"
        report = read_json(path)
        report["completed_epochs"] -= 1
        evidence_fixtures.write_json(path, report)
        self.assert_rejected_before_execution()

    def test_control_does_not_schedule_a_probe(self):
        case, agent = self.fixture.fixture.add_case("control", defect=False)
        self.fixture.bind_pair(case, agent, defect=False)
        self.fixture.manifest["cases"] = [case]
        self.rewrite_proposal()
        plan = self.prepare()
        self.assertEqual(plan["jobs"], [])
        with patch("runsleuth.optimizer_source_verify._execute_case") as execute:
            report_path = self.execute()
        execute.assert_not_called()
        self.assertEqual(read_json(report_path)["status"], "completed")

    def probe_result(self, *, verified=True):
        return {
            "variants": {name: {} for name in ("clean", "stale_head", "source_patched")},
            "checks": {"paired_step_matches": verified},
            "mechanism_verified": verified,
            "optimizer_steps": 3,
            "execution_scope": "extracted_make_probe_optimizer_only",
            "source_patch_executed": True,
            "performance_recovery_evaluated": False,
        }

    def add_fault(self, name):
        case, agent = self.fixture.fixture.add_case(name, defect=True)
        pair = self.fixture.bind_pair(case, agent, defect=True)
        self.bind_baseline(pair)
        self.rewrite_proposal()

    def test_success_records_three_steps_and_does_not_claim_performance_recovery(self):
        before = self.fixture.input_bytes()
        with patch(
            "runsleuth.optimizer_source_verify._execute_case",
            return_value=self.probe_result(),
        ) as execute:
            report = read_json(self.execute())
        execute.assert_called_once()
        self.assertEqual(execute.call_args.args[1], "cpu")
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["summary"]["mechanism_verified"], 1)
        self.assertEqual(report["summary"]["optimizer_steps_recorded"], 3)
        self.assertTrue(report["summary"]["step_accounting_complete"])
        self.assertEqual(report["budget"]["optimizer_step_upper_bound"], 3)
        self.assertEqual(report["budget"]["api_calls"], 0)
        self.assertEqual(report["budget"]["training_epochs"], 0)
        self.assertFalse(report["performance_recovery_evaluated"])
        self.assertIsNone(report["repair_accepted"])
        for path, contents in before.items():
            self.assertEqual(path.read_bytes(), contents, str(path))

    def test_failed_mechanism_check_is_not_reported_as_completed(self):
        with patch(
            "runsleuth.optimizer_source_verify._execute_case",
            return_value=self.probe_result(verified=False),
        ):
            report = read_json(self.execute())
        self.assertEqual(report["status"], "completed_with_failures")
        self.assertEqual(report["cases"][0]["status"], "not_verified")
        self.assertEqual(report["summary"]["mechanism_verified"], 0)
        self.assertEqual(report["summary"]["optimizer_steps_recorded"], 3)
        self.assertTrue(report["summary"]["step_accounting_complete"])

    def test_runtime_exception_leaves_an_honest_partial_receipt(self):
        self.add_fault("second-fault")
        with patch(
            "runsleuth.optimizer_source_verify._execute_case",
            side_effect=[self.probe_result(), RuntimeError("partial runtime failed")],
        ) as execute:
            report = read_json(self.execute())
        self.assertEqual(execute.call_count, 2)
        self.assertEqual(report["status"], "completed_with_failures")
        self.assertEqual(report["summary"]["optimizer_steps_recorded"], 3)
        self.assertFalse(report["summary"]["step_accounting_complete"])
        self.assertEqual(report["cases"][0]["status"], "mechanism_verified")
        failed = report["cases"][1]
        self.assertEqual(failed["status"], "failed")
        self.assertTrue(failed["execution_started"])
        self.assertIsNone(failed["optimizer_steps_recorded"])
        self.assertFalse(failed["step_accounting_complete"])
        self.assertEqual(failed["error"]["type"], "RuntimeError")

    def test_execution_failure_stops_remaining_jobs(self):
        self.add_fault("second-fault")
        with patch(
            "runsleuth.optimizer_source_verify._execute_case",
            side_effect=RuntimeError("first execution failed"),
        ) as execute:
            report = read_json(self.execute())
        execute.assert_called_once()
        self.assertEqual(report["cases"][1]["status"], "not_run")
        self.assertFalse(report["cases"][1]["execution_started"])
        self.assertEqual(report["summary"]["optimizer_steps_recorded"], 0)
        self.assertFalse(report["summary"]["step_accounting_complete"])

    def test_three_patch_cases_are_rejected_before_runtime_budget_is_exceeded(self):
        self.add_fault("second-fault")
        self.add_fault("third-fault")
        self.assert_rejected_before_execution()

    def test_inconsistent_probe_success_flag_is_rejected(self):
        result = self.probe_result(verified=False)
        result["mechanism_verified"] = True
        with patch("runsleuth.optimizer_source_verify._execute_case", return_value=result):
            report = read_json(self.execute())
        self.assertEqual(report["status"], "completed_with_failures")
        self.assertEqual(report["summary"]["mechanism_verified"], 0)
        self.assertEqual(report["cases"][0]["error"]["type"], "ValueError")

    def test_probe_cannot_forge_budget_or_source_execution_claims(self):
        for key, value in (
            ("optimizer_steps", True),
            ("optimizer_steps", 4),
            ("optimizer_steps", 3.0),
            ("source_patch_executed", False),
            ("source_patch_executed", 1),
            ("execution_scope", "whole_historical_module"),
            ("performance_recovery_evaluated", True),
        ):
            with self.subTest(key=key, value=value):
                result = self.probe_result()
                result[key] = value
                with patch("runsleuth.optimizer_source_verify._execute_case", return_value=result):
                    report = read_json(self.execute())
                self.assertEqual(report["status"], "completed_with_failures")
                self.assertEqual(report["summary"]["mechanism_verified"], 0)
                self.assertEqual(report["cases"][0]["error"]["type"], "ValueError")

    def test_cli_exits_nonzero_for_a_rejected_probe(self):
        with patch(
            "runsleuth.optimizer_source_verify._execute_case",
            return_value=self.probe_result(verified=False),
        ):
            path = self.execute()
        arguments = ["optimizer_source_verify", "--proposal-report", str(self.proposal_path)]
        with (
            patch("sys.argv", arguments),
            patch("runsleuth.optimizer_source_verify.run_source_verification", return_value=path),
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            main()
        self.assertEqual(raised.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
