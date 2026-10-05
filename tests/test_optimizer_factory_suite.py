"""Offline suite accounting and real CPU tests of the emitted factory source."""

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from runsleuth.optimizer_factory_suite import (
    main,
    prepare_factory_suite,
    probe_factory_case,
    run_factory_suite,
)

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


class FactorySuiteTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.reference = self.root / "reference"
        self.reference.mkdir()
        self.marker = self.reference / "keep.txt"
        self.marker.write_text("keep recorded baseline unchanged", encoding="utf-8")

    def execute(self, *, verify=False):
        with contextlib.redirect_stdout(io.StringIO()):
            return run_factory_suite(
                workspace_root=self.root,
                reference_run=self.reference,
                verify=verify,
                requested_device="cpu",
            )

    @staticmethod
    def fake_probe(case, _context, runtime):
        runtime.update(
            status="verified",
            optimizer_steps_recorded=3 if case["analysis"]["decision"] == "patch_proposed" else 2,
            step_accounting_complete=True,
            mechanism_verified=True,
            checks={"fixture": True},
        )

    def test_prepare_is_pure_and_labels_are_not_passed_to_analysis(self):
        before = list(self.root.rglob("*"))
        prepared = prepare_factory_suite()
        self.assertEqual(len(prepared["cases"]), 12)
        self.assertTrue(all(case["classification_correct"] for case in prepared["cases"]))
        self.assertEqual(before, list(self.root.rglob("*")))
        fixture = deepcopy(prepared["cases"][0])
        fixture["expected_decision"] = "no_change"
        with patch("runsleuth.optimizer_factory_suite.factory_cases", return_value=[fixture]):
            changed = prepare_factory_suite()["cases"][0]
        self.assertEqual(changed["analysis"]["decision"], "patch_proposed")
        self.assertFalse(changed["classification_correct"])

    def test_static_mode_never_loads_data_or_claims_runtime_success(self):
        with (
            patch("runsleuth.optimizer_factory_suite._load_context") as load,
            patch("runsleuth.optimizer_factory_suite.probe_factory_case") as execute,
        ):
            path = self.execute()
        load.assert_not_called()
        execute.assert_not_called()
        report = read_json(path)
        self.assertEqual(report["status"], "static_completed")
        self.assertEqual(report["summary"]["source_decisions_correct"]["numerator"], 12)
        self.assertEqual(report["summary"]["optimizer_steps_recorded"], 0)
        self.assertIsNone(report["summary"]["fault_repairs_mechanism_verified"]["rate"])
        self.assertFalse(report["performance_recovery_evaluated"])
        self.assertIsNone(report["repair_accepted"])

    def test_exported_source_and_patches_match_the_report(self):
        report = read_json(self.execute())
        for case in report["cases"]:
            self.assertEqual(
                (self.root / case["original_file"]).read_bytes(), case["source"].encode("utf-8")
            )
            if case["analysis"]["decision"] == "patch_proposed":
                self.assertEqual(
                    (self.root / case["proposed_file"]).read_bytes(),
                    case["analysis"]["proposed_source"].encode("utf-8"),
                )
                self.assertEqual(
                    (self.root / case["patch_file"]).read_bytes(),
                    case["analysis"]["unified_diff"].encode("utf-8"),
                )
            else:
                self.assertIsNone(case["proposed_file"])

    def test_twenty_step_budget_and_unsupported_cases_never_execute(self):
        with (
            patch(
                "runsleuth.optimizer_factory_suite._load_context", return_value={"provenance": {}}
            ),
            patch(
                "runsleuth.optimizer_factory_suite.probe_factory_case", side_effect=self.fake_probe
            ) as execute,
        ):
            report = read_json(self.execute(verify=True))
        self.assertEqual(execute.call_count, 8)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["summary"]["optimizer_steps_recorded"], 20)
        self.assertEqual(report["summary"]["fault_repairs_mechanism_verified"]["numerator"], 4)
        self.assertEqual(report["summary"]["healthy_controls_mechanism_verified"]["numerator"], 4)
        self.assertTrue(report["summary"]["step_accounting_complete"])
        self.assertTrue(
            all(
                case["runtime"] is None
                for case in report["cases"]
                if case["analysis"]["decision"] == "unsupported"
            )
        )
        self.assertEqual(
            self.marker.read_text(encoding="utf-8"), "keep recorded baseline unchanged"
        )

    def test_partial_step_error_preserves_unknown_accounting_and_stops_next_case(self):
        def interrupted(_case, _context, runtime):
            runtime.update(
                optimizer_steps_recorded=1,
                step_accounting_complete=False,
                mechanism_verified=None,
                variants={"clean": {"completed": True}},
            )
            raise KeyboardInterrupt("interrupted after beginning second step")

        with (
            patch(
                "runsleuth.optimizer_factory_suite._load_context", return_value={"provenance": {}}
            ),
            patch(
                "runsleuth.optimizer_factory_suite.probe_factory_case", side_effect=interrupted
            ) as execute,
        ):
            report = read_json(self.execute(verify=True))
        execute.assert_called_once()
        self.assertEqual(report["status"], "execution_error")
        self.assertEqual(report["summary"]["optimizer_steps_recorded"], 1)
        self.assertFalse(report["summary"]["step_accounting_complete"])
        self.assertIsNone(report["cases"][1]["runtime"])
        self.assertEqual(report["summary"]["fault_repairs_mechanism_verified"]["numerator"], 0)

    def test_failed_mechanism_checks_do_not_become_successful_repairs(self):
        def failed(case, context, runtime):
            self.fake_probe(case, context, runtime)
            runtime.update(
                status="not_verified", mechanism_verified=False, checks={"head_updates": False}
            )

        with (
            patch(
                "runsleuth.optimizer_factory_suite._load_context", return_value={"provenance": {}}
            ),
            patch("runsleuth.optimizer_factory_suite.probe_factory_case", side_effect=failed),
        ):
            report = read_json(self.execute(verify=True))
        self.assertEqual(report["status"], "mechanism_failed")
        self.assertEqual(report["summary"]["fault_repairs_mechanism_verified"]["numerator"], 0)
        self.assertEqual(report["summary"]["optimizer_steps_recorded"], 20)

    def test_data_loading_failure_is_recorded_without_reporting_training_steps(self):
        with patch(
            "runsleuth.optimizer_factory_suite._load_context",
            side_effect=ValueError("wrong dataset hash"),
        ):
            report = read_json(self.execute(verify=True))
        self.assertEqual(report["status"], "execution_error")
        self.assertEqual(report["summary"]["optimizer_steps_recorded"], 0)
        self.assertTrue(report["summary"]["step_accounting_complete"])
        self.assertIn("wrong dataset", report["error"]["message"])

    def test_changed_expected_labels_cannot_authorize_execution(self):
        plan = prepare_factory_suite()
        plan["cases"][0]["classification_correct"] = False
        with (
            patch("runsleuth.optimizer_factory_suite.prepare_factory_suite", return_value=plan),
            patch("runsleuth.optimizer_factory_suite._load_context") as load,
        ):
            report = read_json(self.execute(verify=True))
        load.assert_not_called()
        self.assertEqual(report["status"], "classification_failed")

    def test_missing_reference_and_bad_devices_rejected_before_outputs(self):
        with self.assertRaises(ValueError):
            run_factory_suite(workspace_root=self.root, verify=True)
        with self.assertRaises(ValueError):
            run_factory_suite(workspace_root=self.root, requested_device="cuda:77")
        self.assertFalse((self.root / "artifacts").exists())

    def test_cli_runtime_failure_has_nonzero_exit_and_report_location(self):
        path = self.root / "failure.json"
        path.write_text(
            json.dumps(
                {
                    "status": "execution_error",
                    "summary": {},
                    "cases": [],
                    "error": {"message": "failed"},
                }
            ),
            encoding="utf-8",
        )
        output = io.StringIO()
        with (
            patch("sys.argv", ["factory-suite"]),
            patch("runsleuth.optimizer_factory_suite.run_factory_suite", return_value=path),
            contextlib.redirect_stdout(output),
        ):
            with self.assertRaises(SystemExit) as result:
                main()
        self.assertEqual(result.exception.code, 1)
        self.assertIn(f"factory_suite_report={path}", output.getvalue())


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch required for real parameter identity/update tests")
class FactoryRuntimeCPUTests(unittest.TestCase):
    def setUp(self):
        import torch

        from runsleuth.optimizer_probe import state_dict_sha256

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Sequential(
                    torch.nn.Linear(2, 4), torch.nn.BatchNorm1d(4), torch.nn.Tanh()
                )
                self.fc = torch.nn.Linear(4, 2)

            def forward(self, inputs):
                return self.fc(self.backbone(inputs))

        torch.manual_seed(29)
        state = deepcopy(Model().state_dict())
        self.context = {
            "torch": torch,
            "config": SimpleNamespace(seed=42, learning_rate=0.001, weight_decay=0.01),
            "device": torch.device("cpu"),
            "state": state,
            "state_sha256": state_dict_sha256(state),
            "inputs": torch.tensor([[1.0, 2.0], [-1.0, 1.0], [0.5, -2.0], [2.0, 0.2]]),
            "targets": torch.tensor([0, 1, 1, 0]),
            "model_factory": Model,
        }
        self.cases = {case["id"]: case for case in prepare_factory_suite()["cases"]}

    def test_all_four_faults_reproduce_zero_head_updates_and_patch_matches_clean(self):
        for case in self.cases.values():
            if case["expected_decision"] != "patch_proposed":
                continue
            with self.subTest(case=case["id"]):
                runtime = {}
                probe_factory_case(case, self.context, runtime)
                self.assertTrue(runtime["mechanism_verified"], runtime["checks"])
                self.assertEqual(runtime["optimizer_steps_recorded"], 3)
                original = runtime["variants"]["original"]
                patched = runtime["variants"]["source_patched"]
                self.assertGreater(original["head"]["gradient_l2_norm"], 0)
                self.assertEqual(original["head"]["parameter_update_l2_norm"], 0)
                self.assertGreater(patched["head"]["parameter_update_l2_norm"], 0)
                self.assertEqual(
                    patched["post_step_state_sha256"],
                    runtime["variants"]["clean"]["post_step_state_sha256"],
                )

    def test_all_four_controls_including_lazy_iterators_match_clean(self):
        for case in self.cases.values():
            if case["expected_decision"] != "no_change":
                continue
            with self.subTest(case=case["id"]):
                runtime = {}
                probe_factory_case(case, self.context, runtime)
                self.assertTrue(runtime["mechanism_verified"], runtime["checks"])
                self.assertEqual(runtime["optimizer_steps_recorded"], 2)
                self.assertNotIn("source_patched", runtime["variants"])

    def test_batchnorm_buffers_are_included_in_comparison(self):
        case = self.cases["eager_parameter_list_before_head"]
        runtime = {}
        probe_factory_case(case, self.context, runtime)
        self.assertTrue(runtime["checks"]["patched_step_matches_clean_including_buffers"])
        self.assertNotEqual(
            runtime["variants"]["clean"]["post_step_state_sha256"], self.context["state_sha256"]
        )

    def test_modified_proposal_cannot_execute(self):
        case = deepcopy(self.cases["eager_parameter_list_before_head"])
        case["analysis"]["proposed_source"] += "\nraise RuntimeError('injected')\n"
        with patch("runsleuth.optimizer_factory_suite._probe_step") as step:
            with self.assertRaises(ValueError):
                probe_factory_case(case, self.context, {})
        step.assert_not_called()

    def test_interruption_after_unreported_step_keeps_lower_bound_accounting(self):
        from runsleuth.optimizer_source_runtime import _probe_step

        calls = 0

        def interrupt(*args, **kwargs):
            nonlocal calls
            calls += 1
            result = _probe_step(*args, **kwargs)
            if calls == 2:
                raise RuntimeError("step executed but receipt not returned")
            return result

        runtime = {}
        with patch("runsleuth.optimizer_factory_suite._probe_step", side_effect=interrupt):
            with self.assertRaises(RuntimeError):
                probe_factory_case(
                    self.cases["direct_constructor_before_head"], self.context, runtime
                )
        self.assertEqual(runtime["optimizer_steps_recorded"], 1)
        self.assertFalse(runtime["step_accounting_complete"])
        self.assertIsNone(runtime["mechanism_verified"])


if __name__ == "__main__":
    unittest.main()
