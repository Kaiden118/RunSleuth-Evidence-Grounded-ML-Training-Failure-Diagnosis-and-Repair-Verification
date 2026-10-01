"""Check fixed-model orchestration without making API requests or training models."""

import contextlib
import io
import json
import unittest
from copy import deepcopy
from unittest.mock import patch

import test_evaluate_optimizer as fixtures

from runsleuth import rerun_optimizer_comparison as runner


class OptimizerRerunTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.EvaluateOptimizerTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.calls = []
        self.reports = {}
        self.register(self.fixture.case, self.fixture.agent)

    def register(self, case, report):
        value = deepcopy(report)
        value["model"] = "fixed-model"
        self.reports[(case["reference_run"], case["candidate_run"])] = value

    @contextlib.contextmanager
    def fake_session(self, model, root):
        self.assertEqual(model, "fixed-model")
        self.assertEqual(root, self.root)

        def run(reference_run, candidate_run):
            key = (reference_run, candidate_run)
            self.calls.append(key)
            value = self.reports[key]
            if isinstance(value, Exception):
                raise value
            return deepcopy(value)

        yield run

    def invoke(self):
        self.fixture.manifest_path.write_text(json.dumps(self.fixture.manifest), encoding="utf-8")
        with patch.object(runner, "_agent_session", side_effect=self.fake_session):
            with patch.object(runner.time, "sleep") as sleep:
                with contextlib.redirect_stdout(io.StringIO()):
                    path = runner.rerun_comparison(
                        self.fixture.manifest_path,
                        workspace_root=self.root,
                        model="fixed-model",
                        case_delay=10,
                    )
        return path, sleep

    def test_four_fresh_attempts_use_one_model_and_preserve_old_reports(self):
        for name, defect in (("control-a", False), ("defect-b", True), ("control-b", False)):
            self.register(*self.fixture.add_case(name, defect=defect))
        old_reports = {
            self.root / case["agent_reports"][0]: (
                self.root / case["agent_reports"][0]
            ).read_bytes()
            for case in self.fixture.manifest["cases"]
        }
        path, sleep = self.invoke()
        result = fixtures.read_json(path)
        fresh = fixtures.read_json(path.parent / "manifest.json")
        plan = fixtures.read_json(path.parent / "run_plan.json")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["models"], ["fixed-model"])
        self.assertEqual(result["all_attempt_usage"]["attempt_report_count"], 4)
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(sleep.call_count, 3)
        self.assertEqual(plan["budget"]["model_call_upper_bound"], 24)
        self.assertEqual(plan["budget"]["training_runs"], 0)
        self.assertEqual(plan["case_retries"], 0)
        for old, new in zip(self.fixture.manifest["cases"], fresh["cases"], strict=True):
            self.assertEqual(new["expected"], old["expected"])
            self.assertEqual(len(new["agent_reports"]), 1)
            self.assertNotEqual(new["agent_reports"], old["agent_reports"])
            self.assertTrue((self.root / new["agent_reports"][0]).is_file())
        for old, content in old_reports.items():
            self.assertEqual(old.read_bytes(), content)

    def test_api_failure_is_retained_without_case_retry_and_other_cases_continue(self):
        case, report = self.fixture.add_case("control", defect=False)
        self.register(case, report)
        failed = self.fixture.fail_attempt(self.fixture.agent)
        self.register(self.fixture.case, failed)
        path, _ = self.invoke()
        result = fixtures.read_json(path)
        self.assertEqual(result["status"], "completed_with_failures")
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(
            result["summary"]["attempt_status_counts"], {"api_error": 1, "completed": 1}
        )
        self.assertEqual(result["all_attempt_usage"]["model_calls"], 3)
        self.assertFalse(result["all_attempt_usage"]["usage_complete"])

    def test_all_artifacts_are_preflighted_before_opening_api_session(self):
        case, _ = self.fixture.add_case("late-case", defect=False)
        (self.root / case["candidate_run"] / "run_report.json").unlink()
        self.fixture.manifest_path.write_text(json.dumps(self.fixture.manifest), encoding="utf-8")
        with patch.object(runner, "_agent_session") as session:
            with self.assertRaises(ValueError):
                runner.rerun_comparison(
                    self.fixture.manifest_path, workspace_root=self.root, model="fixed-model"
                )
        session.assert_not_called()
        self.assertFalse((self.root / "artifacts" / "optimizer_reruns").exists())

    def test_unexpected_exception_stops_and_keeps_manifest_without_fake_usage(self):
        key = (self.fixture.case["reference_run"], self.fixture.case["candidate_run"])
        self.reports[key] = RuntimeError("local runner failure")
        with self.assertRaises(RuntimeError):
            self.invoke()
        directory = next((self.root / "artifacts" / "optimizer_reruns").iterdir())
        self.assertTrue((directory / "manifest.json").is_file())
        self.assertEqual(
            fixtures.read_json(directory / "runner_error.json")["error_type"], "RuntimeError"
        )
        self.assertFalse((directory / "case-001" / "agent_report.json").exists())
        self.assertEqual(len(self.calls), 1)

    def test_changed_requested_model_is_saved_then_rejected(self):
        next(iter(self.reports.values()))["model"] = "unexpected-model"
        with self.assertRaisesRegex(ValueError, "changed its requested model"):
            self.invoke()
        directory = next((self.root / "artifacts" / "optimizer_reruns").iterdir())
        saved = fixtures.read_json(directory / "case-001" / "agent_report.json")
        self.assertEqual(saved["model"], "unexpected-model")
        self.assertEqual(saved["input_tokens"], 2100)
        self.assertTrue((directory / "runner_error.json").is_file())


if __name__ == "__main__":
    unittest.main()
