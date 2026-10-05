"""Bounded source panel provenance, failed-attempt accounting and CLI tests."""

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from test_optimizer_factory_diagnosis import diagnosis_for, trace_for

from runsleuth.optimizer_factory_agent import LIMITS, main, prepare_cases, run_factory_agent
from runsleuth.optimizer_factory_analysis import analyze_optimizer_factory
from runsleuth.optimizer_factory_cases import SUITE_VERSION, factory_cases
from runsleuth.optimizer_factory_diagnosis import factory_evidence


def write_suite(root):
    cases = []
    for index, fixture in enumerate(factory_cases(), 1):
        path = root / f"original-{index:03d}.py"
        payload = fixture["source"].encode()
        path.write_bytes(payload)
        cases.append(
            {
                **fixture,
                "original_file": path.name,
                "source_sha256": hashlib.sha256(payload).hexdigest(),
                "analysis": analyze_optimizer_factory(fixture["source"]),
            }
        )
    path = root / "suite.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task": "optimizer_factory_development_suite",
                "suite_version": SUITE_VERSION,
                "status": "completed",
                "cases": cases,
            }
        ),
        encoding="utf-8",
    )
    return path


def fake_attempt(root, source_path, *, corrected=False):
    payload = (root / source_path).read_bytes()
    data = factory_evidence(payload, source_path=source_path)
    diagnosis = diagnosis_for(data)
    raw = json.dumps(diagnosis)
    attempts = [{"model_call": 2, "raw_final_text": raw, "valid": True, "error": None}]
    if corrected:
        attempts.insert(
            0,
            {
                "model_call": 2,
                "raw_final_text": "{}",
                "valid": False,
                "error": "fixture invalid JSON schema",
            },
        )
        attempts[-1]["model_call"] = 3
    return {
        "model": "fixture-model",
        "limits": dict(LIMITS),
        "status": "completed",
        "profile": "optimizer_factory",
        "diagnosis": diagnosis,
        "tool_trace": trace_for(data),
        "final_text": raw,
        "raw_final_text": raw,
        "pending_evidence": [],
        "error": None,
        "model_calls": 3 if corrected else 2,
        "tool_calls": 1,
        "input_tokens": 33 if corrected else 22,
        "output_tokens": 21 if corrected else 14,
        "usage_complete": True,
        "first_pass_valid": not corrected,
        "max_validation_retries": 1,
        "validation_retries": int(corrected),
        "validation_attempts": attempts,
    }


class FactoryAgentRunnerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.suite = write_suite(self.root)
        self.selected = []

    @contextmanager
    def session(self, root, model):
        self.assertEqual(model, "fixture-model")

        def run(source):
            self.selected.append(source)
            return fake_attempt(root, source, corrected=len(self.selected) == 2)

        yield run

    def execute(self, *, session=None, source=None):
        with (
            patch("runsleuth.optimizer_factory_agent._agent_session", session or self.session),
            patch("runsleuth.optimizer_factory_agent.time.sleep"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            path = run_factory_agent(
                workspace_root=self.root,
                model="fixture-model",
                case_delay=0,
                source=source,
                suite_report=None if source else self.suite,
            )
        return path, json.loads(path.read_text(encoding="utf-8"))

    def test_full_panel_scores_every_case_and_keeps_corrections_and_costs(self):
        original = {p.name: p.read_bytes() for p in self.root.glob("original-*.py")}
        path, report = self.execute()
        self.assertEqual(report["status"], "completed", report["error"])
        self.assertEqual(report["summary"]["validated_reports"]["numerator"], 12)
        self.assertEqual(report["summary"]["first_pass_valid"]["numerator"], 11)
        self.assertEqual(report["summary"]["completed_after_correction"], 1)
        self.assertEqual(report["summary"]["structured_decisions_correct"]["numerator"], 12)
        self.assertEqual(report["summary"]["unsupported_manual_review"]["numerator"], 4)
        self.assertEqual(report["all_attempt_usage"]["model_calls"], 25)
        self.assertEqual(report["all_attempt_usage"]["tool_calls"], 12)
        self.assertEqual(report["budget"]["model_call_upper_bound"], 36)
        self.assertEqual(report["patches_applied"], 0)
        self.assertFalse(report["performance_evaluated"])
        self.assertTrue((path.parent / "source_snapshot/agent.py").is_file())
        self.assertEqual(
            original, {p.name: p.read_bytes() for p in self.root.glob("original-*.py")}
        )
        for index, source in enumerate(self.selected, 1):
            self.assertTrue(source.endswith(f"case-{index:03d}/factory.py"))
            self.assertNotIn(factory_cases()[index - 1]["id"], source)

    def test_single_source_has_no_benchmark_label_or_scored_accuracy(self):
        _, report = self.execute(source=Path("original-001.py"))
        self.assertEqual(report["status"], "completed")
        self.assertIsNone(report["suite"])
        self.assertIsNone(report["summary"]["structured_decisions_correct"]["rate"])

    def test_503_attempt_is_saved_counted_and_not_automatically_retried(self):
        @contextmanager
        def session(root, model):
            count = 0

            def run(source):
                nonlocal count
                count += 1
                report = fake_attempt(root, source)
                if count == 1:
                    report.update(
                        status="api_error",
                        diagnosis=None,
                        first_pass_valid=None,
                        error="InternalServerError",
                        usage_complete=False,
                        input_tokens=11,
                        output_tokens=7,
                        validation_attempts=[],
                    )
                return report

            yield run

        path, report = self.execute(session=session)
        self.assertEqual(report["status"], "completed_with_errors")
        self.assertEqual(report["summary"]["validated_reports"]["denominator"], 12)
        self.assertEqual(report["summary"]["validated_reports"]["numerator"], 11)
        self.assertEqual(report["all_attempt_usage"]["attempt_report_count"], 12)
        self.assertEqual(report["all_attempt_usage"]["model_calls"], 24)
        self.assertFalse(report["all_attempt_usage"]["usage_complete"])
        self.assertTrue((path.parent / "case-001/agent_report.json").is_file())

    def test_unrecorded_exception_retains_previous_attempts_and_marks_costs_incomplete(self):
        @contextmanager
        def session(root, model):
            count = 0

            def run(source):
                nonlocal count
                count += 1
                if count == 2:
                    raise RuntimeError("credential-must-not-be-printed")
                return fake_attempt(root, source)

            yield run

        path, report = self.execute(session=session)
        self.assertEqual(report["status"], "execution_error")
        self.assertEqual(report["summary"]["not_run"], 10)
        self.assertEqual(report["all_attempt_usage"]["attempt_report_count"], 1)
        self.assertFalse(report["all_attempt_usage"]["usage_complete"])
        self.assertTrue(report["all_attempt_usage"]["unrecorded_attempt_exists"])
        self.assertNotIn("credential-must-not-be-printed", path.read_text())

    def test_changed_suite_labels_hashes_or_files_fail_before_client_creation(self):
        saved = self.suite.read_text()
        for mutation in ("label", "hash", "bytes", "missing_case", "duplicate"):
            with self.subTest(mutation=mutation):
                self.suite.write_text(saved)
                content = json.loads(saved)
                if mutation == "label":
                    content["cases"][0]["expected_decision"] = "no_change"
                elif mutation == "hash":
                    content["cases"][0]["source_sha256"] = "b" * 64
                elif mutation == "bytes":
                    content["cases"][0]["source"] += "\n"
                elif mutation == "missing_case":
                    content["cases"].pop()
                else:
                    content["cases"][1] = deepcopy(content["cases"][0])
                self.suite.write_text(json.dumps(content))
                with (
                    patch("runsleuth.optimizer_factory_agent._agent_session") as client,
                    self.assertRaises(ValueError),
                ):
                    run_factory_agent(
                        workspace_root=self.root, model="fixture-model", suite_report=self.suite
                    )
                client.assert_not_called()

    def test_replayed_report_with_forged_source_citations_is_rejected_after_preserving_usage(self):
        @contextmanager
        def session(root, model):
            def run(source):
                report = fake_attempt(root, source)
                report["tool_trace"][0]["result"]["data"]["analysis"]["binding_site"]["code"] = (
                    "forged()"
                )
                return report

            yield run

        path, report = self.execute(session=session)
        self.assertEqual(report["status"], "execution_error")
        self.assertEqual(report["all_attempt_usage"]["model_calls"], 2)
        self.assertTrue((path.parent / "case-001/agent_report.json").is_file())
        self.assertEqual(report["summary"]["validated_reports"]["numerator"], 0)

    def test_claimed_first_pass_success_is_replayed_against_original_model_json(self):
        @contextmanager
        def session(root, model):
            def run(source):
                report = fake_attempt(root, source)
                report["validation_attempts"][0]["raw_final_text"] = "{}"
                return report

            yield run

        _, report = self.execute(session=session)
        self.assertEqual(report["status"], "execution_error")
        self.assertEqual(report["summary"]["first_pass_valid"]["numerator"], 0)

    def test_model_mismatch_is_not_mixed_into_valid_panel(self):
        @contextmanager
        def session(root, model):
            def run(source):
                result = fake_attempt(root, source)
                result["model"] = "other-model"
                return result

            yield run

        _, report = self.execute(session=session)
        self.assertEqual(report["status"], "execution_error")
        self.assertEqual(report["summary"]["validated_reports"]["numerator"], 0)

    def test_invalid_paths_and_delay_fail_before_api(self):
        for kwargs in ({"case_delay": -1}, {"case_delay": 61}, {"case_delay": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                run_factory_agent(
                    workspace_root=self.root,
                    model="fixture-model",
                    suite_report=self.suite,
                    **kwargs,
                )
        with self.assertRaises(ValueError):
            prepare_cases(self.root, source=Path("../outside.py"), suite_report=None)
        with self.assertRaises(ValueError):
            prepare_cases(self.root, source=None, suite_report=None)

    def test_cli_supports_single_source_and_suite_but_requires_exactly_one(self):
        output = self.root / "summary.json"
        output.write_text(
            json.dumps({"status": "completed", "summary": {}, "all_attempt_usage": {}})
        )
        with (
            patch(
                "sys.argv",
                ["factory", "--suite-report", str(self.suite), "--model", "fixture-model"],
            ),
            patch(
                "runsleuth.optimizer_factory_agent.run_factory_agent", return_value=output
            ) as run,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            main()
        self.assertEqual(run.call_args.kwargs["suite_report"], self.suite)
        self.assertIsNone(run.call_args.kwargs["source"])


if __name__ == "__main__":
    unittest.main()
