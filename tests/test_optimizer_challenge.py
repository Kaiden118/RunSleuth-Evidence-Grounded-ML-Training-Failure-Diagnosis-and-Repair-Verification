"""Frozen three-arm panel, independent label preflight and honest error denominators."""

import contextlib
import io
import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from test_optimizer_challenge_llm import FakeAPIError, FakeTools, prediction_for, response

from runsleuth.optimizer_challenge import main, run_challenge, score_method
from runsleuth.optimizer_challenge_cases import challenge_cases
from runsleuth.optimizer_challenge_llm import BASELINE_PROTOCOL, REVISED_PROTOCOL, run_llm_method


class ChallengeRunnerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.calls = []
        self.labels = {row["source"]: row["expected"] for row in challenge_cases()}
        self.wrong_first_source = False
        self.fail_first_source = False

    @staticmethod
    def fake_oracle(case, receipt):
        if case["expected"] == "inconclusive":
            raise AssertionError("Unknown fixture must never be executed")
        receipt.update(
            status="verified",
            label_verified=True,
            optimizer_steps_recorded=1,
            step_accounting_complete=True,
        )

    @contextmanager
    def session(self, root, model, *, protocol=BASELINE_PROTOCOL):
        self.assertEqual(root, self.root)
        self.assertEqual(model, "fixture-model")

        def run(method, source_path, source):
            self.calls.append((method, source_path, source))
            binding = self.labels[source]
            if (
                method == "llm_source"
                and source == challenge_cases()[0]["source"]
                and self.wrong_first_source
            ):
                binding = "current"
            replies = []
            if method == "tool_agent":
                replies.append(response(tool_path=source_path))
            if (
                method == "llm_source"
                and source == challenge_cases()[0]["source"]
                and self.fail_first_source
            ):
                replies.append(FakeAPIError("provider-secret"))
            else:
                replies.append(response(final=prediction_for(source, binding)))
            client = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=Mock(side_effect=replies)))
            )
            with patch.dict("sys.modules", {"openai": SimpleNamespace(APIError=FakeAPIError)}):
                return run_llm_method(
                    client,
                    model,
                    method=method,
                    source_path=source_path,
                    source=source,
                    tools=FakeTools(source) if method == "tool_agent" else None,
                    protocol=protocol,
                )

        yield run

    def execute(self, *, live=True, oracle=None, session=None, protocol=BASELINE_PROTOCOL):
        with (
            patch("runsleuth.optimizer_challenge.verify_known_case", oracle or self.fake_oracle),
            patch("runsleuth.optimizer_challenge._session", session or self.session),
            patch("runsleuth.optimizer_challenge.time.sleep"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            path = run_challenge(
                workspace_root=self.root,
                model="fixture-model",
                execute=live,
                case_delay=0,
                protocol=protocol,
            )
        return path, json.loads(path.read_text(encoding="utf-8"))

    def test_full_panel_runs_independent_fresh_arms_and_counterbalances_order(self):
        path, report = self.execute()
        self.assertEqual(report["status"], "completed", report["error"])
        self.assertEqual(report["oracle_summary"]["known_labels_verified"]["numerator"], 8)
        self.assertEqual(report["oracle_summary"]["optimizer_steps_recorded"], 8)
        self.assertEqual(report["oracle_summary"]["unknown_cases_executed"], 0)
        self.assertEqual(report["summary"]["rules"]["correct"]["numerator"], 8)
        self.assertEqual(report["summary"]["rules"]["known_over_refusal"]["numerator"], 4)
        for method in ("llm_source", "tool_agent"):
            self.assertEqual(report["summary"][method]["correct"]["numerator"], 12)
            self.assertEqual(report["usage_by_method"][method]["attempt_report_count"], 12)
        self.assertEqual(report["all_attempt_usage"]["model_calls"], 36)
        self.assertEqual(report["all_attempt_usage"]["tool_calls"], 12)
        self.assertTrue(report["all_attempt_usage"]["usage_complete"])
        self.assertEqual(
            [method for method, _, _ in self.calls[:4]],
            ["llm_source", "tool_agent", "tool_agent", "llm_source"],
        )
        for _, source_path, source in self.calls:
            self.assertEqual((self.root / source_path).read_text(), source)
        self.assertTrue((path.parent / "source_snapshot/optimizer_factory_analysis.py").is_file())
        self.assertEqual(report["paired_completed_only"]["both_correct"], 12)

    def test_completed_wrong_answer_stays_wrong_instead_of_getting_label_correction(self):
        self.wrong_first_source = True
        _, report = self.execute()
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["summary"]["llm_source"]["correct"]["numerator"], 11)
        self.assertEqual(
            report["summary"]["llm_source"]["first_pass_format_valid"]["numerator"], 12
        )
        self.assertEqual(report["paired_completed_only"]["tool_agent_only_correct"], 1)

    def test_api_failure_stays_in_primary_denominator_and_has_no_automatic_retry(self):
        self.fail_first_source = True
        path, report = self.execute()
        self.assertEqual(report["status"], "completed_with_errors")
        self.assertEqual(
            report["summary"]["llm_source"]["correct"],
            {"numerator": 11, "denominator": 12, "rate": 11 / 12},
        )
        self.assertEqual(report["summary"]["llm_source"]["error_cases"], 1)
        self.assertEqual(report["all_attempt_usage"]["attempt_report_count"], 24)
        self.assertFalse(report["all_attempt_usage"]["usage_complete"])
        self.assertFalse(report["all_attempt_usage"]["unrecorded_attempt_exists"])
        self.assertTrue((path.parent / "case-001/llm_source.json").is_file())
        self.assertEqual(len(self.calls), 24)

    def test_default_mode_verifies_oracle_and_rules_without_creating_client(self):
        with patch("runsleuth.optimizer_challenge._session", side_effect=AssertionError) as session:
            _, report = self.execute(live=False, session=session)
        session.assert_not_called()
        self.assertEqual(report["status"], "oracle_verified")
        self.assertEqual(report["budget"]["model_call_upper_bound"], 0)
        self.assertEqual(report["summary"]["llm_source"]["not_run"], 12)

    def test_oracle_failure_stops_before_paid_inference_and_marks_step_uncertainty(self):
        def broken(case, receipt):
            receipt.update(optimizer_steps_recorded=0, step_accounting_complete=False)
            raise RuntimeError("stopped after a possible CPU step")

        with patch("runsleuth.optimizer_challenge._session", side_effect=AssertionError) as session:
            _, report = self.execute(oracle=broken, session=session)
        session.assert_not_called()
        self.assertEqual(report["status"], "execution_error")
        self.assertEqual(report["error"]["phase"], "oracle")
        self.assertFalse(report["oracle_summary"]["step_accounting_complete"])
        self.assertEqual(report["all_attempt_usage"]["model_calls"], 0)

    def test_unrecorded_transport_exception_retains_known_usage_and_marks_unknown_cost(self):
        @contextmanager
        def session(root, model, *, protocol=BASELINE_PROTOCOL):
            def run(*args):
                raise RuntimeError("secret-transport-token")

            yield run

        path, report = self.execute(session=session)
        self.assertEqual(report["status"], "execution_error")
        self.assertTrue(report["all_attempt_usage"]["unrecorded_attempt_exists"])
        self.assertFalse(report["all_attempt_usage"]["usage_complete"])
        self.assertNotIn("secret-transport-token", path.read_text())
        self.assertEqual(report["summary"]["tool_agent"]["not_run"], 12)

    def test_source_mutation_during_model_call_is_detected_without_losing_report(self):
        @contextmanager
        def session(root, model, *, protocol=BASELINE_PROTOCOL):
            with self.session(root, model, protocol=protocol) as original:

                def run(method, source_path, source):
                    attempt = original(method, source_path, source)
                    (root / source_path).write_text("changed source")
                    return attempt

                yield run

        path, report = self.execute(session=session)
        self.assertEqual(report["status"], "execution_error")
        self.assertEqual(report["all_attempt_usage"]["model_calls"], 1)
        self.assertTrue((path.parent / "case-001/llm_source.json").is_file())

    def test_scoring_separates_coverage_selective_accuracy_and_unknown_overclaim(self):
        rows = [
            {
                "expected": label,
                "methods": {"rules": {"status": "completed", "binding": prediction}},
            }
            for label, prediction in (
                ("stale", "stale"),
                ("current", "inconclusive"),
                ("inconclusive", "current"),
            )
        ]
        score = score_method(rows, "rules")
        self.assertEqual(score["correct"]["rate"], 1 / 3)
        self.assertEqual(score["known_coverage"]["rate"], 1 / 2)
        self.assertEqual(score["selective_accuracy"]["rate"], 1 / 2)
        self.assertEqual(score["unknown_overclaim"]["rate"], 1)

    def test_second_run_uses_new_directory_and_preserves_prior_attempts(self):
        first, _ = self.execute(live=False)
        before = first.read_bytes()
        second, _ = self.execute(live=False)
        self.assertNotEqual(first, second)
        self.assertEqual(first.read_bytes(), before)

    def test_invalid_options_fail_without_creating_artifacts(self):
        for options in (
            {"case_delay": True},
            {"case_delay": -1},
            {"execute": "yes"},
            {"protocol": "unknown"},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                run_challenge(workspace_root=self.root, model="fixture-model", **options)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_cli_requires_explicit_execute_for_api_arms(self):
        output = self.root / "summary.json"
        output.write_text(
            json.dumps(
                {
                    "status": "oracle_verified",
                    "oracle_summary": {},
                    "summary": {},
                    "usage_by_method": {},
                    "all_attempt_usage": {},
                    "error": None,
                }
            )
        )
        with (
            patch("sys.argv", ["challenge", "--model", "fixture-model"]),
            patch("runsleuth.optimizer_challenge.run_challenge", return_value=output) as run,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            main()
        self.assertFalse(run.call_args.kwargs["execute"])
        self.assertEqual(run.call_args.kwargs["protocol"], BASELINE_PROTOCOL)

    def test_revised_panel_is_marked_as_development_retest_and_preserves_baseline(self):
        first, baseline = self.execute()
        original = first.read_bytes()
        second, revised = self.execute(protocol=REVISED_PROTOCOL)
        self.assertNotEqual(first, second)
        self.assertEqual(first.read_bytes(), original)
        self.assertEqual(revised["challenge_sha256"], baseline["challenge_sha256"])
        self.assertEqual(revised["budget"], baseline["budget"])
        self.assertEqual(revised["summary"]["rules"], baseline["summary"]["rules"])
        self.assertEqual(revised["inference_protocol"], REVISED_PROTOCOL)
        self.assertTrue(revised["protocol"]["adapted_on_this_case_panel"])
        self.assertEqual(revised["protocol"]["format_feedback"], "all_independent_errors")
        self.assertIn("Development retest", revised["scope"])
        self.assertEqual(revised["status"], "completed", revised["error"])
        for row in revised["cases"]:
            for method in ("llm_source", "tool_agent"):
                attempt = json.loads(
                    (self.root / row["methods"][method]["attempt_report"]).read_text()
                )
                self.assertEqual(attempt["inference_protocol"], REVISED_PROTOCOL)

    def test_wrong_labels_remain_wrong_on_revised_panel(self):
        self.wrong_first_source = True
        _, report = self.execute(protocol=REVISED_PROTOCOL)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["summary"]["llm_source"]["correct"]["numerator"], 11)
        self.assertEqual(
            report["summary"]["llm_source"]["first_pass_format_valid"]["numerator"], 12
        )

    def test_runner_stops_if_revised_session_returns_a_baseline_attempt(self):
        @contextmanager
        def session(root, model, *, protocol):
            with self.session(root, model, protocol=BASELINE_PROTOCOL) as run:
                yield run

        _, report = self.execute(protocol=REVISED_PROTOCOL, session=session)
        self.assertEqual(report["status"], "execution_error")
        self.assertEqual(report["all_attempt_usage"]["attempt_report_count"], 1)
        self.assertEqual(report["summary"]["llm_source"]["correct"]["numerator"], 0)

    def test_cli_passes_revised_protocol_and_execute_flag(self):
        output = self.root / "summary.json"
        output.write_text(
            json.dumps(
                {
                    "status": "oracle_verified",
                    "oracle_summary": {},
                    "summary": {},
                    "usage_by_method": {},
                    "all_attempt_usage": {},
                    "error": None,
                }
            )
        )
        with (
            patch(
                "sys.argv",
                [
                    "challenge",
                    "--model",
                    "fixture-model",
                    "--protocol",
                    REVISED_PROTOCOL,
                    "--execute",
                ],
            ),
            patch("runsleuth.optimizer_challenge.run_challenge", return_value=output) as run,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            main()
        self.assertTrue(run.call_args.kwargs["execute"])
        self.assertEqual(run.call_args.kwargs["protocol"], REVISED_PROTOCOL)


if __name__ == "__main__":
    unittest.main()
