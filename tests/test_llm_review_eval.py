"""Offline tests for the multi-model LLM review evaluation (fake clients, fake reports)."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from test_signature_matching import SCENARIOS

from runsleuth import llm_review_eval as module

VARIANTS = ("clean", "high_learning_rate", "frozen_head")


class ServiceUnavailable(RuntimeError):
    status_code = 503


class EchoClient:
    """Answers with the matcher's diagnosis, or a fixed one, citing one exact evidence value."""

    def __init__(self, diagnosis=None, *, error=False, bad_first_reply=False, bad=False):
        self.bad = bad
        self.diagnosis = diagnosis
        self.error = error
        self.bad_first_reply = bad_first_reply
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        self.requests.append(request)
        if self.error:
            raise ServiceUnavailable("model overloaded")
        payload = json.loads(request["messages"][1]["content"])
        name, value = next(
            (name, value)
            for name, value in payload["evidence"].items()
            if isinstance(value, (bool, int, float, str))
        )
        reply = json.dumps(
            {
                "diagnosis": self.diagnosis or payload["matcher_diagnosis"],
                "explanation": "The cited evidence decides it.",
                "citations": [{"name": name, "value": value, "role": "supports"}],
                "repair": "Apply the signature's repair.",
                "confidence": "high",
                "disagreement_with_matcher": None,
            }
        )
        if self.bad or (self.bad_first_reply and len(self.requests) == 1):
            reply = "not json"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
        )


class ReviewEvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.reports = [self.report("run-a", 7), self.report("run-b", 2026)]
        self.output = self.root / "evaluations"

    def report(self, name, seed):
        directory = self.root / name
        directory.mkdir()
        (directory / "environment.json").write_text('{"optimizer": "AdamW"}', encoding="utf-8")
        path = directory / "experiment_report.json"
        variants = {variant: SCENARIOS[variant] for variant in VARIANTS}
        report = {"status": "completed", "seed": seed, "variants": variants}
        path.write_text(json.dumps(report), encoding="utf-8")
        return path

    def run_reviews(self, client, reports=None, model="fake-model", **options):
        with contextlib.redirect_stdout(io.StringIO()):
            return module.run_reviews(
                self.reports if reports is None else reports,
                client,
                "ollama",
                model,
                self.output,
                **options,
            )

    def test_every_run_is_reviewed_in_both_modes_and_a_rerun_resumes(self):
        directory = self.run_reviews(EchoClient())
        summary = module.summarize(directory)
        self.assertTrue(summary["complete"])
        self.assertEqual(directory.name, "ollama-fake-model-prompt-v2")
        self.assertEqual(
            (summary["prompt_version"], summary["payload_version"], summary["response_format"]),
            (2, 2, "json_schema"),
        )
        overall = summary["overall"]
        self.assertEqual((overall["cases"], overall["completed"]), (12, 12))
        self.assertEqual(overall["agrees_with_matcher"], 12)
        self.assertEqual(overall["valid_on_first_attempt"], 12)
        self.assertEqual(overall["input_tokens"], 1200)
        # Without a reference the frozen head is deferred; the matcher misses no other case.
        self.assertEqual(overall["matcher_matches_truth"], 10)
        self.assertEqual(overall["llm_defers_with_true_fault"], 2)
        self.assertEqual(summary["by_mode"]["with_reference"]["llm_matches_truth"], 6)
        self.assertEqual(summary["by_health"]["healthy"]["cases"], 4)
        record = json.loads((directory / "cases.jsonl").read_text("utf-8").splitlines()[0])
        self.assertEqual(record["case_id"], "run-a:clean:reference_free")
        self.assertEqual(record["review"]["attempts"][0]["raw_text"][0], "{")

        rerun = EchoClient()
        self.run_reviews(rerun)
        self.assertEqual(rerun.requests, [])

    def test_conflicts_are_counted_by_kind_and_retries_by_first_attempt(self):
        summary = module.summarize(
            self.run_reviews(EchoClient("frozen_head", bad_first_reply=True), limit=6)
        )
        overall = summary["overall"]
        self.assertFalse(summary["complete"])
        self.assertEqual((overall["completed"], summary["expected_cases"]), (6, 12))
        self.assertIn("    6/12", module.summary_table([summary]).splitlines()[1])
        self.assertEqual(overall["valid_on_first_attempt"], 5)
        self.assertEqual(overall["issue_types"], {"schema": 1})
        self.assertEqual(overall["agrees_with_matcher"], 1)
        self.assertEqual(
            overall["conflicts"], {"different_diagnosis": 4, "assumed_reference_condition": 1}
        )
        self.assertEqual(overall["llm_matches_truth"], 2)

    def test_api_errors_wait_stop_the_run_and_are_retried_on_resume(self):
        failing, waits = EchoClient(error=True), []
        directory = self.run_reviews(failing, pause=4.0, error_wait=30.0, sleep=waits.append)
        self.assertEqual(len(failing.requests), module.MAX_CONSECUTIVE_API_ERRORS)
        self.assertEqual(waits, [30.0, 30.0])
        record = json.loads((directory / "cases.jsonl").read_text("utf-8").splitlines()[0])
        self.assertEqual(
            record["review"]["error"], {"type": "ServiceUnavailable", "status_code": 503}
        )
        self.assertEqual(module.summarize(directory)["overall"]["api_error"], 3)
        resumed = EchoClient()
        self.run_reviews(resumed)
        self.assertEqual(len(resumed.requests), 12)
        summary = module.summarize(directory)
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["overall"]["api_error"], 0)
        lines = (directory / "cases.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 15)

    def test_pause_between_cases_and_settings_cannot_change_midway(self):
        pauses = []
        self.run_reviews(EchoClient(), limit=3, pause=4.0, sleep=pauses.append)
        self.assertEqual(pauses, [4.0, 4.0])
        for change in ({"reports": self.reports[:1]}, {"response_format": "json_object"}):
            with self.subTest(change=list(change)):
                with self.assertRaisesRegex(ValueError, "other settings or reports"):
                    self.run_reviews(EchoClient(), **change)

    def test_only_the_failed_reviews_of_an_earlier_run_are_reviewed_again(self):
        earlier = self.run_reviews(EchoClient(bad=True), limit=2)
        self.assertEqual(module.summarize(earlier)["overall"]["invalid_output"], 2)
        arguments = ["eval", "run", "--reports", *map(str, self.reports), "--provider", "ollama"]
        arguments += ["--model", "fake-model", "--output-dir", str(self.root / "again")]
        arguments += ["--only-invalid-from", str(earlier)]
        client = EchoClient()
        with (
            patch.object(sys, "argv", arguments),
            patch("runsleuth.llm_client.create_llm_client", return_value=(client, "fake-model")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            module.main()
        self.assertEqual(len(client.requests), 2)
        again = module.summarize(module.run_directory(self.root / "again", "ollama", "fake-model"))
        self.assertTrue(again["complete"])
        self.assertEqual((again["expected_cases"], again["overall"]["completed"]), (2, 2))
        self.assertEqual(
            [case["case_id"] for case in again["cases"]],
            ["run-a:clean:reference_free", "run-a:clean:with_reference"],
        )
        for only_cases in (set(), {"run-z:clean:reference_free"}):
            with self.subTest(only_cases=only_cases), self.assertRaises(ValueError):
                self.run_reviews(EchoClient(), only_cases=only_cases)

    def test_runs_before_prompt_versioning_summarize_as_prompt_1(self):
        directory = self.run_reviews(EchoClient(), response_format="json_object", limit=1)
        settings = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(settings["response_format"], "json_object")
        for key in ("prompt_version", "payload_version", "system_prompt_sha256", "response_format"):
            settings.pop(key)
        (directory / "run.json").write_text(json.dumps(settings), encoding="utf-8")
        summary = module.summarize(directory)
        self.assertEqual(
            (summary["prompt_version"], summary["payload_version"], summary["response_format"]),
            (1, 1, "json_object"),
        )
        self.assertIn("fake-model p1", module.summary_table([summary]))

    def test_reports_can_come_from_a_signature_matching_record(self):
        cases = [
            {"report": str(path).replace("/", "\\")} for path in self.reports for _ in VARIANTS
        ]
        record = self.root / "signature-matching.json"
        record.write_text(json.dumps({"cases": cases}), encoding="utf-8")
        self.assertEqual(
            [path.resolve() for path in module.record_reports(record)],
            [path.resolve() for path in self.reports],
        )

    def test_cli_runs_one_model_and_summarizes_several(self):
        directories = []
        for model in ("model-a", "model-b"):
            arguments = ["eval", "run", "--reports", *map(str, self.reports)]
            arguments += ["--provider", "ollama"]
            arguments += ["--model", model, "--output-dir", str(self.output)]
            client = EchoClient()
            with (
                patch.object(sys, "argv", arguments),
                patch("runsleuth.llm_client.create_llm_client", return_value=(client, model)),
                contextlib.redirect_stdout(io.StringIO()) as printed,
            ):
                module.main()
            self.assertIn("run_directory=", printed.getvalue())
            directories.append(module.run_directory(self.output, "ollama", model))
        output = self.root / "results" / "llm.json"
        arguments = ["eval", "summarize", *map(str, directories), "--output", str(output)]
        with (
            patch.object(sys, "argv", arguments),
            contextlib.redirect_stdout(io.StringIO()) as printed,
        ):
            module.main()
        result = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual([run["model"] for run in result["runs"]], ["model-a", "model-b"])
        self.assertEqual(len(result["runs"][0]["cases"]), 12)
        self.assertIn("one sampled reply per case", result["scope"])
        table = printed.getvalue().splitlines()
        self.assertEqual(len(table), 3)
        self.assertIn("12/12", table[1])


if __name__ == "__main__":
    unittest.main()
