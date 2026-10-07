"""Offline tests for run diagnosis: matcher first, checked LLM review second (fake client)."""

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

from runsleuth import diagnose_run as module


class FakeClient:
    """Mimics client.chat.completions.create; an Exception reply is raised."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        self.requests.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
        )


def reply(diagnosis="high_learning_rate", value=0.01, disagreement=None):
    return json.dumps(
        {
            "diagnosis": diagnosis,
            "explanation": "The inferred first-step learning rate is far above the range.",
            "citations": [
                {"name": "backbone_inferred_first_step_lr", "value": value, "role": "supports"},
                {"name": "optimizer_family", "value": "adam", "role": "supports"},
            ],
            "repair": "Lower the learning rate to the reference value.",
            "confidence": "high",
            "disagreement_with_matcher": disagreement,
        }
    )


class DiagnoseRunTests(unittest.TestCase):
    high_run = {**SCENARIOS["high_learning_rate"], "optimizer": "AdamW"}

    def review(self, replies):
        client = FakeClient(replies)
        result = module.diagnose_run(self.high_run, client=client, model="fake-model")
        return result, client

    def test_without_a_client_the_matcher_decides(self):
        result = module.diagnose_run(self.high_run)
        self.assertEqual(result["final_diagnosis"], "high_learning_rate")
        self.assertIsNone(result["llm"])
        self.assertEqual(result["decided_by"], "signature_matcher")

    def test_agreeing_review_is_recorded_with_usage(self):
        result, client = self.review([reply()])
        self.assertEqual(result["llm"]["status"], "completed")
        self.assertFalse(result["conflict"])
        self.assertEqual(result["decided_by"], "signature_matcher_and_llm")
        self.assertEqual((result["llm"]["model_calls"], result["llm"]["input_tokens"]), (1, 100))
        self.assertEqual(client.requests[0]["response_format"], {"type": "json_object"})
        payload = json.loads(client.requests[0]["messages"][1]["content"])
        self.assertEqual(payload["matcher_diagnosis"], "high_learning_rate")
        json.dumps(result, allow_nan=False)

    def test_payload_keeps_requires_and_contradicts_apart(self):
        # A real review misread a false contradicting condition as a failed requirement
        # when both lists were merged; roles must stay explicit.
        _, client = self.review([reply()])
        system, user = (message["content"] for message in client.requests[0]["messages"])
        self.assertIn("does not count against the signature", system)
        verdicts = {item["id"]: item for item in json.loads(user)["matcher_verdicts"]}
        contradiction = verdicts["high_learning_rate"]["contradicts"][0]
        self.assertEqual(
            (contradiction["evidence"], contradiction["result"]),
            ("max_optimizer_steps_per_forward", False),
        )
        self.assertNotIn("conditions", verdicts["high_learning_rate"])

    def test_pending_reference_can_be_agreed_and_skipping_it_is_a_named_conflict(self):
        # A real review of a frozen head without a reference had no way to agree:
        # pending_reference was not an allowed answer.
        def frozen_reply(diagnosis):
            return json.dumps(
                {
                    "diagnosis": diagnosis,
                    "explanation": "The head is frozen; only a reference can show intent.",
                    "citations": [
                        {"name": "head_max_trainable_fraction", "value": 0.0, "role": "supports"}
                    ],
                    "repair": "Confirm intent with a reference run.",
                    "confidence": "medium",
                    "disagreement_with_matcher": None,
                }
            )

        frozen = SCENARIOS["frozen_head"]
        for diagnosis, conflict, kind in (
            ("pending_reference", False, None),
            ("frozen_head", True, "assumed_reference_condition"),
        ):
            with self.subTest(diagnosis=diagnosis):
                client = FakeClient([frozen_reply(diagnosis)])
                result = module.diagnose_run(frozen, client=client, model="fake-model")
                self.assertEqual(result["final_diagnosis"], "pending_reference")
                self.assertEqual(result["conflict"], conflict)
                self.assertEqual(result.get("conflict_kind"), kind)

    def test_wrong_citation_is_retried_with_the_issues(self):
        result, client = self.review([reply(value=0.5), reply()])
        self.assertEqual(result["llm"]["status"], "completed")
        self.assertEqual(result["llm"]["model_calls"], 2)
        self.assertEqual(result["llm"]["attempts"][0]["issues"][0]["type"], "wrong_value")
        self.assertIn("wrong_value", client.requests[1]["messages"][-1]["content"])

    def test_disagreement_is_flagged_and_the_matcher_stays_final(self):
        result, _ = self.review([reply("missing_optimizer_step", disagreement="I think so.")])
        self.assertTrue(result["conflict"])
        self.assertEqual(result["final_diagnosis"], "high_learning_rate")
        self.assertEqual(result["decided_by"], "signature_matcher")
        self.assertIn("human review", result["llm_note"])

    def test_invalid_replies_and_api_errors_fall_back_to_the_matcher(self):
        cases = (
            (["not json", reply("made_up_fault")], "invalid_output"),
            ([RuntimeError("quota exceeded")], "api_error"),
        )
        for replies, status in cases:
            with self.subTest(status=status):
                result, _ = self.review(replies)
                self.assertEqual(result["llm"]["status"], status)
                self.assertEqual(result["final_diagnosis"], "high_learning_rate")
                self.assertFalse(result["conflict"])

    def test_validation_names_unknown_evidence_and_schema_errors(self):
        evidence = {"optimizer_family": "adam"}
        allowed = {"high_learning_rate"}
        parsed, issues = module.validate_llm_diagnosis(reply(), evidence, allowed)
        self.assertIsNone(parsed)
        self.assertEqual(
            issues[0], {"type": "unknown_evidence", "detail": "backbone_inferred_first_step_lr"}
        )
        parsed, issues = module.validate_llm_diagnosis('{"diagnosis": 1}', evidence, allowed)
        self.assertIsNone(parsed)
        self.assertTrue(all(issue["type"] == "schema" for issue in issues))

    def test_cli_reads_the_optimizer_from_the_experiment_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment = Path(directory) / "experiment"
            (experiment / "high_learning_rate").mkdir(parents=True)
            (experiment / "environment.json").write_text('{"optimizer": "AdamW"}', "utf-8")
            run_path = experiment / "high_learning_rate" / "run_report.json"
            run_path.write_text(json.dumps(SCENARIOS["high_learning_rate"]), encoding="utf-8")
            arguments = ["diagnose", "--run", str(run_path), "--no-llm", "--output-dir", directory]
            with patch.object(sys, "argv", arguments), contextlib.redirect_stdout(io.StringIO()):
                module.main()
            (report,) = Path(directory).glob("diagnosis-*.json")
            saved = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual(saved["final_diagnosis"], "high_learning_rate")

    def test_cli_without_llm_writes_a_report_and_prints_a_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            run_path = Path(directory) / "run_report.json"
            run_path.write_text(json.dumps(SCENARIOS["missing_optimizer_step"]), encoding="utf-8")
            output = io.StringIO()
            arguments = ["diagnose", "--run", str(run_path), "--no-llm", "--output-dir", directory]
            with patch.object(sys, "argv", arguments), contextlib.redirect_stdout(output):
                module.main()
            reports = list(Path(directory).glob("diagnosis-*.json"))
            self.assertEqual(len(reports), 1)
            saved = json.loads(reports[0].read_text(encoding="utf-8"))
        text = output.getvalue()
        self.assertEqual(saved["final_diagnosis"], "missing_optimizer_step")
        self.assertIn("Diagnosis: missing_optimizer_step", text)
        self.assertIn("+ max_optimizer_steps_per_forward = 0.0", text)
        self.assertIn("Repair: Call optimizer.step()", text)


if __name__ == "__main__":
    unittest.main()
