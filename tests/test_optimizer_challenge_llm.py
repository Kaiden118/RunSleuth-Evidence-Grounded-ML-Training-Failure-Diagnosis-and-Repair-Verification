"""Offline API doubles and identical label-blind validation for both LLM arms."""

import json
import unittest
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock, patch

from runsleuth.optimizer_challenge_cases import challenge_cases
from runsleuth.optimizer_challenge_llm import (
    BASELINE_PROTOCOL,
    REVISED_PROTOCOL,
    PredictionValidationError,
    replay_prediction,
    run_llm_method,
    system_prompt_for,
    validate_prediction,
)
from runsleuth.optimizer_factory_diagnosis import TOOL_NAME, factory_evidence

SOURCE_PATH = "inputs/factory.py"


def prediction_for(source, binding="current"):
    lines = source.splitlines()
    indices = [i for i, line in enumerate(lines, 1) if "model.fc =" in line or "updater =" in line]
    if len(indices) < 2:
        indices = [i for i, line in enumerate(lines, 1) if line.strip()][:2]
    return {
        "binding": binding,
        "explanation": "A source-based hypothesis.",
        "citations": [{"start_line": i, "end_line": i, "code": lines[i - 1]} for i in indices[:2]],
        "uncertainty": "No training execution is observed.",
    }


def response(*, final=None, tool_path=None, with_usage=True, call_name=TOOL_NAME):
    calls = []
    payload = {"role": "assistant", "content": json.dumps(final) if final is not None else None}
    if tool_path is not None:
        calls = [
            SimpleNamespace(
                id="source-call",
                type="function",
                function=SimpleNamespace(
                    name=call_name, arguments=json.dumps({"source_path": tool_path})
                ),
            )
        ]
        payload["tool_calls"] = [
            {
                "id": "source-call",
                "type": "function",
                "function": {"name": call_name, "arguments": calls[0].function.arguments},
                "extra_content": {"google": {"thought_signature": "signature"}},
            }
        ]
    counts = {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
    return SimpleNamespace(
        id="offline-response",
        model="fixture-model",
        usage=SimpleNamespace(**counts, model_dump=lambda: counts) if with_usage else None,
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls" if calls else "stop",
                message=SimpleNamespace(
                    refusal=None,
                    content=payload["content"],
                    tool_calls=calls,
                    model_dump=lambda **kw: deepcopy(payload),
                ),
            )
        ],
    )


@dataclass
class Result:
    tool_name: str
    ok: bool
    data: dict
    error_code: str | None = None
    error_message: str | None = None


class FakeTools:
    def __init__(self, source):
        self.source = source
        self.calls = []

    def execute(self, name, arguments):
        self.calls.append((name, arguments))
        return Result(
            name, True, factory_evidence(self.source.encode(), source_path=arguments["source_path"])
        )


class FakeAPIError(Exception):
    status_code = 503


def invoke(responses, *, source, method, tools=None, protocol=BASELINE_PROTOCOL):
    create = Mock(side_effect=responses)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with patch.dict("sys.modules", {"openai": SimpleNamespace(APIError=FakeAPIError)}):
        report = run_llm_method(
            client,
            "fixture-model",
            method=method,
            source_path=SOURCE_PATH,
            source=source,
            tools=tools,
            protocol=protocol,
        )
    return report, create


class ChallengeLLMTests(unittest.TestCase):
    def setUp(self):
        self.source = challenge_cases()[0]["source"]

    def test_wrong_semantic_label_with_valid_quotes_is_not_silently_corrected(self):
        payload = prediction_for(self.source, "current")  # True label is stale.
        report, create = invoke([response(final=payload)], source=self.source, method="llm_source")
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["prediction"]["binding"], "current")
        self.assertEqual(create.call_count, 1)
        self.assertTrue(report["first_pass_valid"])
        replay_prediction(
            report,
            source=self.source,
            source_path=SOURCE_PATH,
            method="llm_source",
            model="fixture-model",
        )

    def test_source_only_has_no_tool_output_expected_label_or_cross_case_history(self):
        report, create = invoke(
            [response(final=prediction_for(self.source))], source=self.source, method="llm_source"
        )
        request = create.call_args.kwargs
        self.assertNotIn("tools", request)
        self.assertEqual(len(request["messages"]), 2)
        user = json.loads(request["messages"][1]["content"])
        self.assertEqual(set(user), {"source_path", "numbered_source"})
        self.assertEqual(report["tool_trace"], [])

    def test_tool_arm_uses_fixed_scope_and_preserves_vendor_metadata(self):
        tools = FakeTools(self.source)
        report, create = invoke(
            [response(tool_path=SOURCE_PATH), response(final=prediction_for(self.source, "stale"))],
            source=self.source,
            method="tool_agent",
            tools=tools,
        )
        self.assertEqual(report["status"], "completed", report["error"])
        self.assertEqual((report["model_calls"], report["tool_calls"]), (2, 1))
        self.assertEqual(len(tools.calls), 1)
        requests = [call.kwargs for call in create.call_args_list]
        self.assertEqual(
            requests[0]["tools"][0]["function"]["parameters"]["properties"]["source_path"]["enum"],
            [SOURCE_PATH],
        )
        self.assertEqual(
            requests[1]["messages"][2]["tool_calls"][0]["extra_content"]["google"][
                "thought_signature"
            ],
            "signature",
        )
        replay_prediction(
            report,
            source=self.source,
            source_path=SOURCE_PATH,
            method="tool_agent",
            model="fixture-model",
        )

    def test_mismatched_source_in_tool_result_cannot_be_used(self):
        tools = FakeTools(challenge_cases()[1]["source"])
        report, _ = invoke(
            [response(tool_path=SOURCE_PATH)], source=self.source, method="tool_agent", tools=tools
        )
        self.assertEqual(report["status"], "tool_error")

    def test_out_of_scope_tool_path_never_reaches_dispatch(self):
        tools = FakeTools(self.source)
        report, _ = invoke(
            [response(tool_path="secret.py")], source=self.source, method="tool_agent", tools=tools
        )
        self.assertEqual(report["status"], "tool_error")
        self.assertEqual(tools.calls, [])

    def test_only_format_errors_get_one_correction_without_label_feedback(self):
        invalid = prediction_for(self.source, "current")
        invalid["citations"][0]["code"] = "invented()"
        correct_format = prediction_for(self.source, "current")
        report, create = invoke(
            [response(final=invalid), response(final=correct_format)],
            source=self.source,
            method="llm_source",
        )
        self.assertEqual(report["status"], "completed")
        self.assertFalse(report["first_pass_valid"])
        self.assertEqual(report["validation_retries"], 1)
        self.assertEqual(report["prediction"]["binding"], "current")
        feedback = create.call_args.kwargs["messages"][-1]["content"]
        self.assertIn("source", feedback)
        self.assertNotIn("expected_status", feedback)
        self.assertEqual((report["input_tokens"], report["output_tokens"]), (40, 20))

    def test_two_invalid_responses_stop_without_extra_requests(self):
        report, create = invoke(
            [response(final={}), response(final={})], source=self.source, method="llm_source"
        )
        self.assertEqual(report["status"], "invalid_prediction")
        self.assertEqual(create.call_count, 2)

    def test_503_records_partial_usage_without_automatic_retry_or_error_secret(self):
        report, _ = invoke(
            [response(tool_path=SOURCE_PATH), FakeAPIError("secret-token")],
            source=self.source,
            method="tool_agent",
            tools=FakeTools(self.source),
        )
        self.assertEqual(report["status"], "api_error")
        self.assertFalse(report["usage_complete"])
        self.assertEqual((report["input_tokens"], report["output_tokens"]), (20, 10))
        self.assertNotIn("secret-token", json.dumps(report))

    def test_missing_usage_stays_incomplete(self):
        report, _ = invoke(
            [response(final=prediction_for(self.source), with_usage=False)],
            source=self.source,
            method="llm_source",
        )
        self.assertFalse(report["usage_complete"])

    def test_citation_schema_rejects_wrong_types_duplicate_spans_and_extra_fields(self):
        for mode in ("bool_line", "duplicate", "bad_end", "extra", "one_span"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                value = prediction_for(self.source)
                if mode == "bool_line":
                    value["citations"][0]["start_line"] = True
                elif mode == "duplicate":
                    value["citations"][1] = deepcopy(value["citations"][0])
                elif mode == "bad_end":
                    value["citations"][0]["end_line"] = 1000
                elif mode == "extra":
                    value["proposed_patch"] = "remove head"
                else:
                    value["citations"].pop()
                validate_prediction(json.dumps(value), self.source)

    def test_replay_rejects_changed_semantics_or_model_identity(self):
        report, _ = invoke(
            [response(final=prediction_for(self.source))], source=self.source, method="llm_source"
        )
        report["prediction"]["binding"] = "stale"
        with self.assertRaises(ValueError):
            replay_prediction(
                report,
                source=self.source,
                source_path=SOURCE_PATH,
                method="llm_source",
                model="fixture-model",
            )

    def test_tool_unsupported_does_not_force_model_abstention(self):
        source = challenge_cases()[3]["source"]  # Local helper is fully specified.
        report, _ = invoke(
            [response(tool_path=SOURCE_PATH), response(final=prediction_for(source, "stale"))],
            source=source,
            method="tool_agent",
            tools=FakeTools(source),
        )
        self.assertEqual(report["status"], "completed")
        self.assertEqual(
            report["tool_trace"][0]["result"]["data"]["analysis"]["decision"], "unsupported"
        )
        self.assertEqual(report["prediction"]["binding"], "stale")

    def test_revised_feedback_reports_empty_uncertainty_and_one_span_together(self):
        source = challenge_cases()[6]["source"]
        # Reproduce the two simultaneous failures in the uploaded case-007 report.
        invalid = prediction_for(source, "stale")
        invalid["uncertainty"] = ""
        invalid["citations"] = [
            {"start_line": 5, "end_line": 6, "code": "\n".join(source.split("\n")[4:6])}
        ]
        with self.assertRaises(PredictionValidationError) as raised:
            validate_prediction(json.dumps(invalid), source, protocol=REVISED_PROTOCOL)
        self.assertEqual(
            {tuple(item["path"]) for item in raised.exception.issues},
            {("uncertainty",), ("citations",)},
        )
        fixed_format = prediction_for(source, "stale")  # Still the wrong semantic answer.
        report, create = invoke(
            [response(final=invalid), response(final=fixed_format)],
            source=source,
            method="llm_source",
            protocol=REVISED_PROTOCOL,
        )
        feedback = create.call_args.kwargs["messages"][-1]["content"]
        self.assertIn("uncertainty", feedback)
        self.assertIn("two distinct source spans", feedback)
        self.assertNotIn("expected", feedback)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["prediction"]["binding"], "stale")
        self.assertFalse(report["first_pass_valid"])
        self.assertEqual(report["validation_retries"], 1)
        replay_prediction(
            report,
            source=source,
            source_path=SOURCE_PATH,
            method="llm_source",
            model="fixture-model",
            protocol=REVISED_PROTOCOL,
        )

    def test_multiple_bad_quotes_are_reported_despite_another_bad_field(self):
        value = prediction_for(self.source)
        value["uncertainty"] = ""
        for quote in value["citations"]:
            quote["code"] = "incorrect quotation"
        with self.assertRaises(PredictionValidationError) as raised:
            validate_prediction(json.dumps(value), self.source, protocol=REVISED_PROTOCOL)
        self.assertEqual(
            {tuple(item["path"]) for item in raised.exception.issues},
            {("uncertainty",), ("citations", 0, "code"), ("citations", 1, "code")},
        )

    def test_protocols_accept_the_same_json_but_revised_reveals_all_format_errors(self):
        payload = prediction_for(self.source)
        examples = [
            payload,
            {},
            [],
            {**payload, "uncertainty": ""},
            {**payload, "citations": payload["citations"][:1]},
            {**payload, "binding": "unsupported"},
        ]
        for value in examples:
            with self.subTest(value=value):
                outcomes = []
                for protocol in (BASELINE_PROTOCOL, REVISED_PROTOCOL):
                    try:
                        result = validate_prediction(
                            json.dumps(value), self.source, protocol=protocol
                        )
                    except ValueError:
                        outcomes.append(None)
                    else:
                        outcomes.append(result.model_dump())
                self.assertEqual(outcomes[0], outcomes[1])
        value = {**payload, "uncertainty": "", "citations": payload["citations"][:1]}
        with self.assertRaises(ValueError) as raised:
            validate_prediction(json.dumps(value), self.source)
        self.assertNotIn("two distinct source spans", str(raised.exception))

    def test_revised_tool_disagreement_is_not_overwritten_or_given_answer_feedback(self):
        source = challenge_cases()[1]["source"]
        report, create = invoke(
            [response(tool_path=SOURCE_PATH), response(final=prediction_for(source, "stale"))],
            source=source,
            method="tool_agent",
            tools=FakeTools(source),
            protocol=REVISED_PROTOCOL,
        )
        self.assertEqual(report["status"], "completed")
        self.assertEqual(
            report["tool_trace"][0]["result"]["data"]["analysis"]["decision"], "no_change"
        )
        self.assertEqual(report["prediction"]["binding"], "stale")
        self.assertEqual(create.call_count, 2)
        self.assertEqual(report["validation_retries"], 0)

    def test_revised_uses_common_source_semantics_without_case_answers_in_both_arms(self):
        for method in ("llm_source", "tool_agent"):
            with self.subTest(method=method):
                replies = [response(tool_path=SOURCE_PATH)] if method == "tool_agent" else []
                replies.append(response(final=prediction_for(self.source)))
                report, create = invoke(
                    replies,
                    source=self.source,
                    method=method,
                    tools=FakeTools(self.source),
                    protocol=REVISED_PROTOCOL,
                )
                prompt = create.call_args_list[0].kwargs["messages"][0]["content"]
                self.assertTrue(prompt.startswith(system_prompt_for(REVISED_PROTOCOL)))
                self.assertNotIn("case-001", prompt)
                self.assertNotIn("tuple_alias", prompt)
                self.assertEqual(report["inference_protocol"], REVISED_PROTOCOL)
                self.assertEqual(len(report["system_prompt_sha256"]), 64)

    def test_replay_accepts_legacy_and_rejects_cross_protocol_reports(self):
        legacy, _ = invoke(
            [response(final=prediction_for(self.source))], source=self.source, method="llm_source"
        )
        legacy.pop("inference_protocol")
        legacy.pop("system_prompt_sha256")
        replay_prediction(
            legacy,
            source=self.source,
            source_path=SOURCE_PATH,
            method="llm_source",
            model="fixture-model",
        )
        for protocol in (REVISED_PROTOCOL, "unknown-protocol"):
            with self.subTest(protocol=protocol), self.assertRaises(ValueError):
                replay_prediction(
                    legacy,
                    source=self.source,
                    source_path=SOURCE_PATH,
                    method="llm_source",
                    model="fixture-model",
                    protocol=protocol,
                )

    def test_revised_failure_budget_and_usage_are_not_relaxed(self):
        for mode in ("bad_format", "api_error"):
            with self.subTest(mode=mode):
                replies = (
                    [response(final={}), response(final={})]
                    if mode == "bad_format"
                    else [FakeAPIError("secret-token")]
                )
                report, create = invoke(
                    replies, source=self.source, method="llm_source", protocol=REVISED_PROTOCOL
                )
                self.assertEqual(create.call_count, len(replies))
                self.assertEqual(report["limits"]["max_model_calls"], 2)
                self.assertEqual(
                    report["status"], "invalid_prediction" if mode == "bad_format" else "api_error"
                )
                if mode == "api_error":
                    self.assertFalse(report["usage_complete"])
                    self.assertNotIn("secret-token", json.dumps(report))


if __name__ == "__main__":
    unittest.main()
