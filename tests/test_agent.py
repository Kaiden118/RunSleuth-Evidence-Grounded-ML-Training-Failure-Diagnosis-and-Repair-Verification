"""Offline checks for the Gemini-compatible diagnostic agent."""

import json
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from runsleuth.diagnostic_tools import DiagnosticTools, ToolResult

pytest.importorskip("openai")
agent_module = import_module("runsleuth.agent")
ChatCompletion = import_module("openai.types.chat").ChatCompletion

EVIDENCE_REQUESTS = [
    ("inspect_training_source", {"source_path": "train.py"}),
    ("load_run_config", {"run_directory": "clean"}),
    ("load_run_config", {"run_directory": "candidate"}),
    ("compare_runs", {"reference_run": "clean", "candidate_run": "candidate"}),
]


@pytest.fixture
def tools(tmp_path, monkeypatch):
    diagnostic_tools = DiagnosticTools(tmp_path)

    def execute(name, arguments):
        if name == "load_run_config":
            directory = arguments["run_directory"]
            data = {
                "config_path": f"{directory}/config.json",
                "config": {
                    "learning_rate": 0.001,
                    "optimizer_step_enabled": directory == "clean",
                },
            }
        elif name == "compare_runs":
            data = {
                "clean": {"final_epoch_parameter_update_norm": 0.043},
                "candidate": {"final_epoch_parameter_update_norm": 0.0},
            }
        elif name == "inspect_training_source":
            data = {
                "source_path": arguments["source_path"],
                "calls": [
                    {
                        "function": "train_one_epoch",
                        "call": "optimizer.step",
                        "line": 85,
                        "code": "optimizer.step()",
                        "conditions": ["optimizer_step_enabled"],
                    }
                ],
            }
        else:
            raise AssertionError(f"Unexpected fixture tool: {name}")
        return ToolResult(tool_name=name, ok=True, data=data)

    monkeypatch.setattr(diagnostic_tools, "execute", Mock(side_effect=execute))
    return diagnostic_tools


def diagnosis_text():
    return json.dumps(
        {
            "status": "failure_detected",
            "summary": "Recorded evidence supports skipped optimizer updates.",
            "ranked_causes": [
                {
                    "root_cause": "missing_optimizer_step",
                    "confidence": "high",
                    "explanation": "Updates are disabled and recorded updates are zero.",
                    "evidence": [
                        {
                            "call_id": "call-2",
                            "path": ["config", "optimizer_step_enabled"],
                            "expected_value": False,
                        },
                        {
                            "call_id": "call-3",
                            "path": ["candidate", "final_epoch_parameter_update_norm"],
                            "expected_value": 0.0,
                        },
                    ],
                }
            ],
            "uncertainties": ["Current source is not a historical execution trace."],
            "proposed_patch": {
                "config_call_id": "call-2",
                "field": "optimizer_step_enabled",
                "old_value": False,
                "new_value": True,
            },
        }
    )


def tool_call(index):
    name, arguments = EVIDENCE_REQUESTS[index]
    return {
        "id": f"call-{index}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
        "extra_content": {"google": {"thought_signature": "fake-signature"}},
    }


def reply(calls=(), text="", finish_reason=None, with_usage=True):
    return ChatCompletion.model_validate(
        {
            "id": "response-fixture",
            "object": "chat.completion",
            "created": 0,
            "model": "fake-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason or ("tool_calls" if calls else "stop"),
                    "message": {
                        "role": "assistant",
                        "content": text,
                        "tool_calls": list(calls) or None,
                    },
                }
            ],
            "usage": (
                {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
                if with_usage
                else None
            ),
        }
    )


def run_fake(tools, responses, **limits):
    create = Mock(side_effect=responses)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    report = agent_module.run_diagnostic_agent(
        client,
        "fake-model",
        tools,
        source_path="train.py",
        reference_run="clean",
        candidate_run="candidate",
        **limits,
    )
    return report, create


@pytest.mark.parametrize("finish_reason", ["tool_calls", "stop"])
def test_agent_collects_evidence_and_preserves_signatures(tools, finish_reason):
    report, create = run_fake(
        tools,
        [
            reply([tool_call(i) for i in range(4)], finish_reason=finish_reason),
            reply(text=diagnosis_text()),
        ],
        max_model_calls=2,
        max_tool_calls=4,
    )

    assert report.status == "completed"
    assert report.pending_evidence == []
    assert report.diagnosis == json.loads(diagnosis_text())
    assert json.loads(report.final_text) == report.diagnosis
    assert report.raw_final_text == diagnosis_text()
    assert (report.model_calls, report.tool_calls) == (2, 4)
    assert (report.input_tokens, report.output_tokens) == (20, 4)
    assert report.usage_complete
    assert create.call_args_list[0].kwargs["tool_choice"] == "required"
    collection_request = create.call_args_list[0].kwargs
    final_request = create.call_args.kwargs
    assert "response_format" not in collection_request
    assert final_request["response_format"]["type"] == "json_schema"
    assert "tools" not in final_request
    assert "tool_choice" not in final_request

    messages = create.call_args.kwargs["messages"]
    assistant = next(message for message in messages if message["role"] == "assistant")
    signature = assistant["tool_calls"][0]["extra_content"]["google"]["thought_signature"]
    assert signature == "fake-signature"

    outputs = [message for message in messages if message["role"] == "tool"]
    assert [output["tool_call_id"] for output in outputs] == [f"call-{i}" for i in range(4)]
    assert all(json.loads(output["content"])["ok"] for output in outputs)
    for output in outputs:
        payload = json.loads(output["content"])
        assert payload["call_id"] == output["tool_call_id"]
    assert "fake-signature" not in report.to_json()


@pytest.mark.parametrize(
    ("limits", "expected_calls"),
    [
        ({"max_model_calls": 2}, 1),
        ({"max_tool_calls": 2}, 2),
    ],
)
def test_agent_stops_when_evidence_budget_is_exhausted(tools, limits, expected_calls):
    report, create = run_fake(
        tools,
        [reply([tool_call(i)]) for i in range(4)],
        **limits,
    )
    assert report.status == "budget_exhausted"
    assert report.pending_evidence
    assert report.final_text is None
    assert report.model_calls == report.tool_calls == expected_calls
    assert create.call_count == expected_calls


def test_agent_rejects_entire_oversized_batch(tools):
    report, _ = run_fake(
        tools,
        [reply([tool_call(i) for i in range(4)])],
        max_tool_calls=3,
    )
    assert report.status == "budget_exhausted"
    assert report.tool_calls == 0
    tools.execute.assert_not_called()


def test_agent_can_recover_from_invalid_arguments_within_budget(tools):
    bad_call = tool_call(0)
    bad_call["id"] = "bad-call"
    bad_call["function"]["arguments"] = "{"

    report, create = run_fake(
        tools,
        [reply([bad_call])]
        + [reply([tool_call(i)]) for i in range(4)]
        + [reply(text=diagnosis_text())],
    )
    assert report.status == "completed"
    assert (report.model_calls, report.tool_calls) == (6, 5)
    assert tools.execute.call_count == 4
    assert report.tool_trace[0]["result"]["error_code"] == "invalid_arguments"

    messages = create.call_args_list[1].kwargs["messages"]
    output = next(message for message in messages if message["role"] == "tool")
    assert output["tool_call_id"] == "bad-call"
    assert json.loads(output["content"])["ok"] is False


@pytest.mark.parametrize(
    ("finish_reason", "expected_status"),
    [
        ("length", "response_not_completed"),
        ("content_filter", "refused"),
    ],
)
def test_agent_does_not_execute_tools_from_blocked_responses(tools, finish_reason, expected_status):
    report, _ = run_fake(tools, [reply([tool_call(0)], finish_reason=finish_reason)])
    assert report.status == expected_status
    assert report.tool_calls == 0
    tools.execute.assert_not_called()


def test_agent_rejects_duplicate_call_ids(tools):
    report, _ = run_fake(tools, [reply([tool_call(0), tool_call(0)])])
    assert report.status == "protocol_error"
    tools.execute.assert_not_called()


def test_failed_tool_does_not_satisfy_required_evidence(tools):
    tools.execute.side_effect = lambda name, arguments: ToolResult(
        tool_name=name,
        ok=False,
        error_code="not_found",
        error_message="Fixture file is unavailable",
    )
    report, _ = run_fake(tools, [reply([tool_call(0)]), reply(text="Premature diagnosis.")])
    assert report.status == "protocol_error"
    assert len(report.pending_evidence) == 4
    assert report.final_text is None


def test_agent_rejects_tools_during_finalization(tools):
    report, _ = run_fake(
        tools,
        [
            reply([tool_call(i) for i in range(4)]),
            reply([tool_call(0)]),
        ],
    )
    assert report.status == "protocol_error"
    assert report.tool_calls == tools.execute.call_count == 4


def test_agent_records_api_failure_without_retry(tools, monkeypatch):
    class FakeAPIError(Exception):
        code = "quota_exhausted"

    monkeypatch.setattr(agent_module, "APIError", FakeAPIError)
    report, create = run_fake(tools, [FakeAPIError()])

    assert report.status == "api_error"
    assert "quota_exhausted" in report.error
    assert report.usage_complete is False
    assert create.call_count == report.model_calls == 1
    tools.execute.assert_not_called()


def test_agent_marks_missing_usage_as_incomplete(tools):
    report, _ = run_fake(
        tools,
        [
            reply([tool_call(i) for i in range(4)], with_usage=False),
            reply(text=diagnosis_text()),
        ],
    )
    assert report.status == "completed"
    assert report.usage_complete is False
    assert (report.input_tokens, report.output_tokens) == (10, 2)


@pytest.mark.parametrize(
    ("case", "error_fragment"),
    [
        ("invalid_json", "Invalid JSON"),
        ("missing_field", "summary"),
        ("unknown_id", "Unknown evidence call ID"),
        ("wrong_value", "Evidence value mismatch"),
        ("bool_as_number", "Evidence value mismatch"),
        ("missing_path", "Evidence path does not exist"),
        ("invalid_index", "Evidence path does not exist"),
        ("failed_call", "Evidence call did not succeed"),
        ("reference_patch", "Patch must target the candidate"),
        ("uncited_patch", "First-ranked cause must cite"),
        ("wrong_patch_field", "Patch field must match"),
    ],
)
def test_agent_rejects_invalid_diagnoses(tools, case, error_fragment):
    payload = json.loads(diagnosis_text())
    cause = payload["ranked_causes"][0]
    citation = cause["evidence"][0]
    calls = [tool_call(i) for i in range(4)]

    if case == "missing_field":
        del payload["summary"]
    elif case == "unknown_id":
        citation["call_id"] = "invented-call"
    elif case == "wrong_value":
        cause["evidence"][1]["expected_value"] = 0.5
    elif case == "bool_as_number":
        citation["expected_value"] = 0
    elif case == "missing_path":
        citation["path"] = ["config", "nonexistent_field"]
    elif case == "invalid_index":
        citation.update(
            call_id="call-0",
            path=["calls", 99, "line"],
            expected_value=85,
        )
    elif case == "failed_call":
        bad_call = tool_call(0)
        bad_call["id"] = "failed-call"
        bad_call["function"]["arguments"] = "{"
        calls.insert(0, bad_call)
        citation["call_id"] = "failed-call"
    elif case == "reference_patch":
        payload["proposed_patch"]["config_call_id"] = "call-1"
    elif case == "uncited_patch":
        cause["evidence"].pop(0)
    elif case == "wrong_patch_field":
        payload["proposed_patch"].update(field="learning_rate", old_value=0.1, new_value=0.001)

    raw_text = "{" if case == "invalid_json" else json.dumps(payload)
    report, create = run_fake(tools, [reply(calls), reply(text=raw_text)])

    assert report.status == "invalid_diagnosis"
    assert error_fragment in report.error
    assert report.raw_final_text == raw_text
    assert report.diagnosis is None
    assert report.final_text is None
    assert report.model_calls == create.call_count == 2
    assert report.tool_calls == len(calls)


def test_agent_accepts_explicit_abstention(tools):
    payload = {
        "status": "no_supported_failure",
        "summary": "No supported cause is established.",
        "ranked_causes": [],
        "uncertainties": ["Other failure mechanisms remain possible."],
        "proposed_patch": None,
    }
    report, _ = run_fake(
        tools,
        [
            reply([tool_call(i) for i in range(4)]),
            reply(text=json.dumps(payload)),
        ],
    )
    assert report.status == "completed"
    assert report.diagnosis == payload
    assert report.diagnosis["proposed_patch"] is None


def test_agent_rejects_json_when_generation_was_truncated(tools):
    report, _ = run_fake(
        tools,
        [
            reply([tool_call(i) for i in range(4)]),
            reply(text=diagnosis_text(), finish_reason="length"),
        ],
    )
    assert report.status == "response_not_completed"
    assert report.diagnosis is None
    assert report.final_text is None
