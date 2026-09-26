"""Regression coverage for fixed task paths and bounded evidence collection."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

import runsleuth.agent as agent
from runsleuth.diagnostic_tools import ToolResult

SOURCE = "src/runsleuth/train.py"
REFERENCE = "artifacts/runs/reference"
CANDIDATE = "artifacts/runs/candidate"
HEALTHY_DIAGNOSIS = {
    "status": "no_supported_failure",
    "summary": "No supported failure was established from the collected evidence.",
    "ranked_causes": [],
    "uncertainties": ["Static inspection is not a runtime execution trace."],
    "proposed_patch": None,
}


def request(name, **arguments):
    return {"name": name, "arguments": arguments}


def required_requests(candidate=CANDIDATE):
    requests = [
        request("inspect_training_source", source_path=SOURCE),
        request("load_run_config", run_directory=REFERENCE),
        request("load_run_config", run_directory=candidate),
        request("compare_runs", reference_run=REFERENCE, candidate_run=candidate),
    ]
    unique = []
    for item in requests:
        if item not in unique:
            unique.append(item)
    return unique


def tool_definitions():
    fields = {
        "inspect_training_source": ["source_path"],
        "load_run_config": ["run_directory"],
        "compare_runs": ["reference_run", "candidate_run"],
    }
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"Read evidence with {name}.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        field: {"type": "string", "description": "Exact task path."}
                        for field in names
                    },
                    "required": names,
                },
            },
        }
        for name, names in fields.items()
    ]


def response(requests=(), *, prefix="call", final=False, signature=None):
    calls = [
        SimpleNamespace(
            id=f"{prefix}-{index}",
            type="function",
            function=SimpleNamespace(name=item["name"], arguments=json.dumps(item["arguments"])),
        )
        for index, item in enumerate(requests)
    ]
    payload = {"role": "assistant"}
    if final:
        payload["content"] = json.dumps(HEALTHY_DIAGNOSIS)
    else:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in calls
        ]
    if signature is not None:
        payload["tool_calls"][0]["extra_content"] = {"google": {"thought_signature": signature}}
    message = SimpleNamespace(
        refusal=None,
        content=payload.get("content"),
        tool_calls=calls or None,
        model_dump=lambda **kwargs: deepcopy(payload),
    )
    usage_values = {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}
    return SimpleNamespace(
        id=f"response-{prefix}",
        model="test-model",
        usage=SimpleNamespace(**usage_values, model_dump=lambda: dict(usage_values)),
        choices=[SimpleNamespace(finish_reason="stop" if final else "tool_calls", message=message)],
    )


class FakeTools:
    def __init__(self):
        self.calls = []

    def execute(self, name, arguments):
        self.calls.append(request(name, **deepcopy(arguments)))
        return ToolResult(
            tool_name=name,
            ok=True,
            data={},
            error_code=None,
            error_message=None,
        )


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        return next(self.responses)


@pytest.fixture
def environment(monkeypatch):
    definitions = tool_definitions()
    validations = []
    monkeypatch.setattr(agent, "build_openai_tools", lambda tools: definitions)

    def validate(diagnosis, trace, *, candidate_run):
        validations.append((diagnosis, deepcopy(trace), candidate_run))

    monkeypatch.setattr(agent, "validate_diagnosis_evidence", validate)
    return SimpleNamespace(definitions=definitions, validations=validations)


def run_agent(client, tools, *, candidate=CANDIDATE, **limits):
    return agent.run_diagnostic_agent(
        client,
        "test-model",
        tools,
        source_path=SOURCE,
        reference_run=REFERENCE,
        candidate_run=candidate,
        **limits,
    )


@pytest.mark.parametrize("raw_arguments", ["{broken", "[]", '"path"'])
def test_malformed_arguments_never_reach_tools(raw_arguments):
    tools = FakeTools()
    arguments, result = agent.execute_tool_call(
        tools,
        "load_run_config",
        raw_arguments,
        allowed_requests=required_requests(),
    )
    assert arguments is None
    assert result.ok is False
    assert result.error_code == "invalid_arguments"
    assert tools.calls == []


@pytest.mark.parametrize(
    "requested",
    [
        request("load_run_config", run_directory="artifacts/runs/invented"),
        request("compare_runs", reference_run=CANDIDATE, candidate_run=REFERENCE),
        request("load_run_config", run_directory=REFERENCE, extra="unexpected"),
        request("unregistered_tool", run_directory=REFERENCE),
    ],
)
def test_scope_rejection_retains_arguments_without_dispatch(requested):
    tools = FakeTools()
    raw = json.dumps(requested["arguments"])
    arguments, result = agent.execute_tool_call(
        tools, requested["name"], raw, allowed_requests=required_requests()
    )
    assert arguments == requested["arguments"]
    assert result.ok is False
    assert result.error_code == "out_of_scope_request"
    assert tools.calls == []
    assert raw == json.dumps(requested["arguments"])


def test_default_execution_remains_compatible_and_approved_repeats_are_allowed():
    tools = FakeTools()
    item = request("load_run_config", run_directory=REFERENCE)
    raw = json.dumps(item["arguments"])
    arguments, result = agent.execute_tool_call(tools, item["name"], raw)
    assert arguments == item["arguments"]
    assert result.ok is True
    for allowed in (required_requests(), required_requests()):
        arguments, result = agent.execute_tool_call(
            tools, item["name"], raw, allowed_requests=allowed
        )
        assert arguments == item["arguments"]
        assert result.ok is True
    assert tools.calls == [item, item, item]


def test_scoped_schema_uses_pending_names_and_unique_values_without_mutation():
    original = tool_definitions()
    snapshot = deepcopy(original)
    pending = [
        request("load_run_config", run_directory=REFERENCE),
        request("load_run_config", run_directory=CANDIDATE),
        request("load_run_config", run_directory=REFERENCE),
    ]
    scoped = agent._scoped_tool_definitions(original, pending)
    assert len(scoped) == 1
    function = scoped[0]["function"]
    assert function["name"] == "load_run_config"
    parameters = function["parameters"]
    assert parameters["properties"]["run_directory"]["enum"] == [REFERENCE, CANDIDATE]
    assert parameters["additionalProperties"] is False
    assert parameters["type"] == "object"
    assert parameters["required"] == ["run_directory"]
    assert parameters["properties"]["run_directory"]["type"] == "string"
    assert parameters["properties"]["run_directory"]["description"] == "Exact task path."
    assert function["description"] == "Read evidence with load_run_config."
    assert original == snapshot
    parameters["required"].append("unexpected")
    parameters["properties"]["run_directory"]["enum"].append("unexpected")
    assert original == snapshot


def test_scoped_schema_recomputed_from_remaining_requests():
    original = tool_definitions()
    first = agent._scoped_tool_definitions(original, required_requests())
    second = agent._scoped_tool_definitions(
        original, [request("load_run_config", run_directory=CANDIDATE)]
    )
    first_config = next(item for item in first if item["function"]["name"] == "load_run_config")
    assert first_config["function"]["parameters"]["properties"]["run_directory"]["enum"] == [
        REFERENCE,
        CANDIDATE,
    ]
    assert second[0]["function"]["parameters"]["properties"]["run_directory"]["enum"] == [CANDIDATE]
    assert agent._scoped_tool_definitions(original, []) == []


@pytest.mark.parametrize("candidate, expected_calls", [(REFERENCE, 3), (CANDIDATE, 4)])
def test_complete_evidence_then_final_diagnosis_with_exact_budget(
    environment, candidate, expected_calls
):
    expected = required_requests(candidate)
    client = FakeClient([response(expected), response(final=True)])
    tools = FakeTools()
    report = run_agent(
        client, tools, candidate=candidate, max_model_calls=2, max_tool_calls=expected_calls
    )
    assert report.status == "completed"
    assert report.model_calls == 2
    assert report.tool_calls == expected_calls
    assert report.pending_evidence == []
    assert tools.calls == expected
    assert report.diagnosis == HEALTHY_DIAGNOSIS
    assert len(environment.validations) == 1
    assert environment.validations[0][2] == candidate
    assert client.requests[0]["tool_choice"] == "required"
    assert "tools" not in client.requests[1]
    assert client.requests[1]["response_format"]["type"] == "json_schema"


def test_task_paths_persist_after_source_and_configs_leave_pending(environment):
    evidence = required_requests()
    client = FakeClient(
        [
            response(evidence[:3], prefix="initial"),
            response(evidence[3:], prefix="compare"),
            response(final=True),
        ]
    )
    report = run_agent(client, FakeTools(), max_model_calls=3, max_tool_calls=4)
    assert report.status == "completed"
    for api_request in client.requests:
        system_text = api_request["messages"][0]["content"]
        assert SOURCE in system_text
        assert REFERENCE in system_text
        assert CANDIDATE in system_text
    remaining = client.requests[1]["tools"]
    assert [item["function"]["name"] for item in remaining] == ["compare_runs"]
    properties = remaining[0]["function"]["parameters"]["properties"]
    assert properties["reference_run"]["enum"] == [REFERENCE]
    assert properties["candidate_run"]["enum"] == [CANDIDATE]


def test_same_path_comparison_is_still_required_before_finalization(environment):
    evidence = required_requests(REFERENCE)
    client = FakeClient([response(evidence[:2])])
    tools = FakeTools()
    report = run_agent(client, tools, candidate=REFERENCE, max_model_calls=2, max_tool_calls=3)
    assert report.status == "budget_exhausted"
    assert report.pending_evidence == [evidence[2]]
    assert len(environment.validations) == 0
    assert report.diagnosis is None
    assert report.model_calls == 1


def test_provider_ignoring_path_enum_is_rejected_and_consumes_budget(environment):
    invented = request("load_run_config", run_directory="artifacts/runs/invented")
    client = FakeClient([response([invented], prefix="bad")])
    tools = FakeTools()
    report = run_agent(client, tools, max_model_calls=3, max_tool_calls=1)
    assert report.status == "budget_exhausted"
    assert report.model_calls == 1
    assert report.tool_calls == 1
    assert tools.calls == []
    assert report.pending_evidence == required_requests()
    assert report.tool_trace[0]["raw_arguments"] == json.dumps(invented["arguments"])
    assert report.tool_trace[0]["result"]["error_code"] == "out_of_scope_request"


def test_approved_repeated_request_in_parallel_batch_does_not_break_scope(environment):
    evidence = required_requests(REFERENCE)
    repeated = [evidence[0], evidence[0], *evidence[1:]]
    client = FakeClient([response(repeated), response(final=True)])
    tools = FakeTools()
    report = run_agent(client, tools, candidate=REFERENCE, max_model_calls=2, max_tool_calls=4)
    assert report.status == "completed"
    assert tools.calls == repeated
    assert report.tool_calls == 4
    assert all(item["result"]["ok"] for item in report.tool_trace)


def test_oversized_tool_batch_is_rejected_without_dispatch(environment):
    client = FakeClient([response(required_requests())])
    tools = FakeTools()
    report = run_agent(client, tools, max_model_calls=2, max_tool_calls=3)
    assert report.status == "budget_exhausted"
    assert report.error == "Tool call batch exceeds the remaining budget"
    assert report.tool_calls == 0
    assert report.tool_trace == []
    assert tools.calls == []


def test_final_evidence_validation_failure_is_not_bypassed(environment, monkeypatch):
    def reject(diagnosis, trace, *, candidate_run):
        assert candidate_run == REFERENCE
        assert len(trace) == 3
        raise ValueError("Synthetic evidence mismatch")

    monkeypatch.setattr(agent, "validate_diagnosis_evidence", reject)
    client = FakeClient([response(required_requests(REFERENCE)), response(final=True)])
    report = run_agent(
        client, FakeTools(), candidate=REFERENCE, max_model_calls=2, max_tool_calls=3
    )
    assert report.status == "invalid_diagnosis"
    assert report.error == "Synthetic evidence mismatch"
    assert report.raw_final_text == json.dumps(HEALTHY_DIAGNOSIS)
    assert report.diagnosis is None


def test_assistant_tool_metadata_and_thought_signature_survive_history(environment):
    signature = "opaque-test-thought-signature"
    client = FakeClient(
        [response(required_requests(REFERENCE), signature=signature), response(final=True)]
    )
    report = run_agent(
        client, FakeTools(), candidate=REFERENCE, max_model_calls=2, max_tool_calls=3
    )
    assert report.status == "completed"
    assistant = next(
        message for message in client.requests[1]["messages"] if message["role"] == "assistant"
    )
    assert assistant["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == signature
    assert assistant["tool_calls"][0]["function"]["name"] == "inspect_training_source"
