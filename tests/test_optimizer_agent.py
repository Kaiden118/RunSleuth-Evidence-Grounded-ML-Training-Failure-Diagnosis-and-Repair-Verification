"""Offline integration coverage for the opt-in optimizer-binding agent profile."""

import json
import unittest
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch

import runsleuth.agent as agent
from runsleuth.camelyon_config import CamelyonConfig
from runsleuth.diagnostic_tools import ToolResult

SOURCE = "src/runsleuth/train.py"
REFERENCE = "artifacts/optimizer_experiments/pair/reference"
CANDIDATE = "artifacts/optimizer_experiments/pair/candidate"


def request(path):
    return {"name": "inspect_optimizer_run", "arguments": {"run_directory": path}}


def definitions():
    fields = {
        "inspect_optimizer_run": ["run_directory"],
        "inspect_training_source": ["source_path"],
        "load_run_config": ["run_directory"],
        "compare_runs": ["reference_run", "candidate_run"],
    }
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": "Read recorded evidence.",
                "parameters": {
                    "type": "object",
                    "properties": {field: {"type": "string"} for field in names},
                    "required": names,
                },
            },
        }
        for name, names in fields.items()
    ]


def response(requests=(), *, prefix="call", final=None):
    calls = [
        SimpleNamespace(
            id=f"{prefix}-{index}",
            type="function",
            function=SimpleNamespace(name=item["name"], arguments=json.dumps(item["arguments"])),
        )
        for index, item in enumerate(requests)
    ]
    payload = {"role": "assistant"}
    if final is not None:
        payload["content"] = json.dumps(final)
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
    message = SimpleNamespace(
        refusal=None,
        content=payload.get("content"),
        tool_calls=calls or None,
        model_dump=lambda **kwargs: deepcopy(payload),
    )
    counts = {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}
    return SimpleNamespace(
        id=f"response-{prefix}",
        model="test-model",
        usage=SimpleNamespace(**counts, model_dump=lambda: dict(counts)),
        choices=[
            SimpleNamespace(
                finish_reason="stop" if final is not None else "tool_calls", message=message
            )
        ],
    )


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        return next(self.responses)


class FakeTools:
    def __init__(self, data=None):
        self.data = data or {}
        self.calls = []

    def execute(self, name, arguments):
        self.calls.append({"name": name, "arguments": deepcopy(arguments)})
        value = self.data.get(arguments.get("run_directory"), {})
        return ToolResult(tool_name=name, ok=True, data=deepcopy(value))


def run_data(path, *, mismatch=False):
    return {
        "run_directory": path,
        "config": asdict(CamelyonConfig(epochs=1, device="cpu")),
        "initial_state_sha256": "a" * 64,
        "data_identity": {"selected_sample_ids_sha256": {"train": "b" * 64}},
        "optimizer_audit": {
            "missing_trainable_parameter_tensors": 2 if mismatch else 0,
            "missing_trainable_names": ["fc.bias", "fc.weight"] if mismatch else [],
            "foreign_parameter_tensors": 2 if mismatch else 0,
            "duplicate_parameter_occurrences": 0,
        },
        "epochs": [
            {
                "epoch": 1,
                "optimizer_steps": 2,
                "id_validation_accuracy": 0.98 if mismatch else 0.97,
                "id_validation_loss": 0.07 if mismatch else 0.08,
                "ood_validation_accuracy": 0.91 if mismatch else 0.90,
                "ood_validation_loss": 0.32 if mismatch else 0.38,
                "head": {
                    "mean_gradient_l2_norm": 0.3,
                    "mean_parameter_update_l2_norm": 0.0 if mismatch else 0.001,
                },
                "backbone": {
                    "mean_gradient_l2_norm": 1.6,
                    "mean_parameter_update_l2_norm": 0.04,
                },
            }
        ],
    }


def citation(call_id, data, path):
    value = data
    for key in path:
        value = value[key]
    return {"call_id": call_id, "path": path, "expected_value": value}


def diagnosis_payload(*, same_run=False):
    reference = run_data(REFERENCE)
    candidate = reference if same_run else run_data(CANDIDATE, mismatch=True)
    candidate_call = "call-0" if same_run else "call-1"
    audit_fields = [
        "missing_trainable_parameter_tensors",
        "foreign_parameter_tensors",
        "duplicate_parameter_occurrences",
    ]
    implementation = [
        citation(candidate_call, candidate, ["optimizer_audit", field]) for field in audit_fields
    ]
    if not same_run:
        implementation.extend(
            [
                citation(candidate_call, candidate, ["epochs", 0, "head", field])
                for field in ("mean_gradient_l2_norm", "mean_parameter_update_l2_norm")
            ]
        )
    metrics = [
        "id_validation_accuracy",
        "id_validation_loss",
        "ood_validation_accuracy",
        "ood_validation_loss",
    ]
    observed = (
        [("call-0", reference)]
        if same_run
        else [
            ("call-0", reference),
            (candidate_call, candidate),
        ]
    )
    return {
        "profile": "optimizer_binding",
        "implementation_status": "no_optimizer_binding_defect"
        if same_run
        else "optimizer_binding_defect",
        "summary": "The recorded optimizer audit and metrics were checked separately.",
        "implementation_evidence": implementation,
        "ranked_causes": []
        if same_run
        else [
            {
                "root_cause": "optimizer_parameter_mismatch",
                "confidence": "high",
                "explanation": "The optimizer omits current head parameters and contains foreign parameters.",
                "evidence": [deepcopy(implementation[0])],
            }
        ],
        "performance": {
            "status": "no_observed_degradation",
            "explanation": "No final validation metric worsened in this controlled pair.",
            "evidence": [
                citation(call_id, data, ["epochs", 0, metric])
                for call_id, data in observed
                for metric in metrics
            ],
        },
        "uncertainties": [
            "This is one recorded experiment; generalization has not been established."
        ],
        "recommended_action": "no_change" if same_run else "review_optimizer_binding",
        "proposed_patch": None,
    }


def run_agent(client, tools, *, candidate=CANDIDATE, **options):
    return agent.run_diagnostic_agent(
        client,
        "test-model",
        tools,
        source_path=SOURCE,
        reference_run=REFERENCE,
        candidate_run=candidate,
        profile="optimizer_binding",
        **options,
    )


class OptimizerAgentTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(agent, "build_openai_tools", lambda tools: definitions())
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_defect_with_better_performance_completes_without_a_repair(self):
        data = {REFERENCE: run_data(REFERENCE), CANDIDATE: run_data(CANDIDATE, mismatch=True)}
        tools = FakeTools(data)
        client = FakeClient(
            [
                response([request(REFERENCE), request(CANDIDATE)]),
                response(final=diagnosis_payload()),
            ]
        )
        report = run_agent(client, tools, max_model_calls=2, max_tool_calls=2)
        self.assertEqual(report.status, "completed", report.error)
        self.assertEqual((report.model_calls, report.tool_calls), (2, 2))
        self.assertEqual(report.pending_evidence, [])
        self.assertEqual(report.diagnosis["profile"], "optimizer_binding")
        self.assertEqual(report.diagnosis["implementation_status"], "optimizer_binding_defect")
        self.assertEqual(report.diagnosis["performance"]["status"], "no_observed_degradation")
        self.assertIsNone(report.diagnosis["proposed_patch"])
        self.assertNotIn("tools", client.requests[-1])
        self.assertEqual(client.requests[-1]["response_format"], {"type": "json_object"})
        self.assertEqual(report.output_mode, "json_object")

        final_prompt = client.requests[-1]["messages"][0]["content"]
        prompt_schema = json.loads(final_prompt.split("\nRequired JSON Schema:\n", 1)[1])
        self.assertEqual(prompt_schema["properties"]["proposed_patch"]["type"], "null")
        self.assertFalse(prompt_schema["additionalProperties"])
        self.assertIn("OptimizerEvidence", prompt_schema["$defs"])
        self.assertEqual(
            [item["name"] for item in tools.calls],
            [
                "inspect_optimizer_run",
                "inspect_optimizer_run",
            ],
        )
        self.assertEqual(report.input_tokens, 24)
        self.assertTrue(report.usage_complete)
        self.assertEqual(report.max_validation_retries, 1)
        self.assertEqual(report.validation_retries, 0)
        self.assertIs(report.first_pass_valid, True)
        self.assertEqual(
            report.validation_attempts,
            [
                {
                    "model_call": 2,
                    "raw_final_text": json.dumps(diagnosis_payload()),
                    "valid": True,
                    "error": None,
                }
            ],
        )

    def test_same_run_requires_one_inspection_and_reuses_its_citations(self):
        tools = FakeTools({REFERENCE: run_data(REFERENCE)})
        client = FakeClient(
            [
                response([request(REFERENCE)]),
                response(final=diagnosis_payload(same_run=True)),
            ]
        )
        report = run_agent(client, tools, candidate=REFERENCE, max_model_calls=2, max_tool_calls=1)
        self.assertEqual(report.status, "completed", report.error)
        self.assertEqual(tools.calls, [request(REFERENCE)])
        self.assertEqual(report.diagnosis["implementation_status"], "no_optimizer_binding_defect")
        self.assertEqual(report.diagnosis["recommended_action"], "no_change")

    def test_performance_claim_cannot_override_recorded_metrics(self):
        payload = diagnosis_payload()
        payload["performance"]["status"] = "degraded"
        tools = FakeTools(
            {REFERENCE: run_data(REFERENCE), CANDIDATE: run_data(CANDIDATE, mismatch=True)}
        )
        client = FakeClient(
            [
                response([request(REFERENCE), request(CANDIDATE)]),
                response(final=payload),
            ]
        )
        report = run_agent(client, tools, max_model_calls=2)
        self.assertEqual(report.status, "invalid_diagnosis")
        self.assertIsNone(report.diagnosis)
        self.assertIsNotNone(report.raw_final_text)
        self.assertEqual(report.model_calls, 2)
        self.assertEqual(len(client.requests), 2)
        self.assertEqual(report.validation_retries, 0)
        self.assertIs(report.first_pass_valid, False)
        self.assertEqual(
            report.validation_attempts,
            [
                {
                    "model_call": 2,
                    "raw_final_text": json.dumps(payload),
                    "valid": False,
                    "error": report.error,
                }
            ],
        )

    def test_modified_citation_value_is_not_accepted(self):
        payload = diagnosis_payload()
        payload["implementation_evidence"][0]["expected_value"] = 0
        tools = FakeTools(
            {REFERENCE: run_data(REFERENCE), CANDIDATE: run_data(CANDIDATE, mismatch=True)}
        )
        client = FakeClient(
            [
                response([request(REFERENCE), request(CANDIDATE)]),
                response(final=payload),
            ]
        )
        report = run_agent(client, tools, max_model_calls=2)
        self.assertEqual(report.status, "invalid_diagnosis")
        self.assertIsNone(report.diagnosis)

    def test_invalid_performance_claim_is_corrected_with_recorded_evidence(self):
        invalid = diagnosis_payload()
        invalid["performance"]["status"] = "degraded"
        valid = diagnosis_payload()
        rejected = response(final=invalid, prefix="invalid")
        rejected_text = "  " + json.dumps(invalid) + "\n"
        rejected.choices[0].message.content = rejected_text
        original_message = {
            "role": "assistant",
            "content": rejected_text,
            "provider_specific_fields": {"thought_signature": "preserve-this-metadata"},
        }
        rejected.choices[0].message.model_dump = lambda **kwargs: deepcopy(original_message)
        tools = FakeTools(
            {REFERENCE: run_data(REFERENCE), CANDIDATE: run_data(CANDIDATE, mismatch=True)}
        )
        client = FakeClient(
            [
                response([request(REFERENCE), request(CANDIDATE)]),
                rejected,
                response(final=valid, prefix="corrected"),
            ]
        )

        report = run_agent(client, tools, max_model_calls=3, max_tool_calls=2)

        self.assertEqual(report.status, "completed", report.error)
        self.assertIsNone(report.error)
        self.assertEqual(report.diagnosis, valid)
        self.assertEqual(json.loads(report.final_text), valid)
        self.assertEqual(report.raw_final_text, json.dumps(valid))
        self.assertEqual((report.model_calls, report.tool_calls), (3, 2))
        self.assertEqual((report.input_tokens, report.output_tokens), (36, 15))
        self.assertTrue(report.usage_complete)
        self.assertEqual(len(report.requests), 3)
        self.assertEqual(report.max_validation_retries, 1)
        self.assertEqual(report.validation_retries, 1)
        self.assertIs(report.first_pass_valid, False)
        self.assertEqual(report.pending_evidence, [])
        self.assertEqual(tools.calls, [request(REFERENCE), request(CANDIDATE)])
        self.assertEqual(len(report.tool_trace), 2)
        self.assertEqual([item["call_id"] for item in report.tool_trace], ["call-0", "call-1"])

        failed_attempt, corrected_attempt = report.validation_attempts
        expected_error = (
            "Performance status must be no_observed_degradation for these recorded metrics"
        )
        self.assertEqual(
            failed_attempt,
            {
                "model_call": 2,
                "raw_final_text": rejected_text,
                "valid": False,
                "error": expected_error,
            },
        )
        self.assertEqual(
            corrected_attempt,
            {"model_call": 3, "raw_final_text": json.dumps(valid), "valid": True, "error": None},
        )
        self.assertEqual(
            json.loads(report.to_json())["validation_attempts"], report.validation_attempts
        )

        initial_final_request, correction_request = client.requests[1:]
        self.assertEqual(correction_request["messages"][:-2], initial_final_request["messages"])
        self.assertEqual(correction_request["messages"][-2], original_message)
        feedback = correction_request["messages"][-1]
        self.assertEqual(feedback["role"], "user")
        self.assertIn(failed_attempt["error"], feedback["content"])
        for final_request in (initial_final_request, correction_request):
            self.assertNotIn("tools", final_request)
            self.assertNotIn("tool_choice", final_request)
            self.assertEqual(final_request["response_format"], {"type": "json_object"})
            self.assertIn(REFERENCE, final_request["messages"][0]["content"])
            self.assertIn(CANDIDATE, final_request["messages"][0]["content"])

    def test_second_invalid_final_stops_before_remaining_model_budget_is_used(self):
        first = diagnosis_payload()
        first["performance"]["status"] = "degraded"
        second = diagnosis_payload()
        second["implementation_evidence"][0]["expected_value"] = 0
        tools = FakeTools(
            {REFERENCE: run_data(REFERENCE), CANDIDATE: run_data(CANDIDATE, mismatch=True)}
        )
        client = FakeClient(
            [
                response([request(REFERENCE), request(CANDIDATE)]),
                response(final=first),
                response(final=second),
                response(final=diagnosis_payload()),
            ]
        )

        report = run_agent(client, tools, max_model_calls=6)

        self.assertEqual(report.status, "invalid_diagnosis")
        self.assertIsNone(report.diagnosis)
        self.assertIsNone(report.final_text)
        self.assertEqual(report.raw_final_text, json.dumps(second))
        self.assertEqual((report.model_calls, report.tool_calls), (3, 2))
        self.assertEqual(len(client.requests), 3)
        self.assertEqual(report.validation_retries, 1)
        self.assertIs(report.first_pass_valid, False)
        self.assertEqual(len(report.validation_attempts), 2)
        self.assertEqual([item["model_call"] for item in report.validation_attempts], [2, 3])
        self.assertEqual(
            [item["raw_final_text"] for item in report.validation_attempts],
            [json.dumps(first), json.dumps(second)],
        )
        self.assertTrue(all(item["valid"] is False for item in report.validation_attempts))
        self.assertTrue(all(item["error"] for item in report.validation_attempts))
        self.assertEqual(report.error, report.validation_attempts[-1]["error"])
        self.assertEqual(tools.calls, [request(REFERENCE), request(CANDIDATE)])

    def test_invalid_json_can_be_corrected_once(self):
        invalid_response = response(final=diagnosis_payload())
        invalid_response.choices[0].message.content = "{"
        invalid_response.choices[0].message.model_dump = lambda **kwargs: {
            "role": "assistant",
            "content": "{",
        }
        tools = FakeTools(
            {REFERENCE: run_data(REFERENCE), CANDIDATE: run_data(CANDIDATE, mismatch=True)}
        )
        client = FakeClient(
            [
                response([request(REFERENCE), request(CANDIDATE)]),
                invalid_response,
                response(final=diagnosis_payload()),
            ]
        )

        report = run_agent(client, tools, max_model_calls=3)

        self.assertEqual(report.status, "completed", report.error)
        self.assertEqual(report.validation_retries, 1)
        self.assertIs(report.first_pass_valid, False)
        self.assertEqual(report.validation_attempts[0]["raw_final_text"], "{")
        self.assertIs(report.validation_attempts[0]["valid"], False)
        self.assertIs(report.validation_attempts[1]["valid"], True)

    def test_nonvalidation_final_failures_do_not_trigger_correction(self):
        cases = (
            ("refusal", "refused"),
            ("content_filter", "refused"),
            ("length", "response_not_completed"),
            ("empty", "empty_response"),
            ("tool_calls", "protocol_error"),
            ("multiple_choices", "protocol_error"),
        )
        for case, expected_status in cases:
            with self.subTest(case=case):
                final_response = response(final=diagnosis_payload())
                choice = final_response.choices[0]
                if case == "refusal":
                    choice.message.refusal = "I cannot provide a diagnosis."
                elif case in ("content_filter", "length"):
                    choice.finish_reason = case
                elif case == "empty":
                    choice.message.content = "  "
                elif case == "tool_calls":
                    final_response = response([request(CANDIDATE)], prefix="unexpected")
                else:
                    final_response.choices.append(deepcopy(choice))
                tools = FakeTools(
                    {
                        REFERENCE: run_data(REFERENCE),
                        CANDIDATE: run_data(CANDIDATE, mismatch=True),
                    }
                )
                client = FakeClient(
                    [
                        response([request(REFERENCE), request(CANDIDATE)]),
                        final_response,
                        response(final=diagnosis_payload()),
                    ]
                )

                report = run_agent(client, tools, max_model_calls=6)

                self.assertEqual(report.status, expected_status)
                self.assertEqual((report.model_calls, report.tool_calls), (2, 2))
                self.assertEqual(len(client.requests), 2)
                self.assertEqual(report.validation_retries, 0)
                self.assertIsNone(report.first_pass_valid)
                self.assertEqual(report.validation_attempts, [])
                self.assertIsNone(report.diagnosis)
                self.assertEqual(tools.calls, [request(REFERENCE), request(CANDIDATE)])

    def test_final_api_error_does_not_trigger_validation_correction(self):
        class FakeAPIError(Exception):
            code = "quota_exhausted"

        tools = FakeTools(
            {REFERENCE: run_data(REFERENCE), CANDIDATE: run_data(CANDIDATE, mismatch=True)}
        )
        client = FakeClient([])
        with (
            patch.object(agent, "APIError", FakeAPIError),
            patch.object(
                client.chat.completions,
                "create",
                side_effect=[
                    response([request(REFERENCE), request(CANDIDATE)]),
                    FakeAPIError("Test quota error"),
                    response(final=diagnosis_payload()),
                ],
            ) as create,
        ):
            report = run_agent(client, tools, max_model_calls=6)

        self.assertEqual(report.status, "api_error")
        self.assertIn("quota_exhausted", report.error)
        self.assertEqual(create.call_count, 2)
        self.assertEqual((report.model_calls, report.tool_calls), (2, 2))
        self.assertFalse(report.usage_complete)
        self.assertEqual((report.input_tokens, report.output_tokens), (12, 5))
        self.assertEqual(report.validation_retries, 0)
        self.assertIsNone(report.first_pass_valid)
        self.assertEqual(report.validation_attempts, [])
        self.assertIsNone(report.diagnosis)

    def test_collects_only_optimizer_evidence_with_exact_task_paths(self):
        tools = FakeTools()
        client = FakeClient([response([request(REFERENCE), request(CANDIDATE)])])
        report = run_agent(client, tools, max_model_calls=1)
        self.assertEqual(report.status, "budget_exhausted")
        self.assertEqual(report.model_calls, 0)
        self.assertEqual(tools.calls, [])
        self.assertEqual(report.pending_evidence, [request(REFERENCE), request(CANDIDATE)])
        self.assertEqual(report.validation_retries, 0)
        self.assertIsNone(report.first_pass_valid)
        self.assertEqual(report.validation_attempts, [])

    def test_invented_candidate_is_not_dispatched_after_reference_success(self):
        client = FakeClient(
            [
                response([request(REFERENCE)], prefix="reference"),
                response([request("artifacts/invented")], prefix="invented"),
            ]
        )
        tools = FakeTools()
        report = run_agent(client, tools, max_model_calls=3, max_tool_calls=3)
        self.assertEqual(report.status, "budget_exhausted")
        self.assertEqual(tools.calls, [request(REFERENCE)])
        self.assertEqual(report.tool_trace[-1]["result"]["error_code"], "out_of_scope_request")
        self.assertEqual(report.pending_evidence, [request(CANDIDATE)])
        offered = client.requests[1]["tools"]
        self.assertEqual(len(offered), 1)
        function = offered[0]["function"]
        self.assertEqual(function["name"], "inspect_optimizer_run")
        self.assertEqual(function["parameters"]["properties"]["run_directory"]["enum"], [CANDIDATE])

    def test_premature_final_json_cannot_bypass_evidence_collection(self):
        client = FakeClient([response(final={"profile": "optimizer_binding"})])
        tools = FakeTools()
        report = run_agent(client, tools)
        self.assertEqual(report.status, "protocol_error")
        self.assertIsNone(report.diagnosis)
        self.assertEqual(tools.calls, [])

    def test_over_budget_tool_batch_is_rejected_before_execution(self):
        client = FakeClient([response([request(REFERENCE), request(CANDIDATE)])])
        tools = FakeTools()
        report = run_agent(client, tools, max_tool_calls=1)
        self.assertEqual(report.status, "budget_exhausted")
        self.assertEqual(tools.calls, [])

    def test_unknown_profile_is_rejected_without_api_calls(self):
        client = FakeClient([])
        with self.assertRaises(ValueError):
            agent.run_diagnostic_agent(
                client,
                "test-model",
                FakeTools(),
                source_path=SOURCE,
                reference_run=REFERENCE,
                candidate_run=CANDIDATE,
                profile="invented",
            )
        self.assertEqual(client.requests, [])

    def test_legacy_default_keeps_original_evidence_requirements(self):
        client = FakeClient([])
        report = agent.run_diagnostic_agent(
            client,
            "test-model",
            FakeTools(),
            source_path=SOURCE,
            reference_run=REFERENCE,
            candidate_run=CANDIDATE,
            max_model_calls=1,
        )
        self.assertEqual(report.status, "budget_exhausted")
        self.assertEqual(
            [item["name"] for item in report.pending_evidence],
            [
                "inspect_training_source",
                "load_run_config",
                "load_run_config",
                "compare_runs",
            ],
        )
        self.assertEqual(client.requests, [])
        legacy_fields = json.loads(report.to_json())
        for optimizer_field in (
            "profile",
            "output_mode",
            "max_validation_retries",
            "validation_retries",
            "first_pass_valid",
            "validation_attempts",
        ):
            self.assertNotIn(optimizer_field, legacy_fields)

    def test_non_null_patch_is_rejected_after_final_response(self):
        for invalid_patch in ("change optimizer", {}, False):
            with self.subTest(patch=invalid_patch):
                payload = diagnosis_payload()
                payload["proposed_patch"] = invalid_patch
                tools = FakeTools(
                    {
                        REFERENCE: run_data(REFERENCE),
                        CANDIDATE: run_data(CANDIDATE, mismatch=True),
                    }
                )
                client = FakeClient(
                    [
                        response([request(REFERENCE), request(CANDIDATE)]),
                        response(final=payload),
                    ]
                )
                report = run_agent(client, tools, max_model_calls=2, max_tool_calls=2)
                self.assertEqual(report.status, "invalid_diagnosis")
                self.assertIsNone(report.diagnosis)
                self.assertEqual(report.model_calls, 2)
                self.assertEqual(report.validation_retries, 0)
                self.assertIs(report.first_pass_valid, False)
                self.assertEqual(len(report.validation_attempts), 1)
                self.assertIs(report.validation_attempts[0]["valid"], False)

    def test_corrected_response_still_cannot_propose_a_patch(self):
        invalid = diagnosis_payload()
        invalid["performance"]["status"] = "degraded"
        correction = diagnosis_payload()
        correction["proposed_patch"] = {"field": "optimizer", "new_value": "rebind"}
        tools = FakeTools(
            {REFERENCE: run_data(REFERENCE), CANDIDATE: run_data(CANDIDATE, mismatch=True)}
        )
        client = FakeClient(
            [
                response([request(REFERENCE), request(CANDIDATE)]),
                response(final=invalid),
                response(final=correction),
                response(final=diagnosis_payload()),
            ]
        )

        report = run_agent(client, tools, max_model_calls=6, max_tool_calls=2)

        self.assertEqual(report.status, "invalid_diagnosis")
        self.assertIsNone(report.diagnosis)
        self.assertIsNone(report.final_text)
        self.assertEqual(report.raw_final_text, json.dumps(correction))
        self.assertEqual(report.model_calls, 3)
        self.assertEqual(len(client.requests), 3)
        self.assertEqual(report.validation_retries, 1)
        self.assertIs(report.first_pass_valid, False)
        self.assertEqual(len(report.validation_attempts), 2)
        self.assertIn("proposed_patch", report.error)
        self.assertEqual(report.error, report.validation_attempts[-1]["error"])

    def test_single_correction_feedback_covers_schema_and_performance_errors(self):
        invalid = diagnosis_payload()
        invalid["ranked_causes"][0]["evidence"] = [
            {
                "call_id": "call-1",
                "path": ["optimizer_audit", "missing_trainable_names"],
                "expected_value": ["fc.bias", "fc.weight"],
            }
        ]
        invalid["performance"]["status"] = "mixed"
        valid = diagnosis_payload()
        data = {REFERENCE: run_data(REFERENCE), CANDIDATE: run_data(CANDIDATE, mismatch=True)}
        tools = FakeTools(data)
        client = FakeClient(
            [
                response([request(REFERENCE), request(CANDIDATE)]),
                response(final=invalid),
                response(final=valid),
            ]
        )

        report = run_agent(client, tools, max_model_calls=3, max_tool_calls=2)

        self.assertEqual(report.status, "completed", report.error)
        self.assertEqual(report.diagnosis, valid)
        self.assertEqual((report.model_calls, report.tool_calls), (3, 2))
        self.assertEqual(report.validation_retries, 1)
        self.assertIs(report.first_pass_valid, False)
        self.assertEqual(tools.calls, [request(REFERENCE), request(CANDIDATE)])
        self.assertEqual([item["valid"] for item in report.validation_attempts], [False, True])
        correction_request = client.requests[-1]
        self.assertNotIn("tools", correction_request)
        self.assertNotIn("tool_choice", correction_request)
        feedback_message = correction_request["messages"][-1]
        self.assertEqual(feedback_message["role"], "user")
        feedback = json.loads(
            feedback_message["content"].split("Validator feedback (data):\n", 1)[1]
        )
        self.assertEqual(feedback["error"], report.validation_attempts[0]["error"])
        self.assertIn("expected_value", feedback["error"])
        self.assertIn("scalar", feedback["citation_rule"].lower())
        self.assertEqual(report.validation_feedback, [{"model_call": 2, "feedback": feedback}])
        self.assertEqual(
            json.loads(report.to_json())["validation_feedback"], report.validation_feedback
        )
        performance = feedback["performance_check"]
        self.assertIs(performance["available"], True)
        self.assertIs(performance["comparable"], True)
        self.assertEqual(performance["expected_status"], "no_observed_degradation")
        self.assertEqual(performance["reference_run"], REFERENCE)
        self.assertEqual(performance["candidate_run"], CANDIDATE)
        self.assertEqual(performance["reference_call_ids"], ["call-0"])
        self.assertEqual(performance["candidate_call_ids"], ["call-1"])
        comparisons = performance["final_metric_comparisons"]
        self.assertEqual(
            [item["metric"] for item in comparisons],
            [
                "id_validation_accuracy",
                "id_validation_loss",
                "ood_validation_accuracy",
                "ood_validation_loss",
            ],
        )
        for item in comparisons:
            metric = item["metric"]
            self.assertEqual(item["reference_value"], data[REFERENCE]["epochs"][-1][metric])
            self.assertEqual(item["candidate_value"], data[CANDIDATE]["epochs"][-1][metric])
            self.assertEqual(
                item["better_when"], "higher" if metric.endswith("accuracy") else "lower"
            )
            self.assertGreater(item["signed_improvement"], 0)
            self.assertEqual(item["observation"], "improved")


if __name__ == "__main__":
    unittest.main()
