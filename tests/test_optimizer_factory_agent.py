"""Exercise the shared Agent loop and real tool dispatch with an offline API double."""

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from test_optimizer_factory_diagnosis import diagnosis_for

from runsleuth.optimizer_factory_cases import factory_cases
from runsleuth.optimizer_factory_diagnosis import factory_evidence

try:
    import runsleuth.agent as agent
    from runsleuth.diagnostic_tools import DiagnosticTools
except ModuleNotFoundError as error:
    if error.name not in ("openai", "torch"):
        raise
    AVAILABLE = False
else:
    AVAILABLE = True


def reply(*, source=None, final=None, call_id="factory-call", name="inspect_optimizer_factory"):
    calls = (
        []
        if source is None
        else [
            SimpleNamespace(
                id=call_id,
                type="function",
                function=SimpleNamespace(name=name, arguments=json.dumps({"source_path": source})),
            )
        ]
    )
    payload = {"role": "assistant", "content": json.dumps(final) if final is not None else None}
    if calls:
        payload["tool_calls"] = [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": calls[0].function.arguments},
                "extra_content": {"google": {"thought_signature": "preserve-me"}},
            }
        ]
    counts = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
    return SimpleNamespace(
        id="offline",
        model="fixture-model",
        usage=SimpleNamespace(**counts, model_dump=lambda: counts),
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls" if calls else "stop",
                message=SimpleNamespace(
                    content=payload["content"],
                    tool_calls=calls,
                    refusal=None,
                    model_dump=lambda **kw: deepcopy(payload),
                ),
            )
        ],
    )


@unittest.skipUnless(AVAILABLE, "Agent dispatch needs the project's torch and openai dependencies")
class FactoryAgentTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = "factory.py"
        self.tools = DiagnosticTools(self.root, enable_optimizer_factory=True)
        self.write_case(0)

    def write_case(self, index):
        self.payload = factory_cases()[index]["source"].encode()
        (self.root / self.source).write_bytes(self.payload)
        self.data = factory_evidence(self.payload, source_path=self.source)

    def run_fake(self, responses, *, tools=None, **options):
        create = Mock(side_effect=responses)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        report = agent.run_diagnostic_agent(
            client,
            "fixture-model",
            tools or self.tools,
            source_path=self.source,
            reference_run="",
            candidate_run="",
            profile="optimizer_factory",
            **{"max_model_calls": 3, "max_tool_calls": 2, "max_output_tokens": 2000, **options},
        )
        return report, create

    def test_all_twelve_source_shapes_use_shared_loop_and_checked_source_citations(self):
        for index, case in enumerate(factory_cases()):
            with self.subTest(case=case["id"]):
                self.write_case(index)
                report, create = self.run_fake(
                    [reply(source=self.source), reply(final=diagnosis_for(self.data))]
                )
                self.assertEqual(report.status, "completed", report.error)
                self.assertEqual((report.model_calls, report.tool_calls), (2, 1))
                self.assertEqual(report.profile, "optimizer_factory")
                self.assertTrue(report.first_pass_valid)
                self.assertFalse(report.diagnosis["execution_verified"])
                self.assertEqual((self.root / self.source).read_bytes(), self.payload)
                first = create.call_args_list[0].kwargs
                self.assertEqual(len(first["tools"]), 1)
                self.assertEqual(
                    first["tools"][0]["function"]["parameters"]["properties"]["source_path"][
                        "enum"
                    ],
                    [self.source],
                )
                final = create.call_args_list[1].kwargs
                self.assertEqual(final["response_format"], {"type": "json_object"})
                self.assertNotIn("tools", final)
                self.assertEqual(
                    final["messages"][2]["tool_calls"][0]["extra_content"]["google"][
                        "thought_signature"
                    ],
                    "preserve-me",
                )

    def test_bounded_correction_preserves_failed_json_and_all_usage(self):
        invalid = diagnosis_for(self.data)
        invalid["evidence"].pop()
        report, create = self.run_fake(
            [reply(source=self.source), reply(final=invalid), reply(final=diagnosis_for(self.data))]
        )
        self.assertEqual(report.status, "completed", report.error)
        self.assertFalse(report.first_pass_valid)
        self.assertEqual(report.validation_retries, 1)
        self.assertEqual([r["valid"] for r in report.validation_attempts], [False, True])
        self.assertEqual(
            (report.model_calls, report.tool_calls, report.input_tokens, report.output_tokens),
            (3, 1, 33, 21),
        )
        feedback = create.call_args_list[-1].kwargs["messages"][-1]["content"]
        self.assertIn("required_evidence_paths", feedback)
        self.assertNotIn("performance_check", feedback)

    def test_second_invalid_final_remains_rejected(self):
        invalid = {**diagnosis_for(self.data), "proposed_patch": "rewrite the source"}
        report, _ = self.run_fake(
            [reply(source=self.source), reply(final=invalid), reply(final=invalid)]
        )
        self.assertEqual(report.status, "invalid_diagnosis")
        self.assertEqual(report.validation_retries, 1)
        self.assertIsNone(report.diagnosis)

    def test_out_of_scope_request_never_reads_other_file(self):
        report, _ = self.run_fake(
            [
                reply(source="secret.py", call_id="wrong"),
                reply(source=self.source),
                reply(final=diagnosis_for(self.data)),
            ]
        )
        self.assertEqual(report.status, "completed", report.error)
        self.assertEqual(report.tool_trace[0]["result"]["error_code"], "out_of_scope_request")

    def test_final_api_error_retains_partial_usage_and_no_validation_retry(self):
        class FakeAPIError(Exception):
            status_code = 503
            message = "temporary overload"
            code = None

        with patch.object(agent, "APIError", FakeAPIError):
            report, _ = self.run_fake([reply(source=self.source), FakeAPIError()])
        self.assertEqual(report.status, "api_error")
        self.assertFalse(report.usage_complete)
        self.assertEqual((report.model_calls, report.tool_calls, report.input_tokens), (2, 1, 11))
        self.assertEqual(report.validation_retries, 0)

    def test_insufficient_budget_cannot_skip_required_inspection(self):
        report, create = self.run_fake([], max_model_calls=1)
        self.assertEqual(report.status, "budget_exhausted")
        create.assert_not_called()

    def test_factory_tool_requires_opt_in_and_does_not_change_legacy_definitions(self):
        default = DiagnosticTools(self.root)
        self.assertEqual(len(default.definitions()), 3)
        self.assertEqual(
            len(DiagnosticTools(self.root, enable_optimizer_audit=True).definitions()), 4
        )
        self.assertEqual(len(self.tools.definitions()), 4)
        self.assertEqual(
            default.execute("inspect_optimizer_factory", {"source_path": self.source}).error_code,
            "unknown_tool",
        )
        with self.assertRaisesRegex(ValueError, "enable_optimizer_factory"):
            self.run_fake([], tools=default)

    def test_factory_dispatch_precedes_generic_source_dispatch(self):
        with patch(
            "runsleuth.diagnostic_tools.inspect_training_source", side_effect=AssertionError
        ):
            result = self.tools.execute("inspect_optimizer_factory", {"source_path": self.source})
        self.assertTrue(result.ok, result.error_message)
        self.assertEqual(result.data, self.data)

    def test_boundary_and_schema_violations_do_not_reach_source_reader(self):
        for arguments in (
            {"source_path": "../outside.py"},
            {"source_path": "C:/secret.py"},
            {"source_path": False},
            {"source_path": self.source, "execute": True},
        ):
            with (
                self.subTest(arguments=arguments),
                patch("runsleuth.diagnostic_tools.inspect_optimizer_factory") as read,
            ):
                result = self.tools.execute("inspect_optimizer_factory", arguments)
                self.assertFalse(result.ok)
                read.assert_not_called()

    def test_invalid_encoding_is_reported_without_executing_source(self):
        (self.root / self.source).write_bytes(b"\xff")
        result = self.tools.execute("inspect_optimizer_factory", {"source_path": self.source})
        self.assertEqual(result.error_code, "invalid_artifact")

    def test_shared_loop_report_passes_real_runner_validation(self):
        from runsleuth.optimizer_factory_agent import _validate_attempt

        invalid = diagnosis_for(self.data)
        invalid["evidence"].pop()
        for corrected in (False, True):
            with self.subTest(corrected=corrected):
                responses = [reply(source=self.source)]
                if corrected:
                    responses.append(reply(final=invalid))
                responses.append(reply(final=diagnosis_for(self.data)))
                report, _ = self.run_fake(responses)
                _validate_attempt(
                    json.loads(report.to_json()),
                    model="fixture-model",
                    source_path=self.source,
                    evidence=self.data,
                )


if __name__ == "__main__":
    unittest.main()
