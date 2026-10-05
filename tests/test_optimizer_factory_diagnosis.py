"""Pure source-evidence tests; no model, GPU, API key or network is required."""

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from runsleuth.optimizer_factory_cases import factory_cases
from runsleuth.optimizer_factory_diagnosis import (
    FINDINGS,
    FactoryDiagnosis,
    build_factory_feedback,
    factory_evidence,
    inspect_optimizer_factory,
    required_factory_paths,
    validate_factory_diagnosis,
)

SOURCE_PATH = "inputs/factory.py"


def evidence_for(index=0, source_path=SOURCE_PATH):
    return factory_evidence(factory_cases()[index]["source"].encode(), source_path=source_path)


def trace_for(data, *, call_id="factory-call"):
    return [
        {
            "call_id": call_id,
            "tool_name": "inspect_optimizer_factory",
            "raw_arguments": json.dumps({"source_path": data["source_path"]}),
            "result": {
                "tool_name": "inspect_optimizer_factory",
                "ok": True,
                "data": deepcopy(data),
                "error_code": None,
                "error_message": None,
            },
        }
    ]


def diagnosis_for(data, *, call_id="factory-call"):
    evidence = []
    for path in required_factory_paths(data):
        value = data
        for key in path:
            value = value[key]
        evidence.append({"call_id": call_id, "path": path, "expected_value": value})
    status, action = FINDINGS[data["analysis"]["decision"]]
    return {
        "profile": "optimizer_factory",
        "source_sha256": data["source_sha256"],
        "status": status,
        "summary": "A static finding within the supported subset.",
        "evidence": evidence,
        "uncertainties": ["No source execution or performance evaluation."],
        "recommended_action": action,
        "proposed_patch": None,
        "execution_verified": False,
        "performance_evaluated": False,
    }


class FactoryDiagnosisTests(unittest.TestCase):
    def test_all_development_shapes_have_citable_findings_without_patch_content(self):
        for index, case in enumerate(factory_cases()):
            with self.subTest(case=case["id"]):
                data = evidence_for(index)
                self.assertNotIn("proposed_source", data["analysis"])
                self.assertNotIn("unified_diff", data["analysis"])
                self.assertFalse(data["execution_performed"])
                validate_factory_diagnosis(
                    FactoryDiagnosis.model_validate(diagnosis_for(data)),
                    trace_for(data),
                    source_path=SOURCE_PATH,
                )

    def test_eager_and_lazy_capture_sites_are_distinguished(self):
        eager, lazy = evidence_for(2)["analysis"], evidence_for(6)["analysis"]
        self.assertLess(eager["binding_site"]["line"], eager["replacement_site"]["line"])
        self.assertGreater(eager["constructor_site"]["line"], eager["replacement_site"]["line"])
        self.assertGreater(lazy["binding_site"]["line"], lazy["replacement_site"]["line"])
        self.assertEqual(lazy["binding_site"], lazy["constructor_site"])

    def test_hash_tracks_exact_crlf_and_final_newline_bytes(self):
        source = factory_cases()[0]["source"]
        lf = factory_evidence(source.encode(), source_path=SOURCE_PATH)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "factory.py"
            payload = source.replace("\n", "\r\n").rstrip("\r\n").encode()
            path.write_bytes(payload)
            crlf = inspect_optimizer_factory(path, source_path=SOURCE_PATH)
            self.assertNotEqual(lf["source_sha256"], crlf["source_sha256"])
            self.assertEqual(path.read_bytes(), payload)

    def test_inspection_does_not_execute_imports_or_module_statements(self):
        data = factory_evidence(
            b"raise RuntimeError('must never execute')\n", source_path=SOURCE_PATH
        )
        self.assertEqual(data["analysis"]["decision"], "unsupported")
        self.assertIn("Module-level", data["analysis"]["reason"])

    def test_source_size_and_encoding_fail_before_analysis(self):
        for payload in (b"", b"x" * (128 * 1024 + 1), b"\xff"):
            with self.subTest(length=len(payload)), self.assertRaises(ValueError):
                factory_evidence(payload, source_path=SOURCE_PATH)

    def test_mutated_values_locations_and_incorrect_findings_are_rejected(self):
        data = evidence_for()
        good = diagnosis_for(data)
        changes = [
            ("status", "no_stale_parameter_capture"),
            ("recommended_action", "no_change"),
            ("source_sha256", "a" * 64),
        ]
        for field, value in changes:
            with self.subTest(field=field), self.assertRaises(ValueError):
                payload = {**good, field: value}
                validate_factory_diagnosis(
                    FactoryDiagnosis.model_validate(payload),
                    trace_for(data),
                    source_path=SOURCE_PATH,
                )
        for value in (999, True, "invented source"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                payload = deepcopy(good)
                payload["evidence"][2]["expected_value"] = value
                validate_factory_diagnosis(
                    FactoryDiagnosis.model_validate(payload),
                    trace_for(data),
                    source_path=SOURCE_PATH,
                )

    def test_every_required_line_and_code_citation_is_enforced(self):
        data = evidence_for()
        for index in range(len(required_factory_paths(data))):
            with (
                self.subTest(index=index),
                self.assertRaisesRegex(ValueError, "citation is required"),
            ):
                payload = diagnosis_for(data)
                payload["evidence"].pop(index)
                validate_factory_diagnosis(
                    FactoryDiagnosis.model_validate(payload),
                    trace_for(data),
                    source_path=SOURCE_PATH,
                )

    def test_unknown_failed_foreign_and_duplicate_calls_are_rejected(self):
        data = evidence_for()
        diagnosis = FactoryDiagnosis.model_validate(diagnosis_for(data))
        for kind in ("wrong_path", "failed", "wrong_tool", "duplicate", "modified_data"):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                trace = trace_for(data)
                if kind == "wrong_path":
                    trace[0]["raw_arguments"] = '{"source_path":"other.py"}'
                elif kind == "failed":
                    trace[0]["result"]["ok"] = False
                elif kind == "wrong_tool":
                    trace[0]["tool_name"] = "inspect_training_source"
                elif kind == "duplicate":
                    trace.append(deepcopy(trace[0]))
                else:
                    trace.extend(trace_for({**data, "source_sha256": "f" * 64}, call_id="second"))
                validate_factory_diagnosis(diagnosis, trace, source_path=SOURCE_PATH)

    def test_unsupported_requires_reason_and_cannot_be_reported_as_healthy(self):
        data = evidence_for(8)
        payload = diagnosis_for(data)
        self.assertEqual(payload["recommended_action"], "manual_review")
        payload["status"], payload["recommended_action"] = "no_stale_parameter_capture", "no_change"
        with self.assertRaises(ValueError):
            validate_factory_diagnosis(
                FactoryDiagnosis.model_validate(payload), trace_for(data), source_path=SOURCE_PATH
            )

    def test_read_only_fields_and_non_scalar_citations_are_strict(self):
        data = evidence_for()
        for field in ("execution_verified", "performance_evaluated"):
            for invalid in (True, 0, "false", None):
                with self.subTest(field=field, value=invalid), self.assertRaises(ValueError):
                    FactoryDiagnosis.model_validate({**diagnosis_for(data), field: invalid})
        for invalid in ("repair it", {}, False):
            with self.subTest(patch=invalid), self.assertRaises(ValueError):
                FactoryDiagnosis.model_validate({**diagnosis_for(data), "proposed_patch": invalid})
        payload = diagnosis_for(data)
        payload["evidence"][0]["expected_value"] = ["array"]
        with self.assertRaises(ValueError):
            FactoryDiagnosis.model_validate(payload)

    def test_correction_feedback_uses_existing_calls_and_complete_source_requirements(self):
        data = evidence_for()
        feedback = build_factory_feedback(
            "missing source code", trace_for(data), source_path=SOURCE_PATH
        )
        self.assertEqual(feedback["required_evidence_paths"], required_factory_paths(data))
        self.assertEqual(feedback["inspection_call_ids"], ["factory-call"])
        self.assertNotIn("performance_check", feedback)


if __name__ == "__main__":
    unittest.main()
