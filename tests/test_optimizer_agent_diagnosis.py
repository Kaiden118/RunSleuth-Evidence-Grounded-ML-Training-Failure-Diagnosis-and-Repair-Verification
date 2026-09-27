"""Offline schema, evidence provenance, and optimizer/performance separation tests."""

import json
import unittest
from copy import deepcopy

from pydantic import ValidationError

from runsleuth.optimizer_agent_diagnosis import (
    AUDIT_FIELDS,
    METRIC_FIELDS,
    OptimizerDiagnosis,
    build_optimizer_validation_feedback,
    validate_optimizer_diagnosis_evidence,
)


def make_run(path="artifacts/candidate", *, mismatch=True):
    config = {
        "dataset": "camelyon17",
        "dataset_version": "1.0",
        "model": "resnet18",
        "pretrained_weights": "IMAGENET1K_V1",
        "hf_revision": "a" * 40,
        "seed": 42,
        "subset_seed": 2026,
        "epochs": 2,
        "batch_size": 32,
        "learning_rate": 0.0001,
        "weight_decay": 0.0001,
        "max_train_samples": 20000,
        "max_id_val_samples": 5000,
        "max_ood_val_samples": 5000,
        "num_workers": 0,
        "device": "cuda",
    }
    return {
        "run_directory": path,
        "config": config,
        "optimizer_audit": {
            "model_parameter_tensors": 62,
            "trainable_parameter_tensors": 62,
            "optimizer_unique_parameter_tensors": 62,
            "missing_trainable_names": ["fc.bias", "fc.weight"] if mismatch else [],
            "missing_trainable_parameter_tensors": 2 if mismatch else 0,
            "foreign_parameter_tensors": 2 if mismatch else 0,
            "duplicate_parameter_occurrences": 0,
        },
        "initial_state_sha256": "b" * 64,
        "data_identity": {
            "dataset": "camelyon17",
            "dataset_version": "1.0",
            "subset_seed": 2026,
            "repo": "example/mirror",
            "revision": "a" * 40,
            "metadata_sha256": "c" * 64,
            "selected_sample_ids_sha256": {"train": "d" * 64, "id_val": "e" * 64, "val": "f" * 64},
        },
        "epochs": [
            {
                "epoch": epoch,
                "optimizer_steps": 625,
                "head": {
                    "mean_gradient_l2_norm": 0.25,
                    "mean_parameter_update_l2_norm": 0.0 if mismatch else 0.0003,
                },
                "backbone": {"mean_gradient_l2_norm": 1.6, "mean_parameter_update_l2_norm": 0.04},
                "id_validation_accuracy": 0.9806 if mismatch else 0.9786,
                "id_validation_loss": 0.07118 if mismatch else 0.0748,
                "ood_validation_accuracy": 0.9156 if mismatch else 0.9016,
                "ood_validation_loss": 0.3201 if mismatch else 0.3787,
            }
            for epoch in (1, 2)
        ],
        "provenance": {"kind": "recorded_runtime_audit"},
    }


def make_entry(call_id, data):
    return {
        "call_id": call_id,
        "tool_name": "inspect_optimizer_run",
        "raw_arguments": json.dumps({"run_directory": data["run_directory"]}),
        "result": {"tool_name": "inspect_optimizer_run", "ok": True, "data": data},
    }


def cite(entry, path):
    value = entry["result"]["data"]
    for part in path:
        value = value[part]
    return {"call_id": entry["call_id"], "path": path, "expected_value": value}


def make_payload(trace, *, performance="no_observed_degradation", defect=True):
    candidate = trace[-1]
    last = len(candidate["result"]["data"]["epochs"]) - 1
    implementation = [cite(candidate, ["optimizer_audit", key]) for key in AUDIT_FIELDS]
    if defect:
        implementation += [
            cite(candidate, ["epochs", last, "head", "mean_gradient_l2_norm"]),
            cite(candidate, ["epochs", last, "head", "mean_parameter_update_l2_norm"]),
        ]
    return {
        "profile": "optimizer_binding",
        "implementation_status": "optimizer_binding_defect"
        if defect
        else "no_optimizer_binding_defect",
        "summary": "The recorded audit and parameter-group metrics were inspected.",
        "implementation_evidence": implementation,
        "ranked_causes": [
            {
                "root_cause": "optimizer_parameter_mismatch",
                "confidence": "high",
                "explanation": "The recorded optimizer omits trainable model parameters.",
                "evidence": [implementation[0]],
            }
        ]
        if defect
        else [],
        "performance": {
            "status": performance,
            "explanation": "These are observed final-epoch differences, not a significance claim.",
            "evidence": [
                cite(entry, ["epochs", len(entry["result"]["data"]["epochs"]) - 1, key])
                for entry in trace
                for key in METRIC_FIELDS
            ]
            if performance != "inconclusive"
            else [],
        },
        "uncertainties": [
            "The audit is recorded evidence, not a fresh live audit; confidence is heuristic."
        ],
        "recommended_action": "review_optimizer_binding" if defect else "no_change",
        "proposed_patch": None,
    }


class OptimizerDiagnosisTests(unittest.TestCase):
    def setUp(self):
        self.trace = [
            make_entry("ref-call", make_run("artifacts/reference", mismatch=False)),
            make_entry("candidate-call", make_run()),
        ]
        self.payload = make_payload(self.trace)

    def check(self, payload=None, trace=None, **paths):
        validate_optimizer_diagnosis_evidence(
            OptimizerDiagnosis.model_validate(self.payload if payload is None else payload),
            self.trace if trace is None else trace,
            reference_run=paths.get("reference_run", "artifacts/reference"),
            candidate_run=paths.get("candidate_run", "artifacts/candidate"),
        )

    def test_silent_defect_accepts_better_performance_and_review_only(self):
        self.check()
        parsed = OptimizerDiagnosis.model_validate_json(json.dumps(self.payload))
        self.assertIsNone(parsed.proposed_patch)

    def test_healthy_distinct_run_has_no_cause_and_no_action(self):
        self.trace[-1] = make_entry("candidate-call", make_run(mismatch=False))
        self.check(make_payload(self.trace, defect=False))

    def test_one_successful_inspection_can_cover_identical_paths(self):
        entry = self.trace[0]
        payload = make_payload([entry], defect=False)
        self.check(payload, [entry], candidate_run="artifacts/reference")

    def test_performance_degraded_and_mixed_are_distinct(self):
        for expected in ("degraded", "mixed"):
            with self.subTest(expected=expected):
                trace = deepcopy(self.trace)
                final = trace[-1]["result"]["data"]["epochs"][-1]
                final.update(id_validation_accuracy=0.9, id_validation_loss=0.2)
                if expected == "degraded":
                    final.update(ood_validation_accuracy=0.8, ood_validation_loss=0.5)
                self.check(make_payload(trace, performance=expected), trace)

    def test_comparison_requires_same_initialization_data_and_config(self):
        for key in ("initial_state_sha256", "data_identity", "config"):
            with self.subTest(key=key):
                trace = deepcopy(self.trace)
                data = trace[-1]["result"]["data"]
                if key == "config":
                    data[key]["seed"] = 99
                elif key == "data_identity":
                    data[key]["metadata_sha256"] = "0" * 64
                else:
                    data[key] = "0" * 64
                self.check(make_payload(trace, performance="inconclusive"), trace)
                with self.assertRaisesRegex(ValueError, "inconclusive"):
                    self.check(make_payload(trace), trace)

    def test_cannot_claim_a_degradation_for_this_silent_defect(self):
        self.payload["performance"]["status"] = "degraded"
        with self.assertRaisesRegex(ValueError, "no_observed_degradation"):
            self.check()

    def test_false_does_not_match_numeric_zero(self):
        self.payload["implementation_evidence"][-1]["expected_value"] = False
        with self.assertRaisesRegex(ValueError, "JSON type"):
            self.check()

    def test_integer_zero_does_not_silently_replace_float_zero(self):
        self.payload["implementation_evidence"][-1]["expected_value"] = 0
        with self.assertRaisesRegex(ValueError, "JSON type"):
            self.check()

    def test_unknown_failed_or_wrong_tool_citations_are_rejected(self):
        for kind in ("unknown", "failed", "wrong_tool"):
            with self.subTest(kind=kind):
                trace = deepcopy(self.trace)
                payload = deepcopy(self.payload)
                if kind == "unknown":
                    payload["ranked_causes"][0]["evidence"][0]["call_id"] = "invented"
                elif kind == "failed":
                    trace[-1]["result"]["ok"] = False
                else:
                    trace[-1]["tool_name"] = "load_run_config"
                with self.assertRaises(ValueError):
                    self.check(payload, trace)

    def test_duplicate_call_ids_are_rejected(self):
        self.trace.append(deepcopy(self.trace[-1]))
        with self.assertRaisesRegex(ValueError, "unique"):
            self.check()

    def test_scope_and_data_path_are_checked(self):
        for mutation in ("arguments", "data", "extra_argument"):
            with self.subTest(mutation=mutation):
                trace = deepcopy(self.trace)
                if mutation == "arguments":
                    trace[-1]["raw_arguments"] = '{"run_directory": "artifacts/other"}'
                elif mutation == "data":
                    trace[-1]["result"]["data"]["run_directory"] = "artifacts/other"
                else:
                    trace[-1]["raw_arguments"] = '{"run_directory": "artifacts/candidate", "x": 1}'
                with self.assertRaises(ValueError):
                    self.check(trace=trace)

    def test_lists_negative_indices_and_bool_indices_are_rejected(self):
        for path in (["epochs"], ["epochs", -1, "epoch"], ["epochs", True, "epoch"]):
            with self.subTest(path=path):
                payload = deepcopy(self.payload)
                payload["implementation_evidence"][0]["path"] = path
                with self.assertRaises((ValueError, ValidationError)):
                    self.check(payload)

    def test_relevant_structural_and_performance_citations_are_required(self):
        for key in ("implementation_evidence", "performance"):
            with self.subTest(key=key):
                payload = deepcopy(self.payload)
                if key == "implementation_evidence":
                    payload[key].pop()
                else:
                    payload[key]["evidence"].pop()
                with self.assertRaisesRegex(ValueError, "required"):
                    self.check(payload)

    def test_reference_audit_cannot_substitute_for_candidate_audit(self):
        self.payload["implementation_evidence"][0] = cite(
            self.trace[0], ["optimizer_audit", AUDIT_FIELDS[0]]
        )
        with self.assertRaisesRegex(ValueError, "required"):
            self.check()

    def test_no_defect_cannot_ignore_a_positive_audit(self):
        self.payload = make_payload(self.trace, defect=False)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.check()

    def test_schema_rejects_code_patches_new_causes_and_missing_profile(self):
        for kind in ("patch", "cause", "profile", "confidence", "unknown_key"):
            with self.subTest(kind=kind):
                payload = deepcopy(self.payload)
                if kind == "patch":
                    payload["proposed_patch"] = {"field": "optimizer"}
                elif kind == "cause":
                    payload["ranked_causes"][0]["root_cause"] = "stale_head_construction_order"
                elif kind == "profile":
                    del payload["profile"]
                elif kind == "confidence":
                    payload["ranked_causes"][0]["confidence"] = 0.95
                else:
                    payload["repair_accepted"] = True
                with self.assertRaises(ValidationError):
                    OptimizerDiagnosis.model_validate(payload)

    def test_all_public_schema_fields_are_required(self):
        schema = OptimizerDiagnosis.model_json_schema()
        self.assertEqual(set(schema["required"]), set(schema["properties"]))

    def test_counts_nonfinite_metrics_and_epoch_order_are_validated(self):
        for kind in ("count", "nan", "epoch"):
            with self.subTest(kind=kind):
                trace = deepcopy(self.trace)
                data = trace[-1]["result"]["data"]
                if kind == "count":
                    data["optimizer_audit"][AUDIT_FIELDS[0]] = True
                elif kind == "nan":
                    data["epochs"][0]["id_validation_loss"] = float("nan")
                else:
                    data["epochs"][0]["epoch"] = 3
                with self.assertRaises(ValueError):
                    self.check(trace=trace)

    def test_feedback_uses_final_metric_directions_and_scoped_call_ids(self):
        cases = (
            ("improved", (0.6, 0.4, 0.7, 0.3), "no_observed_degradation"),
            ("unchanged", (0.5, 0.5, 0.5, 0.5), "no_observed_degradation"),
            ("worsened", (0.4, 0.6, 0.3, 0.7), "degraded"),
            ("mixed", (0.6, 0.4, 0.3, 0.7), "mixed"),
            ("same_path", (0.5, 0.5, 0.5, 0.5), "no_observed_degradation"),
        )
        for kind, candidate_values, expected_status in cases:
            with self.subTest(kind=kind):
                trace = deepcopy(self.trace)
                reference = trace[0]["result"]["data"]
                candidate = trace[-1]["result"]["data"]
                for metric, value in zip(METRIC_FIELDS, candidate_values, strict=True):
                    reference["epochs"][-1][metric] = 0.5
                    candidate["epochs"][-1][metric] = value
                reference_path = reference["run_directory"]
                candidate_path = candidate["run_directory"]
                if kind == "same_path":
                    trace = [trace[0]]
                    candidate_path = reference_path
                    candidate = reference
                else:
                    duplicate = deepcopy(trace[0])
                    duplicate["call_id"] = "another-ref-call"
                    trace.append(duplicate)
                original_trace = deepcopy(trace)

                feedback = build_optimizer_validation_feedback(
                    "The final JSON failed validation",
                    trace,
                    reference_run=reference_path,
                    candidate_run=candidate_path,
                )

                self.assertEqual(trace, original_trace)
                self.assertEqual(feedback["error"], "The final JSON failed validation")
                self.assertIn("scalar", feedback["citation_rule"].lower())
                check = feedback["performance_check"]
                self.assertIs(check["available"], True)
                self.assertIs(check["comparable"], True)
                self.assertEqual(check["expected_status"], expected_status)
                self.assertEqual(check["reference_run"], reference_path)
                self.assertEqual(check["candidate_run"], candidate_path)
                self.assertEqual(
                    check["reference_call_ids"],
                    ["ref-call"] if kind == "same_path" else ["another-ref-call", "ref-call"],
                )
                self.assertEqual(
                    check["candidate_call_ids"],
                    ["ref-call"] if kind == "same_path" else ["candidate-call"],
                )
                comparisons = check["final_metric_comparisons"]
                self.assertEqual([item["metric"] for item in comparisons], list(METRIC_FIELDS))
                for item in comparisons:
                    metric = item["metric"]
                    direction = 1 if metric.endswith("accuracy") else -1
                    expected_value = candidate["epochs"][-1][metric]
                    improvement = (expected_value - 0.5) * direction
                    self.assertEqual(item["reference_value"], 0.5)
                    self.assertEqual(item["candidate_value"], expected_value)
                    self.assertEqual(item["better_when"], "higher" if direction == 1 else "lower")
                    self.assertAlmostEqual(item["signed_improvement"], improvement)
                    self.assertEqual(
                        item["observation"],
                        (
                            "improved"
                            if improvement > 0
                            else "worsened"
                            if improvement < 0
                            else "unchanged"
                        ),
                    )
                json.dumps(feedback, allow_nan=False)

    def test_feedback_marks_incomparable_runs_without_metric_comparisons(self):
        for key in ("initial_state_sha256", "data_identity", "config"):
            with self.subTest(key=key):
                trace = deepcopy(self.trace)
                candidate = trace[-1]["result"]["data"]
                if key == "config":
                    candidate[key]["seed"] = 99
                elif key == "data_identity":
                    candidate[key]["metadata_sha256"] = "0" * 64
                else:
                    candidate[key] = "0" * 64

                feedback = build_optimizer_validation_feedback(
                    "Invalid final JSON",
                    trace,
                    reference_run="artifacts/reference",
                    candidate_run="artifacts/candidate",
                )

                check = feedback["performance_check"]
                self.assertIs(check["available"], True)
                self.assertIs(check["comparable"], False)
                self.assertEqual(check["expected_status"], "inconclusive")
                self.assertEqual(check["final_metric_comparisons"], [])

    def test_feedback_does_not_derive_performance_from_invalid_trace(self):
        for kind in ("missing", "malformed", "failed", "out_of_scope", "nonfinite", "duplicate"):
            with self.subTest(kind=kind):
                trace = deepcopy(self.trace)
                if kind == "missing":
                    trace.pop()
                elif kind == "malformed":
                    trace[-1]["raw_arguments"] = "{"
                elif kind == "failed":
                    trace[-1]["result"]["ok"] = False
                elif kind == "out_of_scope":
                    trace[-1]["raw_arguments"] = '{"run_directory": "artifacts/unrelated"}'
                elif kind == "nonfinite":
                    trace[-1]["result"]["data"]["epochs"][-1]["id_validation_loss"] = float("nan")
                else:
                    trace.append(deepcopy(trace[-1]))

                feedback = build_optimizer_validation_feedback(
                    "Original schema error",
                    trace,
                    reference_run="artifacts/reference",
                    candidate_run="artifacts/candidate",
                )

                self.assertEqual(feedback["error"], "Original schema error")
                self.assertIn("scalar", feedback["citation_rule"].lower())
                check = feedback["performance_check"]
                self.assertIs(check["available"], False)
                self.assertIsInstance(check["error"], str)
                self.assertTrue(check["error"])
                self.assertNotIn("expected_status", check)
                self.assertNotIn("final_metric_comparisons", check)
                json.dumps(feedback, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
