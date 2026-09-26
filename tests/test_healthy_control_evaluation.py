"""Offline healthy-control integration coverage; no API calls or training."""

import copy
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import runsleuth.evaluate as evaluate
from runsleuth.compare import compare_runs
from runsleuth.config import TrainingConfig
from runsleuth.evaluation import SmokeCase, SmokeSuite


class HealthyControlEvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source_path = "src/runsleuth/train.py"
        source = self.root / self.source_path
        source.parent.mkdir(parents=True)
        source.write_text("def train_one_epoch():\n    pass\n", encoding="utf-8")
        self.reference = "artifacts/runs/reference"
        self.candidate = "artifacts/runs/candidate"
        self._write_run(self.reference, learning_rate=0.001, accuracy=0.90, loss=0.40)
        self._write_run(self.candidate, learning_rate=0.002, accuracy=0.885, loss=0.44)

    def _write_json(self, relative_path, payload):
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _read_json(self, relative_path):
        return json.loads((self.root / relative_path).read_text(encoding="utf-8"))

    def _write_run(self, run, *, learning_rate, accuracy, loss):
        config = TrainingConfig(
            run_name=Path(run).name,
            output_dir=f"artifacts/config-output/{Path(run).name}",
            seed=7,
            epochs=3,
            learning_rate=learning_rate,
            optimizer_step_enabled=True,
        )
        self._write_json(f"{run}/config.json", asdict(config))
        rows = [
            {
                "epoch": epoch,
                "train_loss": 0.60 - epoch * 0.05,
                "train_accuracy": 0.80 + epoch * 0.02,
                "validation_loss": loss + (config.epochs - epoch) * 0.04,
                "validation_accuracy": accuracy - (config.epochs - epoch) * 0.02,
                "mean_gradient_norm": 1.25,
                "mean_parameter_update_norm": 0.05,
            }
            for epoch in range(1, config.epochs + 1)
        ]
        self._write_metrics(run, rows)

    def _read_metrics(self, run):
        return [
            json.loads(line)
            for line in (self.root / run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        ]

    def _write_metrics(self, run, rows):
        (self.root / run / "metrics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    def _case(self, **changes):
        values = {
            "id": "healthy-lr-change",
            "kind": "healthy_control",
            "reference_run": self.reference,
            "candidate_run": self.candidate,
            "expected_diagnosis_status": "no_supported_failure",
            "expected_root_cause": None,
            "agent_report": "reports/healthy.json",
            "repair_report": None,
        }
        values.update(changes)
        return SmokeCase.model_validate(values)

    def _suite(self, *cases):
        return SmokeSuite(
            suite="healthy-control-regression",
            max_repair_epochs=2,
            notes=[],
            source_path=self.source_path,
            cases=list(cases),
        )

    @staticmethod
    def _trace_entry(index, tool_name, arguments, data):
        return {
            "call_id": f"call-{index}",
            "tool_name": tool_name,
            "raw_arguments": json.dumps(arguments),
            "result": {"ok": True, "tool_name": tool_name, "data": data},
        }

    def _report(self, case):
        trace = [
            self._trace_entry(
                1,
                "inspect_training_source",
                {"source_path": self.source_path},
                {"source_path": self.source_path, "calls": []},
            )
        ]
        # A self-comparison legitimately has only one unique config request.
        for run in dict.fromkeys((case.reference_run, case.candidate_run)):
            config = TrainingConfig(**self._read_json(f"{run}/config.json"))
            trace.append(
                self._trace_entry(
                    len(trace) + 1,
                    "load_run_config",
                    {"run_directory": run},
                    {"run_directory": run, "config": asdict(config)},
                )
            )
        comparison = asdict(
            compare_runs(self.root / case.reference_run, self.root / case.candidate_run)
        )
        # Tool payloads use workspace-relative paths, unlike this direct call.
        comparison["clean"]["run_directory"] = case.reference_run
        comparison["candidate"]["run_directory"] = case.candidate_run
        trace.append(
            self._trace_entry(
                len(trace) + 1,
                "compare_runs",
                {"reference_run": case.reference_run, "candidate_run": case.candidate_run},
                comparison,
            )
        )
        return {
            "model": "offline-fixture",
            "status": "completed",
            "diagnosis": {
                "status": "no_supported_failure",
                "summary": "Both completed runs meet the healthy-control criteria.",
                "ranked_causes": [],
                "uncertainties": [],
                "proposed_patch": None,
            },
            "pending_evidence": [],
            "tool_calls": len(trace),
            "model_calls": 2,
            "input_tokens": 10,
            "output_tokens": 5,
            "usage_complete": True,
            "tool_trace": trace,
        }

    def _save_report(self, case):
        report = self._report(case)
        self._write_json(case.agent_report, report)
        return report

    def assertFraction(self, actual, numerator, denominator):
        self.assertEqual(
            actual,
            {
                "numerator": numerator,
                "denominator": denominator,
                "rate": numerator / denominator if denominator else None,
            },
        )

    def test_two_valid_controls_have_their_own_summary_denominator(self):
        first = self._case()
        third_run = "artifacts/runs/candidate-two"
        self._write_run(third_run, learning_rate=0.003, accuracy=0.89, loss=0.45)
        second = self._case(
            id="healthy-lr-change-two",
            candidate_run=third_run,
            agent_report="reports/healthy-two.json",
        )
        for case in (first, second):
            self._save_report(case)
        result = evaluate.evaluate_suite(self._suite(first, second), self.root)
        self.assertEqual(result["summary"]["passed_cases"], 2)
        self.assertFraction(result["summary"]["healthy_control_eligible"], 2, 2)
        self.assertFraction(result["summary"]["healthy_control_no_action"], 2, 2)
        self.assertFraction(result["summary"]["healthy_no_action"], 0, 0)
        for row in result["cases"]:
            self.assertEqual(row["outcome"], "passed")
            self.assertTrue(row["diagnosis"]["correct"])
            self.assertFalse(row["repair"]["required"])
            self.assertTrue(row["repair"]["passed"])
            health = row["healthy_control"]
            self.assertTrue(health["required"])
            self.assertTrue(health["passed"])
            self.assertIsNone(health["error"])
            self.assertEqual(set(health["config_changes"]), {"learning_rate"})
            self.assertEqual(health["policy"]["max_validation_accuracy_shortfall"], 0.03)
            self.assertEqual(health["policy"]["max_validation_loss_ratio"], 1.25)
            self.assertIn("final_validation_accuracy", health["reference"])
            self.assertIn("final_validation_loss", health["candidate"])
            self.assertTrue(all(check["passed"] for check in health["checks"]))
        self.assertEqual(result["selected_report_usage"]["tool_calls"], 8)

    def test_api_failure_stays_in_denominator_and_health_is_still_recomputed(self):
        successful = self._case()
        failed = self._case(id="api-error", agent_report="reports/api-error.json")
        self._save_report(successful)
        report = self._report(failed)
        report.update(status="api_error", diagnosis=None, tool_trace=[], tool_calls=0)
        self._write_json(failed.agent_report, report)
        result = evaluate.evaluate_suite(self._suite(successful, failed), self.root)
        self.assertFraction(result["summary"]["healthy_control_eligible"], 2, 2)
        self.assertFraction(result["summary"]["healthy_control_no_action"], 1, 2)
        self.assertEqual(result["summary"]["total_cases"], 2)
        self.assertEqual(result["summary"]["error_cases"], 1)
        row = result["cases"][1]
        self.assertTrue(row["healthy_control"]["passed"])
        self.assertEqual(row["outcome"], "error")
        self.assertFalse(row["diagnosis"]["correct"])

    def test_legacy_reference_config_uses_training_defaults(self):
        case = self._case()
        config_path = f"{self.reference}/config.json"
        config = self._read_json(config_path)
        del config["optimizer_step_enabled"]
        self._write_json(config_path, config)
        self._save_report(case)
        result = evaluate.evaluate_suite(self._suite(case), self.root)
        self.assertEqual(result["cases"][0]["outcome"], "passed")

    def test_invalid_intermediate_metric_invalidates_an_otherwise_correct_diagnosis(self):
        case = self._case()
        self._save_report(case)
        rows = self._read_metrics(self.candidate)
        rows[1]["validation_loss"] = float("nan")
        self._write_metrics(self.candidate, rows)
        result = evaluate.evaluate_suite(self._suite(case), self.root)
        row = result["cases"][0]
        self.assertEqual(row["outcome"], "error")
        self.assertFalse(row["diagnosis"]["correct"])
        self.assertFalse(row["healthy_control"]["passed"])
        self.assertIsNotNone(row["healthy_control"]["error"])
        self.assertFraction(result["summary"]["healthy_control_eligible"], 0, 1)
        self.assertFraction(result["summary"]["healthy_control_no_action"], 0, 1)

    def test_failed_health_threshold_does_not_count_as_a_correct_healthy_label(self):
        case = self._case()
        self._write_run(self.candidate, learning_rate=0.002, accuracy=0.70, loss=0.80)
        self._save_report(case)
        row = evaluate.evaluate_suite(self._suite(case), self.root)["cases"][0]
        self.assertEqual(row["outcome"], "error")
        self.assertFalse(row["healthy_control"]["passed"])
        self.assertFalse(row["diagnosis"]["correct"])
        self.assertTrue(any(not check["passed"] for check in row["healthy_control"]["checks"]))

    def test_stale_comparison_or_config_evidence_is_rejected(self):
        case = self._case()
        original = self._report(case)
        for tool_name in ("compare_runs", "load_run_config"):
            with self.subTest(tool_name=tool_name):
                report = copy.deepcopy(original)
                entry = next(
                    item for item in report["tool_trace"] if item["tool_name"] == tool_name
                )
                if tool_name == "compare_runs":
                    entry["result"]["data"]["candidate"]["final_validation_accuracy"] = 0.999
                else:
                    entry["result"]["data"]["config"]["learning_rate"] = 0.123
                self._write_json(case.agent_report, report)
                row = evaluate.evaluate_suite(self._suite(case), self.root)["cases"][0]
                self.assertTrue(row["healthy_control"]["passed"])
                self.assertEqual(row["outcome"], "error")
                self.assertFalse(row["diagnosis"]["correct"])
                self.assertIn("evidence", row["diagnosis"]["error"].lower())

    def test_config_changed_after_the_trace_is_rejected_even_when_still_healthy(self):
        case = self._case()
        self._save_report(case)
        config = self._read_json(f"{self.candidate}/config.json")
        config["learning_rate"] = 0.003
        self._write_json(f"{self.candidate}/config.json", config)
        row = evaluate.evaluate_suite(self._suite(case), self.root)["cases"][0]
        self.assertTrue(row["healthy_control"]["passed"])
        self.assertEqual(row["outcome"], "error")
        self.assertFalse(row["diagnosis"]["correct"])

    def test_resolved_path_alias_is_not_a_distinct_control_run(self):
        case = self._case(candidate_run=f"{self.reference}/../reference")
        score = evaluate.score_healthy_control(case, workspace_root=self.root)
        self.assertTrue(score["required"])
        self.assertFalse(score["passed"])
        self.assertIn("distinct", score["error"].lower())

    def test_wrong_diagnosis_fails_without_inflating_no_action_count(self):
        case = self._case()
        self._save_report(case)
        score = {
            "agent_status": "completed",
            "predicted_status": "failure_detected",
            "predicted_root_cause": "high_learning_rate",
            "correct": False,
            "citation_count": 0,
            "citation_values_valid": None,
            "error": None,
        }
        # Isolate aggregation from the separate fault-diagnosis citation schema.
        with patch.object(evaluate, "score_diagnosis", return_value=score):
            result = evaluate.evaluate_suite(self._suite(case), self.root)
        row = result["cases"][0]
        self.assertEqual(row["outcome"], "failed")
        self.assertTrue(row["healthy_control"]["passed"])
        self.assertFalse(row["diagnosis"]["correct"])
        self.assertFraction(result["summary"]["healthy_control_eligible"], 1, 1)
        self.assertFraction(result["summary"]["healthy_control_no_action"], 0, 1)

    def test_healthy_control_must_not_have_a_repair_report(self):
        case = self._case(repair_report="reports/unexpected-repair.json")
        self._save_report(case)
        self._write_json(case.repair_report, {})
        result = evaluate.evaluate_suite(self._suite(case), self.root)
        row = result["cases"][0]
        self.assertEqual(row["outcome"], "error")
        self.assertTrue(row["healthy_control"]["passed"])
        self.assertFalse(row["repair"]["passed"])
        self.assertIn("repair report", row["repair"]["error"].lower())
        self.assertFraction(result["summary"]["healthy_control_no_action"], 0, 1)

    def test_existing_self_comparison_metric_and_report_shape_are_preserved(self):
        case = self._case(kind="clean_self_comparison", candidate_run=self.reference)
        self._save_report(case)
        result = evaluate.evaluate_suite(self._suite(case), self.root)
        row = result["cases"][0]
        self.assertEqual(row["outcome"], "passed")
        self.assertFraction(result["summary"]["healthy_no_action"], 1, 1)
        self.assertNotIn("healthy_control_eligible", result["summary"])
        self.assertNotIn("healthy_control_no_action", result["summary"])
        self.assertNotIn("healthy_control", row)
        self.assertEqual(result["selected_report_usage"]["tool_calls"], 3)

    def test_healthy_control_schema_rejects_failure_expectations_or_same_raw_path(self):
        for changes in (
            {"expected_diagnosis_status": "failure_detected"},
            {"expected_root_cause": "high_learning_rate"},
            {"expected_root_cause": "missing_optimizer_step"},
            {"candidate_run": self.reference},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self._case(**changes)


if __name__ == "__main__":
    unittest.main()
