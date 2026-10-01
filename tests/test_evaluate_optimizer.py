"""Replay actual JSON artifacts to test offline comparison and honest accounting."""

import contextlib
import io
import json
import shutil
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import test_optimizer_agent_diagnosis as diagnosis_fixtures
import test_optimizer_evidence as evidence_fixtures

from runsleuth.evaluate_optimizer import evaluate_optimizer_suite, main
from runsleuth.optimizer_evidence import inspect_optimizer_run


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


class EvaluateOptimizerTests(unittest.TestCase):
    def setUp(self):
        # Reuse artifact creation without inheriting its TestCase or collecting it twice.
        self.fixture = evidence_fixtures.OptimizerEvidenceTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.clean = self.root / "clean"
        shutil.copytree(self.fixture.directory, self.clean)
        report = read_json(self.clean / "run_report.json")
        report["variant"] = "clean"
        report["optimizer_audit"].update(missing_trainable_names=[], foreign_parameter_tensors=0)
        for row in report["parameter_group_epochs"]:
            row["head"]["mean_parameter_update_l2_norm"] = 0.001
        evidence_fixtures.write_rows(
            self.clean / "parameter_group_metrics.jsonl", report["parameter_group_epochs"]
        )
        evidence_fixtures.write_json(self.clean / "run_report.json", report)
        self.manifest_path = self.root / "manifest.json"
        self.manifest = {"schema_version": 1, "name": "fixture-comparison", "cases": []}
        self.case, self.agent = self.add_case("defect", defect=True)

    def add_case(self, case_id, *, defect):
        directory = self.root / "cases" / case_id
        reference, candidate = directory / "reference", directory / "candidate"
        shutil.copytree(self.clean, reference)
        shutil.copytree(self.fixture.directory if defect else self.clean, candidate)
        case = {
            "id": case_id,
            "seed": self.fixture.config.seed,
            "reference_run": reference.relative_to(self.root).as_posix(),
            "candidate_run": candidate.relative_to(self.root).as_posix(),
            "expected": {
                "implementation_status": "optimizer_binding_defect"
                if defect
                else "no_optimizer_binding_defect",
                "root_cause": "optimizer_parameter_mismatch" if defect else None,
                "performance_status": "no_observed_degradation",
                "recommended_action": "review_optimizer_binding" if defect else "no_change",
            },
            "agent_reports": [f"agent-{case_id}.json"],
        }
        trace = []
        for call_id, name in (
            ("reference-call", case["reference_run"]),
            ("candidate-call", case["candidate_run"]),
        ):
            data = inspect_optimizer_run(self.root / name, require_file=self.fixture.require_file)
            data["run_directory"] = name
            entry = diagnosis_fixtures.make_entry(call_id, data)
            entry["result"].update(error_code=None, error_message=None)
            trace.append(entry)
        payload = diagnosis_fixtures.make_payload(trace, defect=defect)
        agent = {
            "profile": "optimizer_binding",
            "model": "fixture-model",
            "status": "completed",
            "error": None,
            "pending_evidence": [],
            "model_calls": 2,
            "tool_calls": 2,
            "input_tokens": 2100,
            "output_tokens": 350,
            "usage_complete": True,
            "limits": {"max_model_calls": 6, "max_tool_calls": 5, "max_output_tokens": 3000},
            "first_pass_valid": True,
            "validation_retries": 0,
            "max_validation_retries": 1,
            "tool_trace": trace,
            "diagnosis": payload,
            "final_text": json.dumps(payload),
            "validation_attempts": [
                {
                    "model_call": 2,
                    "raw_final_text": json.dumps(payload),
                    "valid": True,
                    "error": None,
                }
            ],
        }
        self.manifest["cases"].append(case)
        self.save_agent(agent, case)
        return case, agent

    def save_agent(self, agent=None, case=None):
        selected_case = self.case if case is None else case
        selected_agent = self.agent if agent is None else agent
        evidence_fixtures.write_json(self.root / selected_case["agent_reports"][-1], selected_agent)

    def evaluate(self):
        evidence_fixtures.write_json(self.manifest_path, self.manifest)
        return evaluate_optimizer_suite(self.manifest_path, workspace_root=self.root)

    def fail_attempt(self, agent, *, trace_count=0):
        result = deepcopy(agent)
        result.update(
            status="api_error",
            error="InternalServerError: UNAVAILABLE",
            diagnosis=None,
            final_text=None,
            first_pass_valid=None,
            validation_retries=0,
            validation_attempts=[],
            tool_trace=deepcopy(agent["tool_trace"][:trace_count]),
            model_calls=2 if trace_count else 1,
            tool_calls=trace_count,
            input_tokens=909 if trace_count else 0,
            output_tokens=165 if trace_count else 0,
            usage_complete=False,
        )
        return result

    def test_four_cases_seven_attempts_preserve_failures_and_cumulative_cost(self):
        pairs = [(self.case, self.agent)]
        pairs += [
            self.add_case(name, defect=defect)
            for name, defect in (("control-a", False), ("defect-b", True), ("control-b", False))
        ]
        for index, (case, agent) in enumerate(pairs[1:]):
            failed_name = f"failed-{case['id']}.json"
            evidence_fixtures.write_json(
                self.root / failed_name,
                self.fail_attempt(agent, trace_count=2 if index == 0 else 0),
            )
            case["agent_reports"].insert(0, failed_name)
        report = self.evaluate()
        summary, usage = report["summary"], report["all_attempt_usage"]
        self.assertEqual(report["status"], "completed")
        self.assertEqual(summary["total_cases"], 4)
        for method, correct in (
            ("rules", 4),
            ("agent_first_attempt", 1),
            ("agent_after_recovery", 4),
        ):
            self.assertEqual(summary[method]["correct"]["numerator"], correct)
            self.assertEqual(summary[method]["correct"]["denominator"], 4)
        self.assertEqual(summary["attempt_status_counts"], {"completed": 4, "api_error": 3})
        self.assertEqual(summary["completed_reports_first_pass_valid"]["numerator"], 4)
        self.assertEqual(usage["attempt_report_count"], 7)
        self.assertEqual(usage["model_calls"], 12)
        self.assertEqual(usage["tool_calls"], 10)
        self.assertEqual(usage["input_tokens"], 9309)
        self.assertEqual(usage["output_tokens"], 1565)
        self.assertFalse(usage["usage_complete"])
        self.assertEqual(usage["unreadable_usage_reports"], 0)
        self.assertEqual(report["new_execution"], {"api_calls": 0, "training_runs": 0})
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        evaluate_optimizer_suite(self.manifest_path, workspace_root=self.root)
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()})

    def test_valid_completed_but_wrong_expected_label_is_scored_wrong_not_error(self):
        case, _ = self.add_case("control", defect=False)
        case["expected"] = deepcopy(self.case["expected"])
        self.manifest["cases"] = [case]
        report = self.evaluate()
        self.assertEqual(report["status"], "completed_with_failures")
        attempt = report["cases"][0]["agent_attempts"][0]
        self.assertEqual(attempt["status"], "completed")
        self.assertIsNone(attempt["error"])
        self.assertEqual(attempt["labels"]["implementation_status"], "no_optimizer_binding_defect")
        for method in ("rules", "agent_first_attempt", "agent_after_recovery"):
            self.assertEqual(report["summary"][method]["correct"]["numerator"], 0)
            self.assertEqual(report["summary"][method]["unscored_cases"], 0)

    def test_completed_attempt_cannot_be_replaced_even_if_it_scored_wrong(self):
        self.case["expected"]["performance_status"] = "mixed"
        first_path = self.root / self.case["agent_reports"][0]
        self.case["agent_reports"].append("later.json")
        self.save_agent()
        for corrupt_counter in (False, True):
            with self.subTest(corrupt_counter=corrupt_counter):
                first = deepcopy(self.agent)
                if corrupt_counter:
                    first["model_calls"] = True
                evidence_fixtures.write_json(first_path, first)
                with self.assertRaisesRegex(ValueError, "replace a completed attempt"):
                    self.evaluate()

    def test_duplicate_canonical_report_paths_and_case_pairs_are_rejected(self):
        original = deepcopy(self.manifest)
        self.case["agent_reports"].append("./" + self.case["agent_reports"][0])
        with self.assertRaisesRegex(ValueError, "counted more than once"):
            self.evaluate()
        self.manifest = deepcopy(original)
        duplicate = deepcopy(self.manifest["cases"][0])
        duplicate.update(id="different-id", reference_run="./" + duplicate["reference_run"])
        self.manifest["cases"].append(duplicate)
        with self.assertRaisesRegex(ValueError, "pairs must be unique"):
            self.evaluate()

    def test_changed_artifact_bytes_invalidate_saved_evidence_but_preserve_rules(self):
        path = self.root / self.case["candidate_run"] / "config.json"
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        report = self.evaluate()
        self.assertEqual(report["summary"]["rules"]["correct"]["numerator"], 1)
        attempt = report["cases"][0]["agent_attempts"][0]
        self.assertEqual(attempt["status"], "report_error")
        self.assertIn("current optimizer artifacts", attempt["error"])
        self.assertEqual(report["all_attempt_usage"]["input_tokens"], 2100)

    def test_missing_report_stays_in_denominator_and_makes_usage_incomplete(self):
        (self.root / self.case["agent_reports"][0]).unlink()
        report = self.evaluate()
        self.assertEqual(report["summary"]["rules"]["correct"]["numerator"], 1)
        score = report["summary"]["agent_after_recovery"]
        self.assertEqual(score["correct"], {"numerator": 0, "denominator": 1, "rate": 0.0})
        self.assertEqual(score["unscored_cases"], 1)
        self.assertEqual(report["all_attempt_usage"]["unreadable_usage_reports"], 1)
        self.assertFalse(report["all_attempt_usage"]["usage_complete"])
        self.case["agent_reports"].append("recovery.json")
        self.save_agent()
        recovered = self.evaluate()
        self.assertEqual(recovered["summary"]["agent_after_recovery"]["correct"]["numerator"], 1)
        self.assertEqual(recovered["status"], "completed_with_failures")
        self.assertFalse(recovered["all_attempt_usage"]["usage_complete"])

    def test_api_errors_distinguish_manifest_only_from_partly_refreshed_evidence(self):
        second_case, second_agent = self.add_case("control", defect=False)
        self.save_agent(self.fail_attempt(self.agent), self.case)
        self.save_agent(self.fail_attempt(second_agent, trace_count=1), second_case)
        report = self.evaluate()
        attempts = [case["agent_attempts"][0] for case in report["cases"]]
        self.assertEqual([a["status"] for a in attempts], ["api_error", "api_error"])
        self.assertEqual(
            [a["task_binding"] for a in attempts], ["manifest_only", "partial_evidence_refreshed"]
        )
        self.assertEqual(report["summary"]["agent_after_recovery"]["unscored_cases"], 2)
        self.assertFalse(report["all_attempt_usage"]["usage_complete"])

    def test_boolean_usage_counters_and_limits_are_not_valid_integers(self):
        original = deepcopy(self.agent)
        for field in ("model_calls", "tool_calls", "input_tokens", "output_tokens"):
            with self.subTest(counter=field):
                self.agent = deepcopy(original)
                self.agent[field] = True
                self.save_agent()
                report = self.evaluate()
                self.assertEqual(report["cases"][0]["agent_attempts"][0]["status"], "report_error")
                self.assertFalse(report["all_attempt_usage"]["usage_complete"])
        self.agent = deepcopy(original)
        self.agent["limits"]["max_model_calls"] = True
        self.save_agent()
        report = self.evaluate()
        self.assertEqual(report["cases"][0]["agent_attempts"][0]["status"], "report_error")
        failed = self.fail_attempt(original)
        failed["usage_complete"] = True
        self.save_agent(failed)
        report = self.evaluate()
        self.assertEqual(report["status"], "completed_with_failures")
        self.assertFalse(report["all_attempt_usage"]["usage_complete"])

    def test_mixed_requested_models_or_call_limits_cannot_be_pooled(self):
        second_case, second_agent = self.add_case("control", defect=False)
        for change in ("model", "limits"):
            with self.subTest(change=change):
                altered = deepcopy(second_agent)
                if change == "model":
                    altered["model"] = "other-model"
                else:
                    altered["limits"]["max_output_tokens"] += 1
                self.save_agent(altered, second_case)
                with self.assertRaisesRegex(ValueError, "one requested model"):
                    self.evaluate()

    def test_paths_outside_workspace_are_rejected_before_evaluation(self):
        original = deepcopy(self.case)
        for field in ("reference_run", "candidate_run", "agent_reports"):
            with self.subTest(field=field):
                self.manifest["cases"] = [deepcopy(original)]
                self.manifest["cases"][0][field] = (
                    ["../outside.json"] if field == "agent_reports" else "../outside"
                )
                with self.assertRaisesRegex(ValueError, "inside the workspace"):
                    self.evaluate()
        with self.assertRaisesRegex(ValueError, "inside the workspace"):
            evaluate_optimizer_suite(self.root.parent / "outside.json", workspace_root=self.root)

    def test_malformed_json_fails_manifest_and_retains_invalid_report_case(self):
        self.manifest["schema_version"] = True
        with self.assertRaises(ValueError):
            self.evaluate()
        self.manifest["schema_version"] = 1
        for malformed in ('{"schema_version":1,"schema_version":1}', "[]", '{"x": NaN}'):
            with self.subTest(manifest=malformed):
                self.manifest_path.write_text(malformed, encoding="utf-8")
                with self.assertRaises(ValueError):
                    evaluate_optimizer_suite(self.manifest_path, workspace_root=self.root)
        for malformed in ("{", '{"status":"completed","status":"api_error"}', "[]"):
            with self.subTest(report=malformed):
                (self.root / self.case["agent_reports"][0]).write_text(malformed, encoding="utf-8")
                report = self.evaluate()
                self.assertEqual(report["summary"]["total_cases"], 1)
                self.assertEqual(report["summary"]["attempt_status_counts"], {"report_error": 1})
                self.assertFalse(report["all_attempt_usage"]["usage_complete"])
        self.case["agent_reports"].append("recovery.json")
        self.save_agent()
        recovered = self.evaluate()
        self.assertEqual(recovered["summary"]["agent_after_recovery"]["correct"]["numerator"], 1)
        self.assertEqual(recovered["status"], "completed_with_failures")

    def test_mismatched_manifest_seed_cannot_score_either_method(self):
        self.case["seed"] += 1
        report = self.evaluate()
        self.assertIn("training seed", report["cases"][0]["rules"]["error"])
        for method in ("rules", "agent_first_attempt", "agent_after_recovery"):
            self.assertEqual(report["summary"][method]["correct"]["numerator"], 0)
            self.assertEqual(report["summary"][method]["correct"]["denominator"], 1)

    def test_validation_provenance_is_replayed_instead_of_trusting_flags(self):
        original = deepcopy(self.agent)
        mutations = (
            lambda r: r.update(validation_attempts=[]),
            lambda r: r.update(first_pass_valid=False),
            lambda r: r["validation_attempts"][0].update(valid=False),
            lambda r: r["validation_attempts"][0].update(model_call=True),
            lambda r: r["validation_attempts"][0].update(raw_final_text="{}"),
            lambda r: r["diagnosis"].update(summary="Different saved diagnosis"),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                self.agent = deepcopy(original)
                mutate(self.agent)
                self.save_agent()
                report = self.evaluate()
                self.assertEqual(report["cases"][0]["agent_attempts"][0]["status"], "report_error")
                self.assertEqual(report["summary"]["agent_after_recovery"]["unscored_cases"], 1)

    def test_bounded_validation_correction_is_separate_from_api_recovery(self):
        self.agent.update(first_pass_valid=False, validation_retries=1, model_calls=3)
        self.agent["validation_attempts"][0]["model_call"] = 3
        self.agent["validation_attempts"].insert(
            0, {"model_call": 2, "raw_final_text": "{}", "valid": False, "error": "invalid schema"}
        )
        self.save_agent()
        report = self.evaluate()
        self.assertEqual(report["status"], "completed")
        summary = report["summary"]
        self.assertEqual(summary["agent_first_attempt"]["correct"]["numerator"], 1)
        self.assertEqual(
            summary["completed_reports_first_pass_valid"],
            {"numerator": 0, "denominator": 1, "rate": 0.0},
        )
        self.assertEqual(report["all_attempt_usage"]["attempt_report_count"], 1)
        self.assertEqual(report["all_attempt_usage"]["model_calls"], 3)

    def test_cli_writes_a_new_report_and_returns_failure_for_unscored_cases(self):
        evidence_fixtures.write_json(self.manifest_path, self.manifest)
        output = io.StringIO()
        args = ["evaluate_optimizer", "--manifest", str(self.manifest_path)]
        with patch("sys.argv", args), patch.object(Path, "cwd", return_value=self.root):
            with contextlib.redirect_stdout(output):
                main()
            directory = self.root / "artifacts" / "optimizer_evaluations"
            first = next(directory.glob("*.json"))
            before = first.read_bytes()
            self.assertEqual(read_json(first)["status"], "completed")
            (self.root / self.case["agent_reports"][0]).unlink()
            with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as stopped:
                main()
            self.assertEqual(stopped.exception.code, 1)
            self.assertEqual(len(list(directory.glob("*.json"))), 2)
            self.assertEqual(first.read_bytes(), before)
        self.assertIn("optimizer_evaluation_report=", output.getvalue())


if __name__ == "__main__":
    unittest.main()
