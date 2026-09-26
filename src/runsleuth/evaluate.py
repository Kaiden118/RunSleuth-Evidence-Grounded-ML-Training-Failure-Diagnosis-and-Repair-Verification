"""Evaluate selected saved agent runs without API calls or retraining."""

import argparse
import json
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath

from runsleuth.agent_diagnosis import AgentDiagnosis
from runsleuth.config import TrainingConfig
from runsleuth.diagnosis import RootCause
from runsleuth.evaluation import SmokeCase, SmokeSuite, load_smoke_suite
from runsleuth.evidence_validation import validate_diagnosis_evidence
from runsleuth.repair import apply_config_patch
from runsleuth.verification import RepairDecision, VerificationPolicy, verify_repair


def _workspace_path(path: str, workspace_root: Path) -> Path:
    """Resolve an artifact path while keeping it inside the workspace."""
    if not isinstance(path, str) or not path.strip():
        raise ValueError("Artifact path must be a nonempty string")
    windows_path = PureWindowsPath(path)
    if windows_path.drive or windows_path.root:
        raise ValueError("Artifact paths must be relative to the workspace")
    root = workspace_root.resolve()
    artifact_path = (root / path.replace("\\", "/")).resolve()
    if not artifact_path.is_relative_to(root):
        raise ValueError("Artifact path escapes the workspace")
    return artifact_path


def read_report(path: str, workspace_root: Path) -> dict[str, object]:
    """Read a JSON object located inside the project workspace."""
    report = json.loads(_workspace_path(path, workspace_root).read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("Report must contain a JSON object")
    return report


def _same_json(left: object, right: object) -> bool:
    """Keep Boolean and numeric values distinct."""
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False
    )


def _request_key(tool_name: str, arguments: dict[str, object]) -> tuple[str, str]:
    return tool_name, json.dumps(arguments, sort_keys=True, allow_nan=False)


def _check_tool_scope(trace: list[dict[str, object]], case: SmokeCase, source_path: str) -> None:
    requests = [
        ("inspect_training_source", {"source_path": source_path}),
        ("load_run_config", {"run_directory": case.reference_run}),
        ("load_run_config", {"run_directory": case.candidate_run}),
        (
            "compare_runs",
            {"reference_run": case.reference_run, "candidate_run": case.candidate_run},
        ),
    ]
    expected = {_request_key(name, arguments) for name, arguments in requests}
    observed: set[tuple[str, str]] = set()
    for entry in trace:
        name, result = entry.get("tool_name"), entry.get("result")
        if not isinstance(name, str) or not name:
            raise ValueError("Tool trace contains an invalid tool name")
        if not isinstance(result, dict) or type(result.get("ok")) is not bool:
            raise ValueError("Tool trace contains an invalid result")
        if result.get("tool_name") != name:
            raise ValueError("Tool name differs between request and result")
        if result["ok"] is False:
            continue
        if not isinstance(result.get("data"), dict):
            raise ValueError("Successful tool result must contain object data")
        raw_arguments = entry.get("raw_arguments")
        if not isinstance(raw_arguments, str):
            raise ValueError("Tool arguments must be recorded as JSON text")
        arguments = json.loads(raw_arguments)
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must contain a JSON object")
        request = _request_key(name, arguments)
        if request not in expected:
            raise ValueError("Successful tool request does not belong to this case")
        observed.add(request)
    if expected - observed:
        raise ValueError("Report is missing required successful evidence requests")


def _empty_diagnosis_score() -> dict[str, object]:
    return {
        "agent_status": None,
        "predicted_status": None,
        "predicted_root_cause": None,
        "correct": False,
        "citation_count": 0,
        "citation_values_valid": None,
        "error": None,
    }


def score_diagnosis(
    case: SmokeCase, report: dict[str, object], *, source_path: str
) -> dict[str, object]:
    """Score status and Top-1 cause after checking recorded evidence."""
    score = _empty_diagnosis_score()
    try:
        status = report.get("status")
        if not isinstance(status, str) or not status:
            raise ValueError("Agent report must contain a status")
        score["agent_status"] = status
        if status != "completed":
            raise ValueError(f"Agent run did not complete: {status}")
        if report.get("pending_evidence") != []:
            raise ValueError("Agent report has missing or unfinished evidence")
        trace = report.get("tool_trace")
        if not isinstance(trace, list) or not all(isinstance(item, dict) for item in trace):
            raise ValueError("Agent report must contain a tool trace")
        if type(report.get("tool_calls")) is not int or report["tool_calls"] != len(trace):
            raise ValueError("Recorded tool count does not match the trace")
        diagnosis = AgentDiagnosis.model_validate(report.get("diagnosis"))
        top_cause = diagnosis.ranked_causes[0].root_cause if diagnosis.ranked_causes else None
        citation_count = sum(len(cause.evidence) for cause in diagnosis.ranked_causes)
        score.update(
            predicted_status=diagnosis.status,
            predicted_root_cause=top_cause,
            citation_count=citation_count,
        )
        _check_tool_scope(trace, case, source_path)
        if citation_count:
            score["citation_values_valid"] = False
        validate_diagnosis_evidence(diagnosis, trace, candidate_run=case.candidate_run)
        if citation_count:
            score["citation_values_valid"] = True
        score["correct"] = (
            diagnosis.status == case.expected_diagnosis_status
            and top_cause == case.expected_root_cause
        )
    except ValueError as error:
        score["error"] = str(error)
    return score


def _check_recorded_epochs(metrics_path: Path, expected_epochs: int) -> None:
    rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != expected_epochs:
        raise ValueError("Recorded epoch count does not match the repaired config")
    for expected_epoch, row in enumerate(rows, start=1):
        if (
            not isinstance(row, dict)
            or type(row.get("epoch")) is not int
            or row["epoch"] != expected_epoch
        ):
            raise ValueError("Recorded epochs must be consecutive and start at 1")


def _empty_repair_score(case: SmokeCase) -> dict[str, object]:
    return {
        "required": case.kind == "injected_fault",
        "passed": False,
        "accepted": None,
        "within_budget": None,
        "error": None,
        "skip_reason": None,
    }


def score_repair(
    case: SmokeCase,
    agent_report: dict[str, object],
    *,
    workspace_root: Path,
    max_repair_epochs: int,
) -> dict[str, object]:
    """Check saved repair artifacts; recompute acceptance without training."""
    score = _empty_repair_score(case)
    try:
        if not score["required"]:
            if case.repair_report is not None:
                raise ValueError("Healthy self-comparison must not have a repair report")
            score["passed"] = True
            return score
        score.update(accepted=False, within_budget=False)
        if case.repair_report is None:
            raise ValueError("Fault case is missing its repair report")
        if type(max_repair_epochs) is not int or max_repair_epochs < 1:
            raise ValueError("Suite repair budget must be a positive integer")
        report = read_report(case.repair_report, workspace_root)
        if _workspace_path(report.get("agent_report"), workspace_root) != _workspace_path(
            case.agent_report, workspace_root
        ):
            raise ValueError("Repair report belongs to a different agent report")
        model = agent_report.get("model")
        if not isinstance(model, str) or not model or report.get("model") != model:
            raise ValueError("Repair model does not match the agent report")
        if agent_report.get("status") != "completed":
            raise ValueError("Repair requires a completed agent diagnosis")
        diagnosis = AgentDiagnosis.model_validate(agent_report.get("diagnosis"))
        if not _same_json(report.get("diagnosis"), diagnosis.model_dump(mode="json")):
            raise ValueError("Repair diagnosis differs from the agent diagnosis")
        patch = diagnosis.proposed_patch
        if diagnosis.status != "failure_detected" or patch is None:
            raise ValueError("Repair requires a detected failure and a proposed patch")
        trace = agent_report.get("tool_trace")
        if not isinstance(trace, list) or not all(isinstance(item, dict) for item in trace):
            raise ValueError("Agent report must contain a tool trace")
        validate_diagnosis_evidence(diagnosis, trace, candidate_run=case.candidate_run)
        if report.get("training_mode") != "from_scratch":
            raise ValueError("This suite requires repair training from scratch")
        budget = report.get("max_epochs")
        if type(budget) is not int or not 1 <= budget <= max_repair_epochs:
            raise ValueError("Recorded repair budget exceeds the suite limit")
        failed_config = TrainingConfig(
            **read_report(f"{case.candidate_run}/config.json", workspace_root)
        )
        config_call = next(
            (item for item in trace if item.get("call_id") == patch.config_call_id), None
        )
        result = config_call.get("result") if isinstance(config_call, dict) else None
        data = result.get("data") if isinstance(result, dict) else None
        if not isinstance(data, dict) or not _same_json(asdict(failed_config), data.get("config")):
            raise ValueError("Failed run config differs from the agent's recorded evidence")
        if type(failed_config.epochs) is not int or failed_config.epochs < 1:
            raise ValueError("Failed run must have a positive integer epoch count")
        expected_config = apply_config_patch(
            failed_config,
            root_cause=RootCause(diagnosis.ranked_causes[0].root_cause),
            field=patch.field,
            old_value=patch.old_value,
            new_value=patch.new_value,
        )
        expected_config = replace(
            expected_config,
            run_name=f"{failed_config.run_name}-agent-repair",
            epochs=min(failed_config.epochs, budget),
        )
        repaired_name = report.get("repaired_run")
        repaired_run = _workspace_path(repaired_name, workspace_root)
        reference_run = _workspace_path(case.reference_run, workspace_root)
        failed_run = _workspace_path(case.candidate_run, workspace_root)
        if repaired_run in (reference_run, failed_run):
            raise ValueError("Repair must have its own run directory")
        repaired_config = TrainingConfig(
            **read_report(f"{repaired_name}/config.json", workspace_root)
        )
        expected_values = asdict(expected_config)
        if not _same_json(asdict(repaired_config), expected_values) or not _same_json(
            report.get("repaired_config"), expected_values
        ):
            raise ValueError("Repaired config contains missing or unexpected changes")
        for run_name in (case.reference_run, case.candidate_run, repaired_name):
            _workspace_path(f"{run_name}/metrics.jsonl", workspace_root)
        _check_recorded_epochs(
            _workspace_path(f"{repaired_name}/metrics.jsonl", workspace_root),
            expected_config.epochs,
        )
        score["within_budget"] = True
        saved = report.get("verification")
        if not isinstance(saved, dict):
            raise ValueError("Repair report is missing its verification")
        paths = {
            "reference_run": reference_run,
            "failed_run": failed_run,
            "repaired_run": repaired_run,
        }
        for field, expected_path in paths.items():
            if _workspace_path(saved.get(field), workspace_root) != expected_path:
                raise ValueError(f"Verification uses a different {field}")
        verification = verify_repair(
            reference_run=reference_run,
            failed_run=failed_run,
            repaired_run=repaired_run,
            policy=VerificationPolicy(
                min_validation_accuracy_gain=0.20,
                max_validation_accuracy_shortfall=0.03,
                max_validation_loss_ratio=1.25,
            ),
        )
        saved_values = {key: value for key, value in saved.items() if key not in paths}
        fresh_values = {
            key: value for key, value in asdict(verification).items() if key not in paths
        }
        if not _same_json(saved_values, fresh_values):
            raise ValueError("Saved verification does not match recomputed metrics")
        accepted = verification.decision is RepairDecision.ACCEPTED
        score.update(accepted=accepted, passed=accepted)
    except (OSError, ValueError, TypeError) as error:
        score["error"] = str(error)
    return score


def _usage_metadata(report: dict[str, object]) -> dict[str, object]:
    """Preserve available counters even when a selected attempt failed."""
    usage: dict[str, object] = {}
    invalid = []
    for field in ("model_calls", "tool_calls", "input_tokens", "output_tokens"):
        value = report.get(field)
        valid = type(value) is int and value >= 0
        usage[field] = value if valid else None
        if not valid:
            invalid.append(field)
    complete = report.get("usage_complete")
    if type(complete) is not bool:
        invalid.append("usage_complete")
    usage["usage_complete"] = complete is True and not invalid
    usage["error"] = f"Invalid usage fields: {', '.join(invalid)}" if invalid else None
    return usage


def _fraction(numerator: int, denominator: int) -> dict[str, object]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def evaluate_suite(suite: SmokeSuite, workspace_root: Path) -> dict[str, object]:
    """Keep every manifest case in the evaluation, including failed attempts."""
    cases = []
    for case in suite.cases:
        diagnosis = _empty_diagnosis_score()
        repair = _empty_repair_score(case)
        usage = _usage_metadata({})
        model = None
        try:
            agent_report = read_report(case.agent_report, workspace_root)
        except (OSError, ValueError) as error:
            diagnosis["error"] = f"Cannot read agent report: {error}"
            repair["skip_reason"] = "Agent report is unavailable"
        else:
            model = agent_report.get("model")
            usage = _usage_metadata(agent_report)
            diagnosis = score_diagnosis(case, agent_report, source_path=suite.source_path)
            if diagnosis["error"] is None:
                repair = score_repair(
                    case,
                    agent_report,
                    workspace_root=workspace_root,
                    max_repair_epochs=suite.max_repair_epochs,
                )
            else:
                repair["skip_reason"] = "Diagnosis integrity or completion check failed"
        errors = [item["error"] for item in (diagnosis, repair, usage) if item["error"]]
        if errors:
            outcome = "error"
        else:
            outcome = "passed" if diagnosis["correct"] and repair["passed"] else "failed"
        cases.append(
            {
                "id": case.id,
                "kind": case.kind,
                "expected_status": case.expected_diagnosis_status,
                "expected_root_cause": case.expected_root_cause,
                "agent_report": case.agent_report,
                "repair_report": case.repair_report,
                "model": model,
                "outcome": outcome,
                "diagnosis": diagnosis,
                "repair": repair,
                "usage": usage,
            }
        )
    faults = [case for case in cases if case["kind"] == "injected_fault"]
    healthy = [case for case in cases if case["kind"] == "clean_self_comparison"]
    cited = [case for case in cases if case["diagnosis"]["citation_count"] > 0]
    token_totals = {
        field: sum(case["usage"][field] or 0 for case in cases)
        for field in ("model_calls", "tool_calls", "input_tokens", "output_tokens")
    }
    token_totals["usage_complete"] = all(case["usage"]["usage_complete"] for case in cases)
    token_totals["scope"] = "Only the reports selected in this manifest"
    return {
        "suite": suite.suite,
        "evaluation_type": "selected_run_smoke",
        "max_repair_epochs": suite.max_repair_epochs,
        "notes": suite.notes,
        "summary": {
            "total_cases": len(cases),
            "passed_cases": sum(case["outcome"] == "passed" for case in cases),
            "failed_cases": sum(case["outcome"] == "failed" for case in cases),
            "error_cases": sum(case["outcome"] == "error" for case in cases),
            "diagnosis_correct": _fraction(
                sum(case["diagnosis"]["correct"] for case in cases), len(cases)
            ),
            "fault_top1_correct": _fraction(
                sum(case["diagnosis"]["correct"] for case in faults), len(faults)
            ),
            "verified_fault_repairs": _fraction(
                sum(case["repair"]["accepted"] is True for case in faults), len(faults)
            ),
            "healthy_no_action": _fraction(
                sum(case["outcome"] == "passed" for case in healthy), len(healthy)
            ),
            "reports_with_valid_citation_values": _fraction(
                sum(case["diagnosis"]["citation_values_valid"] is True for case in cited),
                len(cited),
            ),
        },
        "selected_report_usage": token_totals,
        "cases": cases,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="evaluations/smoke_cases.json")
    parser.add_argument("--output-dir", default="artifacts/evaluations")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    root = Path.cwd()
    try:
        suite = load_smoke_suite(_workspace_path(args.manifest, root))
        output_dir = _workspace_path(args.output_dir, root)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    report = evaluate_suite(suite, root)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    report_path = output_dir / f"smoke-{timestamp}.json"
    with report_path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"evaluation_report={report_path.relative_to(root)}")
    print(json.dumps(report["summary"], indent=2))
    print(json.dumps(report["selected_report_usage"], indent=2))
    if report["summary"]["passed_cases"] != report["summary"]["total_cases"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
