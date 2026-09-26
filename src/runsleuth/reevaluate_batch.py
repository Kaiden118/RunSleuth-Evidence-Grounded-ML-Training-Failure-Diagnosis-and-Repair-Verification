"""Evaluate another LLM on saved training runs, without regenerating the dataset."""

import argparse
import json
import math
import re
import time
from datetime import UTC, datetime
from pathlib import Path

from runsleuth.agent import run_diagnostic_agent
from runsleuth.agent_repair import run_bounded_agent_repair
from runsleuth.batch_evaluate import (
    BatchSettings,
    _error_record,
    _relative_run,
    _save_json,
    _save_text,
)
from runsleuth.config import TrainingConfig
from runsleuth.diagnostic_tools import DiagnosticTools
from runsleuth.evaluate import _usage_metadata, _workspace_path, evaluate_suite, read_report
from runsleuth.evaluation import load_smoke_suite
from runsleuth.llm_client import create_llm_client
from runsleuth.retry_batch import _retryable, _validate_limits


def prepare_reevaluation(batch_summary_path: str, workspace_root: Path) -> dict[str, object]:
    """Validate every original case locally; never select cases by diagnosis outcome."""
    root = workspace_root.resolve()
    original = read_report(batch_summary_path, root)
    settings_data = original.get("settings")
    if not isinstance(settings_data, dict):
        raise ValueError("Batch summary is missing settings")
    try:
        settings = BatchSettings(**settings_data)
    except TypeError as error:
        raise ValueError("Batch settings do not match the expected schema") from error
    batch_directory = _workspace_path(original.get("batch_directory"), root)
    manifest_path = _workspace_path(original.get("manifest_path"), root)
    if not manifest_path.is_relative_to(batch_directory):
        raise ValueError("Original manifest is outside its batch directory")
    suite = load_smoke_suite(manifest_path)
    if any(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", case.id) is None for case in suite.cases
    ):
        raise ValueError("Case IDs must be safe filename identifiers")
    if suite.source_path != settings.source_path:
        raise ValueError("Source path differs between manifest and batch settings")
    if suite.max_repair_epochs != settings.max_repair_epochs:
        raise ValueError("Repair budget differs between manifest and batch settings")
    if not _workspace_path(suite.source_path, root).is_file():
        raise ValueError("Training source is missing")
    progress = original.get("progress")
    evaluation = original.get("evaluation")
    if not isinstance(progress, dict) or not isinstance(evaluation, dict):
        raise ValueError("Batch summary is missing progress or evaluation")
    original_model = progress.get("model")
    if not isinstance(original_model, str) or not original_model.strip():
        raise ValueError("Original batch must identify its model")
    ids = {case.id for case in suite.cases}
    records = progress.get("cases")
    for rows in (records, evaluation.get("cases")):
        if (
            not isinstance(rows, list)
            or not all(isinstance(row, dict) and isinstance(row.get("id"), str) for row in rows)
            or len(rows) != len(ids)
            or {row["id"] for row in rows} != ids
        ):
            raise ValueError("Manifest, progress, and evaluation must contain identical case IDs")
    if any(row.get("training_status") != "completed" for row in records):
        raise ValueError("All original cases must have completed training before reevaluation")
    run_root = (batch_directory / "runs").resolve()
    for case in suite.cases:
        for run in (case.reference_run, case.candidate_run):
            directory = _workspace_path(run, root)
            if not directory.is_relative_to(run_root):
                raise ValueError(f"Case {case.id} has a run outside the original batch")
            if not all((directory / name).is_file() for name in ("config.json", "metrics.jsonl")):
                raise ValueError(f"Case {case.id} is missing saved training artifacts")
            config = TrainingConfig.load(directory / "config.json")
            if _workspace_path(config.output_dir, root) != run_root:
                raise ValueError(f"Case {case.id} has an unexpected repair output directory")
    return {
        "original": original,
        "settings": settings,
        "suite": suite,
        "original_model": original_model,
        "batch_directory": batch_directory,
    }


def reevaluation_budget(plan: dict[str, object], max_attempts: int) -> dict[str, int]:
    settings = plan["settings"]
    count = len(plan["suite"].cases)
    attempts = count * max_attempts
    return {
        "total_cases": count,
        "agent_attempt_upper_bound": attempts,
        "model_call_upper_bound": attempts * settings.max_model_calls,
        "agent_tool_call_upper_bound": attempts * settings.max_tool_calls,
        "initial_training_runs": 0,
        # Any model-proposed repair reaches the same gate, including healthy false positives.
        "repair_training_epoch_upper_bound": count * settings.max_repair_epochs,
    }


def _validate_options(
    expected_model: str, max_attempts: int, retry_delay: float, case_delay: float
) -> None:
    _validate_limits(max_attempts, retry_delay)
    if not isinstance(expected_model, str) or not expected_model.strip():
        raise ValueError("An explicit model name is required")
    if (
        isinstance(case_delay, bool)
        or not isinstance(case_delay, (int, float))
        or not math.isfinite(case_delay)
        or not 0 <= case_delay <= 60
    ):
        raise ValueError("case_delay must be a finite number between 0 and 60 seconds")


def _attempt_usage(attempts: list[dict[str, object]], root: Path) -> dict[str, object]:
    totals = dict.fromkeys(("model_calls", "tool_calls", "input_tokens", "output_tokens"), 0)
    complete = True
    unknown = False
    report_count = 0
    for event in attempts:
        path = event["report_path"]
        if path is None:
            unknown = True
            complete = False
            continue
        report_count += 1
        try:
            usage = _usage_metadata(read_report(path, root))
        except (OSError, ValueError, TypeError):
            complete = False
            continue
        for field in totals:
            totals[field] += usage[field] or 0
        complete = complete and usage["usage_complete"]
        if event["status"] in {"running", "exception"}:
            complete = False
    return {
        **totals,
        "usage_complete": complete,
        "attempt_report_count": report_count,
        "unrecorded_attempt_exists": unknown,
        "scope": "Every new Agent attempt in this model evaluation; original batch excluded",
    }


def run_reevaluation(
    batch_summary_path: str,
    workspace_root: Path,
    *,
    expected_model: str,
    max_attempts: int = 1,
    retry_delay: float = 10.0,
    case_delay: float = 30.0,
    output_dir: str = "artifacts/model_evaluations",
) -> dict[str, object]:
    """Run all saved cases with the selected model and independently verify repairs."""
    _validate_options(expected_model, max_attempts, retry_delay, case_delay)
    root = workspace_root.resolve()
    if root != Path.cwd().resolve():
        raise ValueError("Run reevaluation from the workspace root")
    plan = prepare_reevaluation(batch_summary_path, root)
    settings = plan["settings"]
    suite = plan["suite"].model_copy(deep=True)
    suite.suite = "saved_runs_model_evaluation"
    suite.notes = [
        "Every original case is diagnosed again; no original Agent or repair reports are reused.",
        "Training data splits and telemetry are reused from the recorded runs.",
        "The inspected source is the current source, not a historical source snapshot.",
        "Healthy cases are self-comparisons, not independent healthy training runs.",
        f"Up to {max_attempts} Agent attempts per case; only transient API failures are retried.",
        "A completed diagnosis or rejected repair is never retried to improve its outcome.",
    ]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    directory = _workspace_path(output_dir, root) / timestamp
    directory.mkdir(parents=True)
    relative_directory = directory.relative_to(root).as_posix()
    for case in suite.cases:
        case.agent_report = f"{relative_directory}/agent_reports/{case.id}-attempt-1.json"
        case.repair_report = None
    first_suite = suite.model_copy(deep=True)
    first_suite.suite = "saved_runs_model_first_attempt"
    first_suite.notes.append("Only attempt 1 contributes to this evaluation.")
    attempts = []
    progress = {
        "original_batch_summary": batch_summary_path,
        "original_model": plan["original_model"],
        "requested_model": expected_model,
        "actual_model": None,
        "startup_error": None,
        "attempts": attempts,
        "cases": [
            {"id": case.id, "status": "planned", "repair_status": "not_requested", "errors": []}
            for case in suite.cases
        ],
    }

    def persist() -> None:
        _save_json(directory / "manifest.json", suite.model_dump(mode="json"))
        _save_json(directory / "first_attempt_manifest.json", first_suite.model_dump(mode="json"))
        _save_json(directory / "progress.json", progress)

    budget = reevaluation_budget(plan, max_attempts)
    _save_json(
        directory / "settings.json",
        {
            "original_batch_summary": batch_summary_path,
            "model": expected_model,
            "max_attempts": max_attempts,
            "retry_delay": retry_delay,
            "case_delay": case_delay,
            "max_model_calls": settings.max_model_calls,
            "max_tool_calls": settings.max_tool_calls,
            "max_output_tokens": settings.max_output_tokens,
            "max_repair_epochs": settings.max_repair_epochs,
            "budget": budget,
        },
    )
    persist()
    print(f"model_evaluation_directory={relative_directory}", flush=True)
    try:
        client, model = create_llm_client()
        progress["actual_model"] = model
        with client:
            if model != expected_model:
                raise ValueError("Current client model does not match the requested --model")
            tools = DiagnosticTools(root)
            for index, (case, record) in enumerate(
                zip(suite.cases, progress["cases"], strict=True)
            ):
                if index and case_delay:
                    time.sleep(case_delay)
                for number in range(1, max_attempts + 1):
                    if number > 1 and retry_delay:
                        time.sleep(min(30.0, retry_delay * 2 ** (number - 2)))
                    event = {
                        "case_id": case.id,
                        "attempt": number,
                        "status": "running",
                        "report_path": None,
                        "error": None,
                    }
                    attempts.append(event)
                    record["status"] = "running"
                    persist()
                    try:
                        agent_report = run_diagnostic_agent(
                            client,
                            model,
                            tools,
                            source_path=suite.source_path,
                            reference_run=case.reference_run,
                            candidate_run=case.candidate_run,
                            max_model_calls=settings.max_model_calls,
                            max_tool_calls=settings.max_tool_calls,
                            max_output_tokens=settings.max_output_tokens,
                        )
                        report_path = (
                            directory / "agent_reports" / f"{case.id}-attempt-{number}.json"
                        )
                        _save_text(report_path, agent_report.to_json())
                        event["report_path"] = report_path.relative_to(root).as_posix()
                        payload = read_report(event["report_path"], root)
                        if payload.get("model") != expected_model:
                            raise ValueError(
                                "Agent report model does not match the requested model"
                            )
                        if not isinstance(payload.get("status"), str) or not payload["status"]:
                            raise ValueError("Agent report is missing status")
                        case.agent_report = event["report_path"]
                        event["status"] = payload["status"]
                        event["error"] = payload.get("error")
                        record["status"] = payload["status"]
                        persist()
                    except Exception as error:
                        event["status"] = "exception"
                        event["error"] = _error_record(error)
                        record["status"] = "exception"
                        record["errors"].append(event["error"])
                        # A failed attempt must never fall back to an earlier completed answer.
                        case.agent_report = (
                            f"{relative_directory}/invalid_reports/{case.id}-attempt-{number}.json"
                        )
                        if number == 1:
                            first_suite.cases[index].agent_report = case.agent_report
                        persist()
                        break
                    if payload["status"] == "completed":
                        diagnosis = payload.get("diagnosis")
                        if (
                            isinstance(diagnosis, dict)
                            and diagnosis.get("status") == "failure_detected"
                            and diagnosis.get("proposed_patch") is not None
                        ):
                            record["repair_status"] = "running"
                            persist()
                            try:
                                repaired = run_bounded_agent_repair(
                                    Path(event["report_path"]),
                                    source_path=Path(suite.source_path),
                                    reference_run=Path(case.reference_run),
                                    failed_run=Path(case.candidate_run),
                                    max_epochs=settings.max_repair_epochs,
                                )
                                # Existing repair code inherits output_dir from the saved config.
                                run = _relative_run(
                                    Path(repaired.repaired_run), root, plan["batch_directory"]
                                )
                                repair_path = _workspace_path(
                                    f"{run}/agent_repair_report.json", root
                                )
                                _save_text(repair_path, repaired.to_json())
                                case.repair_report = repair_path.relative_to(root).as_posix()
                                if number == 1:
                                    first_suite.cases[index].repair_report = case.repair_report
                                record["repair_status"] = "completed"
                            except Exception as error:
                                record["repair_status"] = "error"
                                record["errors"].append(_error_record(error))
                            persist()
                        break
                    if not _retryable(payload):
                        break
                print(
                    f"case={case.id}, diagnosis={record['status']}, repair={record['repair_status']}",
                    flush=True,
                )
    except Exception as error:
        progress["startup_error"] = _error_record(error)
        for record in progress["cases"]:
            if record["status"] in {"planned", "running"}:
                record["status"] = "blocked"
        persist()

    persist()
    first_evaluation = evaluate_suite(first_suite, root)
    final_evaluation = evaluate_suite(suite, root)
    _save_json(directory / "first_attempt_evaluation.json", first_evaluation)
    _save_json(directory / "evaluation.json", final_evaluation)
    result = {
        "directory": relative_directory,
        "original_batch_summary": batch_summary_path,
        "original_model": plan["original_model"],
        "model": expected_model,
        "budget": budget,
        "first_attempt_evaluation": first_evaluation,
        "final_evaluation": final_evaluation,
        "all_attempt_usage": _attempt_usage(attempts, root),
        "progress": progress,
    }
    _save_json(directory / "model_evaluation_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a new model on all saved batch cases.")
    parser.add_argument("--batch-summary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--retry-delay", type=float, default=10.0)
    parser.add_argument("--case-delay", type=float, default=30.0)
    parser.add_argument("--output-dir", default="artifacts/model_evaluations")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        _validate_options(args.model, args.max_attempts, args.retry_delay, args.case_delay)
        plan = prepare_reevaluation(args.batch_summary, Path.cwd())
        _workspace_path(args.output_dir, Path.cwd())
    except (OSError, ValueError, TypeError) as error:
        parser.error(_error_record(error)["message"])
    print(
        json.dumps(
            {
                "original_model": plan["original_model"],
                "requested_model": args.model,
                "budget": reevaluation_budget(plan, args.max_attempts),
            },
            indent=2,
        )
    )
    if args.dry_run:
        return
    result = run_reevaluation(
        args.batch_summary,
        Path.cwd(),
        expected_model=args.model,
        max_attempts=args.max_attempts,
        retry_delay=args.retry_delay,
        case_delay=args.case_delay,
        output_dir=args.output_dir,
    )
    print(f"model_evaluation_summary={result['directory']}/model_evaluation_summary.json")
    if result["progress"]["startup_error"]:
        print(json.dumps(result["progress"]["startup_error"]))
    print("first_attempt_summary=" + json.dumps(result["first_attempt_evaluation"]["summary"]))
    print("final_summary=" + json.dumps(result["final_evaluation"]["summary"]))
    print("all_attempt_usage=" + json.dumps(result["all_attempt_usage"]))
    summary = result["final_evaluation"]["summary"]
    if summary["passed_cases"] != summary["total_cases"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
