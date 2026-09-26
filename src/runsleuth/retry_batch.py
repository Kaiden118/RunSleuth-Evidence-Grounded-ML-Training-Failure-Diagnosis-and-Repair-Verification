"""Recover transient API failures while preserving the original batch and every attempt."""

import argparse
import json
import math
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
from runsleuth.diagnostic_tools import DiagnosticTools
from runsleuth.evaluate import _usage_metadata, _workspace_path, evaluate_suite, read_report
from runsleuth.evaluation import load_smoke_suite
from runsleuth.llm_client import create_llm_client

TRANSIENT_ERROR_TYPES = frozenset({"InternalServerError", "APIConnectionError", "APITimeoutError"})


def _retryable(report: dict[str, object]) -> bool:
    error = report.get("error")
    return (
        report.get("status") == "api_error"
        and isinstance(error, str)
        and error.partition(":")[0] in TRANSIENT_ERROR_TYPES
    )


def _validate_limits(max_attempts: int, retry_delay: float) -> None:
    if type(max_attempts) is not int or not 1 <= max_attempts <= 3:
        raise ValueError("max_attempts must be an integer between 1 and 3")
    if (
        isinstance(retry_delay, bool)
        or not isinstance(retry_delay, (int, float))
        or not math.isfinite(retry_delay)
        or not 0 <= retry_delay <= 30
    ):
        raise ValueError("retry_delay must be a finite number between 0 and 30 seconds")


def prepare_recovery(
    batch_summary_path: str,
    workspace_root: Path,
    case_ids: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Validate a saved batch and select API failures without making network calls."""
    root = workspace_root.resolve()
    original = read_report(batch_summary_path, root)
    settings_data = original.get("settings")
    if not isinstance(settings_data, dict):
        raise ValueError("Batch summary is missing settings")
    try:
        settings = BatchSettings(**settings_data)
    except TypeError as error:
        raise ValueError("Batch settings do not match the expected schema") from error
    manifest_path = _workspace_path(original.get("manifest_path"), root)
    suite = load_smoke_suite(manifest_path)
    batch_directory = _workspace_path(original.get("batch_directory"), root)
    if not manifest_path.is_relative_to(batch_directory):
        raise ValueError("Original manifest is outside its batch directory")
    if suite.source_path != settings.source_path:
        raise ValueError("Source path differs between manifest and batch settings")
    if suite.max_repair_epochs != settings.max_repair_epochs:
        raise ValueError("Repair budget differs between manifest and batch settings")
    if not _workspace_path(settings.source_path, root).is_file():
        raise ValueError("Training source is missing")
    progress = original.get("progress")
    evaluation = original.get("evaluation")
    if not isinstance(progress, dict) or not isinstance(evaluation, dict):
        raise ValueError("Batch summary is missing progress or evaluation")
    model = progress.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("Original batch must identify the model used")
    ids = {case.id for case in suite.cases}
    records = progress.get("cases")
    evaluated_cases = evaluation.get("cases")
    for rows in (records, evaluated_cases):
        if (
            not isinstance(rows, list)
            or not all(isinstance(row, dict) for row in rows)
            or len(rows) != len(ids)
            or {row.get("id") for row in rows} != ids
        ):
            raise ValueError("Manifest, progress, and evaluation must contain identical case IDs")
    by_id = {row["id"]: row for row in records}
    eligible = []
    report_paths = []
    for case in suite.cases:
        report_paths.append(case.agent_report)
        if by_id[case.id].get("training_status") != "completed":
            continue
        try:
            report = read_report(case.agent_report, root)
        except (OSError, ValueError):
            continue
        if _retryable(report):
            if report.get("model") != model:
                raise ValueError(f"Original model differs for case {case.id}")
            for run in (case.reference_run, case.candidate_run):
                for name in ("config.json", "metrics.jsonl"):
                    if not _workspace_path(f"{run}/{name}", root).is_file():
                        raise ValueError(f"Case {case.id} is missing saved training artifacts")
            eligible.append(case.id)
    if case_ids is not None:
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("Case IDs must be unique")
        if not set(case_ids).issubset(ids):
            raise ValueError("Unknown case ID in recovery request")
        if not set(case_ids).issubset(eligible):
            raise ValueError("Selected cases must have completed training and transient API errors")
        eligible = [case.id for case in suite.cases if case.id in case_ids]
    if not eligible:
        raise ValueError("No eligible transient API failures were found")
    return {
        "original": original,
        "suite": suite,
        "settings": settings,
        "selected_ids": eligible,
        "model": model,
        "batch_directory": batch_directory,
        "original_report_paths": report_paths,
    }


def recovery_budget(plan: dict[str, object], max_attempts: int) -> dict[str, int]:
    settings = plan["settings"]
    selected = len(plan["selected_ids"])
    attempts = selected * max_attempts
    return {
        "selected_cases": selected,
        "additional_agent_attempt_upper_bound": attempts,
        "additional_model_call_upper_bound": attempts * settings.max_model_calls,
        "additional_agent_tool_call_upper_bound": attempts * settings.max_tool_calls,
        "initial_training_runs": 0,
        "repair_training_epoch_upper_bound": selected
        * min(settings.epochs, settings.max_repair_epochs),
    }


def _cumulative_usage(paths: list[str], root: Path, *, unknown_attempt: bool) -> dict[str, object]:
    fields = ("model_calls", "tool_calls", "input_tokens", "output_tokens")
    totals = dict.fromkeys(fields, 0)
    complete = not unknown_attempt
    unique_paths = list(dict.fromkeys(paths))
    for path in unique_paths:
        try:
            usage = _usage_metadata(read_report(path, root))
        except (OSError, ValueError):
            complete = False
            continue
        for field in fields:
            totals[field] += usage[field] or 0
        complete = complete and usage["usage_complete"]
    return {
        **totals,
        "usage_complete": complete,
        "attempt_report_count": len(unique_paths),
        "unrecorded_attempt_exists": unknown_attempt,
        "scope": "Original batch plus every attempt in this recovery; other sessions excluded",
    }


def run_recovery(
    batch_summary_path: str,
    workspace_root: Path,
    *,
    case_ids: tuple[str, ...] | None = None,
    max_attempts: int = 2,
    retry_delay: float = 10.0,
    output_dir: str = "artifacts/recoveries",
) -> dict[str, object]:
    """Retry transient errors, never completed judgments or rejected repairs."""
    _validate_limits(max_attempts, retry_delay)
    root = workspace_root.resolve()
    if root != Path.cwd().resolve():
        raise ValueError("Run recovery from the workspace root")
    plan = prepare_recovery(batch_summary_path, root, case_ids)
    settings = plan["settings"]
    suite = plan["suite"].model_copy(deep=True)
    suite.suite = "multi_seed_after_bounded_recovery"
    suite.notes = [note for note in suite.notes if "one Agent invocation" not in note] + [
        f"Selected transient API failures received up to {max_attempts} additional attempts.",
        "Any completed diagnosis stops retries, whether correct or incorrect.",
        "All original cases remain in the denominator; original results are preserved separately.",
        "Selected-result usage excludes superseded attempts; cumulative_usage includes them.",
    ]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    directory = _workspace_path(output_dir, root) / timestamp
    directory.mkdir(parents=True)
    relative_directory = directory.relative_to(root).as_posix()
    attempts = []
    progress = {
        "original_batch_summary": batch_summary_path,
        "expected_model": plan["model"],
        "actual_model": None,
        "startup_error": None,
        "attempts": attempts,
        "cases": [
            {"id": case_id, "status": "planned", "repair_status": "not_requested", "errors": []}
            for case_id in plan["selected_ids"]
        ],
    }
    manifest_path = directory / "manifest.json"

    def persist() -> None:
        _save_json(manifest_path, suite.model_dump(mode="json"))
        _save_json(directory / "progress.json", progress)

    _save_json(
        directory / "settings.json",
        {
            "original_batch_summary": batch_summary_path,
            "selected_ids": plan["selected_ids"],
            "max_attempts": max_attempts,
            "retry_delay": retry_delay,
            "budget": recovery_budget(plan, max_attempts),
        },
    )
    persist()
    print(f"recovery_directory={relative_directory}", flush=True)
    try:
        client, model = create_llm_client()
        progress["actual_model"] = model
        with client:
            if model != plan["model"]:
                raise ValueError("Recovery must use the same model as the original batch")
            tools = DiagnosticTools(root)
            by_id = {case.id: case for case in suite.cases}
            for record in progress["cases"]:
                case = by_id[record["id"]]
                record["status"] = "running"
                for number in range(1, max_attempts + 1):
                    if number > 1:
                        time.sleep(min(30.0, retry_delay * 2 ** (number - 2)))
                    event = {
                        "case_id": case.id,
                        "attempt": number,
                        "status": "running",
                        "report_path": None,
                        "error": None,
                    }
                    attempts.append(event)
                    persist()
                    print(f"case={case.id}, recovery_attempt={number}", flush=True)
                    try:
                        report = run_diagnostic_agent(
                            client,
                            model,
                            tools,
                            source_path=settings.source_path,
                            reference_run=case.reference_run,
                            candidate_run=case.candidate_run,
                            max_model_calls=settings.max_model_calls,
                            max_tool_calls=settings.max_tool_calls,
                            max_output_tokens=settings.max_output_tokens,
                        )
                        path = f"{relative_directory}/agent_reports/{case.id}-attempt-{number}.json"
                        text = report.to_json()
                        _save_text(root / path, text)
                        event["report_path"] = path
                        payload = json.loads(text)
                        if not isinstance(payload, dict) or not isinstance(
                            payload.get("status"), str
                        ):
                            raise ValueError("Agent returned an invalid report")
                        event.update(status=payload["status"], error=payload.get("error"))
                        record["status"] = payload["status"]
                        case.agent_report = path
                        case.repair_report = None
                        persist()
                    except Exception as error:
                        event.update(status="exception", error=_error_record(error))
                        record["status"] = "exception"
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
                                repair = run_bounded_agent_repair(
                                    Path(path),
                                    source_path=Path(settings.source_path),
                                    reference_run=Path(case.reference_run),
                                    failed_run=Path(case.candidate_run),
                                    max_epochs=settings.max_repair_epochs,
                                )
                                run = _relative_run(
                                    Path(repair.repaired_run), root, plan["batch_directory"]
                                )
                                repair_path = f"{run}/agent_repair_report.json"
                                _save_text(root / repair_path, repair.to_json())
                                case.repair_report = repair_path
                                record["repair_status"] = "completed"
                            except Exception as error:
                                record["repair_status"] = "error"
                                record["errors"].append(_error_record(error))
                            persist()
                        break
                    if not _retryable(payload):
                        break
                persist()
    except Exception as error:
        progress["startup_error"] = _error_record(error)
        for record in progress["cases"]:
            if record["status"] in ("planned", "running"):
                record["status"] = "blocked"
        persist()
    recovered = evaluate_suite(suite, root)
    recovered["evaluation_type"] = "after_bounded_api_recovery"
    _save_json(directory / "evaluation.json", recovered)
    new_paths = [event["report_path"] for event in attempts if event["report_path"] is not None]
    cumulative = _cumulative_usage(
        [*plan["original_report_paths"], *new_paths],
        root,
        unknown_attempt=any(event["status"] in ("running", "exception") for event in attempts),
    )
    result = {
        "original_batch_summary": batch_summary_path,
        "recovery_directory": relative_directory,
        "manifest_path": manifest_path.relative_to(root).as_posix(),
        "budget": recovery_budget(plan, max_attempts),
        "original_summary": plan["original"]["evaluation"]["summary"],
        "recovered_evaluation": recovered,
        "attempts": attempts,
        "cumulative_usage": cumulative,
        "progress": progress,
    }
    _save_json(directory / "recovery_summary.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-summary", required=True)
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--retry-delay", type=float, default=10.0)
    parser.add_argument("--output-dir", default="artifacts/recoveries")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    root = Path.cwd()
    case_ids = tuple(args.case_ids) if args.case_ids is not None else None
    try:
        _validate_limits(args.max_attempts, args.retry_delay)
        plan = prepare_recovery(args.batch_summary, root, case_ids)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "selected_ids": plan["selected_ids"],
                "model": plan["model"],
                "budget": recovery_budget(plan, args.max_attempts),
            },
            indent=2,
        )
    )
    if args.dry_run:
        return
    result = run_recovery(
        args.batch_summary,
        root,
        case_ids=case_ids,
        max_attempts=args.max_attempts,
        retry_delay=args.retry_delay,
        output_dir=args.output_dir,
    )
    print(f"recovery_summary={result['recovery_directory']}/recovery_summary.json")
    if result["progress"]["startup_error"]:
        print(json.dumps(result["progress"]["startup_error"], ensure_ascii=False))
    for record in result["progress"]["cases"]:
        print(
            f"case={record['id']}, diagnosis={record['status']}, repair={record['repair_status']}"
        )
    print("first_attempt_summary=" + json.dumps(result["original_summary"]))
    print("after_recovery_summary=" + json.dumps(result["recovered_evaluation"]["summary"]))
    print("cumulative_usage=" + json.dumps(result["cumulative_usage"]))
    summary = result["recovered_evaluation"]["summary"]
    if summary["passed_cases"] != summary["total_cases"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
