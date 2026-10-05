"""Run a bounded, fixed-model Agent over one factory or a saved development suite."""

import argparse
import hashlib
import json
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from runsleuth.evaluate_optimizer import _read, _scoped
from runsleuth.optimizer_agent_diagnosis import _same_value
from runsleuth.optimizer_factory_analysis import MAX_SOURCE_BYTES, analyze_optimizer_factory
from runsleuth.optimizer_factory_cases import SUITE_VERSION, factory_cases
from runsleuth.optimizer_factory_diagnosis import (
    FINDINGS,
    FactoryDiagnosis,
    factory_evidence,
    validate_factory_diagnosis,
)

LIMITS = {"max_model_calls": 3, "max_tool_calls": 2, "max_output_tokens": 2000}
USAGE_FIELDS = ("model_calls", "tool_calls", "input_tokens", "output_tokens")


def _source_bytes(path: Path) -> bytes:
    with path.open("rb") as stream:
        payload = stream.read(MAX_SOURCE_BYTES + 1)
    factory_evidence(payload, source_path="factory.py")
    return payload


def prepare_cases(root: Path, *, source: Path | None, suite_report: Path | None) -> tuple:
    """Validate every selected input before creating a client or making API calls."""
    if (source is None) == (suite_report is None):
        raise ValueError("Select exactly one of --source and --suite-report")
    if source is not None:
        path = _scoped(root, source, file=True)
        if path.suffix.lower() != ".py":
            raise ValueError("Factory source must be a Python file")
        payload = _source_bytes(path)
        return [
            {
                "id": "source-001",
                "original_file": path.relative_to(root).as_posix(),
                "payload": payload,
                "expected_decision": None,
            }
        ], None
    path = _scoped(root, suite_report, file=True)
    report, digest = _read(path)
    if (
        report.get("schema_version") != 1
        or report.get("task") != "optimizer_factory_development_suite"
        or report.get("suite_version") != SUITE_VERSION
        or report.get("status") not in ("completed", "static_completed")
    ):
        raise ValueError("A completed factory development suite report is required")
    fixtures = {case["id"]: case for case in factory_cases()}
    rows = report.get("cases")
    if not isinstance(rows, list) or len(rows) != len(fixtures):
        raise ValueError("The complete twelve-case development suite is required")
    cases, seen = [], set()
    for case in rows:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str):
            raise ValueError("Malformed factory suite case")
        case_id = case["id"]
        if case_id not in fixtures or case_id in seen:
            raise ValueError("Suite cases must have known, unique IDs")
        seen.add(case_id)
        fixture = fixtures[case_id]
        if (
            case.get("expected_decision") != fixture["expected_decision"]
            or case.get("source") != fixture["source"]
        ):
            raise ValueError("Development fixture or expected label changed")
        original = _scoped(root, case["original_file"], file=True)
        payload = _source_bytes(original)
        if payload != fixture["source"].encode("utf-8") or hashlib.sha256(
            payload
        ).hexdigest() != case.get("source_sha256"):
            raise ValueError("Source snapshot does not match the saved suite")
        if not _same_value(analyze_optimizer_factory(payload.decode("utf-8")), case["analysis"]):
            raise ValueError("Source analysis changed; regenerate the development suite first")
        cases.append(
            {
                "id": case_id,
                "original_file": original.relative_to(root).as_posix(),
                "payload": payload,
                "expected_decision": fixture["expected_decision"],
            }
        )
    return cases, {
        "path": path.relative_to(root).as_posix(),
        "sha256": digest,
        "scope": "Hand-authored development fixtures; not a held-out benchmark",
    }


@contextmanager
def _agent_session(root: Path, model: str):
    from runsleuth.agent import run_diagnostic_agent
    from runsleuth.diagnostic_tools import DiagnosticTools
    from runsleuth.llm_client import create_llm_client

    client, selected_model = create_llm_client()
    if selected_model != model:
        client.close()
        raise ValueError("GEMINI_MODEL must match the explicit --model argument")
    with client:
        tools = DiagnosticTools(root, enable_optimizer_factory=True)

        def run(source_path: str) -> dict:
            report = run_diagnostic_agent(
                client,
                model,
                tools,
                source_path=source_path,
                reference_run="",
                candidate_run="",
                profile="optimizer_factory",
                **LIMITS,
            )
            return json.loads(report.to_json())

        yield run


def _write(path: Path, data: dict) -> None:
    text = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _rate(n: int, d: int) -> dict:
    return {"numerator": n, "denominator": d, "rate": n / d if d else None}


def _summarize(report: dict) -> None:
    rows = report["cases"]
    scored = [row for row in rows if row["expected_decision"] is not None]
    valid = [row for row in rows if row["validated"]]
    report["summary"] = {
        "total_cases": len(rows),
        "validated_reports": _rate(len(valid), len(rows)),
        "first_pass_valid": _rate(sum(row["first_pass_valid"] is True for row in rows), len(rows)),
        "completed_after_correction": sum(
            row["validated"] and row["first_pass_valid"] is False for row in rows
        ),
        "structured_decisions_correct": _rate(
            sum(row["decision_correct"] is True for row in scored), len(scored)
        ),
        "unsupported_manual_review": _rate(
            sum(
                row["decision_correct"] is True
                for row in scored
                if row["expected_decision"] == "unsupported"
            ),
            sum(row["expected_decision"] == "unsupported" for row in scored),
        ),
        "error_cases": sum(
            row["status"] not in ("completed", "not_run", "running") for row in rows
        ),
        "not_run": sum(row["status"] == "not_run" for row in rows),
    }


def _record_usage(total: dict, attempt: dict) -> None:
    total["attempt_report_count"] += 1
    for field in USAGE_FIELDS:
        value = attempt.get(field)
        if type(value) is not int or value < 0:
            total["usage_complete"] = False
        else:
            total[field] += value
    total["usage_complete"] &= attempt.get("usage_complete") is True


def _validate_attempt(attempt: dict, *, model: str, source_path: str, evidence: dict) -> None:
    if (
        attempt.get("model") != model
        or not _same_value(attempt.get("limits"), LIMITS)
        or attempt.get("profile") != "optimizer_factory"
    ):
        raise ValueError("Agent report changed its model, profile or call budget")
    for field in ("model_calls", "tool_calls"):
        value = attempt.get(field)
        if type(value) is not int or not 0 <= value <= LIMITS[f"max_{field}"]:
            raise ValueError("Invalid recorded Agent call accounting")
    if attempt.get("status") != "completed":
        return
    diagnosis = FactoryDiagnosis.model_validate(attempt.get("diagnosis"))
    trace = attempt.get("tool_trace")
    if not isinstance(trace, list) or len(trace) != attempt["tool_calls"]:
        raise ValueError("Tool trace does not match the recorded call count")
    validate_factory_diagnosis(diagnosis, trace, source_path=source_path)
    if attempt.get("pending_evidence") != []:
        raise ValueError("Completed report still has pending evidence")
    for entry in trace:
        if entry["result"].get("ok") is True and not _same_value(
            entry["result"].get("data"), evidence
        ):
            raise ValueError("Agent evidence differs from the selected source snapshot")
    attempts = attempt.get("validation_attempts")
    retries = attempt.get("validation_retries")
    if (
        type(retries) is not int
        or retries not in (0, 1)
        or not isinstance(attempts, list)
        or len(attempts) != retries + 1
        or attempt.get("first_pass_valid") is not (retries == 0)
        or attempts[-1].get("valid") is not True
        or any(row.get("valid") is not False for row in attempts[:-1])
    ):
        raise ValueError("Invalid first-pass or correction accounting")
    if attempt.get("max_validation_retries") != 1:
        raise ValueError("Factory Agent must keep its one-correction limit")
    for row in attempts:
        try:
            raw = FactoryDiagnosis.model_validate_json(row.get("raw_final_text"))
            validate_factory_diagnosis(raw, trace, source_path=source_path)
        except ValueError:
            raw_valid = False
        else:
            raw_valid = True
        if row["valid"] is not raw_valid:
            raise ValueError("Recorded validation outcome does not match the saved model JSON")
    final = FactoryDiagnosis.model_validate_json(attempt.get("final_text"))
    raw_final = FactoryDiagnosis.model_validate_json(attempt.get("raw_final_text"))
    if not _same_value(final.model_dump(), diagnosis.model_dump()) or not _same_value(
        raw_final.model_dump(), diagnosis.model_dump()
    ):
        raise ValueError("Final model JSON differs from the stored structured diagnosis")


def run_factory_agent(
    *,
    workspace_root: Path,
    model: str,
    source: Path | None = None,
    suite_report: Path | None = None,
    case_delay: int = 10,
) -> Path:
    """One new Agent attempt per source; one bounded JSON correction, no case retries."""
    if not isinstance(model, str) or not model.strip() or model != model.strip():
        raise ValueError("Provide a nonempty model without outer whitespace")
    if type(case_delay) is not int or not 0 <= case_delay <= 60:
        raise ValueError("case_delay must be an integer from 0 to 60")
    root = workspace_root.resolve()
    selected, suite = prepare_cases(root, source=source, suite_report=suite_report)
    directory = _scoped(root, "artifacts/optimizer_factory_agents") / datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "report.json"
    report = {
        "schema_version": 1,
        "task": "optimizer_factory_agent",
        "status": "prepared",
        "method": "controller_assisted_source_interpretation",
        "model": model,
        "limits": dict(LIMITS),
        "suite": suite,
        "error": None,
        "cases": [],
        "scope": "Measures structured findings and citation contracts; does not certify every prose claim or LLM superiority",
        "case_retries": 0,
        "validation_corrections_per_attempt": 1,
        "case_delay_seconds": case_delay,
        "patches_applied": 0,
        "training_epochs": 0,
        "performance_evaluated": False,
        "budget": {
            "agent_attempts": len(selected),
            "model_call_upper_bound": 3 * len(selected),
            "tool_call_upper_bound": 2 * len(selected),
            "training_runs": 0,
        },
        "all_attempt_usage": {
            **dict.fromkeys(USAGE_FIELDS, 0),
            "attempt_report_count": 0,
            "usage_complete": True,
            "unrecorded_attempt_exists": False,
            "scope": "Every attempt in this new panel; previous sessions excluded",
        },
    }
    snapshot = directory / "source_snapshot"
    snapshot.mkdir()
    report["source_files_sha256"] = {}
    for name in (
        "agent.py",
        "diagnostic_tools.py",
        "llm_client.py",
        "optimizer_factory_analysis.py",
        "optimizer_factory_cases.py",
        "optimizer_factory_diagnosis.py",
        "optimizer_factory_agent.py",
        "optimizer_source_patch.py",
        "optimizer_agent_diagnosis.py",
    ):
        payload = (Path(__file__).parent / name).read_bytes()
        (snapshot / name).write_bytes(payload)
        report["source_files_sha256"][name] = hashlib.sha256(payload).hexdigest()
    for index, case in enumerate(selected, 1):
        folder = directory / f"case-{index:03d}"
        folder.mkdir()
        source_path = folder / "factory.py"
        source_path.write_bytes(case["payload"])
        report["cases"].append(
            {
                "id": case["id"],
                "original_file": case["original_file"],
                "source_path": source_path.relative_to(root).as_posix(),
                "source_sha256": hashlib.sha256(case["payload"]).hexdigest(),
                "expected_decision": case["expected_decision"],
                "status": "not_run",
                "agent_report": None,
                "validated": False,
                "first_pass_valid": None,
                "decision_correct": None,
                "error": None,
            }
        )
    _summarize(report)
    _write(path, report)
    print(f"factory_agent_directory={directory}", flush=True)
    print(json.dumps({"model": model, "limits": LIMITS, "budget": report["budget"]}), flush=True)
    active, in_flight = None, False
    try:
        with _agent_session(root, model) as run:
            report["status"] = "running"
            for index, row in enumerate(report["cases"]):
                active = row
                if index:
                    time.sleep(case_delay)
                payload = _source_bytes(_scoped(root, row["source_path"], file=True))
                if hashlib.sha256(payload).hexdigest() != row["source_sha256"]:
                    raise ValueError("Source changed after the panel was prepared")
                evidence = factory_evidence(payload, source_path=row["source_path"])
                row["status"] = "running"
                _write(path, report)
                in_flight = True
                attempt = run(row["source_path"])
                attempt_path = directory / f"case-{index + 1:03d}" / "agent_report.json"
                _write(attempt_path, attempt)
                row["agent_report"] = attempt_path.relative_to(root).as_posix()
                in_flight = False
                _record_usage(report["all_attempt_usage"], attempt)
                if _source_bytes(root / row["source_path"]) != payload:
                    raise ValueError("Source snapshot changed during Agent execution")
                _validate_attempt(
                    attempt, model=model, source_path=row["source_path"], evidence=evidence
                )
                row["status"] = attempt["status"]
                row["first_pass_valid"] = attempt.get("first_pass_valid")
                row["validated"] = attempt["status"] == "completed"
                if row["expected_decision"] is not None:
                    row["decision_correct"] = (
                        row["validated"]
                        and attempt["diagnosis"]["status"] == FINDINGS[row["expected_decision"]][0]
                    )
                row["error"] = attempt.get("error")
                _summarize(report)
                _write(path, report)
                print(
                    f"case={row['id']}, status={row['status']}, first_pass_valid={row['first_pass_valid']}",
                    flush=True,
                )
        report["status"] = (
            "completed"
            if all(row["validated"] for row in report["cases"])
            else "completed_with_errors"
        )
    except (Exception, KeyboardInterrupt) as error:
        # Exception messages may contain provider credentials; retain only the type.
        report.update(status="execution_error", error={"type": type(error).__name__})
        if active is not None:
            active.update(status="runner_error", error={"type": type(error).__name__})
        if in_flight:
            report["all_attempt_usage"].update(usage_complete=False, unrecorded_attempt_exists=True)
        if active is None and isinstance(error, (ValueError, RuntimeError)):
            report["error"]["message"] = str(error)
            print(f"setup_error={error}", flush=True)
        print(f"runner_error_type={type(error).__name__}", flush=True)
    _summarize(report)
    _write(path, report)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--source", type=Path)
    inputs.add_argument("--suite-report", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--case-delay", type=int, default=10)
    args = parser.parse_args()
    try:
        path = run_factory_agent(
            workspace_root=Path.cwd(),
            model=args.model,
            source=args.source,
            suite_report=args.suite_report,
            case_delay=args.case_delay,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    report, _ = _read(path)
    print(f"factory_agent_report={path}")
    print(f"status={report['status']}")
    print(json.dumps(report["summary"], indent=2))
    print(json.dumps(report["all_attempt_usage"], indent=2))
    if report["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
