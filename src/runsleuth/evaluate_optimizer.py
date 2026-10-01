"""Compare deterministic audit rules with saved Agent attempts, without API calls."""

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from runsleuth.optimizer_agent_diagnosis import (
    OptimizerDiagnosis,
    _same_value,
    validate_optimizer_diagnosis_evidence,
)
from runsleuth.optimizer_baseline import RULE_VERSION, diagnose_optimizer_rules
from runsleuth.optimizer_evidence import MAX_FILE_BYTES, _parse, inspect_optimizer_run

Text = Annotated[str, Field(strict=True, min_length=1, max_length=4096)]
STATUS = Literal["optimizer_binding_defect", "no_optimizer_binding_defect", "inconclusive"]
PERFORMANCE = Literal["no_observed_degradation", "mixed", "degraded", "inconclusive"]
ACTION = Literal["review_optimizer_binding", "no_change", "collect_more_evidence"]
SCORE_FIELDS = ("implementation_status", "root_cause", "performance_status", "recommended_action")
USAGE_FIELDS = ("model_calls", "tool_calls", "input_tokens", "output_tokens")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Expected(_Strict):
    implementation_status: STATUS
    root_cause: Literal["optimizer_parameter_mismatch"] | None
    performance_status: PERFORMANCE
    recommended_action: ACTION

    @model_validator(mode="after")
    def coherent(self) -> "Expected":
        defect = self.implementation_status == "optimizer_binding_defect"
        if defect != (self.root_cause == "optimizer_parameter_mismatch"):
            raise ValueError("Expected root cause must agree with the implementation label")
        if defect != (self.recommended_action == "review_optimizer_binding"):
            raise ValueError("Expected review action must agree with the implementation label")
        if self.recommended_action == "no_change" and (
            self.implementation_status != "no_optimizer_binding_defect"
            or self.performance_status != "no_observed_degradation"
        ):
            raise ValueError("Expected no_change requires clean implementation and performance")
        return self


class Case(_Strict):
    id: Text
    seed: Annotated[StrictInt, Field(ge=0)]
    reference_run: Text
    candidate_run: Text
    expected: Expected
    agent_reports: Annotated[list[Text], Field(min_length=1, max_length=20)]


class Manifest(_Strict):
    schema_version: Annotated[StrictInt, Field(ge=1, le=1)]
    name: Text
    cases: Annotated[list[Case], Field(min_length=1, max_length=100)]


def _scoped(root: Path, name: str | Path, *, file: bool = False) -> Path:
    path = Path(name)
    if PureWindowsPath(str(name)).drive and not path.is_absolute():
        raise ValueError("Foreign or drive-relative Windows path")
    resolved = (path if path.is_absolute() else root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Evaluation inputs must stay inside the workspace")
    if file and not resolved.is_file():
        raise ValueError(f"Required evaluation file is missing: {name}")
    return resolved


def _read(path: Path) -> tuple[dict, str]:
    with path.open("rb") as stream:
        payload = stream.read(MAX_FILE_BYTES + 1)
    if len(payload) > MAX_FILE_BYTES:
        raise ValueError("Evaluation JSON exceeds the size limit")
    value = _parse(payload.decode("utf-8-sig"))
    json.dumps(value, allow_nan=False)
    return value, hashlib.sha256(payload).hexdigest()


def _integer(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _labels(diagnosis: OptimizerDiagnosis) -> dict:
    return {
        "implementation_status": diagnosis.implementation_status,
        "root_cause": diagnosis.ranked_causes[0].root_cause if diagnosis.ranked_causes else None,
        "performance_status": diagnosis.performance.status,
        "recommended_action": diagnosis.recommended_action,
    }


def _check_trace(trace: object, runs: dict[str, dict]) -> set[str]:
    """Bind successful saved tool results, including hashes, to current run artifacts."""
    if not isinstance(trace, list):
        raise ValueError("Agent tool trace must be a list")
    seen, inspected = set(), set()
    for entry in trace:
        if not isinstance(entry, dict):
            raise ValueError("Malformed Agent tool trace")
        call_id, result = entry.get("call_id"), entry.get("result")
        if not isinstance(call_id, str) or not call_id or call_id in seen:
            raise ValueError("Tool call IDs must be present and unique")
        seen.add(call_id)
        if not isinstance(result, dict) or type(result.get("ok")) is not bool:
            raise ValueError("Malformed Agent tool result")
        if result.get("tool_name") != entry.get("tool_name"):
            raise ValueError("Tool result name does not match its request")
        if not result["ok"]:
            continue
        if entry.get("tool_name") != "inspect_optimizer_run":
            raise ValueError("Unexpected successful tool in optimizer profile")
        raw = entry.get("raw_arguments")
        if not isinstance(raw, str):
            raise ValueError("Tool arguments must be JSON text")
        arguments = _parse(raw)
        name = arguments.get("run_directory")
        if set(arguments) != {"run_directory"} or not isinstance(name, str) or name not in runs:
            raise ValueError("Saved tool call belongs to a different evaluation case")
        expected = {
            "tool_name": "inspect_optimizer_run",
            "ok": True,
            "data": runs[name],
            "error_code": None,
            "error_message": None,
        }
        if not _same_value(result, expected):
            raise ValueError("Saved tool evidence does not match current optimizer artifacts")
        inspected.add(name)
    return inspected


def _validate_completed(report: dict, case: Case) -> tuple[dict, bool]:
    if report.get("error") is not None or report.get("pending_evidence") != []:
        raise ValueError("Completed report still has an error or pending evidence")
    diagnosis = OptimizerDiagnosis.model_validate(report.get("diagnosis"))
    options = {"reference_run": case.reference_run, "candidate_run": case.candidate_run}
    validate_optimizer_diagnosis_evidence(diagnosis, report["tool_trace"], **options)
    attempts = report.get("validation_attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("Completed report requires validation-attempt provenance")
    retries = _integer(report.get("validation_retries"), "validation_retries")
    if len(attempts) != retries + 1 or retries > 1:
        raise ValueError("Validation attempts do not match the bounded correction count")
    validities, previous_call = [], 0
    for attempt in attempts:
        if not isinstance(attempt, dict) or type(attempt.get("valid")) is not bool:
            raise ValueError("Malformed diagnosis validation attempt")
        call = _integer(attempt.get("model_call"), "validation model_call")
        if not previous_call < call <= report["model_calls"]:
            raise ValueError("Validation model calls must be ordered and within the budget")
        previous_call = call
        raw = attempt.get("raw_final_text")
        if not isinstance(raw, str):
            raise ValueError("Raw validation response must be JSON text")
        try:
            parsed = OptimizerDiagnosis.model_validate(_parse(raw))
            validate_optimizer_diagnosis_evidence(parsed, report["tool_trace"], **options)
            valid = True
        except ValueError:
            valid = False
        if valid is not attempt["valid"]:
            raise ValueError("Recorded validation result differs from replayed validation")
        validities.append(valid)
    if not validities[-1] or any(validities[:-1]):
        raise ValueError("Completed diagnosis must follow the first successful validation")
    if report.get("first_pass_valid") is not validities[0]:
        raise ValueError("first_pass_valid differs from the first recorded diagnosis")
    final = OptimizerDiagnosis.model_validate(_parse(attempts[-1]["raw_final_text"]))
    if not _same_value(final.model_dump(), diagnosis.model_dump()):
        raise ValueError("Saved diagnosis differs from its final raw response")
    return _labels(diagnosis), validities[0]


def _attempt(path: Path, root: Path, case: Case, runs: dict[str, dict]) -> dict:
    result = {
        "report": path.relative_to(root).as_posix(),
        "status": "report_error",
        "labels": None,
        "error": None,
        "usage": None,
        "first_pass_valid": None,
        "task_binding": "manifest_only",
    }
    try:
        report, digest = _read(path)
        result["sha256"] = digest
        result["recorded_status"] = report.get("status")
        # Account for every readable attempt even when its diagnosis is rejected below.
        usage = {key: _integer(report.get(key), key) for key in USAGE_FIELDS}
        if type(report.get("usage_complete")) is not bool:
            raise ValueError("usage_complete must be a boolean")
        usage["usage_complete"] = report["usage_complete"]
        result["usage"] = usage
        if report.get("profile") != "optimizer_binding":
            raise ValueError("Expected an optimizer-binding Agent report")
        model = report.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Agent report must identify its requested model")
        result["model"] = model
        result["response_models"] = sorted(
            {
                request["model"]
                for request in report.get("requests", [])
                if isinstance(request, dict) and isinstance(request.get("model"), str)
            }
        )
        limits = report.get("limits")
        if not isinstance(limits, dict) or set(limits) != {
            "max_model_calls",
            "max_tool_calls",
            "max_output_tokens",
        }:
            raise ValueError("Agent report must record all call limits")
        if any(_integer(value, key) < 1 for key, value in limits.items()):
            raise ValueError("Agent limits must be positive")
        if usage["model_calls"] > limits["max_model_calls"] or (
            usage["tool_calls"] > limits["max_tool_calls"]
        ):
            raise ValueError("Recorded usage exceeds the declared Agent call limits")
        result["limits"] = limits
        inspected = _check_trace(report.get("tool_trace"), runs)
        result["task_binding"] = (
            "both_runs_refreshed"
            if inspected == set(runs)
            else "partial_evidence_refreshed"
            if inspected
            else "manifest_only"
        )
        if len(report["tool_trace"]) != usage["tool_calls"]:
            raise ValueError("Tool-call usage does not match the recorded trace")
        status = report.get("status")
        if not isinstance(status, str) or not status or status == "running":
            raise ValueError("Expected a finished Agent attempt")
        if status == "api_error" and usage["usage_complete"]:
            usage["usage_complete"] = False
            raise ValueError("An API-error report cannot claim complete recorded usage")
        result["status"] = status
        if status == "completed":
            labels, first_pass = _validate_completed(report, case)
            result.update(
                labels=labels,
                first_pass_valid=first_pass,
                validation_retries=report["validation_retries"],
            )
        else:
            result["error"] = report.get("error")
    except (OSError, ValueError, TypeError, KeyError) as error:
        result.update(status="report_error", labels=None, error=str(error))
    return result


def _rate(numerator: int, denominator: int) -> dict:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def _score(cases: list[dict], key: str) -> dict:
    def labels(row):
        if key == "rules":
            return row["rules"].get("labels")
        attempts = row["agent_attempts"]
        return attempts[0 if key == "agent_first_attempt" else -1]["labels"]

    def agrees(row, fields=SCORE_FIELDS):
        actual = labels(row)
        return actual is not None and all(
            actual[field] == row["expected"][field] for field in fields
        )

    defects = [
        row
        for row in cases
        if row["expected"]["implementation_status"] == "optimizer_binding_defect"
    ]
    controls = [
        row
        for row in cases
        if row["expected"]["implementation_status"] == "no_optimizer_binding_defect"
    ]
    return {
        "correct": _rate(sum(agrees(row) for row in cases), len(cases)),
        "implementation_correct": _rate(
            sum(agrees(row, ("implementation_status",)) for row in cases), len(cases)
        ),
        "performance_correct": _rate(
            sum(agrees(row, ("performance_status",)) for row in cases), len(cases)
        ),
        "fault_top1_correct": _rate(
            sum(agrees(row, ("implementation_status", "root_cause")) for row in defects),
            len(defects),
        ),
        "control_no_action": _rate(
            sum(
                agrees(row) and labels(row)["recommended_action"] == "no_change" for row in controls
            ),
            len(controls),
        ),
        "control_false_positives": _rate(
            sum(
                labels(row) is not None
                and labels(row)["implementation_status"] == "optimizer_binding_defect"
                for row in controls
            ),
            len(controls),
        ),
        "unscored_cases": sum(labels(row) is None for row in cases),
    }


def evaluate_optimizer_suite(manifest_path: Path, *, workspace_root: Path) -> dict:
    """Evaluate declared attempts in order; a completed attempt is never replaced by another."""
    root = workspace_root.resolve()
    path = _scoped(root, manifest_path, file=True)
    values, manifest_hash = _read(path)
    manifest = Manifest.model_validate(values)
    seen_ids, seen_pairs, seen_reports = set(), set(), set()
    for case in manifest.cases:
        pair = (_scoped(root, case.reference_run), _scoped(root, case.candidate_run))
        if case.id in seen_ids or pair in seen_pairs:
            raise ValueError("Case IDs and reference/candidate pairs must be unique")
        seen_ids.add(case.id)
        seen_pairs.add(pair)
        for name in case.agent_reports:
            report_path = _scoped(root, name)
            if report_path in seen_reports:
                raise ValueError("An Agent report cannot be counted more than once")
            seen_reports.add(report_path)
    rows, cache = [], {}
    for case in manifest.cases:
        row = {
            "id": case.id,
            "seed": case.seed,
            "reference_run": case.reference_run,
            "candidate_run": case.candidate_run,
            "expected": case.expected.model_dump(),
            "rules": {"labels": None, "error": None},
            "agent_attempts": [],
        }
        runs, evidence_error = {}, None
        try:
            for name in (case.reference_run, case.candidate_run):
                if name not in cache:
                    data = inspect_optimizer_run(
                        _scoped(root, name),
                        require_file=lambda item: _scoped(root, item, file=True),
                    )
                    data["run_directory"] = name
                    cache[name] = data
                runs[name] = cache[name]
                if runs[name]["config"]["seed"] != case.seed:
                    raise ValueError("Manifest seed does not match the recorded training seed")
            baseline = diagnose_optimizer_rules(runs[case.reference_run], runs[case.candidate_run])
            row["rules"] = {
                "labels": {key: baseline[key] for key in SCORE_FIELDS},
                "details": baseline,
                "error": None,
            }
            row["evidence_sha256"] = {
                name: data["provenance"]["files_sha256"] for name, data in runs.items()
            }
        except (OSError, ValueError, TypeError, KeyError) as error:
            evidence_error = str(error)
            row["rules"]["error"] = evidence_error
        for name in case.agent_reports:
            attempt = _attempt(_scoped(root, name), root, case, runs)
            if evidence_error:
                attempt.update(status="report_error", labels=None, error=evidence_error)
            row["agent_attempts"].append(attempt)
        if any(a.get("recorded_status") == "completed" for a in row["agent_attempts"][:-1]):
            raise ValueError("Do not replace a completed attempt with a later selected attempt")
        rows.append(row)
    attempts = [attempt for row in rows for attempt in row["agent_attempts"]]
    models = sorted({a["model"] for a in attempts if "model" in a})
    limits = {json.dumps(a["limits"], sort_keys=True) for a in attempts if "limits" in a}
    if len(models) > 1 or len(limits) > 1:
        raise ValueError("Compare attempts from one requested model with identical call limits")
    usage = {
        key: sum(a["usage"][key] for a in attempts if a["usage"] is not None)
        for key in USAGE_FIELDS
    }
    usage.update(
        attempt_report_count=len(attempts),
        unreadable_usage_reports=sum(a["usage"] is None for a in attempts),
        usage_complete=all(
            a["usage"] is not None and a["usage"]["usage_complete"] for a in attempts
        ),
        scope="Every report declared in this manifest; undeclared sessions excluded",
    )
    summary = {
        key: _score(rows, key) for key in ("rules", "agent_first_attempt", "agent_after_recovery")
    }
    completed = [a for a in attempts if a["status"] == "completed"]
    summary.update(
        total_cases=len(rows),
        attempt_status_counts=dict(Counter(a["status"] for a in attempts)),
        completed_reports_first_pass_valid=_rate(
            sum(a["first_pass_valid"] is True for a in completed), len(completed)
        ),
    )
    success = all(
        summary[key]["correct"]["numerator"] == len(rows)
        for key in ("rules", "agent_after_recovery")
    ) and not any(attempt["status"] == "report_error" for attempt in attempts)
    return {
        "schema_version": 1,
        "task": "offline_optimizer_method_comparison",
        "status": "completed" if success else "completed_with_failures",
        "manifest": {"path": path.relative_to(root).as_posix(), "sha256": manifest_hash},
        "name": manifest.name,
        "rule_version": RULE_VERSION,
        "models": models,
        "implementation_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("evaluate_optimizer.py", "optimizer_baseline.py")
        },
        "agent_limits": json.loads(next(iter(limits))) if limits else None,
        "new_execution": {"api_calls": 0, "training_runs": 0},
        "summary": summary,
        "all_attempt_usage": usage,
        "cases": rows,
        "interpretation": [
            "Development comparison on recorded audit evidence, "
            "not an independent held-out benchmark.",
            "The Agent validator already enforces related audit and performance rules; "
            "agreement does not establish an LLM advantage.",
            "Manifest expectations are not supplied to the rule classifier; "
            "directory names are not used in its decisions.",
            "Missing or invalid reports stay in case denominators; "
            "zero false positives does not imply complete coverage.",
            "Attempt order and completeness are declared by the manifest. "
            "Empty traces cannot independently prove task assignment.",
            "Citation checks validate recorded values and classifications, "
            "not every sentence of a generated explanation.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = evaluate_optimizer_suite(args.manifest, workspace_root=Path.cwd())
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    directory = _scoped(Path.cwd().resolve(), "artifacts/optimizer_evaluations")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    with target.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"optimizer_evaluation_report={target}")
    print(json.dumps(report["summary"], indent=2))
    print(json.dumps(report["all_attempt_usage"], indent=2))
    if report["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
