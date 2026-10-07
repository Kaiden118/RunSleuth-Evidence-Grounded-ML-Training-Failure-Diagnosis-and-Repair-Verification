"""Freeze and compare rules, source-only LLM and tool-assisted semantic predictions."""

import argparse
import hashlib
import json
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from runsleuth.evaluate_optimizer import _read, _scoped
from runsleuth.optimizer_challenge_cases import (
    ASSUMPTIONS,
    VERSION,
    challenge_cases,
    challenge_sha256,
    verify_known_case,
)
from runsleuth.optimizer_challenge_llm import (
    BASELINE_PROTOCOL,
    METHODS,
    PROTOCOLS,
    REVISED_PROTOCOL,
    BindingPrediction,
    limits_for,
    replay_prediction,
    run_llm_method,
    system_prompt_for,
)
from runsleuth.optimizer_factory_agent import _record_usage, _write
from runsleuth.optimizer_factory_analysis import analyze_optimizer_factory

SNAPSHOT_FILES = (
    "optimizer_challenge.py",
    "optimizer_challenge_cases.py",
    "optimizer_challenge_llm.py",
    "optimizer_factory_analysis.py",
    "optimizer_source_patch.py",
    "optimizer_factory_diagnosis.py",
    "optimizer_factory_agent.py",
    "diagnostic_tools.py",
    "llm_client.py",
    "optimizer_source_runtime.py",
    "optimizer_probe.py",
    "optimizer_audit.py",
    "tool_contracts.py",
    "optimizer_agent_diagnosis.py",
)
RULE_MAPPING = {"patch_proposed": "stale", "no_change": "current", "unsupported": "inconclusive"}


def _rate(n: int, d: int) -> dict:
    return {"numerator": n, "denominator": d, "rate": n / d if d else None}


def _usage() -> dict:
    return {
        "model_calls": 0,
        "tool_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "attempt_report_count": 0,
        "usage_complete": True,
        "unrecorded_attempt_exists": False,
    }


def score_method(cases: list[dict], method: str) -> dict:
    """Errors remain in all-case denominators; refusal is not silently dropped."""
    rows = [(case["expected"], case["methods"][method]) for case in cases]
    known = [(label, row) for label, row in rows if label != "inconclusive"]
    unknown = [(label, row) for label, row in rows if label == "inconclusive"]
    healthy = [(label, row) for label, row in rows if label == "current"]

    def predicted(row):
        return row.get("binding") if row["status"] == "completed" else None

    answered = [(label, row) for label, row in rows if predicted(row) in ("stale", "current")]
    return {
        "correct": _rate(sum(predicted(row) == label for label, row in rows), len(rows)),
        "known_binding_correct": _rate(
            sum(predicted(row) == label for label, row in known), len(known)
        ),
        "unknown_abstention": _rate(
            sum(predicted(row) == "inconclusive" for _, row in unknown), len(unknown)
        ),
        "unknown_overclaim": _rate(
            sum(predicted(row) in ("stale", "current") for _, row in unknown), len(unknown)
        ),
        "known_coverage": _rate(
            sum(predicted(row) in ("stale", "current") for _, row in known), len(known)
        ),
        "known_over_refusal": _rate(
            sum(predicted(row) == "inconclusive" for _, row in known), len(known)
        ),
        "selective_accuracy": _rate(
            sum(predicted(row) == label for label, row in answered), len(answered)
        ),
        "healthy_false_positive": _rate(
            sum(predicted(row) == "stale" for _, row in healthy), len(healthy)
        ),
        "completed": sum(row["status"] == "completed" for _, row in rows),
        "error_cases": sum(
            row["status"] not in ("completed", "not_run", "running") for _, row in rows
        ),
        "not_run": sum(row["status"] == "not_run" for _, row in rows),
        "first_pass_format_valid": _rate(
            sum(row.get("first_pass_valid") is True for _, row in rows), len(rows)
        )
        if method != "rules"
        else None,
    }


def summarize(report: dict) -> None:
    report["summary"] = {
        method: score_method(report["cases"], method) for method in ("rules", *METHODS)
    }
    known = [row["oracle"] for row in report["cases"] if row["expected"] != "inconclusive"]
    report["oracle_summary"] = {
        "known_labels_verified": _rate(
            sum(row.get("label_verified") is True for row in known), len(known)
        ),
        "optimizer_steps_recorded": sum(row.get("optimizer_steps_recorded", 0) for row in known),
        "step_accounting_complete": all(row.get("step_accounting_complete", True) for row in known),
        "unknown_cases_executed": 0,
    }
    paired = []
    for case in report["cases"]:
        plain, agent = (case["methods"][method] for method in METHODS)
        if plain["status"] == agent["status"] == "completed":
            paired.append(
                (plain["binding"] == case["expected"], agent["binding"] == case["expected"])
            )
    report["paired_completed_only"] = {
        "denominator": len(paired),
        "both_correct": sum(a and b for a, b in paired),
        "source_only_correct": sum(a and not b for a, b in paired),
        "tool_agent_only_correct": sum(not a and b for a, b in paired),
        "both_wrong": sum(not a and not b for a, b in paired),
        "note": "Supplement only; primary accuracy above includes errors and unrun cases",
    }


@contextmanager
def _session(root: Path, model: str, *, protocol: str = BASELINE_PROTOCOL):
    from runsleuth.diagnostic_tools import DiagnosticTools
    from runsleuth.llm_client import create_llm_client

    client, selected = create_llm_client()
    if selected != model:
        client.close()
        raise ValueError("GEMINI_MODEL must equal --model")
    with client:
        tools = DiagnosticTools(root, enable_optimizer_factory=True)

        def run(method, source_path, source):
            return run_llm_method(
                client,
                model,
                method=method,
                source_path=source_path,
                source=source,
                tools=tools if method == "tool_agent" else None,
                protocol=protocol,
            )

        yield run


def run_challenge(
    *,
    workspace_root: Path,
    model: str,
    execute: bool = False,
    case_delay: int = 10,
    protocol: str = BASELINE_PROTOCOL,
) -> Path:
    if type(execute) is not bool or type(case_delay) is not int or not 0 <= case_delay <= 60:
        raise ValueError("Invalid execute flag or case delay (0 to 60 seconds)")
    if not isinstance(model, str) or not model.strip() or model != model.strip():
        raise ValueError("Specify a model without outer whitespace")
    base_prompt = system_prompt_for(protocol)
    root = workspace_root.resolve()
    cases = challenge_cases()
    directory = _scoped(root, "artifacts/optimizer_challenges") / datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "report.json"
    report = {
        "schema_version": 2,
        "task": "optimizer_semantic_method_comparison",
        "version": VERSION,
        "challenge_sha256": challenge_sha256(),
        "status": "prepared",
        "error": None,
        "inference_protocol": protocol,
        "model": model,
        "execute_requested": execute,
        "cases": [],
        "case_delay_seconds": case_delay,
        "scope": (
            "Development retest on the same 12 cases after inspecting their baseline failures; not held-out evidence"
            if protocol == REVISED_PROTOCOL
            else "Author-constructed semantic challenge, not an independently blinded or real-repository benchmark"
        ),
        "protocol": {
            "id": protocol,
            "adapted_on_this_case_panel": protocol == REVISED_PROTOCOL,
            "assumptions": ASSUMPTIONS,
            "prompt": base_prompt,
            "output_schema": BindingPrediction.model_json_schema(),
            "limits": {method: limits_for(method) for method in METHODS},
            "rules": "Existing conservative AST decision mapping; unsupported is an abstention, not a truth label",
            "rule_mapping": RULE_MAPPING,
            "validation": "Schema and exact source quotations only; no answer-label correction feedback",
            "format_feedback": "all_independent_errors"
            if protocol == REVISED_PROTOCOL
            else "first_error",
            "tool_access": "Only tool_agent receives AST findings; both LLM arms receive identical complete source and assumptions",
            "order": "Alternate which LLM arm runs first by case; independent conversation for every arm",
            "case_retries": 0,
            "performance_evaluated": False,
            "patches_applied": 0,
        },
        "budget": {
            "cpu_oracle_steps": 8,
            "gpu_training_epochs": 0,
            "model_call_upper_bound": 60 if execute else 0,
            "tool_call_upper_bound": 12 if execute else 0,
            "llm_attempts": 24 if execute else 0,
        },
        "usage_by_method": {method: _usage() for method in METHODS},
        "all_attempt_usage": {
            **_usage(),
            "scope": "All new attempts in this comparison; historical runs excluded",
        },
    }
    snapshot = directory / "source_snapshot"
    snapshot.mkdir()
    report["source_files_sha256"] = {}
    for name in SNAPSHOT_FILES:
        payload = (Path(__file__).parent / name).read_bytes()
        (snapshot / name).write_bytes(payload)
        report["source_files_sha256"][name] = hashlib.sha256(payload).hexdigest()
    for case in cases:
        folder = directory / case["id"]
        folder.mkdir()
        source_path = folder / "factory.py"
        source_path.write_bytes(case["source"].encode())
        report["cases"].append(
            {
                **case,
                "source_path": source_path.relative_to(root).as_posix(),
                "oracle": {"status": "not_run"},
                "methods": {
                    method: {
                        "status": "not_run",
                        "binding": None,
                        "first_pass_valid": None,
                        "attempt_report": None,
                    }
                    for method in ("rules", *METHODS)
                },
            }
        )
    summarize(report)
    _write(path, report)
    print(f"challenge_directory={directory}", flush=True)
    print(
        json.dumps(
            {
                "model": model,
                "inference_protocol": protocol,
                "challenge_sha256": report["challenge_sha256"],
                "budget": report["budget"],
            }
        ),
        flush=True,
    )
    active = method = None
    in_flight = False
    phase = "oracle"
    try:
        for case, row in zip(cases, report["cases"], strict=True):
            if case["expected"] == "inconclusive":
                row["oracle"] = {
                    "status": "not_executed",
                    "reason": "Unknown helper behavior or model input is deliberately absent",
                }
            else:
                active = row
                print(f"phase=cpu_label_check case={case['id']}", flush=True)
                verify_known_case(case, row["oracle"])
            summarize(report)
            _write(path, report)
        active = None
        for row in report["cases"]:
            analysis = analyze_optimizer_factory(row["source"])
            row["methods"]["rules"].update(
                status="completed", binding=RULE_MAPPING[analysis["decision"]], analysis=analysis
            )
        report["status"] = "oracle_verified"
        summarize(report)
        _write(path, report)
        if execute:
            phase = "inference"
            with _session(root, model, protocol=protocol) as run:
                report["status"] = "running"
                attempt_index = 0
                for index, row in enumerate(report["cases"]):
                    active = row
                    order = METHODS if index % 2 == 0 else tuple(reversed(METHODS))
                    for method in order:
                        if attempt_index:
                            time.sleep(case_delay)
                        attempt_index += 1
                        selected = row["methods"][method]
                        current_source = _scoped(root, row["source_path"], file=True)
                        if (
                            hashlib.sha256(current_source.read_bytes()).hexdigest()
                            != row["source_sha256"]
                        ):
                            raise ValueError("Challenge source changed before inference")
                        for name, digest in report["source_files_sha256"].items():
                            if (
                                hashlib.sha256(
                                    (Path(__file__).parent / name).read_bytes()
                                ).hexdigest()
                                != digest
                            ):
                                raise ValueError(
                                    "Implementation changed during the frozen comparison"
                                )
                        selected["status"] = "running"
                        _write(path, report)
                        in_flight = True
                        attempt = run(method, row["source_path"], row["source"])
                        attempt_path = directory / row["id"] / f"{method}.json"
                        _write(attempt_path, attempt)
                        selected["attempt_report"] = attempt_path.relative_to(root).as_posix()
                        in_flight = False
                        _record_usage(report["all_attempt_usage"], attempt)
                        _record_usage(report["usage_by_method"][method], attempt)
                        if (
                            hashlib.sha256(current_source.read_bytes()).hexdigest()
                            != row["source_sha256"]
                        ):
                            raise ValueError("Challenge source changed during inference")
                        prediction = replay_prediction(
                            attempt,
                            source=row["source"],
                            source_path=row["source_path"],
                            method=method,
                            model=model,
                            protocol=protocol,
                        )
                        selected.update(
                            status=attempt["status"],
                            first_pass_valid=attempt.get("first_pass_valid"),
                            binding=prediction["binding"] if prediction else None,
                            error=attempt.get("error"),
                        )
                        summarize(report)
                        _write(path, report)
                        print(
                            f"case={row['id']}, method={method}, status={selected['status']}, binding={selected['binding']}",
                            flush=True,
                        )
            report["status"] = (
                "completed"
                if all(
                    row["methods"][m]["status"] == "completed"
                    for row in report["cases"]
                    for m in METHODS
                )
                else "completed_with_errors"
            )
    except (Exception, KeyboardInterrupt) as error:
        report.update(
            status="execution_error", error={"type": type(error).__name__, "phase": phase}
        )
        if phase == "oracle" and active is not None:
            active["oracle"].update(status="error", error_type=type(error).__name__)
        if phase == "inference" and active is not None and method is not None:
            active["methods"][method].update(status="runner_error", error_type=type(error).__name__)
        if in_flight:
            for usage in (report["all_attempt_usage"], report["usage_by_method"][method]):
                usage.update(usage_complete=False, unrecorded_attempt_exists=True)
        if active is None and isinstance(error, (ValueError, RuntimeError)):
            report["error"]["message"] = str(error)
        print(f"stopped phase={phase}, error_type={type(error).__name__}", flush=True)
    summarize(report)
    _write(path, report)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--protocol",
        choices=PROTOCOLS,
        default=BASELINE_PROTOCOL,
        help="binding-v2 retests these development cases with explicit binding semantics and complete format feedback.",
    )
    parser.add_argument(
        "--execute", action="store_true", help="After CPU preflight, run the paid LLM arms."
    )
    parser.add_argument("--case-delay", type=int, default=10)
    args = parser.parse_args()
    try:
        path = run_challenge(
            workspace_root=Path.cwd(),
            model=args.model,
            execute=args.execute,
            case_delay=args.case_delay,
            protocol=args.protocol,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    report, _ = _read(path)
    print(f"challenge_report={path}")
    print(f"status={report['status']}")
    print(json.dumps(report["oracle_summary"], indent=2))
    print(json.dumps(report["summary"], indent=2))
    print(json.dumps(report["usage_by_method"], indent=2))
    print(json.dumps(report["all_attempt_usage"], indent=2))
    if report["error"] is not None:
        print(json.dumps(report["error"]))
    if report["status"] not in ("completed", "oracle_verified"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
