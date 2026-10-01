"""Run a fresh, fixed-model optimizer diagnosis panel on existing training evidence."""

import argparse
import json
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from runsleuth.evaluate_optimizer import Manifest, _read, _scoped, evaluate_optimizer_suite
from runsleuth.optimizer_baseline import diagnose_optimizer_rules
from runsleuth.optimizer_evidence import inspect_optimizer_run

LIMITS = {"max_model_calls": 6, "max_tool_calls": 5, "max_output_tokens": 3000}


def _write(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


@contextmanager
def _agent_session(model: str, root: Path):
    # Keep API dependencies lazy so preflight and offline tests need no client.
    from runsleuth.agent import run_diagnostic_agent
    from runsleuth.diagnostic_tools import DiagnosticTools
    from runsleuth.llm_client import create_llm_client

    client, selected_model = create_llm_client()
    if selected_model != model:
        client.close()
        raise ValueError("GEMINI_MODEL must match the explicit --model argument")
    with client:
        tools = DiagnosticTools(root, enable_optimizer_audit=True)

        def run(reference_run: str, candidate_run: str) -> dict:
            report = run_diagnostic_agent(
                client,
                selected_model,
                tools,
                source_path="src/runsleuth/train.py",
                reference_run=reference_run,
                candidate_run=candidate_run,
                profile="optimizer_binding",
                **LIMITS,
            )
            return json.loads(report.to_json())

        yield run


def rerun_comparison(
    manifest_path: Path,
    *,
    workspace_root: Path,
    model: str,
    case_delay: int = 10,
) -> Path:
    """Make one new Agent attempt per declared case and evaluate only those attempts."""
    if not isinstance(model, str) or not model.strip() or model != model.strip():
        raise ValueError("Provide an explicit nonempty model name without outer whitespace")
    if type(case_delay) is not int or not 0 <= case_delay <= 60:
        raise ValueError("case_delay must be an integer between 0 and 60 seconds")
    root = workspace_root.resolve()
    source_path = _scoped(root, manifest_path, file=True)
    source_json, source_hash = _read(source_path)
    source = Manifest.model_validate(source_json)
    if len(source.cases) > 4:
        raise ValueError("This bounded rerun supports at most four cases")
    seen_ids, seen_pairs, evidence = set(), set(), {}
    for case in source.cases:
        pair = tuple(_scoped(root, name) for name in (case.reference_run, case.candidate_run))
        if case.id in seen_ids or pair in seen_pairs:
            raise ValueError("Case IDs and reference/candidate pairs must be unique")
        seen_ids.add(case.id)
        seen_pairs.add(pair)
        for name in (case.reference_run, case.candidate_run):
            if name not in evidence:
                evidence[name] = inspect_optimizer_run(
                    _scoped(root, name), require_file=lambda p: _scoped(root, p, file=True)
                )
            if evidence[name]["config"]["seed"] != case.seed:
                raise ValueError("Manifest seed does not match the recorded training seed")
        rule = diagnose_optimizer_rules(evidence[case.reference_run], evidence[case.candidate_run])
        if rule["performance_status"] == "inconclusive":
            raise ValueError("Rerun requires comparable paired training evidence")

    # Client creation sends no request. All input checks finish before the first paid call.
    with _agent_session(model, root) as run:
        directory = _scoped(root, "artifacts/optimizer_reruns") / datetime.now(UTC).strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
        directory.mkdir(parents=True)
        cases = []
        for index, case in enumerate(source.cases, 1):
            report_path = directory / f"case-{index:03d}" / "agent_report.json"
            report_path.parent.mkdir()
            cases.append(
                {**case.model_dump(), "agent_reports": [report_path.relative_to(root).as_posix()]}
            )
        fresh_manifest = directory / "manifest.json"
        _write(
            fresh_manifest,
            {
                "schema_version": 1,
                "name": f"{source.name}-fresh",
                "cases": cases,
            },
        )
        budget = {
            "agent_attempts": len(cases),
            "model_call_upper_bound": 6 * len(cases),
            "tool_call_upper_bound": 5 * len(cases),
            "training_runs": 0,
        }
        _write(
            directory / "run_plan.json",
            {
                "source_manifest": source_path.relative_to(root).as_posix(),
                "source_manifest_sha256": source_hash,
                "model": model,
                "limits": LIMITS,
                "budget": budget,
                "case_delay_seconds": case_delay,
                "case_retries": 0,
                "validation_corrections_per_attempt": 1,
                "usage_scope": "New panel only; historical attempts excluded and preserved",
                "evidence_sha256": {
                    name: data["provenance"]["files_sha256"] for name, data in evidence.items()
                },
            },
        )
        print(f"fresh_manifest={fresh_manifest}", flush=True)
        print(json.dumps({"model": model, "limits": LIMITS, "budget": budget}), flush=True)
        active_case = None
        try:
            for index, case in enumerate(cases):
                active_case = case["id"]
                if index:
                    time.sleep(case_delay)
                report = run(case["reference_run"], case["candidate_run"])
                # Save the real report before checking identity, retaining any recorded costs.
                _write(root / case["agent_reports"][0], report)
                if report.get("model") != model or report.get("limits") != LIMITS:
                    raise ValueError("Agent changed its requested model or budget during the panel")
                print(f"case={case['id']}, status={report.get('status')}", flush=True)
        except (Exception, KeyboardInterrupt) as error:
            _write(
                directory / "runner_error.json",
                {
                    "case": active_case,
                    "error_type": type(error).__name__,
                    "note": "Stopped; no synthetic Agent report or zero-usage estimate was created.",
                },
            )
            raise

    evaluation = evaluate_optimizer_suite(fresh_manifest, workspace_root=root)
    evaluation_path = directory / "evaluation_report.json"
    _write(evaluation_path, evaluation)
    print(f"optimizer_evaluation_report={evaluation_path}")
    print(json.dumps(evaluation["summary"], indent=2))
    print(json.dumps(evaluation["all_attempt_usage"], indent=2))
    return evaluation_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--case-delay", type=int, default=10)
    args = parser.parse_args()
    try:
        path = rerun_comparison(
            args.manifest,
            workspace_root=Path.cwd(),
            model=args.model,
            case_delay=args.case_delay,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    report, _ = _read(path)
    if report["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
