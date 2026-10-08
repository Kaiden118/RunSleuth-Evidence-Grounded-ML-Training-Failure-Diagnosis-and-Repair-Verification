"""Re-score recorded gate v1 repair verifications under gate v2, without retraining.

    python -m runsleuth.rescore_repairs <experiment report.json> ... --output <result.json>

Gate v2 (camelyon_variant_runner.REPAIR_GATE) was revised after gate v1 rejected
repairs in these same runs, so re-scoring them shows what the revision changes; it
is not independent evidence that v2 is right. The recorded checks against clean are
first recomputed from the recorded metrics, and a report whose checks disagree with
its own metrics is refused.
"""

import argparse
import json
from pathlib import Path

from runsleuth.camelyon_optimizer_probe import file_sha256
from runsleuth.camelyon_variant_runner import REPAIR_GATE, verify_repair

REFERENCE = "clean"


def _verifications(report: dict):
    """Yield (faulty variant, candidate, recorded verification) for every repair."""
    recorded = report["repair_verification"]
    if "decision" not in recorded:
        for fault, verification in recorded.items():
            yield fault, verification["candidate"], verification
        return
    # Single-repair reports name the faulty run only as their second comparator.
    (faulty,) = {check["comparator"] for check in recorded["performance_checks"]} - {REFERENCE}
    yield faulty, recorded.get("candidate", f"{faulty}_repaired"), recorded


def _failed(checks: list) -> list[str]:
    return [
        f"{check['comparator']}:{check['domain']}_{check['metric']}"
        for check in checks
        if not check["passed"]
    ]


def rescore_report(report: dict) -> list[dict]:
    """One case per recorded repair with its v1 and v2 decisions."""
    if report.get("status") != "completed":
        raise ValueError(f"Experiment report is not completed: {report.get('status')}")
    cases = []
    for faulty, candidate, recorded in _verifications(report):
        if "gate" in recorded:
            raise ValueError(f"{candidate} was already verified under gate v2")
        compared = {check["comparator"] for check in recorded["performance_checks"]}
        gate = verify_repair(
            report["variants"],
            candidate,
            reference=REFERENCE,
            faulty=faulty if faulty in compared else None,
        )
        stored = [
            check for check in recorded["performance_checks"] if check["comparator"] == REFERENCE
        ]
        if stored != gate["performance_checks"]:
            raise ValueError(f"Recorded checks for {candidate} disagree with the recorded metrics")
        accepted = recorded["structure_verified"] and gate["performance_nonregression"]
        cases.append(
            {
                "fault": faulty,
                "candidate": candidate,
                "structure_verified": recorded["structure_verified"],
                "v1_decision": recorded["decision"],
                "v2_decision": "accepted" if accepted else "rejected",
                "v1_failed_checks": _failed(recorded["performance_checks"]),
                "v2_failed_checks": _failed(gate["performance_checks"]),
                "faulty_passes_gate_vs_reference": gate["faulty_passes_gate_vs_reference"],
            }
        )
    return cases


def _reference_seed(report: dict) -> int | None:
    """The stale-binding reports record their seed only in the reference run's config."""
    if not report.get("reference_run"):
        return None
    config = Path(report["reference_run"]) / "config.json"
    return json.loads(config.read_text(encoding="utf-8")).get("seed") if config.is_file() else None


def rescore_reports(report_paths: list[Path]) -> dict:
    cases, sources = [], []
    for path in report_paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        seed = report["seed"] if "seed" in report else _reference_seed(report)
        for case in rescore_report(report):
            cases.append({"report": str(path), "task": report["task"], "seed": seed, **case})
        sources.append({"path": str(path), "sha256": file_sha256(path)})
    return {
        "schema_version": 1,
        "task": "repair_gate_rescore",
        "gate": dict(REPAIR_GATE),
        "scope": "post hoc: gate v2 was revised after gate v1 rejected repairs in these "
        "runs; not independent validation",
        "reports": sources,
        "summary": {
            "cases": len(cases),
            "v1_accepted": sum(case["v1_decision"] == "accepted" for case in cases),
            "v2_accepted": sum(case["v2_decision"] == "accepted" for case in cases),
            "changed": [
                f"{case['task']} seed {case['seed']} {case['fault']}: "
                f"{case['v1_decision']} -> {case['v2_decision']}"
                for case in cases
                if case["v1_decision"] != case["v2_decision"]
            ],
            "faulty_passes_gate_vs_reference": sum(
                case["faulty_passes_gate_vs_reference"] is True for case in cases
            ),
        },
        "cases": cases,
    }


def summary_table(result: dict) -> str:
    lines = [f"{'seed':<6}{'fault':<36}{'v1':<10}{'v2':<10}faulty passes gate"]
    for case in result["cases"]:
        lines.append(
            f"{case['seed']!s:<6}{case['fault']:<36}{case['v1_decision']:<10}"
            f"{case['v2_decision']:<10}{case['faulty_passes_gate_vs_reference']}"
        )
    summary = result["summary"]
    lines.append(
        f"accepted: v1 {summary['v1_accepted']}/{summary['cases']}, "
        f"v2 {summary['v2_accepted']}/{summary['cases']}"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = rescore_reports(args.reports)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", "utf-8")
    print(summary_table(result))


if __name__ == "__main__":
    main()
