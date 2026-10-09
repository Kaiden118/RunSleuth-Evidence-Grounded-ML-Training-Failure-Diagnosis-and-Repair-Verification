"""The runsleuth command: diagnose a monitored run, verify a repair, or run the demo.

    runsleuth diagnose --run runs/bad/run_report.json --reference runs/good/run_report.json
    runsleuth verify --candidate runs/fixed/run_report.json \\
        --reference runs/good/run_report.json --faulty runs/bad/run_report.json
    runsleuth demo --baseline <seed baseline run> --fault random --blind

verify accepts a repaired run when re-diagnosis against the healthy reference finds
no known fault and gate v2 passes: no regression against the reference on every
validation split both runs logged as <split>_validation_accuracy and
<split>_validation_loss. The faulty run, if given, is compared for information only.
verify exits with status 1 when it rejects the repair, so a script or CI job can stop.
"""

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

from runsleuth.camelyon_variant_runner import verify_repair
from runsleuth.diagnose_run import diagnose_run, run_optimizer

COMMANDS = {
    "diagnose": "Diagnose a monitored run, optionally against a healthy reference",
    "verify": "Verify a repaired run against a healthy reference",
    "demo": "Inject, diagnose, repair and verify a fault on Camelyon17",
}
VALIDATION_METRIC = re.compile(r"^([A-Za-z0-9]+)_validation_(accuracy|loss)$")


def validation_splits(*runs: dict) -> tuple[str, ...]:
    """The splits for which every run logged both validation accuracy and loss."""

    def splits(run: dict) -> set[str]:
        found: dict[str, set[str]] = {}
        for key in run.get("final_metrics", {}):
            if match := VALIDATION_METRIC.match(key):
                found.setdefault(match[1], set()).add(match[2])
        return {split for split, metrics in found.items() if metrics == {"accuracy", "loss"}}

    return tuple(sorted(set.intersection(*(splits(run) for run in runs))))


def verify_runs(
    candidate: dict, reference: dict, faulty: dict | None = None, *, optimizer_name=None
) -> dict:
    """Re-diagnose the candidate against the reference, then apply gate v2."""
    diagnosis = diagnose_run(candidate, reference, optimizer_name=optimizer_name)
    found = diagnosis["final_diagnosis"]
    splits = validation_splits(candidate, reference)
    compare_faulty = (
        faulty is not None and bool(splits) and set(validation_splits(faulty)) >= set(splits)
    )
    runs = {"candidate": candidate, "reference": reference}
    if compare_faulty:
        runs["faulty"] = faulty
    gate = verify_repair(
        runs,
        "candidate",
        reference="reference",
        faulty="faulty" if compare_faulty else None,
        domains=splits,
    )
    reasons = []
    if found != "no_known_fault":
        reasons.append(f"re-diagnosis against the reference found {found}")
    if not splits:
        reasons.append(
            "no validation split with accuracy and loss in both runs; log "
            "<split>_validation_accuracy and <split>_validation_loss"
        )
    for check in gate["performance_checks"]:
        if not check["passed"]:
            observed = check["observed_value"]
            value = "an invalid value" if observed is None else f"{observed:.4g}"
            reasons.append(
                f"{check['domain']} {check['metric']} is {value}, beyond {check['threshold']}"
            )
    return {
        "decision": "rejected" if reasons else "accepted",
        "reasons": reasons,
        "re_diagnosis": found,
        "validation_splits": list(splits),
        **gate,
        "scope": "development policy; no statistical or performance-recovery claim",
    }


def _check_lines(checks: list) -> list[str]:
    lines = []
    for check in checks:
        observed = check["observed_value"]
        value = "invalid" if observed is None else f"{observed:.4g}"
        mark = "+" if check["passed"] else "x"
        lines.append(
            f"  {mark} {check['domain']} {check['metric']} = {value} (<= {check['threshold']})"
        )
    return lines


def verification_text(result: dict) -> str:
    lines = [f"Decision: {result['decision']}", f"Re-diagnosis: {result['re_diagnosis']}"]
    lines.append("No regression against the reference:")
    lines += _check_lines(result["performance_checks"])
    if result["informational_checks"]:
        lines.append("Against the faulty run (informational):")
        lines += _check_lines(result["informational_checks"])
    if result["faulty_passes_gate_vs_reference"]:
        lines.append(
            "Note: the faulty run itself passes the gate; the fault was silent in metrics."
        )
    lines += [f"Reason: {reason}" for reason in result["reasons"]]
    return "\n".join(lines)


def _verify(argv: list[str], prog: str) -> None:
    parser = argparse.ArgumentParser(prog=prog, description=COMMANDS["verify"])
    parser.add_argument("--candidate", type=Path, required=True, help="Run report after the repair")
    parser.add_argument("--reference", type=Path, required=True, help="Run report of a healthy run")
    parser.add_argument("--faulty", type=Path, help="Run report before the repair (informational)")
    parser.add_argument(
        "--optimizer",
        help="Optimizer class name (default: the report, then the experiment's environment.json)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/verifications"))
    args = parser.parse_args(argv)

    def read(path: Path | None) -> dict | None:
        return json.loads(path.read_text(encoding="utf-8")) if path else None

    candidate = read(args.candidate)
    result = verify_runs(
        candidate,
        read(args.reference),
        read(args.faulty),
        optimizer_name=run_optimizer(candidate, args.candidate, args.optimizer),
    )
    result["inputs"] = {
        name: str(path) if path else None
        for name, path in (
            ("candidate", args.candidate),
            ("reference", args.reference),
            ("faulty", args.faulty),
        )
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    path = args.output_dir / f"verification-{timestamp}.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(verification_text(result))
    print(f"verification_report={path}")
    if result["decision"] != "accepted":
        raise SystemExit(1)


def usage() -> str:
    lines = ["usage: runsleuth <command> [options]", "", __doc__.splitlines()[0], "", "commands:"]
    lines += [f"  {name:<10}{description}" for name, description in COMMANDS.items()]
    return "\n".join([*lines, "", "Run 'runsleuth <command> --help' for its options."])


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS:
        if argv in (["-h"], ["--help"]):
            print(usage())
            return
        print(usage(), file=sys.stderr)
        raise SystemExit(2)
    command, options = argv[0], argv[1:]
    prog = f"runsleuth {command}"
    if command == "diagnose":
        from runsleuth.diagnose_run import main as diagnose

        diagnose(options, prog)
    elif command == "verify":
        _verify(options, prog)
    else:
        from runsleuth.demo import main as demo

        demo(options, prog)


if __name__ == "__main__":
    main()
