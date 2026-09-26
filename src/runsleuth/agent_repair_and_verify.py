"""Command-line entry point for agent-proposed repair verification."""

import argparse
from pathlib import Path

from runsleuth.agent_repair import run_bounded_agent_repair
from runsleuth.verification import RepairDecision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate an agent proposal, retrain, and verify recovery.",
    )
    parser.add_argument("--agent-report", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--failed-run", type=Path, required=True)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("src/runsleuth/train.py"),
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=2,
        help="Maximum number of training epochs for this repair attempt.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    report = run_bounded_agent_repair(
        agent_report_path=args.agent_report,
        source_path=args.source,
        reference_run=args.reference_run,
        failed_run=args.failed_run,
        max_epochs=args.max_epochs,
    )

    report_path = Path(report.repaired_run) / "agent_repair_report.json"
    report_json = report.to_json()
    report_path.write_text(f"{report_json}\n", encoding="utf-8")

    print(f"agent_repair_report={report_path}")
    print(report_json)

    if report.verification.decision is RepairDecision.REJECTED:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
