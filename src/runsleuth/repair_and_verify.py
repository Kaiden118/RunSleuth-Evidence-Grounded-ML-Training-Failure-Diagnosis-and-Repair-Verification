"""Command-line entry point for bounded repair and verification."""

import argparse
from pathlib import Path

from runsleuth.repair import run_bounded_repair
from runsleuth.verification import RepairDecision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose, repair, retrain, and verify a failed run."
    )
    parser.add_argument(
        "--reference-run",
        type=Path,
        required=True,
        help="Healthy reference run directory.",
    )
    parser.add_argument(
        "--failed-run",
        type=Path,
        required=True,
        help="Failed candidate run directory.",
    )
    parser.add_argument(
        "--reference-config",
        type=Path,
        required=True,
        help="Source configuration for the healthy run.",
    )
    parser.add_argument(
        "--failed-config",
        type=Path,
        required=True,
        help="Source configuration for the failed run.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=2,
        help="Maximum number of epochs allowed for repair verification.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    report = run_bounded_repair(
        reference_run=args.reference_run,
        failed_run=args.failed_run,
        reference_config_path=args.reference_config,
        failed_config_path=args.failed_config,
        max_epochs=args.max_epochs,
    )

    report_path = Path(report.repaired_run) / "repair_report.json"
    report_json = report.to_json()
    report_path.write_text(f"{report_json}\n", encoding="utf-8")

    print(f"repair_report={report_path}")
    print(report_json)

    if report.verification.decision is RepairDecision.REJECTED:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
