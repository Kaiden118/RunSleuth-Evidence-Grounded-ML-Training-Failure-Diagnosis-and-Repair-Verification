"""Deterministic comparison of two training runs."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from runsleuth.telemetry import EpochMetrics


@dataclass(frozen=True, slots=True)
class RunSummary:
    run_directory: str
    final_validation_accuracy: float
    final_validation_loss: float
    first_epoch_gradient_norm: float
    first_epoch_parameter_update_norm: float
    final_epoch_gradient_norm: float
    final_epoch_parameter_update_norm: float


@dataclass(frozen=True, slots=True)
class RunComparison:
    clean: RunSummary
    candidate: RunSummary
    validation_accuracy_drop: float
    validation_loss_ratio: float
    first_epoch_update_norm_ratio: float
    final_epoch_gradient_norm_ratio: float


def load_epoch_metrics(run_directory: Path) -> list[EpochMetrics]:
    """Load all epoch records from one run."""
    metrics_path = run_directory / "metrics.jsonl"
    records = [
        EpochMetrics(**json.loads(line))
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"No epoch metrics found in {metrics_path}")
    return records


def summarize_run(run_directory: Path) -> RunSummary:
    """Extract the evidence needed for run comparison."""
    records = load_epoch_metrics(run_directory)
    first_epoch = records[0]
    final_epoch = records[-1]

    return RunSummary(
        run_directory=str(run_directory),
        final_validation_accuracy=final_epoch.validation_accuracy,
        final_validation_loss=final_epoch.validation_loss,
        first_epoch_gradient_norm=first_epoch.mean_gradient_norm,
        first_epoch_parameter_update_norm=(
            first_epoch.mean_parameter_update_norm
        ),
        final_epoch_gradient_norm=final_epoch.mean_gradient_norm,
        final_epoch_parameter_update_norm=(
            final_epoch.mean_parameter_update_norm
        ),
    )


def ratio(numerator: float, denominator: float, name: str) -> float:
    """Compute an evidence ratio while rejecting an undefined result."""
    if denominator == 0:
        raise ValueError(f"Cannot compute {name}: denominator is zero")
    return numerator / denominator


def compare_runs(
    clean_run_directory: Path,
    candidate_run_directory: Path,
) -> RunComparison:
    """Compare a candidate run against a known-good clean run."""
    clean = summarize_run(clean_run_directory)
    candidate = summarize_run(candidate_run_directory)

    return RunComparison(
        clean=clean,
        candidate=candidate,
        validation_accuracy_drop=(
            clean.final_validation_accuracy
            - candidate.final_validation_accuracy
        ),
        validation_loss_ratio=ratio(
            candidate.final_validation_loss,
            clean.final_validation_loss,
            "validation loss ratio",
        ),
        first_epoch_update_norm_ratio=ratio(
            candidate.first_epoch_parameter_update_norm,
            clean.first_epoch_parameter_update_norm,
            "first-epoch update norm ratio",
        ),
        final_epoch_gradient_norm_ratio=ratio(
            candidate.final_epoch_gradient_norm,
            clean.final_epoch_gradient_norm,
            "final-epoch gradient norm ratio",
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare candidate telemetry with a clean run.",
    )
    parser.add_argument("--clean-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    comparison = compare_runs(args.clean_run, args.candidate_run)
    print(json.dumps(asdict(comparison), indent=2))


if __name__ == "__main__":
    main()
