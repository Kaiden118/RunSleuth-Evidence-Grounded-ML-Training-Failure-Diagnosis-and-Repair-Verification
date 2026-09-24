"""Rule-based diagnosis grounded in configuration and telemetry evidence."""

import argparse
import json
from difflib import unified_diff
from math import isfinite
from pathlib import Path

from runsleuth.compare import compare_runs, ratio
from runsleuth.config import TrainingConfig
from runsleuth.diagnosis import (
    DiagnosisReport,
    DiagnosisStatus,
    Evidence,
    MinimalPatch,
    RootCause,
    RootCauseCandidate,
)

MIN_LEARNING_RATE_RATIO = 10.0
MIN_ACCURACY_DROP = 0.20
MIN_FIRST_UPDATE_RATIO = 3.0
MAX_FINAL_GRADIENT_RATIO = 0.25


def verify_run_config(
    run_directory: Path,
    expected_config: TrainingConfig,
) -> None:
    """Ensure telemetry came from the configuration being diagnosed."""
    recorded_config = TrainingConfig.load(run_directory / "config.json")
    if recorded_config != expected_config:
        raise ValueError(
            f"Recorded config in {run_directory} does not match the supplied source config"
        )


def build_config_patch(
    candidate_config_path: Path,
    field: str,
    old_value: float | bool,
    new_value: float | bool,
) -> MinimalPatch:
    """Propose a change to one explicitly supported configuration field."""
    if field not in {"learning_rate", "optimizer_step_enabled"}:
        raise ValueError(f"Unsupported repair field: {field}")

    original_text = candidate_config_path.read_text(encoding="utf-8")
    values = json.loads(original_text)

    if values.get(field) != old_value:
        raise ValueError(f"Current value of {field} does not match the proposed patch")

    values[field] = new_value
    repaired_text = json.dumps(values, indent=2)
    if original_text.endswith("\n"):
        repaired_text += "\n"

    diff = "".join(
        unified_diff(
            original_text.splitlines(keepends=True),
            repaired_text.splitlines(keepends=True),
            fromfile=str(candidate_config_path),
            tofile=f"{candidate_config_path} (proposed)",
        )
    )

    return MinimalPatch(
        target_file=str(candidate_config_path),
        field=field,
        old_value=old_value,
        new_value=new_value,
        unified_diff=diff,
    )


def build_learning_rate_patch(
    candidate_config_path: Path,
    old_value: float,
    new_value: float,
) -> MinimalPatch:
    """Keep the existing learning-rate patch interface."""
    return build_config_patch(
        candidate_config_path,
        field="learning_rate",
        old_value=old_value,
        new_value=new_value,
    )


def diagnose_training_failure(
    reference_run: Path,
    candidate_run: Path,
    reference_config_path: Path,
    candidate_config_path: Path,
) -> DiagnosisReport:
    """Diagnose currently supported failures without modifying any files."""
    reference_config = TrainingConfig.load(reference_config_path)
    candidate_config = TrainingConfig.load(candidate_config_path)

    verify_run_config(reference_run, reference_config)
    verify_run_config(candidate_run, candidate_config)

    comparison = compare_runs(reference_run, candidate_run)
    metrics_text = (candidate_run / "metrics.jsonl").read_text(encoding="utf-8")
    records = [json.loads(line) for line in metrics_text.splitlines() if line.strip()]
    if not records:
        raise ValueError("Candidate metrics must contain at least one epoch")

    gradient_norms = [record["mean_gradient_norm"] for record in records]
    update_norms = [record["mean_parameter_update_norm"] for record in records]

    if any(not isfinite(value) or value < 0.0 for value in gradient_norms + update_norms):
        raise ValueError("Gradient and update norms must be finite and nonnegative")

    missing_step_detected = (
        reference_config.optimizer_step_enabled is True
        and candidate_config.optimizer_step_enabled is False
        and candidate_config.learning_rate == reference_config.learning_rate
        and candidate_config.learning_rate > 0.0
        and min(gradient_norms) > 0.0
        and max(update_norms) == 0.0
        and comparison.validation_accuracy_drop >= MIN_ACCURACY_DROP
    )

    if missing_step_detected:
        evidence = (
            Evidence(
                metric="optimizer_step_enabled",
                observed_value=False,
                reference_value=True,
                interpretation="Parameter updates are disabled in the candidate config.",
            ),
            Evidence(
                metric="final_epoch_gradient_norm",
                observed_value=comparison.candidate.final_epoch_gradient_norm,
                reference_value=comparison.clean.final_epoch_gradient_norm,
                interpretation=(f"Gradients are positive in all {len(records)} recorded epochs."),
            ),
            Evidence(
                metric="final_epoch_parameter_update_norm",
                observed_value=comparison.candidate.final_epoch_parameter_update_norm,
                reference_value=comparison.clean.final_epoch_parameter_update_norm,
                interpretation=(
                    f"Parameter updates are zero in all {len(records)} recorded epochs."
                ),
            ),
            Evidence(
                metric="final_validation_accuracy",
                observed_value=comparison.candidate.final_validation_accuracy,
                reference_value=comparison.clean.final_validation_accuracy,
                interpretation=(
                    "Candidate accuracy is "
                    f"{comparison.validation_accuracy_drop * 100:.2f} "
                    "percentage points below the reference."
                ),
            ),
        )

        candidate = RootCauseCandidate(
            rank=1,
            root_cause=RootCause.MISSING_OPTIMIZER_STEP,
            confidence=0.95,
            evidence=evidence,
            proposed_patch=build_config_patch(
                candidate_config_path,
                field="optimizer_step_enabled",
                old_value=False,
                new_value=True,
            ),
        )

        return DiagnosisReport(
            status=DiagnosisStatus.FAILURE_DETECTED,
            reference_run=str(reference_run),
            candidate_run=str(candidate_run),
            ranked_causes=(candidate,),
        )
    learning_rate_ratio = ratio(
        candidate_config.learning_rate,
        reference_config.learning_rate,
        "learning-rate ratio",
    )

    high_lr_detected = (
        learning_rate_ratio >= MIN_LEARNING_RATE_RATIO
        and comparison.validation_accuracy_drop >= MIN_ACCURACY_DROP
        and comparison.first_epoch_update_norm_ratio >= MIN_FIRST_UPDATE_RATIO
        and comparison.final_epoch_gradient_norm_ratio <= MAX_FINAL_GRADIENT_RATIO
    )

    if not high_lr_detected:
        return DiagnosisReport(
            status=DiagnosisStatus.NO_SUPPORTED_FAILURE,
            reference_run=str(reference_run),
            candidate_run=str(candidate_run),
            ranked_causes=(),
        )

    evidence = (
        Evidence(
            metric="learning_rate",
            observed_value=candidate_config.learning_rate,
            reference_value=reference_config.learning_rate,
            interpretation=(
                f"Candidate learning rate is {learning_rate_ratio:.2f}x the clean reference."
            ),
        ),
        Evidence(
            metric="final_validation_accuracy",
            observed_value=(comparison.candidate.final_validation_accuracy),
            reference_value=comparison.clean.final_validation_accuracy,
            interpretation=(
                f"Candidate accuracy is {comparison.validation_accuracy_drop:.2%} lower."
            ),
        ),
        Evidence(
            metric="final_validation_loss",
            observed_value=comparison.candidate.final_validation_loss,
            reference_value=comparison.clean.final_validation_loss,
            interpretation=(
                f"Candidate loss is {comparison.validation_loss_ratio:.2f}x the reference."
            ),
        ),
        Evidence(
            metric="first_epoch_parameter_update_norm",
            observed_value=(comparison.candidate.first_epoch_parameter_update_norm),
            reference_value=(comparison.clean.first_epoch_parameter_update_norm),
            interpretation=(
                "Initial parameter updates are "
                f"{comparison.first_epoch_update_norm_ratio:.2f}x larger."
            ),
        ),
        Evidence(
            metric="final_epoch_gradient_norm",
            observed_value=comparison.candidate.final_epoch_gradient_norm,
            reference_value=comparison.clean.final_epoch_gradient_norm,
            interpretation=(
                "Final gradient norm is "
                f"{comparison.final_epoch_gradient_norm_ratio:.2%} "
                "of the clean reference."
            ),
        ),
    )

    candidate = RootCauseCandidate(
        rank=1,
        root_cause=RootCause.HIGH_LEARNING_RATE,
        confidence=0.95,
        evidence=evidence,
        proposed_patch=build_learning_rate_patch(
            candidate_config_path,
            old_value=candidate_config.learning_rate,
            new_value=reference_config.learning_rate,
        ),
    )

    return DiagnosisReport(
        status=DiagnosisStatus.FAILURE_DETECTED,
        reference_run=str(reference_run),
        candidate_run=str(candidate_run),
        ranked_causes=(candidate,),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose a failed training run from recorded evidence.",
    )
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--reference-config", type=Path, required=True)
    parser.add_argument("--candidate-config", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = diagnose_training_failure(
        reference_run=args.reference_run,
        candidate_run=args.candidate_run,
        reference_config_path=args.reference_config,
        candidate_config_path=args.candidate_config,
    )
    print(report.to_json())


if __name__ == "__main__":
    main()
