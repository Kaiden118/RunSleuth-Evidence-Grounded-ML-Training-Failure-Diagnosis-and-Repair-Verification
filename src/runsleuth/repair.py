"""Controlled execution of bounded training repairs."""

import json
from dataclasses import asdict, dataclass, replace
from math import isfinite
from pathlib import Path

from runsleuth.config import TrainingConfig
from runsleuth.diagnose import diagnose_training_failure
from runsleuth.diagnosis import DiagnosisReport, DiagnosisStatus, RootCause
from runsleuth.experiment import run_experiment
from runsleuth.verification import (
    VerificationPolicy,
    VerificationReport,
    verify_repair,
)


@dataclass(frozen=True)
class RepairAttemptReport:
    """Complete record of one diagnosis, repair, and verification attempt."""

    diagnosis: DiagnosisReport
    repaired_config: TrainingConfig
    repaired_run: str
    verification: VerificationReport

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def apply_config_patch(
    failed_config: TrainingConfig,
    *,
    root_cause: RootCause,
    field: str,
    old_value: object,
    new_value: object,
) -> TrainingConfig:
    """Validate an allowlisted patch and return a new training config."""
    if root_cause is RootCause.HIGH_LEARNING_RATE:
        if field != "learning_rate":
            raise ValueError("High-learning-rate repair must target learning_rate")

        if isinstance(old_value, bool) or not isinstance(old_value, (int, float)):
            raise ValueError("Original learning rate must be a number")

        if (
            not isfinite(old_value)
            or old_value <= 0.0
            or isinstance(failed_config.learning_rate, bool)
            or old_value != failed_config.learning_rate
        ):
            raise ValueError("Learning-rate patch does not match the current config")

        if isinstance(new_value, bool) or not isinstance(new_value, (int, float)):
            raise ValueError("Repaired learning rate must be a number")

        if not isfinite(new_value) or new_value <= 0.0:
            raise ValueError("Repaired learning rate must be finite and positive")

        if new_value >= old_value:
            raise ValueError("High-learning-rate repair must reduce learning_rate")

        return replace(
            failed_config,
            learning_rate=float(new_value),
        )

    if root_cause is RootCause.MISSING_OPTIMIZER_STEP:
        if field != "optimizer_step_enabled":
            raise ValueError("Missing-step repair must target optimizer_step_enabled")

        if (
            failed_config.optimizer_step_enabled is not False
            or old_value is not False
            or new_value is not True
        ):
            raise ValueError("Missing-step repair requires a False -> True change")

        return replace(
            failed_config,
            optimizer_step_enabled=True,
        )

    raise ValueError(f"Unsupported root cause: {root_cause}")


def run_bounded_repair(
    reference_run: Path,
    failed_run: Path,
    reference_config_path: Path,
    failed_config_path: Path,
    max_epochs: int = 2,
    policy: VerificationPolicy | None = None,
) -> RepairAttemptReport:
    """Run an allowlisted repair under a fixed training budget."""

    if type(max_epochs) is not int or max_epochs < 1:
        raise ValueError("max_epochs must be a positive integer")

    diagnosis = diagnose_training_failure(
        reference_run=reference_run,
        candidate_run=failed_run,
        reference_config_path=reference_config_path,
        candidate_config_path=failed_config_path,
    )

    if diagnosis.status is not DiagnosisStatus.FAILURE_DETECTED:
        raise ValueError("repair requires a supported detected failure")

    if not diagnosis.ranked_causes:
        raise ValueError("diagnosis did not provide a repair candidate")

    top_cause = diagnosis.ranked_causes[0]
    proposed_patch = top_cause.proposed_patch
    failed_config = TrainingConfig.load(failed_config_path)

    repaired_config = apply_config_patch(
        failed_config,
        root_cause=top_cause.root_cause,
        field=proposed_patch.field,
        old_value=proposed_patch.old_value,
        new_value=proposed_patch.new_value,
    )

    repaired_config = replace(
        repaired_config,
        run_name=f"{failed_config.run_name}-repair",
        epochs=min(failed_config.epochs, max_epochs),
    )

    repaired_run = run_experiment(repaired_config)

    verification = verify_repair(
        reference_run=reference_run,
        failed_run=failed_run,
        repaired_run=repaired_run,
        policy=policy,
    )

    return RepairAttemptReport(
        diagnosis=diagnosis,
        repaired_config=repaired_config,
        repaired_run=str(repaired_run),
        verification=verification,
    )
