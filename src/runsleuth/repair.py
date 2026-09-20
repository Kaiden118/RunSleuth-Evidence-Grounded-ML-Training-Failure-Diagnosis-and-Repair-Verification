"""Controlled execution of bounded training repairs."""

import json
from dataclasses import asdict, dataclass, replace
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


def run_bounded_repair(
    reference_run: Path,
    failed_run: Path,
    reference_config_path: Path,
    failed_config_path: Path,
    max_epochs: int = 2,
    policy: VerificationPolicy | None = None,
) -> RepairAttemptReport:
    """Run an allowlisted repair under a fixed training budget."""

    if max_epochs < 1:
        raise ValueError("max_epochs must be at least 1")

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

    if top_cause.root_cause is not RootCause.HIGH_LEARNING_RATE:
        raise ValueError(f"unsupported root cause: {top_cause.root_cause}")

    proposed_patch = top_cause.proposed_patch

    if proposed_patch.field != "learning_rate":
        raise ValueError(f"unsupported repair field: {proposed_patch.field}")

    repaired_learning_rate = float(proposed_patch.new_value)
    if repaired_learning_rate <= 0.0:
        raise ValueError("repaired learning rate must be positive")

    failed_config = TrainingConfig.load(failed_config_path)
    repaired_config = replace(
        failed_config,
        run_name=f"{failed_config.run_name}-repair",
        epochs=min(failed_config.epochs, max_epochs),
        learning_rate=repaired_learning_rate,
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
