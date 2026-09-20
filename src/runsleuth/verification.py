"""Evidence-based acceptance policy for repaired training runs."""

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from runsleuth.compare import compare_runs, ratio


class RepairDecision(StrEnum):
    """Possible outcomes of repair verification."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass(frozen=True)
class VerificationPolicy:
    """Explicit thresholds that a repaired run must satisfy."""

    min_validation_accuracy_gain: float = 0.20
    max_validation_accuracy_shortfall: float = 0.03
    max_validation_loss_ratio: float = 1.25

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_validation_accuracy_gain <= 1.0:
            raise ValueError("min_validation_accuracy_gain must be between 0 and 1")
        if not 0.0 <= self.max_validation_accuracy_shortfall <= 1.0:
            raise ValueError("max_validation_accuracy_shortfall must be between 0 and 1")
        if self.max_validation_loss_ratio <= 0.0:
            raise ValueError("max_validation_loss_ratio must be positive")


@dataclass(frozen=True)
class VerificationCheck:
    """One measurable condition used in the final decision."""

    metric: str
    passed: bool
    observed_value: float
    operator: str
    threshold: float


@dataclass(frozen=True)
class VerificationReport:
    """Evidence showing why a repaired run was accepted or rejected."""

    decision: RepairDecision
    reference_run: str
    failed_run: str
    repaired_run: str
    reference_validation_accuracy: float
    failed_validation_accuracy: float
    repaired_validation_accuracy: float
    reference_validation_loss: float
    failed_validation_loss: float
    repaired_validation_loss: float
    policy: VerificationPolicy
    checks: tuple[VerificationCheck, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def verify_repair(
    reference_run: Path,
    failed_run: Path,
    repaired_run: Path,
    policy: VerificationPolicy | None = None,
) -> VerificationReport:
    """Accept a repair only when every recovery requirement passes."""

    selected_policy = policy or VerificationPolicy()

    failed_comparison = compare_runs(reference_run, failed_run)
    repaired_comparison = compare_runs(reference_run, repaired_run)

    reference = failed_comparison.clean
    failed = failed_comparison.candidate
    repaired = repaired_comparison.candidate

    accuracy_gain = repaired.final_validation_accuracy - failed.final_validation_accuracy
    accuracy_shortfall = max(
        0.0,
        reference.final_validation_accuracy - repaired.final_validation_accuracy,
    )
    loss_ratio = ratio(
        repaired.final_validation_loss,
        reference.final_validation_loss,
        "validation loss",
    )

    checks = (
        VerificationCheck(
            metric="validation_accuracy_gain",
            passed=(accuracy_gain >= selected_policy.min_validation_accuracy_gain),
            observed_value=accuracy_gain,
            operator=">=",
            threshold=selected_policy.min_validation_accuracy_gain,
        ),
        VerificationCheck(
            metric="validation_accuracy_shortfall",
            passed=(accuracy_shortfall <= selected_policy.max_validation_accuracy_shortfall),
            observed_value=accuracy_shortfall,
            operator="<=",
            threshold=selected_policy.max_validation_accuracy_shortfall,
        ),
        VerificationCheck(
            metric="validation_loss_ratio_to_reference",
            passed=loss_ratio <= selected_policy.max_validation_loss_ratio,
            observed_value=loss_ratio,
            operator="<=",
            threshold=selected_policy.max_validation_loss_ratio,
        ),
    )

    decision = (
        RepairDecision.ACCEPTED
        if all(check.passed for check in checks)
        else RepairDecision.REJECTED
    )

    return VerificationReport(
        decision=decision,
        reference_run=str(reference_run),
        failed_run=str(failed_run),
        repaired_run=str(repaired_run),
        reference_validation_accuracy=reference.final_validation_accuracy,
        failed_validation_accuracy=failed.final_validation_accuracy,
        repaired_validation_accuracy=repaired.final_validation_accuracy,
        reference_validation_loss=reference.final_validation_loss,
        failed_validation_loss=failed.final_validation_loss,
        repaired_validation_loss=repaired.final_validation_loss,
        policy=selected_policy,
        checks=checks,
    )
