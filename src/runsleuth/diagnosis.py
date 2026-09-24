"""Typed outputs for evidence-grounded failure diagnosis."""

import json
from dataclasses import asdict, dataclass
from enum import StrEnum


class DiagnosisStatus(StrEnum):
    FAILURE_DETECTED = "failure_detected"
    NO_SUPPORTED_FAILURE = "no_supported_failure"


class RootCause(StrEnum):
    HIGH_LEARNING_RATE = "high_learning_rate"
    MISSING_OPTIMIZER_STEP = "missing_optimizer_step"


@dataclass(frozen=True, slots=True)
class Evidence:
    metric: str
    observed_value: float | bool
    reference_value: float | bool
    interpretation: str


@dataclass(frozen=True, slots=True)
class MinimalPatch:
    target_file: str
    field: str
    old_value: float | bool
    new_value: float | bool
    unified_diff: str


@dataclass(frozen=True, slots=True)
class RootCauseCandidate:
    rank: int
    root_cause: RootCause
    confidence: float
    evidence: tuple[Evidence, ...]
    proposed_patch: MinimalPatch

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("rank must be at least 1")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class DiagnosisReport:
    status: DiagnosisStatus
    reference_run: str
    candidate_run: str
    ranked_causes: tuple[RootCauseCandidate, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)
