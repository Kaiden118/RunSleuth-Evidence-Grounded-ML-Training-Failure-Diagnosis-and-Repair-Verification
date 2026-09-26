"""Validated definitions for offline smoke evaluation."""

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SmokeModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )


class SmokeCase(SmokeModel):
    """One evaluation case with independently specified expectations."""

    id: str = Field(min_length=1)
    kind: Literal["injected_fault", "clean_self_comparison"]
    reference_run: str = Field(min_length=1)
    candidate_run: str = Field(min_length=1)
    expected_diagnosis_status: Literal[
        "failure_detected",
        "no_supported_failure",
    ]
    expected_root_cause: Literal[
        "high_learning_rate",
        "missing_optimizer_step",
    ] | None
    agent_report: str = Field(min_length=1)
    repair_report: str | None = Field(min_length=1)

    @model_validator(mode="after")
    def validate_expectations(self) -> Self:
        if self.kind == "injected_fault":
            if (
                self.expected_diagnosis_status != "failure_detected"
                or self.expected_root_cause is None
            ):
                raise ValueError(
                    "Injected faults require failure_detected and a root-cause label"
                )
        else:
            if (
                self.expected_diagnosis_status != "no_supported_failure"
                or self.expected_root_cause is not None
            ):
                raise ValueError(
                    "Clean self-comparisons require no_supported_failure and no root cause"
                )

            if self.reference_run != self.candidate_run:
                raise ValueError(
                    "Clean self-comparisons must use the same reference and candidate path"
                )

        return self


class SmokeSuite(SmokeModel):
    """A fixed collection of smoke cases and their repair budget."""

    suite: str = Field(min_length=1)
    max_repair_epochs: int = Field(ge=1)
    notes: list[str]
    cases: list[SmokeCase] = Field(min_length=1)
    source_path: str = Field(
        default="src/runsleuth/train.py",
        min_length=1,
    )

    @model_validator(mode="after")
    def validate_case_ids(self) -> Self:
        case_ids = [case.id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Evaluation case IDs must be unique")
        return self


def load_smoke_suite(path: Path) -> SmokeSuite:
    """Load the manifest without running diagnosis or training."""
    return SmokeSuite.model_validate_json(path.read_text(encoding="utf-8"))
