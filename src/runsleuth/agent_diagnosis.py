"""Structured LLM diagnoses; evidence is verified separately against tool traces."""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
PathPart = NonEmptyText | Annotated[int, Field(ge=0)]


class StructuredModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class EvidenceCitation(StructuredModel):
    """Locate a scalar inside a successful tool result's data."""

    call_id: NonEmptyText
    path: list[PathPart] = Field(
        min_length=1,
        max_length=8,
        description="Path relative to tool result.data; strings are keys, integers are indices.",
    )
    expected_value: str | int | float | bool | None


class CauseHypothesis(StructuredModel):
    root_cause: Literal["high_learning_rate", "missing_optimizer_step"]
    confidence: Literal["low", "medium", "high"] = Field(
        description="Heuristic confidence, not a calibrated probability.",
    )
    explanation: NonEmptyText
    evidence: list[EvidenceCitation] = Field(min_length=1, max_length=8)


class ConfigPatchProposal(StructuredModel):
    """A proposed change to the candidate configuration, not an executed repair."""

    config_call_id: NonEmptyText = Field(
        description="ID of the successful load_run_config call for the candidate run.",
    )
    field: Literal["learning_rate", "optimizer_step_enabled"]
    old_value: float | bool
    new_value: float | bool

    @model_validator(mode="after")
    def validate_change(self) -> Self:
        if self.field == "optimizer_step_enabled":
            if self.old_value is not False or self.new_value is not True:
                raise ValueError("Optimizer-step repair must change false to true")
        else:
            if type(self.old_value) is not float or type(self.new_value) is not float:
                raise ValueError("Learning-rate values must be numbers, not booleans")
            if not 0 < self.new_value < self.old_value:
                raise ValueError("Learning-rate repair must reduce a positive learning rate")
        return self


class AgentDiagnosis(StructuredModel):
    status: Literal["failure_detected", "no_supported_failure"]
    summary: NonEmptyText
    ranked_causes: list[CauseHypothesis] = Field(
        max_length=2,
        description="Supported hypotheses in descending order of plausibility.",
    )
    uncertainties: list[NonEmptyText] = Field(max_length=6)
    proposed_patch: ConfigPatchProposal | None = Field(
        description="At most one patch for the first-ranked cause; null if unsupported.",
    )

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        if self.status == "no_supported_failure":
            if self.ranked_causes or self.proposed_patch is not None:
                raise ValueError("No supported failure must have no causes and no patch")
            return self

        if not self.ranked_causes:
            raise ValueError("A detected failure requires at least one supported cause")

        causes = [candidate.root_cause for candidate in self.ranked_causes]
        if len(causes) != len(set(causes)):
            raise ValueError("Root causes must not be repeated")

        if self.proposed_patch is not None:
            expected_field = {
                "high_learning_rate": "learning_rate",
                "missing_optimizer_step": "optimizer_step_enabled",
            }[causes[0]]
            if self.proposed_patch.field != expected_field:
                raise ValueError("Patch field must match the first-ranked cause")

        return self
