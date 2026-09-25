"""Validated argument contracts for diagnostic tools."""

from pydantic import BaseModel, ConfigDict, Field


class ToolArguments(BaseModel):
    """Shared validation rules for tool inputs."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )


class InspectSourceArgs(ToolArguments):
    """Arguments for inspecting training source code."""

    source_path: str = Field(
        min_length=1,
        description="Python source file path relative to the project root.",
    )


class LoadRunConfigArgs(ToolArguments):
    """Arguments for reading the configuration recorded with a run."""

    run_directory: str = Field(
        min_length=1,
        description="Run directory relative to the project root; must contain config.json.",
    )


class CompareRunsArgs(ToolArguments):
    """Arguments for comparing candidate telemetry against a healthy reference."""

    reference_run: str = Field(
        min_length=1,
        description="Healthy reference run directory relative to the project root.",
    )
    candidate_run: str = Field(
        min_length=1,
        description="Candidate run directory relative to the project root.",
    )
