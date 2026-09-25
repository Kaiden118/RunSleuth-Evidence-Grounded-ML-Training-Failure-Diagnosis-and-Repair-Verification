"""Controlled, read-only access to diagnostic tools."""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path, PureWindowsPath

from pydantic import ValidationError

from runsleuth.compare import compare_runs
from runsleuth.config import TrainingConfig
from runsleuth.source_inspection import inspect_training_source
from runsleuth.tool_contracts import (
    CompareRunsArgs,
    InspectSourceArgs,
    LoadRunConfigArgs,
    ToolArguments,
)

LOGGER = logging.getLogger(__name__)

TOOL_SPECS: dict[str, tuple[type[ToolArguments], str]] = {
    "inspect_training_source": (
        InspectSourceArgs,
        "Inspect training calls and their enclosing conditions in a Python file.",
    ),
    "load_run_config": (
        LoadRunConfigArgs,
        "Read the configuration recorded with a training run.",
    ),
    "compare_runs": (
        CompareRunsArgs,
        "Compare candidate training telemetry against a healthy reference.",
    ),
}


@dataclass(frozen=True)
class ToolResult:
    """A consistent result envelope for every tool."""

    tool_name: str
    ok: bool
    data: dict[str, object] | None = None
    error_code: str | None = None
    error_message: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, allow_nan=False)


class InvalidToolPath(ValueError):
    """A requested path violates the tool's filesystem boundary."""


def failure(tool_name: str, code: str, message: str) -> ToolResult:
    return ToolResult(
        tool_name=tool_name,
        ok=False,
        error_code=code,
        error_message=message,
    )


class DiagnosticTools:
    """Execute registered tools within an application-selected project root."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve(strict=True)
        if not self.project_root.is_dir():
            raise ValueError("Project root must be a directory")

    def definitions(self) -> list[dict[str, object]]:
        """Describe available tools using their validated argument schemas."""
        return [
            {
                "name": name,
                "description": description,
                "parameters": argument_model.model_json_schema(),
            }
            for name, (argument_model, description) in TOOL_SPECS.items()
        ]

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, object],
    ) -> ToolResult:
        spec = TOOL_SPECS.get(tool_name)
        if spec is None:
            return failure(tool_name, "unknown_tool", "Tool is not registered")

        argument_model, _ = spec
        try:
            validated = argument_model.model_validate(arguments)
        except ValidationError as error:
            issue = error.errors(
                include_input=False,
                include_url=False,
                include_context=False,
            )[0]
            location = ".".join(str(part) for part in issue["loc"]) or "arguments"
            return failure(
                tool_name,
                "invalid_arguments",
                f"{location}: {issue['msg']}",
            )

        try:
            data = self._dispatch(validated)
        except InvalidToolPath as error:
            return failure(tool_name, "invalid_path", str(error))
        except FileNotFoundError:
            return failure(tool_name, "not_found", "Required file or directory is missing")
        except OSError:
            return failure(tool_name, "io_error", "Could not read the requested files")
        except (ValueError, SyntaxError):
            return failure(
                tool_name,
                "invalid_artifact",
                "Source code or recorded artifacts could not be parsed or validated",
            )
        except Exception:
            LOGGER.exception("Diagnostic tool failed: %s", tool_name)
            return failure(tool_name, "tool_error", "Unexpected tool execution failure")

        try:
            normalized = json.loads(json.dumps(data, allow_nan=False))
        except (TypeError, ValueError):
            LOGGER.exception("Tool produced invalid JSON data: %s", tool_name)
            return failure(tool_name, "tool_error", "Tool output is not valid JSON data")

        return ToolResult(tool_name=tool_name, ok=True, data=normalized)

    def _inside_project(self, path: Path) -> Path:
        resolved = path.resolve()
        if not resolved.is_relative_to(self.project_root):
            raise InvalidToolPath("Path must stay within the project root")
        return resolved

    def _input_path(self, value: str) -> Path:
        relative = Path(value.replace("\\", "/"))
        if relative.is_absolute() or PureWindowsPath(value).anchor:
            raise InvalidToolPath("Use a project-relative path")
        if "\0" in value:
            raise InvalidToolPath("Path contains an invalid character")
        return self._inside_project(self.project_root / relative)

    def _require_file(self, path: Path) -> Path:
        resolved = self._inside_project(path)
        if not resolved.exists():
            raise FileNotFoundError
        if not resolved.is_file():
            raise InvalidToolPath("Expected a regular file")
        return resolved

    def _run_directory(self, value: str) -> Path:
        directory = self._input_path(value)
        if not directory.exists():
            raise FileNotFoundError
        if not directory.is_dir():
            raise InvalidToolPath("Expected a run directory")
        return directory

    def _relative_name(self, path: Path) -> str:
        return path.relative_to(self.project_root).as_posix()

    def _dispatch(self, arguments: ToolArguments) -> dict[str, object]:
        if isinstance(arguments, InspectSourceArgs):
            source = self._require_file(self._input_path(arguments.source_path))
            if source.suffix.lower() != ".py":
                raise InvalidToolPath("Source inspection requires a Python file")

            data = asdict(inspect_training_source(source))
            data["source_path"] = self._relative_name(source)
            return data

        if isinstance(arguments, LoadRunConfigArgs):
            directory = self._run_directory(arguments.run_directory)
            config_path = self._require_file(directory / "config.json")
            config = TrainingConfig.load(config_path)
            return {
                "config_path": self._relative_name(config_path),
                "config": asdict(config),
            }

        if isinstance(arguments, CompareRunsArgs):
            reference = self._run_directory(arguments.reference_run)
            candidate = self._run_directory(arguments.candidate_run)

            self._require_file(reference / "metrics.jsonl")
            self._require_file(candidate / "metrics.jsonl")

            data = asdict(compare_runs(reference, candidate))
            data["clean"]["run_directory"] = self._relative_name(reference)
            data["candidate"]["run_directory"] = self._relative_name(candidate)
            return data

        raise TypeError("Argument model has no registered implementation")
