"""Tests for the controlled diagnostic tool interface."""

import json
from pathlib import Path

import pytest

from runsleuth.config import TrainingConfig
from runsleuth.diagnostic_tools import DiagnosticTools


@pytest.fixture
def tools(tmp_path: Path) -> DiagnosticTools:
    project = tmp_path / "project"
    project.mkdir()
    return DiagnosticTools(project)


def test_inspect_source_returns_json_ready_evidence(
    tools: DiagnosticTools,
) -> None:
    source = tools.project_root / "train.py"
    source.write_text(
        "def train():\n"
        "    loss.backward()\n"
        "    if optimizer_step_enabled:\n"
        "        optimizer.step()\n",
        encoding="utf-8",
    )

    result = tools.execute(
        "inspect_training_source",
        {"source_path": "train.py"},
    )

    assert result.ok, result.to_json()
    assert result.data is not None
    assert result.data["source_path"] == "train.py"

    calls = result.data["calls"]
    assert [item["call"] for item in calls] == [
        "loss.backward",
        "optimizer.step",
    ]
    assert calls[1]["conditions"] == ["optimizer_step_enabled"]
    assert json.loads(result.to_json())["data"] == result.data


def test_load_config_preserves_the_original_file(
    tools: DiagnosticTools,
) -> None:
    directory = tools.project_root / "run"
    directory.mkdir()
    config_path = directory / "config.json"
    TrainingConfig(
        run_name="fixture",
        optimizer_step_enabled=False,
    ).save(config_path)
    original = config_path.read_bytes()

    result = tools.execute("load_run_config", {"run_directory": "run"})

    assert result.ok, result.to_json()
    assert result.data is not None
    assert result.data["config_path"] == "run/config.json"
    assert result.data["config"]["run_name"] == "fixture"
    assert result.data["config"]["optimizer_step_enabled"] is False
    assert config_path.read_bytes() == original


def test_compare_runs_uses_recorded_telemetry(
    tools: DiagnosticTools,
) -> None:
    # Each row contains accuracy, loss, gradient norm, and update norm.
    runs = {
        "clean": [
            (0.8, 0.5, 2.0, 0.05),
            (0.9, 0.3, 1.0, 0.04),
        ],
        "candidate": [
            (0.2, 2.5, 3.0, 0.5),
            (0.1, 2.4, 0.1, 0.02),
        ],
    }

    for name, rows in runs.items():
        directory = tools.project_root / name
        directory.mkdir()
        TrainingConfig(run_name=name, epochs=2).save(directory / "config.json")

        records = []
        for epoch, (accuracy, loss, gradient, update) in enumerate(rows, start=1):
            records.append(
                {
                    "epoch": epoch,
                    "train_loss": loss,
                    "train_accuracy": accuracy,
                    "validation_loss": loss,
                    "validation_accuracy": accuracy,
                    "mean_gradient_norm": gradient,
                    "mean_parameter_update_norm": update,
                }
            )

        (directory / "metrics.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    result = tools.execute(
        "compare_runs",
        {"reference_run": "clean", "candidate_run": "candidate"},
    )

    assert result.ok, result.to_json()
    assert result.data is not None
    assert result.data["clean"]["run_directory"] == "clean"
    assert result.data["candidate"]["run_directory"] == "candidate"
    assert result.data["validation_accuracy_drop"] == pytest.approx(0.8)
    assert result.data["validation_loss_ratio"] == pytest.approx(8.0)
    assert result.data["first_epoch_update_norm_ratio"] == pytest.approx(10.0)
    assert result.data["final_epoch_gradient_norm_ratio"] == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("tool_name", "arguments", "expected_code"),
    [
        ("unknown", {}, "unknown_tool"),
        ("inspect_training_source", {}, "invalid_arguments"),
        ("inspect_training_source", {"source_path": 123}, "invalid_arguments"),
        ("inspect_training_source", {"source_path": "   "}, "invalid_arguments"),
        (
            "inspect_training_source",
            {"source_path": "train.py", "extra": True},
            "invalid_arguments",
        ),
        (
            "inspect_training_source",
            {"source_path": "missing.py"},
            "not_found",
        ),
    ],
)
def test_invalid_calls_return_structured_errors(
    tools: DiagnosticTools,
    tool_name: str,
    arguments: dict[str, object],
    expected_code: str,
) -> None:
    result = tools.execute(tool_name, arguments)

    assert not result.ok
    assert result.data is None
    assert result.error_code == expected_code
    assert result.error_message
    assert json.loads(result.to_json())["error_code"] == expected_code


@pytest.mark.parametrize(
    "source_path",
    [
        "../outside.py",
        "/outside.py",
        r"C:\outside.py",
        "C:outside.py",
        r"\outside.py",
    ],
)
def test_rejects_paths_outside_the_project_contract(
    tools: DiagnosticTools,
    source_path: str,
) -> None:
    result = tools.execute(
        "inspect_training_source",
        {"source_path": source_path},
    )

    assert not result.ok
    assert result.error_code == "invalid_path"


def test_malformed_config_returns_artifact_error(
    tools: DiagnosticTools,
) -> None:
    directory = tools.project_root / "broken"
    directory.mkdir()
    (directory / "config.json").write_text("{", encoding="utf-8")

    result = tools.execute("load_run_config", {"run_directory": "broken"})

    assert not result.ok
    assert result.error_code == "invalid_artifact"


def test_rejects_artifact_symlink_outside_project(
    tools: DiagnosticTools,
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.json"
    TrainingConfig().save(outside)

    directory = tools.project_root / "linked"
    directory.mkdir()
    try:
        (directory / "config.json").symlink_to(outside)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"Cannot create symlinks on this system: {error}")

    result = tools.execute("load_run_config", {"run_directory": "linked"})

    assert not result.ok
    assert result.error_code == "invalid_path"
