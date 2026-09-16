from dataclasses import asdict
from pathlib import Path

import pytest

from runsleuth.compare import compare_runs
from runsleuth.config import TrainingConfig
from runsleuth.telemetry import EpochMetrics, append_epoch_metrics


def make_metrics(
    epoch: int,
    validation_accuracy: float,
    validation_loss: float,
    gradient_norm: float,
    update_norm: float,
) -> EpochMetrics:
    return EpochMetrics(
        epoch=epoch,
        train_loss=1.0,
        train_accuracy=0.5,
        validation_loss=validation_loss,
        validation_accuracy=validation_accuracy,
        mean_gradient_norm=gradient_norm,
        mean_parameter_update_norm=update_norm,
    )


def test_high_lr_config_changes_only_controlled_fields() -> None:
    project_root = Path(__file__).resolve().parents[1]
    clean = TrainingConfig.load(project_root / "configs" / "clean.json")
    candidate = TrainingConfig.load(
        project_root / "configs" / "high_learning_rate.json"
    )

    clean_values = asdict(clean)
    candidate_values = asdict(candidate)
    changed_fields = {
        name
        for name, clean_value in clean_values.items()
        if clean_value != candidate_values[name]
    }

    assert changed_fields == {"run_name", "learning_rate"}
    assert clean.learning_rate == pytest.approx(0.001)
    assert candidate.learning_rate == pytest.approx(0.1)


def test_compare_runs_calculates_evidence_ratios(tmp_path: Path) -> None:
    clean_run = tmp_path / "clean"
    candidate_run = tmp_path / "candidate"

    append_epoch_metrics(
        clean_run / "metrics.jsonl",
        make_metrics(1, 0.80, 0.50, 2.5, 0.05),
    )
    append_epoch_metrics(
        clean_run / "metrics.jsonl",
        make_metrics(2, 0.90, 0.25, 1.5, 0.04),
    )
    append_epoch_metrics(
        candidate_run / "metrics.jsonl",
        make_metrics(1, 0.10, 3.00, 2.7, 0.50),
    )
    append_epoch_metrics(
        candidate_run / "metrics.jsonl",
        make_metrics(2, 0.10, 2.50, 0.1, 0.06),
    )

    comparison = compare_runs(clean_run, candidate_run)

    assert comparison.validation_accuracy_drop == pytest.approx(0.80)
    assert comparison.validation_loss_ratio == pytest.approx(10.0)
    assert comparison.first_epoch_update_norm_ratio == pytest.approx(10.0)
    assert comparison.final_epoch_gradient_norm_ratio == pytest.approx(1 / 15)
