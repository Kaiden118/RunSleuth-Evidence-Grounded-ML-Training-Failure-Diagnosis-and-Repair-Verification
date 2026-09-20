import json
from pathlib import Path

import pytest

import runsleuth.repair as repair_module
from runsleuth.config import TrainingConfig
from runsleuth.verification import RepairDecision, verify_repair

MetricRecord = dict[str, int | float]


def _metrics(
    *,
    final_accuracy: float,
    final_loss: float,
    first_update_norm: float,
    final_gradient_norm: float,
) -> list[MetricRecord]:
    return [
        {
            "epoch": 1,
            "train_loss": final_loss + 0.5,
            "train_accuracy": max(0.0, final_accuracy - 0.05),
            "validation_loss": final_loss + 0.2,
            "validation_accuracy": max(0.0, final_accuracy - 0.05),
            "mean_gradient_norm": 2.0,
            "mean_parameter_update_norm": first_update_norm,
        },
        {
            "epoch": 2,
            "train_loss": final_loss,
            "train_accuracy": final_accuracy,
            "validation_loss": final_loss,
            "validation_accuracy": final_accuracy,
            "mean_gradient_norm": final_gradient_norm,
            "mean_parameter_update_norm": 0.04,
        },
    ]


def _healthy_metrics() -> list[MetricRecord]:
    return _metrics(
        final_accuracy=0.90,
        final_loss=0.25,
        first_update_norm=0.05,
        final_gradient_norm=1.5,
    )


def _failed_metrics() -> list[MetricRecord]:
    return _metrics(
        final_accuracy=0.10,
        final_loss=2.50,
        first_update_norm=0.50,
        final_gradient_norm=0.10,
    )


def _recovered_metrics() -> list[MetricRecord]:
    return _metrics(
        final_accuracy=0.88,
        final_loss=0.30,
        first_update_norm=0.05,
        final_gradient_norm=1.3,
    )


def _write_metrics(
    run_directory: Path,
    metrics: list[MetricRecord],
) -> None:
    run_directory.mkdir(parents=True, exist_ok=True)
    payload = "".join(f"{json.dumps(record)}\n" for record in metrics)
    (run_directory / "metrics.jsonl").write_text(
        payload,
        encoding="utf-8",
    )


def _write_configured_run(
    run_directory: Path,
    source_config_path: Path,
    config: TrainingConfig,
    metrics: list[MetricRecord],
) -> None:
    run_directory.mkdir(parents=True, exist_ok=True)
    source_config_path.parent.mkdir(parents=True, exist_ok=True)

    config.save(source_config_path)
    config.save(run_directory / "config.json")
    _write_metrics(run_directory, metrics)


def test_verification_rejects_when_loss_has_not_recovered(
    tmp_path: Path,
) -> None:
    reference_run = tmp_path / "reference-run"
    failed_run = tmp_path / "failed-run"
    repaired_run = tmp_path / "repaired-run"

    _write_metrics(reference_run, _healthy_metrics())
    _write_metrics(failed_run, _failed_metrics())
    _write_metrics(
        repaired_run,
        _metrics(
            final_accuracy=0.88,
            final_loss=0.60,
            first_update_norm=0.05,
            final_gradient_norm=1.3,
        ),
    )

    report = verify_repair(
        reference_run=reference_run,
        failed_run=failed_run,
        repaired_run=repaired_run,
    )

    checks = {check.metric: check.passed for check in report.checks}

    assert report.decision is RepairDecision.REJECTED
    assert checks["validation_accuracy_gain"]
    assert checks["validation_accuracy_shortfall"]
    assert not checks["validation_loss_ratio_to_reference"]


def test_bounded_repair_caps_epochs_and_accepts_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference_run = tmp_path / "reference-run"
    failed_run = tmp_path / "failed-run"
    repaired_run = tmp_path / "repaired-run"
    reference_config_path = tmp_path / "configs" / "clean.json"
    failed_config_path = tmp_path / "configs" / "high-lr.json"

    reference_config = TrainingConfig(
        run_name="clean",
        epochs=3,
        learning_rate=0.001,
        device="cpu",
    )
    failed_config = TrainingConfig(
        run_name="high-lr",
        epochs=3,
        learning_rate=0.1,
        device="cpu",
    )

    _write_configured_run(
        reference_run,
        reference_config_path,
        reference_config,
        _healthy_metrics(),
    )
    _write_configured_run(
        failed_run,
        failed_config_path,
        failed_config,
        _failed_metrics(),
    )

    captured_configs: list[TrainingConfig] = []

    def fake_run_experiment(config: TrainingConfig) -> Path:
        captured_configs.append(config)
        repaired_run.mkdir(parents=True, exist_ok=True)
        config.save(repaired_run / "config.json")
        _write_metrics(repaired_run, _recovered_metrics())
        return repaired_run

    monkeypatch.setattr(
        repair_module,
        "run_experiment",
        fake_run_experiment,
    )

    report = repair_module.run_bounded_repair(
        reference_run=reference_run,
        failed_run=failed_run,
        reference_config_path=reference_config_path,
        failed_config_path=failed_config_path,
        max_epochs=2,
    )

    assert len(captured_configs) == 1

    repaired_config = captured_configs[0]
    assert repaired_config.run_name == "high-lr-repair"
    assert repaired_config.epochs == 2
    assert repaired_config.learning_rate == 0.001

    original_config = TrainingConfig.load(failed_config_path)
    assert original_config.learning_rate == 0.1
    assert original_config.epochs == 3

    assert report.verification.decision is RepairDecision.ACCEPTED
    assert all(check.passed for check in report.verification.checks)
