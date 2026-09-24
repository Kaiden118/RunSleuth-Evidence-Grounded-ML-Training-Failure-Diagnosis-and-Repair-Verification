import json
from pathlib import Path

import pytest

from runsleuth.config import TrainingConfig
from runsleuth.diagnose import diagnose_training_failure
from runsleuth.diagnosis import DiagnosisStatus, RootCause

MetricRecord = dict[str, int | float]


def _healthy_metrics() -> list[MetricRecord]:
    return [
        {
            "epoch": 1,
            "train_loss": 0.50,
            "train_accuracy": 0.80,
            "validation_loss": 0.40,
            "validation_accuracy": 0.85,
            "mean_gradient_norm": 2.0,
            "mean_parameter_update_norm": 0.05,
        },
        {
            "epoch": 2,
            "train_loss": 0.25,
            "train_accuracy": 0.91,
            "validation_loss": 0.25,
            "validation_accuracy": 0.90,
            "mean_gradient_norm": 1.5,
            "mean_parameter_update_norm": 0.04,
        },
    ]


def _high_lr_metrics() -> list[MetricRecord]:
    return [
        {
            "epoch": 1,
            "train_loss": 3.0,
            "train_accuracy": 0.10,
            "validation_loss": 2.4,
            "validation_accuracy": 0.10,
            "mean_gradient_norm": 2.2,
            "mean_parameter_update_norm": 0.50,
        },
        {
            "epoch": 2,
            "train_loss": 2.3,
            "train_accuracy": 0.10,
            "validation_loss": 2.5,
            "validation_accuracy": 0.10,
            "mean_gradient_norm": 0.10,
            "mean_parameter_update_norm": 0.06,
        },
    ]


def _write_metrics(run_directory: Path, metrics: list[MetricRecord]) -> None:
    payload = "".join(f"{json.dumps(record)}\n" for record in metrics)
    (run_directory / "metrics.jsonl").write_text(payload, encoding="utf-8")


def _write_run(
    run_directory: Path,
    source_config_path: Path,
    config: TrainingConfig,
    metrics: list[MetricRecord],
) -> None:
    run_directory.mkdir(parents=True)
    source_config_path.parent.mkdir(parents=True, exist_ok=True)

    config.save(source_config_path)
    config.save(run_directory / "config.json")
    _write_metrics(run_directory, metrics)


def test_diagnoses_high_learning_rate_and_proposes_minimal_patch(
    tmp_path: Path,
) -> None:
    reference_run = tmp_path / "reference-run"
    candidate_run = tmp_path / "candidate-run"
    reference_config_path = tmp_path / "configs" / "clean.json"
    candidate_config_path = tmp_path / "configs" / "high-lr.json"

    reference_config = TrainingConfig(
        run_name="clean",
        learning_rate=0.001,
        device="cpu",
    )
    candidate_config = TrainingConfig(
        run_name="high-lr",
        learning_rate=0.1,
        device="cpu",
    )

    _write_run(
        reference_run,
        reference_config_path,
        reference_config,
        _healthy_metrics(),
    )
    _write_run(
        candidate_run,
        candidate_config_path,
        candidate_config,
        _high_lr_metrics(),
    )

    report = diagnose_training_failure(
        reference_run=reference_run,
        candidate_run=candidate_run,
        reference_config_path=reference_config_path,
        candidate_config_path=candidate_config_path,
    )

    assert report.status is DiagnosisStatus.FAILURE_DETECTED
    assert len(report.ranked_causes) == 1

    cause = report.ranked_causes[0]
    assert cause.rank == 1
    assert cause.root_cause is RootCause.HIGH_LEARNING_RATE
    assert cause.confidence == 0.95

    patch = cause.proposed_patch
    assert patch.field == "learning_rate"
    assert patch.old_value == 0.1
    assert patch.new_value == 0.001
    assert '-  "learning_rate": 0.1' in patch.unified_diff
    assert '+  "learning_rate": 0.001' in patch.unified_diff

    changed_lines = [
        line
        for line in patch.unified_diff.splitlines()
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    ]
    assert changed_lines == [
        '-  "learning_rate": 0.1,',
        '+  "learning_rate": 0.001,',
    ]


def test_does_not_report_failure_for_healthy_candidate(tmp_path: Path) -> None:
    reference_run = tmp_path / "reference-run"
    candidate_run = tmp_path / "candidate-run"
    reference_config_path = tmp_path / "configs" / "reference.json"
    candidate_config_path = tmp_path / "configs" / "candidate.json"

    reference_config = TrainingConfig(
        run_name="reference",
        learning_rate=0.001,
        device="cpu",
    )
    candidate_config = TrainingConfig(
        run_name="candidate",
        learning_rate=0.001,
        device="cpu",
    )

    _write_run(
        reference_run,
        reference_config_path,
        reference_config,
        _healthy_metrics(),
    )
    _write_run(
        candidate_run,
        candidate_config_path,
        candidate_config,
        _healthy_metrics(),
    )

    report = diagnose_training_failure(
        reference_run=reference_run,
        candidate_run=candidate_run,
        reference_config_path=reference_config_path,
        candidate_config_path=candidate_config_path,
    )

    assert report.status is DiagnosisStatus.NO_SUPPORTED_FAILURE
    assert report.ranked_causes == ()


def test_rejects_source_config_that_does_not_match_recorded_run(
    tmp_path: Path,
) -> None:
    reference_run = tmp_path / "reference-run"
    candidate_run = tmp_path / "candidate-run"
    reference_config_path = tmp_path / "configs" / "reference.json"
    candidate_config_path = tmp_path / "configs" / "candidate.json"

    reference_config = TrainingConfig(
        run_name="reference",
        learning_rate=0.001,
        device="cpu",
    )
    candidate_source_config = TrainingConfig(
        run_name="candidate",
        learning_rate=0.1,
        device="cpu",
    )
    candidate_recorded_config = TrainingConfig(
        run_name="candidate",
        learning_rate=0.01,
        device="cpu",
    )

    _write_run(
        reference_run,
        reference_config_path,
        reference_config,
        _healthy_metrics(),
    )

    candidate_run.mkdir(parents=True)
    candidate_config_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_source_config.save(candidate_config_path)
    candidate_recorded_config.save(candidate_run / "config.json")
    _write_metrics(candidate_run, _high_lr_metrics())

    with pytest.raises(ValueError, match="does not match"):
        diagnose_training_failure(
            reference_run=reference_run,
            candidate_run=candidate_run,
            reference_config_path=reference_config_path,
            candidate_config_path=candidate_config_path,
        )


@pytest.fixture
def missing_step_case(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "reference_run": tmp_path / "reference-run",
        "candidate_run": tmp_path / "candidate-run",
        "reference_config_path": tmp_path / "configs" / "clean.json",
        "candidate_config_path": tmp_path / "configs" / "missing-step.json",
    }

    healthy_metrics = _healthy_metrics()
    healthy_metrics.append({**healthy_metrics[-1], "epoch": 3})

    failed_metrics = [
        {
            "epoch": epoch,
            "train_loss": 2.3,
            "train_accuracy": 0.1,
            "validation_loss": 2.3,
            "validation_accuracy": 0.1,
            "mean_gradient_norm": 0.6,
            "mean_parameter_update_norm": 0.0,
        }
        for epoch in range(1, 4)
    ]

    _write_run(
        paths["reference_run"],
        paths["reference_config_path"],
        TrainingConfig(
            run_name="clean",
            learning_rate=0.001,
            optimizer_step_enabled=True,
            device="cpu",
        ),
        healthy_metrics,
    )
    _write_run(
        paths["candidate_run"],
        paths["candidate_config_path"],
        TrainingConfig(
            run_name="missing-step",
            learning_rate=0.001,
            optimizer_step_enabled=False,
            device="cpu",
        ),
        failed_metrics,
    )

    return paths


def test_diagnoses_missing_step_and_proposes_boolean_patch(
    missing_step_case: dict[str, Path],
) -> None:
    config_path = missing_step_case["candidate_config_path"]
    original_text = config_path.read_text(encoding="utf-8")

    report = diagnose_training_failure(**missing_step_case)

    assert report.status is DiagnosisStatus.FAILURE_DETECTED
    assert len(report.ranked_causes) == 1

    cause = report.ranked_causes[0]
    assert cause.root_cause is RootCause.MISSING_OPTIMIZER_STEP

    patch = cause.proposed_patch
    assert patch.field == "optimizer_step_enabled"
    assert patch.old_value is False
    assert patch.new_value is True

    changed_lines = [
        line
        for line in patch.unified_diff.splitlines()
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    ]
    assert changed_lines == [
        '-  "optimizer_step_enabled": false,',
        '+  "optimizer_step_enabled": true,',
    ]
    assert config_path.read_text(encoding="utf-8") == original_text


@pytest.mark.parametrize(
    ("metric", "value"),
    [
        ("mean_parameter_update_norm", 0.01),
        ("mean_gradient_norm", 0.0),
    ],
)
def test_missing_step_requires_consistent_evidence_in_every_epoch(
    missing_step_case: dict[str, Path],
    metric: str,
    value: float,
) -> None:
    metrics_path = missing_step_case["candidate_run"] / "metrics.jsonl"
    records = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    records[1][metric] = value
    _write_metrics(missing_step_case["candidate_run"], records)

    report = diagnose_training_failure(**missing_step_case)

    assert report.status is DiagnosisStatus.NO_SUPPORTED_FAILURE
    assert report.ranked_causes == ()
