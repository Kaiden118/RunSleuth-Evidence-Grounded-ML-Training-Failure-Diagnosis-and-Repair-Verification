import json
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from runsleuth import agent_repair, agent_repair_and_verify
from runsleuth.config import TrainingConfig
from runsleuth.diagnostic_tools import DiagnosticTools
from runsleuth.verification import RepairDecision


def write_run(
    run_directory: Path,
    config: TrainingConfig,
    *,
    accuracy: float,
    loss: float,
    update_norm: float = 0.05,
) -> Path:
    """Create synthetic experiment artifacts for offline verification."""
    run_directory.mkdir(parents=True)
    config.save(run_directory / "config.json")

    metrics = [
        {
            "epoch": epoch,
            "train_loss": loss,
            "train_accuracy": accuracy,
            "validation_loss": loss,
            "validation_accuracy": accuracy,
            "mean_gradient_norm": 1.0,
            "mean_parameter_update_norm": update_norm,
        }
        for epoch in range(1, config.epochs + 1)
    ]
    (run_directory / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in metrics),
        encoding="utf-8",
    )
    return run_directory


@pytest.fixture
def repair_case(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    source_path = Path("train.py")
    reference_run = Path("reference")
    failed_run = Path("candidate")
    report_path = Path("agent_report.json")

    source_path.write_text(
        "def train_one_epoch(optimizer, loss, optimizer_step_enabled):\n"
        "    optimizer.zero_grad()\n"
        "    loss.backward()\n"
        "    if optimizer_step_enabled:\n"
        "        optimizer.step()\n",
        encoding="utf-8",
    )

    clean_config = TrainingConfig(run_name="reference", device="cpu")
    failed_config = replace(
        clean_config,
        run_name="candidate",
        optimizer_step_enabled=False,
    )
    write_run(reference_run, clean_config, accuracy=0.90, loss=0.30)
    write_run(
        failed_run,
        failed_config,
        accuracy=0.10,
        loss=2.30,
        update_norm=0.0,
    )

    requests = [
        ("inspect_training_source", {"source_path": source_path.as_posix()}),
        ("load_run_config", {"run_directory": reference_run.as_posix()}),
        ("load_run_config", {"run_directory": failed_run.as_posix()}),
        (
            "compare_runs",
            {
                "reference_run": reference_run.as_posix(),
                "candidate_run": failed_run.as_posix(),
            },
        ),
    ]

    tools = DiagnosticTools(tmp_path)
    trace = []
    for index, (name, arguments) in enumerate(requests):
        result = tools.execute(name, arguments)
        assert result.ok, result.to_json()
        trace.append(
            {
                "call_id": f"call-{index}",
                "tool_name": name,
                "raw_arguments": json.dumps(arguments),
                "result": asdict(result),
            }
        )

    diagnosis = {
        "status": "failure_detected",
        "summary": "Optimizer updates are disabled in the candidate configuration.",
        "ranked_causes": [
            {
                "root_cause": "missing_optimizer_step",
                "confidence": "high",
                "explanation": "Updates are disabled and the recorded update norm is zero.",
                "evidence": [
                    {
                        "call_id": "call-2",
                        "path": ["config", "optimizer_step_enabled"],
                        "expected_value": False,
                    },
                    {
                        "call_id": "call-3",
                        "path": ["candidate", "final_epoch_parameter_update_norm"],
                        "expected_value": 0.0,
                    },
                ],
            }
        ],
        "uncertainties": ["Static source is not a historical execution trace."],
        "proposed_patch": {
            "config_call_id": "call-2",
            "field": "optimizer_step_enabled",
            "old_value": False,
            "new_value": True,
        },
    }
    report_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "model": "test-model",
                "pending_evidence": [],
                "diagnosis": diagnosis,
                "tool_trace": trace,
            }
        ),
        encoding="utf-8",
    )

    return {
        "agent_report_path": report_path,
        "source_path": source_path,
        "reference_run": reference_run,
        "failed_run": failed_run,
    }


@pytest.fixture
def training_spy(monkeypatch):
    training = Mock()
    monkeypatch.setattr(agent_repair, "run_experiment", training)
    return training


def test_agent_repair_caps_epochs_and_accepts_recovery(repair_case, training_spy):
    training_spy.side_effect = lambda config: write_run(
        Path("repaired"),
        config,
        accuracy=0.88,
        loss=0.33,
    )

    report = agent_repair.run_bounded_agent_repair(
        **repair_case,
        max_epochs=2,
    )

    training_spy.assert_called_once()
    training_config = training_spy.call_args.args[0]
    assert training_config.epochs == 2
    assert training_config.optimizer_step_enabled is True
    assert report.verification.decision is RepairDecision.ACCEPTED

    original = TrainingConfig.load(repair_case["failed_run"] / "config.json")
    assert original.optimizer_step_enabled is False
    assert training_config.seed == original.seed
    assert training_config.learning_rate == original.learning_rate

    saved = json.loads(report.to_json())
    assert saved["max_epochs"] == 2
    assert saved["diagnosis"]["proposed_patch"]["new_value"] is True


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_invalid_budget_stops_before_preparation(repair_case, training_spy, monkeypatch, budget):
    preparation = Mock()
    monkeypatch.setattr(agent_repair, "prepare_agent_repair", preparation)

    with pytest.raises(ValueError, match="max_epochs"):
        agent_repair.run_bounded_agent_repair(**repair_case, max_epochs=budget)

    preparation.assert_not_called()
    training_spy.assert_not_called()


@pytest.mark.parametrize("changed", ["config", "metrics", "source"])
def test_stale_evidence_prevents_training(repair_case, training_spy, changed):
    if changed == "config":
        path = repair_case["failed_run"] / "config.json"
        config = TrainingConfig.load(path)
        replace(config, optimizer_step_enabled=True).save(path)

    elif changed == "metrics":
        path = repair_case["failed_run"] / "metrics.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[0]["mean_parameter_update_norm"] = 0.01
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    else:
        path = repair_case["source_path"]
        source = path.read_text(encoding="utf-8")
        path.write_text(
            source.replace("if optimizer_step_enabled:", "if True:"),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="Saved evidence differs"):
        agent_repair.run_bounded_agent_repair(**repair_case, max_epochs=2)

    training_spy.assert_not_called()


def test_cli_saves_rejected_repair(repair_case, training_spy, monkeypatch):
    training_spy.side_effect = lambda config: write_run(
        Path("repaired"),
        config,
        accuracy=0.88,
        loss=0.90,
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "agent_repair_and_verify",
            "--agent-report",
            str(repair_case["agent_report_path"]),
            "--source",
            str(repair_case["source_path"]),
            "--reference-run",
            str(repair_case["reference_run"]),
            "--failed-run",
            str(repair_case["failed_run"]),
            "--max-epochs",
            "2",
        ],
    )

    with pytest.raises(SystemExit) as error:
        agent_repair_and_verify.main()

    assert error.value.code == 1
    training_spy.assert_called_once()

    report_path = Path("repaired/agent_repair_report.json")
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["verification"]["decision"] == "rejected"
    assert saved["repaired_config"]["optimizer_step_enabled"] is True
