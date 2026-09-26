"""Offline evaluation checks with synthetic run records; no API or training."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from runsleuth.agent_diagnosis import AgentDiagnosis
from runsleuth.config import TrainingConfig
from runsleuth.evaluate import evaluate_suite, main, read_report, score_diagnosis
from runsleuth.evaluation import SmokeCase, SmokeSuite
from runsleuth.verification import VerificationPolicy, verify_repair


def write_json(root: Path, relative_path: str, value: object) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_run(
    root: Path,
    relative_path: str,
    config: TrainingConfig,
    accuracy: float,
    loss: float,
    update_norm: float,
) -> None:
    write_json(root, f"{relative_path}/config.json", asdict(config))
    rows = [
        {
            "epoch": epoch,
            "train_loss": loss,
            "train_accuracy": accuracy,
            "validation_loss": loss,
            "validation_accuracy": accuracy,
            "mean_gradient_norm": 0.5,
            "mean_parameter_update_norm": update_norm,
        }
        for epoch in range(1, config.epochs + 1)
    ]
    (root / relative_path / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def make_agent_report(root: Path, case: SmokeCase) -> dict[str, object]:
    trace = []

    def add(call_id: str, name: str, arguments: dict, data: dict) -> None:
        trace.append(
            {
                "call_id": call_id,
                "tool_name": name,
                "raw_arguments": json.dumps(arguments),
                "result": {
                    "tool_name": name,
                    "ok": True,
                    "data": data,
                    "error_code": None,
                    "error_message": None,
                },
            }
        )

    add(
        "source",
        "inspect_training_source",
        {"source_path": "src/runsleuth/train.py"},
        {"source_path": "src/runsleuth/train.py", "calls": []},
    )
    for index, run in enumerate(dict.fromkeys([case.reference_run, case.candidate_run])):
        add(
            f"config-{index}",
            "load_run_config",
            {"run_directory": run},
            {"config": asdict(TrainingConfig.load(root / run / "config.json"))},
        )
    add(
        "comparison",
        "compare_runs",
        {"reference_run": case.reference_run, "candidate_run": case.candidate_run},
        {"candidate": {"final_epoch_parameter_update_norm": 0.0}},
    )
    diagnosis = {
        "status": "no_supported_failure",
        "summary": "Healthy self-comparison.",
        "ranked_causes": [],
        "uncertainties": [],
        "proposed_patch": None,
    }
    if case.kind == "injected_fault":
        diagnosis = {
            "status": "failure_detected",
            "summary": "Optimizer updates are disabled.",
            "ranked_causes": [
                {
                    "root_cause": "missing_optimizer_step",
                    "confidence": "high",
                    "explanation": "Recorded configuration disables parameter updates.",
                    "evidence": [
                        {
                            "call_id": "config-1",
                            "path": ["config", "optimizer_step_enabled"],
                            "expected_value": False,
                        }
                    ],
                }
            ],
            "uncertainties": ["This test uses synthetic recorded evidence."],
            "proposed_patch": {
                "config_call_id": "config-1",
                "field": "optimizer_step_enabled",
                "old_value": False,
                "new_value": True,
            },
        }
    return {
        "model": "test-model",
        "status": "completed",
        "pending_evidence": [],
        "tool_trace": trace,
        "diagnosis": AgentDiagnosis.model_validate(diagnosis).model_dump(mode="json"),
        "model_calls": 2,
        "tool_calls": len(trace),
        "input_tokens": 100,
        "output_tokens": 40,
        "usage_complete": True,
    }


@pytest.fixture
def smoke_suite(tmp_path: Path) -> SmokeSuite:
    reference = TrainingConfig(run_name="clean", device="cpu")
    failed = replace(reference, run_name="missing-step", optimizer_step_enabled=False)
    repaired = replace(reference, run_name="missing-step-agent-repair", epochs=2)
    write_run(tmp_path, "runs/reference", reference, 0.90, 0.30, 0.05)
    write_run(tmp_path, "runs/failed", failed, 0.10, 2.30, 0.0)
    write_run(tmp_path, "runs/repaired", repaired, 0.88, 0.33, 0.05)
    fault = SmokeCase(
        id="fault",
        kind="injected_fault",
        reference_run="runs/reference",
        candidate_run="runs/failed",
        expected_diagnosis_status="failure_detected",
        expected_root_cause="missing_optimizer_step",
        agent_report="reports/fault.json",
        repair_report="runs/repaired/agent_repair_report.json",
    )
    healthy = SmokeCase(
        id="healthy",
        kind="clean_self_comparison",
        reference_run="runs/reference",
        candidate_run="runs/reference",
        expected_diagnosis_status="no_supported_failure",
        expected_root_cause=None,
        agent_report="reports/healthy.json",
        repair_report=None,
    )
    fault_report = make_agent_report(tmp_path, fault)
    write_json(tmp_path, fault.agent_report, fault_report)
    write_json(tmp_path, healthy.agent_report, make_agent_report(tmp_path, healthy))
    verification = asdict(
        verify_repair(
            reference_run=tmp_path / fault.reference_run,
            failed_run=tmp_path / fault.candidate_run,
            repaired_run=tmp_path / "runs/repaired",
            policy=VerificationPolicy(
                min_validation_accuracy_gain=0.20,
                max_validation_accuracy_shortfall=0.03,
                max_validation_loss_ratio=1.25,
            ),
        )
    )
    # Saved relative Windows paths must match freshly computed absolute paths.
    verification.update(
        reference_run="runs\\reference",
        failed_run="runs\\failed",
        repaired_run="runs\\repaired",
    )
    write_json(
        tmp_path,
        fault.repair_report,
        {
            "agent_report": "reports\\fault.json",
            "model": "test-model",
            "diagnosis": fault_report["diagnosis"],
            "training_mode": "from_scratch",
            "max_epochs": 2,
            "repaired_config": asdict(repaired),
            "repaired_run": "runs\\repaired",
            "verification": verification,
        },
    )
    return SmokeSuite(
        suite="test_smoke",
        max_repair_epochs=2,
        notes=["Synthetic records used by offline tests."],
        cases=[fault, healthy],
    )


def test_suite_verifies_repair_and_healthy_abstention(tmp_path, smoke_suite):
    report = evaluate_suite(smoke_suite, tmp_path)
    summary = report["summary"]
    assert summary["passed_cases"] == 2
    assert summary["diagnosis_correct"] == {"numerator": 2, "denominator": 2, "rate": 1.0}
    assert summary["fault_top1_correct"]["denominator"] == 1
    assert summary["verified_fault_repairs"]["numerator"] == 1
    assert summary["reports_with_valid_citation_values"]["denominator"] == 1
    assert report["cases"][1]["diagnosis"]["citation_values_valid"] is None
    assert report["cases"][1]["repair"]["accepted"] is None
    assert report["selected_report_usage"]["input_tokens"] == 200
    assert report["selected_report_usage"]["tool_calls"] == 7
    assert report["selected_report_usage"]["usage_complete"] is True


def test_valid_wrong_root_is_not_a_schema_error(tmp_path, smoke_suite):
    case = smoke_suite.cases[0]
    agent = read_report(case.agent_report, tmp_path)
    agent["diagnosis"]["ranked_causes"][0]["root_cause"] = "high_learning_rate"
    agent["diagnosis"]["proposed_patch"] = None
    score = score_diagnosis(case, agent, source_path=smoke_suite.source_path)
    assert score["correct"] is False
    assert score["error"] is None
    assert score["citation_values_valid"] is True


@pytest.mark.parametrize("failure", ["missing", "api_error"])
def test_failed_attempts_stay_in_denominator(tmp_path, smoke_suite, failure):
    case = smoke_suite.cases[0]
    if failure == "missing":
        (tmp_path / case.agent_report).unlink()
    else:
        agent = read_report(case.agent_report, tmp_path)
        agent.update(status="api_error", usage_complete=False, input_tokens=12, output_tokens=3)
        write_json(tmp_path, case.agent_report, agent)
    report = evaluate_suite(smoke_suite, tmp_path)
    assert report["summary"]["diagnosis_correct"]["denominator"] == 2
    assert report["summary"]["verified_fault_repairs"]["denominator"] == 1
    assert report["summary"]["error_cases"] == 1
    assert report["selected_report_usage"]["usage_complete"] is False
    assert report["selected_report_usage"]["input_tokens"] == (
        112 if failure == "api_error" else 100
    )


@pytest.mark.parametrize("tamper", ["scope", "boolean_citation"])
def test_invalid_evidence_blocks_scoring(tmp_path, smoke_suite, tamper):
    case = smoke_suite.cases[0]
    agent = read_report(case.agent_report, tmp_path)
    if tamper == "scope":
        agent["tool_trace"][1]["raw_arguments"] = json.dumps({"run_directory": "runs/other"})
    else:
        agent["diagnosis"]["ranked_causes"][0]["evidence"][0]["expected_value"] = 0
    write_json(tmp_path, case.agent_report, agent)
    report = evaluate_suite(smoke_suite, tmp_path)
    row = report["cases"][0]
    assert row["outcome"] == "error"
    assert row["diagnosis"]["correct"] is False
    assert row["repair"]["accepted"] is None


@pytest.mark.parametrize("tamper", ["loss", "seed", "budget", "epochs", "report_link"])
def test_repair_tampering_is_not_accepted(tmp_path, smoke_suite, tamper):
    case = smoke_suite.cases[0]
    repair = read_report(case.repair_report, tmp_path)
    config = TrainingConfig.load(tmp_path / "runs/repaired/config.json")
    if tamper == "loss":
        write_run(tmp_path, "runs/repaired", config, 0.88, 0.90, 0.05)
    elif tamper == "seed":
        write_run(tmp_path, "runs/repaired", replace(config, seed=99), 0.88, 0.33, 0.05)
    elif tamper == "budget":
        repair["max_epochs"] = 3
    elif tamper == "epochs":
        path = tmp_path / "runs/repaired/metrics.jsonl"
        rows = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join(rows + [rows[-1]]) + "\n", encoding="utf-8")
    else:
        repair["agent_report"] = "reports/healthy.json"
    write_json(tmp_path, case.repair_report, repair)
    row = evaluate_suite(smoke_suite, tmp_path)["cases"][0]
    assert row["repair"]["accepted"] is False
    assert row["repair"]["error"] is not None
    assert row["outcome"] == "error"


def test_healthy_case_must_not_have_a_repair(tmp_path, smoke_suite):
    case = smoke_suite.cases[1]
    smoke_suite.cases[1] = case.model_copy(update={"repair_report": "unexpected.json"})
    row = evaluate_suite(smoke_suite, tmp_path)["cases"][1]
    assert row["diagnosis"]["correct"] is True
    assert row["repair"]["passed"] is False
    assert row["outcome"] == "error"


@pytest.mark.parametrize("path", ["../outside.json", "C:\\outside.json", "/outside.json"])
def test_report_paths_stay_inside_workspace(tmp_path, path):
    with pytest.raises(ValueError):
        read_report(path, tmp_path)


def test_cli_saves_error_summary_before_exiting(tmp_path, smoke_suite, monkeypatch):
    write_json(tmp_path, "evaluations/smoke_cases.json", smoke_suite.model_dump(mode="json"))
    (tmp_path / smoke_suite.cases[0].agent_report).unlink()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["runsleuth.evaluate"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1
    summaries = list((tmp_path / "artifacts/evaluations").glob("smoke-*.json"))
    assert len(summaries) == 1
    report = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert report["summary"]["total_cases"] == 2
    assert report["summary"]["error_cases"] == 1
