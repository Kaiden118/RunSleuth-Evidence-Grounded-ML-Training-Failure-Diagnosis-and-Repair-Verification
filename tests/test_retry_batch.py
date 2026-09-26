"""Recovery orchestration tests; no model API or GPU training is performed."""

import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

import runsleuth.retry_batch as retry
from runsleuth.batch_evaluate import BatchSettings
from runsleuth.config import TrainingConfig


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


@pytest.fixture
def recovery_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "src/runsleuth/train.py"
    source.parent.mkdir(parents=True)
    source.write_text("def train_one_epoch():\n    pass\n", encoding="utf-8")
    directory = "artifacts/batches/original"
    settings = BatchSettings(seeds=(7,))
    cases, stages, rows = [], [], []
    for index, cause in enumerate((None, "high_learning_rate", "missing_optimizer_step"), 1):
        case_id = f"case-{index:03d}"
        run = f"{directory}/runs/{case_id}"
        config = TrainingConfig(
            run_name=case_id,
            seed=7,
            output_dir=f"{directory}/runs",
            learning_rate=0.1 if cause == "high_learning_rate" else 0.001,
            optimizer_step_enabled=cause != "missing_optimizer_step",
        )
        write_json(tmp_path / run / "config.json", asdict(config))
        metrics = [
            {
                "epoch": epoch,
                "train_loss": 1.0,
                "train_accuracy": 0.5,
                "validation_loss": 1.0,
                "validation_accuracy": 0.5,
                "mean_gradient_norm": 1.0,
                "mean_parameter_update_norm": 0.0,
            }
            for epoch in range(1, config.epochs + 1)
        ]
        (tmp_path / run / "metrics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in metrics), encoding="utf-8"
        )
        report_path = f"{directory}/agent_reports/{case_id}.json"
        payload = {
            "model": "test-model",
            "status": "completed" if cause is None else "api_error",
            "diagnosis": {
                "status": "no_supported_failure",
                "summary": "Healthy",
                "ranked_causes": [],
                "uncertainties": [],
                "proposed_patch": None,
            }
            if cause is None
            else None,
            "error": None if cause is None else "InternalServerError: no_error_code",
            "model_calls": 2 if cause is None else 1,
            "tool_calls": 3 if cause is None else 0,
            "input_tokens": 20 if cause is None else 10,
            "output_tokens": 5 if cause is None else 2,
            "usage_complete": cause is None,
            "pending_evidence": [],
            "tool_trace": [],
        }
        write_json(tmp_path / report_path, payload)
        cases.append(
            {
                "id": case_id,
                "kind": "clean_self_comparison" if cause is None else "injected_fault",
                "reference_run": f"{directory}/runs/case-001",
                "candidate_run": run,
                "expected_diagnosis_status": "no_supported_failure"
                if cause is None
                else "failure_detected",
                "expected_root_cause": cause,
                "agent_report": report_path,
                "repair_report": None,
            }
        )
        stages.append(
            {
                "id": case_id,
                "seed": 7,
                "kind": cases[-1]["kind"],
                "training_status": "completed",
                "diagnosis_status": payload["status"],
                "repair_status": "not_requested",
                "errors": [],
            }
        )
        rows.append({"id": case_id, "outcome": "passed" if cause is None else "error"})

    manifest_path = f"{directory}/manifest.json"
    suite = {
        "suite": "multi_seed_paired_smoke",
        "max_repair_epochs": 2,
        "source_path": settings.source_path,
        "notes": [],
        "cases": cases,
    }
    write_json(tmp_path / manifest_path, suite)
    evaluation = {
        "summary": {"total_cases": 3, "passed_cases": 1, "failed_cases": 0, "error_cases": 2},
        "selected_report_usage": {},
        "cases": rows,
    }
    progress = {
        "model": "test-model",
        "startup_error": None,
        "reference_runs": [
            {"seed": 7, "status": "completed", "run": cases[0]["reference_run"], "error": None}
        ],
        "cases": stages,
    }
    original = {
        "batch_directory": directory,
        "manifest_path": manifest_path,
        "evaluation_path": f"{directory}/evaluation.json",
        "settings": asdict(settings),
        "budget": settings.describe_budget(),
        "progress": progress,
        "evaluation": evaluation,
    }
    summary_path = f"{directory}/batch_summary.json"
    write_json(tmp_path / summary_path, original)
    write_json(tmp_path / directory / "progress.json", progress)
    write_json(tmp_path / directory / "evaluation.json", evaluation)
    write_json(tmp_path / directory / "settings.json", {"settings": asdict(settings)})

    state = SimpleNamespace(
        root=tmp_path,
        summary_path=summary_path,
        original=original,
        suite=suite,
        requests=[],
        repairs=[],
        evaluated=[],
        sleeps=[],
        client_model="test-model",
        closed=False,
        responses={},
        repair_rejected=False,
    )

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            state.closed = True

    def diagnose(client, model, tools, **kwargs):
        state.requests.append(kwargs)
        case_id = Path(kwargs["candidate_run"]).name
        ledgers = list((tmp_path / "artifacts/recoveries").glob("*/progress.json"))
        assert len(ledgers) == 1
        ledger = json.loads(ledgers[0].read_text(encoding="utf-8"))
        assert ledger["attempts"][-1]["case_id"] == case_id
        assert ledger["attempts"][-1]["status"] == "running"
        assert ledger["attempts"][-1]["report_path"] is None
        responses = state.responses.setdefault(case_id, ["completed"])
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        config = json.loads((tmp_path / kwargs["candidate_run"] / "config.json").read_text())
        if not config["optimizer_step_enabled"]:
            cause, field, old, new = "missing_optimizer_step", "optimizer_step_enabled", False, True
        else:
            cause, field, old, new = (
                "high_learning_rate",
                "learning_rate",
                config["learning_rate"],
                0.0001,
            )
        diagnosis = {
            "status": "failure_detected",
            "summary": "Test proposal",
            "ranked_causes": [{"root_cause": cause}],
            "uncertainties": [],
            "proposed_patch": {
                "config_call_id": "c",
                "field": field,
                "old_value": old,
                "new_value": new,
            },
        }
        if response == "wrong":
            diagnosis = {
                "status": "no_supported_failure",
                "summary": "Wrong completed answer",
                "ranked_causes": [],
                "uncertainties": [],
                "proposed_patch": None,
            }
        status = "completed" if response in ("completed", "wrong") else "api_error"
        payload = {
            "model": model,
            "status": status,
            "error": None if status == "completed" else response,
            "diagnosis": diagnosis if status == "completed" else None,
            "model_calls": 2,
            "tool_calls": 4,
            "input_tokens": 30,
            "output_tokens": 7,
            "usage_complete": status == "completed",
            "pending_evidence": [],
            "tool_trace": [],
        }
        return SimpleNamespace(**payload, to_json=lambda: json.dumps(payload))

    def repair(agent_report_path, **kwargs):
        assert Path(agent_report_path).is_file()
        state.repairs.append({"agent_report_path": agent_report_path, **kwargs})
        directory_path = tmp_path / directory / "runs" / f"repair-{len(state.repairs)}"
        directory_path.mkdir(parents=True)
        decision = "rejected" if state.repair_rejected else "accepted"
        payload = {
            "agent_report": str(agent_report_path),
            "repaired_run": str(directory_path),
            "verification": {"decision": decision},
        }
        return SimpleNamespace(
            repaired_run=str(directory_path),
            verification=SimpleNamespace(decision=decision),
            to_json=lambda: json.dumps(payload),
        )

    def evaluate(manifest, workspace_root):
        state.evaluated.append(manifest.model_copy(deep=True))
        results = []
        for case in manifest.cases:
            payload = json.loads((workspace_root / case.agent_report).read_text())
            if payload["status"] != "completed":
                outcome = "error"
            elif payload["diagnosis"]["status"] != case.expected_diagnosis_status:
                outcome = "failed"
            elif case.kind == "injected_fault":
                if case.repair_report is None:
                    outcome = "error"
                else:
                    repaired = json.loads((workspace_root / case.repair_report).read_text())
                    outcome = (
                        "passed" if repaired["verification"]["decision"] == "accepted" else "failed"
                    )
            else:
                outcome = "passed" if case.repair_report is None else "failed"
            results.append({"id": case.id, "outcome": outcome})
        return {
            "summary": {
                "total_cases": len(results),
                **{
                    f"{kind}_cases": sum(row["outcome"] == kind for row in results)
                    for kind in ("passed", "failed", "error")
                },
            },
            "cases": results,
            "selected_report_usage": {},
        }

    monkeypatch.setattr(retry, "create_llm_client", lambda: (FakeClient(), state.client_model))
    monkeypatch.setattr(retry, "DiagnosticTools", lambda root: object())
    monkeypatch.setattr(retry, "run_diagnostic_agent", diagnose)
    monkeypatch.setattr(retry, "run_bounded_agent_repair", repair)
    monkeypatch.setattr(retry, "evaluate_suite", evaluate)
    monkeypatch.setattr(retry.time, "sleep", state.sleeps.append)
    return state


def test_plan_selects_only_original_transient_api_failures(recovery_harness):
    state = recovery_harness
    plan = retry.prepare_recovery(state.summary_path, state.root)
    assert plan["selected_ids"] == ["case-002", "case-003"]
    assert len(plan["suite"].cases) == 3
    assert plan["model"] == "test-model"
    assert not state.requests


@pytest.mark.parametrize("case_ids", [("unknown",), ("case-002", "case-002"), ("case-001",)])
def test_invalid_selection_fails_before_api(recovery_harness, case_ids):
    state = recovery_harness
    with pytest.raises(ValueError):
        retry.prepare_recovery(state.summary_path, state.root, case_ids=case_ids)
    assert not state.requests


def test_retry_preserves_originals_counts_all_attempts_and_keeps_denominator(recovery_harness):
    state = recovery_harness
    before = {
        path: path.read_bytes()
        for path in (state.root / "artifacts/batches/original").rglob("*.json")
    }
    state.responses["case-002"] = ["InternalServerError: no_error_code", "completed"]
    result = retry.run_recovery(state.summary_path, state.root, max_attempts=2, retry_delay=10.0)

    assert all(path.read_bytes() == content for path, content in before.items())
    assert result["original_summary"]["passed_cases"] == 1
    assert result["recovered_evaluation"]["summary"]["total_cases"] == 3
    assert result["recovered_evaluation"]["summary"]["passed_cases"] == 3
    assert len(state.requests) == 3
    assert len(state.repairs) == 2
    assert len(result["attempts"]) == 3
    assert state.sleeps == [10.0]
    assert state.closed
    for attempt in result["attempts"]:
        assert (state.root / attempt["report_path"]).is_file()
    assert result["cumulative_usage"]["model_calls"] == 10
    assert result["cumulative_usage"]["input_tokens"] == 130
    assert result["cumulative_usage"]["output_tokens"] == 30
    assert result["cumulative_usage"]["usage_complete"] is False
    unchanged = state.evaluated[-1].cases[0]
    assert unchanged.agent_report == state.suite["cases"][0]["agent_report"]
    assert all(request["max_epochs"] == 2 for request in state.repairs)
    for case in state.evaluated[-1].cases[1:]:
        assert "/batches/original/runs/repair-" in case.repair_report


def test_completed_wrong_answer_is_not_retried(recovery_harness):
    state = recovery_harness
    state.responses["case-002"] = ["wrong", "completed"]
    result = retry.run_recovery(
        state.summary_path, state.root, case_ids=("case-002",), retry_delay=0
    )
    assert len(state.requests) == 1
    assert not state.repairs
    assert result["attempts"][0]["status"] == "completed"
    assert result["recovered_evaluation"]["summary"]["failed_cases"] == 1
    assert result["recovered_evaluation"]["summary"]["total_cases"] == 3


def test_nontransient_api_response_is_saved_without_retry(recovery_harness):
    state = recovery_harness
    state.responses["case-002"] = ["RateLimitError: credit_balance_exhausted", "completed"]
    result = retry.run_recovery(
        state.summary_path, state.root, case_ids=("case-002",), retry_delay=0
    )
    assert len(state.requests) == 1
    assert len(result["attempts"]) == 1
    assert result["attempts"][0]["status"] == "api_error"
    assert not state.repairs


def test_retry_limit_keeps_every_failed_attempt(recovery_harness):
    state = recovery_harness
    state.responses["case-002"] = [
        "APIConnectionError: no_error_code",
        "APITimeoutError: no_error_code",
        "completed",
    ]
    result = retry.run_recovery(
        state.summary_path,
        state.root,
        case_ids=("case-002",),
        max_attempts=2,
        retry_delay=0,
    )
    assert len(state.requests) == 2
    assert len(result["attempts"]) == 2
    assert not state.repairs
    assert result["recovered_evaluation"]["summary"]["error_cases"] == 2
    assert result["cumulative_usage"]["attempt_report_count"] == 5


def test_unrecorded_exception_marks_usage_incomplete_and_stops_case(recovery_harness):
    state = recovery_harness
    state.responses["case-002"] = [RuntimeError("unexpected provider exception"), "completed"]
    result = retry.run_recovery(
        state.summary_path, state.root, case_ids=("case-002",), retry_delay=0
    )
    assert len(state.requests) == 1
    assert result["attempts"][0]["status"] == "exception"
    assert result["attempts"][0]["report_path"] is None
    assert result["attempts"][0]["error"]
    assert result["cumulative_usage"]["usage_complete"] is False
    assert not state.repairs


def test_changed_model_is_rejected_before_agent_invocation(recovery_harness):
    state = recovery_harness
    state.client_model = "different-model"
    result = retry.run_recovery(state.summary_path, state.root, retry_delay=0)
    assert not state.requests
    assert not state.repairs
    assert result["progress"]["startup_error"]
    assert result["recovered_evaluation"]["summary"]["passed_cases"] == 1


def test_rejected_repair_does_not_trigger_another_diagnosis(recovery_harness):
    state = recovery_harness
    state.repair_rejected = True
    result = retry.run_recovery(
        state.summary_path, state.root, case_ids=("case-002",), retry_delay=0
    )
    assert len(state.requests) == len(state.repairs) == 1
    assert result["recovered_evaluation"]["summary"]["failed_cases"] == 1


def test_healthy_false_positive_is_not_suppressed_by_case_label(recovery_harness):
    state = recovery_harness
    healthy = state.suite["cases"][0]
    report_path = state.root / healthy["agent_report"]
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload.update(status="api_error", error="APITimeoutError: no_error_code", diagnosis=None)
    write_json(report_path, payload)
    state.original["progress"]["cases"][0]["diagnosis_status"] = "api_error"
    state.original["evaluation"]["cases"][0]["outcome"] = "error"
    state.original["evaluation"]["summary"].update(passed_cases=0, error_cases=3)
    write_json(state.root / state.summary_path, state.original)

    result = retry.run_recovery(
        state.summary_path, state.root, case_ids=("case-001",), retry_delay=0
    )
    assert len(state.requests) == 1
    assert len(state.repairs) == 1
    assert state.evaluated[-1].cases[0].repair_report is not None
    assert result["recovered_evaluation"]["summary"]["failed_cases"] == 1


def test_dry_run_reads_plan_without_starting_recovery(recovery_harness, monkeypatch, capsys):
    state = recovery_harness

    def forbidden(*args, **kwargs):
        pytest.fail("Dry run attempted provider initialization or recovery")

    monkeypatch.setattr(retry, "create_llm_client", forbidden)
    monkeypatch.setattr(retry, "run_recovery", forbidden)
    monkeypatch.setattr(
        sys, "argv", ["retry_batch", "--batch-summary", state.summary_path, "--dry-run"]
    )
    retry.main()
    output = capsys.readouterr().out
    assert "case-002" in output and "case-003" in output
    assert not state.requests


@pytest.mark.parametrize("max_attempts", [0, -1, True, 1.5, 4])
def test_invalid_retry_budget_fails_before_api(recovery_harness, max_attempts):
    state = recovery_harness
    with pytest.raises(ValueError):
        retry.run_recovery(state.summary_path, state.root, max_attempts=max_attempts)
    assert not state.requests


@pytest.mark.parametrize("delay", [-1, 31, float("nan"), float("inf"), True])
def test_invalid_delay_fails_before_api(recovery_harness, delay):
    state = recovery_harness
    with pytest.raises(ValueError):
        retry.run_recovery(state.summary_path, state.root, retry_delay=delay)
    assert not state.requests


def test_mismatched_case_sets_fail_before_api(recovery_harness):
    state = recovery_harness
    state.original["evaluation"]["cases"].pop()
    write_json(state.root / state.summary_path, state.original)
    with pytest.raises(ValueError):
        retry.prepare_recovery(state.summary_path, state.root)
    assert not state.requests


def test_incomplete_training_is_not_eligible_for_diagnosis_retry(recovery_harness):
    state = recovery_harness
    state.original["progress"]["cases"][1]["training_status"] = "error"
    write_json(state.root / state.summary_path, state.original)
    with pytest.raises(ValueError):
        retry.prepare_recovery(state.summary_path, state.root, case_ids=("case-002",))
    assert not state.requests
