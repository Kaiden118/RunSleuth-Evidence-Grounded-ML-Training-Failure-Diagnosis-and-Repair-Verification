"""Model reevaluation orchestration tests without API calls or GPU training."""

import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

import runsleuth.reevaluate_batch as reevaluate
from runsleuth.batch_evaluate import BatchSettings
from runsleuth.config import TrainingConfig


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


@pytest.fixture
def model_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
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
            "model": "original-model",
            "status": "completed" if cause is None else "api_error",
            "diagnosis": {
                "status": "no_supported_failure",
                "ranked_causes": [],
                "proposed_patch": None,
            }
            if cause is None
            else None,
            "error": None if cause is None else "InternalServerError: no_error_code",
            "model_calls": 200,
            "tool_calls": 300,
            "input_tokens": 2000,
            "output_tokens": 500,
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
        "model": "original-model",
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

    state = SimpleNamespace(
        root=tmp_path,
        summary_path=summary_path,
        original=original,
        suite=suite,
        requests=[],
        repairs=[],
        evaluated=[],
        sleeps=[],
        client_model="new-model",
        closed=False,
        responses={},
        repair_rejected=False,
        client_calls=0,
    )

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            state.closed = True

    def client_factory():
        state.client_calls += 1
        return FakeClient(), state.client_model

    def diagnose(client, model, tools, **kwargs):
        state.requests.append(kwargs)
        case_id = Path(kwargs["candidate_run"]).name
        ledgers = list((tmp_path / "artifacts/model_evaluations").glob("*/progress.json"))
        assert len(ledgers) == 1
        ledger = json.loads(ledgers[0].read_text(encoding="utf-8"))
        assert ledger["attempts"][-1]["case_id"] == case_id
        assert ledger["attempts"][-1]["status"] == "running"
        assert ledger["attempts"][-1]["report_path"] is None
        for filename in ("first_attempt_manifest.json", "manifest.json"):
            planned = json.loads((ledgers[0].parent / filename).read_text(encoding="utf-8"))
            assert len(planned["cases"]) == 3
            assert all("/model_evaluations/" in row["agent_report"] for row in planned["cases"])
        responses = state.responses.setdefault(case_id, ["completed"])
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if response == "malformed":
            return SimpleNamespace(to_json=lambda: "[]")
        config = json.loads((tmp_path / kwargs["candidate_run"] / "config.json").read_text())
        healthy = kwargs["candidate_run"] == kwargs["reference_run"]
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
        if response == "wrong" or (healthy and response != "false_positive"):
            diagnosis = {
                "status": "no_supported_failure",
                "summary": "No supported fault",
                "ranked_causes": [],
                "uncertainties": [],
                "proposed_patch": None,
            }
        status = (
            "completed" if response in ("completed", "wrong", "false_positive") else "api_error"
        )
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
        path = tmp_path / directory / "runs" / f"repair-{len(state.repairs)}"
        path.mkdir(parents=True)
        decision = "rejected" if state.repair_rejected else "accepted"
        payload = {
            "agent_report": str(agent_report_path),
            "repaired_run": str(path),
            "verification": {"decision": decision},
        }
        return SimpleNamespace(
            repaired_run=str(path),
            verification=SimpleNamespace(decision=decision),
            to_json=lambda: json.dumps(payload),
        )

    def evaluate(manifest, workspace_root):
        state.evaluated.append(manifest.model_copy(deep=True))
        results = []
        for case in manifest.cases:
            try:
                payload = json.loads((workspace_root / case.agent_report).read_text())
                if not isinstance(payload, dict) or payload.get("status") != "completed":
                    outcome = "error"
                elif payload["diagnosis"]["status"] != case.expected_diagnosis_status:
                    outcome = "failed"
                elif case.kind == "injected_fault":
                    if case.repair_report is None:
                        outcome = "error"
                    else:
                        repaired = json.loads((workspace_root / case.repair_report).read_text())
                        outcome = (
                            "passed"
                            if repaired["verification"]["decision"] == "accepted"
                            else "failed"
                        )
                else:
                    outcome = "passed" if case.repair_report is None else "failed"
            except (OSError, ValueError, KeyError, TypeError):
                outcome = "error"
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

    monkeypatch.setattr(reevaluate, "create_llm_client", client_factory)
    monkeypatch.setattr(reevaluate, "DiagnosticTools", lambda root: object())
    monkeypatch.setattr(reevaluate, "run_diagnostic_agent", diagnose)
    monkeypatch.setattr(reevaluate, "run_bounded_agent_repair", repair)
    monkeypatch.setattr(reevaluate, "evaluate_suite", evaluate)
    monkeypatch.setattr(reevaluate.time, "sleep", state.sleeps.append)
    return state


def run_cases(state, **kwargs):
    options = {"expected_model": "new-model", "case_delay": 0, "retry_delay": 0, **kwargs}
    return reevaluate.run_reevaluation(state.summary_path, state.root, **options)


def test_plan_keeps_every_case_for_new_model(model_harness):
    state = model_harness
    plan = reevaluate.prepare_reevaluation(state.summary_path, state.root)
    assert plan["original_model"] == "original-model"
    assert [case.id for case in plan["suite"].cases] == ["case-001", "case-002", "case-003"]
    assert not state.client_calls


def test_all_cases_are_rerun_and_original_files_remain_unchanged(model_harness):
    state = model_harness
    original_dir = state.root / "artifacts/batches/original"
    before = {path: path.read_bytes() for path in original_dir.rglob("*") if path.is_file()}
    result = run_cases(state)
    assert all(path.read_bytes() == content for path, content in before.items())
    assert len(state.requests) == 3
    assert len(state.repairs) == 2
    assert result["first_attempt_evaluation"]["summary"]["passed_cases"] == 3
    assert result["final_evaluation"]["summary"]["passed_cases"] == 3
    assert result["model"] == "new-model"
    assert result["original_model"] == "original-model"
    assert result["budget"]["initial_training_runs"] == 0
    assert result["budget"]["agent_attempt_upper_bound"] == 3
    assert result["budget"]["repair_training_epoch_upper_bound"] == 6
    assert result["all_attempt_usage"]["model_calls"] == 6
    assert result["all_attempt_usage"]["input_tokens"] == 90
    assert result["all_attempt_usage"]["output_tokens"] == 21
    assert result["all_attempt_usage"]["usage_complete"] is True
    assert result["all_attempt_usage"]["attempt_report_count"] == 3
    assert state.closed
    assert all(request["max_epochs"] == 2 for request in state.repairs)
    for request in state.repairs:
        for key in ("agent_report_path", "source_path", "reference_run", "failed_run"):
            assert not Path(request[key]).is_absolute()
    assert all(request["source_path"] == "src/runsleuth/train.py" for request in state.requests)
    assert all("expected_root_cause" not in request for request in state.requests)
    for case in state.evaluated[-1].cases[1:]:
        assert "/batches/original/runs/repair-" in case.repair_report


def test_transient_retry_preserves_first_attempt_scores_and_all_usage(model_harness):
    state = model_harness
    state.responses["case-002"] = ["InternalServerError: no_error_code", "completed"]
    result = run_cases(state, max_attempts=2, retry_delay=10)
    assert len(state.requests) == 4
    assert len(state.repairs) == 2
    assert state.sleeps == [10]
    assert result["first_attempt_evaluation"]["summary"]["error_cases"] == 1
    assert result["final_evaluation"]["summary"]["passed_cases"] == 3
    assert result["all_attempt_usage"]["model_calls"] == 8
    assert result["all_attempt_usage"]["input_tokens"] == 120
    assert result["all_attempt_usage"]["usage_complete"] is False
    assert result["all_attempt_usage"]["attempt_report_count"] == 4
    first = json.loads(
        (state.root / result["directory"] / "first_attempt_manifest.json").read_text()
    )
    final = json.loads((state.root / result["directory"] / "manifest.json").read_text())
    assert first["cases"][1]["agent_report"].endswith("case-002-attempt-1.json")
    assert first["cases"][1]["repair_report"] is None
    assert final["cases"][1]["agent_report"].endswith("case-002-attempt-2.json")
    assert final["cases"][1]["repair_report"] is not None


def test_completed_wrong_answer_is_not_retried(model_harness):
    state = model_harness
    state.responses["case-002"] = ["wrong", "completed"]
    result = run_cases(state, max_attempts=3)
    assert len(state.requests) == 3
    assert len(state.repairs) == 1
    assert result["final_evaluation"]["summary"]["failed_cases"] == 1
    assert len(result["progress"]["attempts"]) == 3


@pytest.mark.parametrize(
    "error", ["RateLimitError: credit_balance_exhausted", "BadRequestError: invalid_model"]
)
def test_nontransient_errors_stay_in_denominator_without_retry(model_harness, error):
    state = model_harness
    state.responses["case-002"] = [error, "completed"]
    result = run_cases(state, max_attempts=3)
    assert len(state.requests) == 3
    assert result["final_evaluation"]["summary"]["total_cases"] == 3
    assert result["final_evaluation"]["summary"]["error_cases"] == 1


def test_transient_retries_stop_at_budget(model_harness):
    state = model_harness
    state.responses["case-002"] = ["APITimeoutError: no_error_code"] * 3
    result = run_cases(state, max_attempts=2)
    assert len(state.requests) == 4
    assert len(result["progress"]["attempts"]) == 4
    assert result["final_evaluation"]["summary"]["error_cases"] == 1
    assert result["all_attempt_usage"]["attempt_report_count"] == 4


def test_changed_actual_model_blocks_all_cases_without_reusing_old_success(model_harness):
    state = model_harness
    state.client_model = "unexpected-model"
    result = run_cases(state)
    assert not state.requests and not state.repairs
    assert result["progress"]["startup_error"]
    assert result["progress"]["actual_model"] == "unexpected-model"
    assert result["first_attempt_evaluation"]["summary"]["error_cases"] == 3
    assert result["final_evaluation"]["summary"]["passed_cases"] == 0
    assert result["all_attempt_usage"]["model_calls"] == 0
    assert result["all_attempt_usage"]["attempt_report_count"] == 0


def test_healthy_false_positive_reaches_repair_gate(model_harness):
    state = model_harness
    state.responses["case-001"] = ["false_positive"]
    result = run_cases(state)
    assert len(state.repairs) == 3
    assert result["final_evaluation"]["summary"]["failed_cases"] == 1
    assert state.evaluated[-1].cases[0].repair_report is not None


def test_rejected_repairs_do_not_trigger_more_diagnoses(model_harness):
    state = model_harness
    state.repair_rejected = True
    result = run_cases(state, max_attempts=3)
    assert len(state.requests) == 3
    assert len(state.repairs) == 2
    assert result["final_evaluation"]["summary"]["failed_cases"] == 2


@pytest.mark.parametrize("response", [RuntimeError("unexpected provider exception"), "malformed"])
def test_unrecorded_or_malformed_response_marks_usage_incomplete(model_harness, response):
    state = model_harness
    state.responses["case-002"] = [response, "completed"]
    result = run_cases(state, max_attempts=2)
    assert len(state.requests) == 3
    assert result["final_evaluation"]["summary"]["error_cases"] == 1
    assert result["all_attempt_usage"]["usage_complete"] is False
    assert result["progress"]["attempts"][1]["status"] == "exception"


def test_case_delay_is_applied_only_between_cases(model_harness):
    state = model_harness
    run_cases(state, case_delay=30)
    assert state.sleeps == [30, 30]


def test_dry_run_does_not_initialize_provider(model_harness, monkeypatch, capsys):
    state = model_harness

    def forbidden(*args, **kwargs):
        pytest.fail("Dry run attempted provider initialization or execution")

    monkeypatch.setattr(reevaluate, "create_llm_client", forbidden)
    monkeypatch.setattr(reevaluate, "run_reevaluation", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reevaluate_batch",
            "--batch-summary",
            state.summary_path,
            "--model",
            "new-model",
            "--dry-run",
        ],
    )
    reevaluate.main()
    assert "new-model" in capsys.readouterr().out
    assert not state.requests


@pytest.mark.parametrize(
    "options",
    [
        {"max_attempts": 0},
        {"max_attempts": 4},
        {"max_attempts": True},
        {"retry_delay": -1},
        {"retry_delay": 31},
        {"retry_delay": float("nan")},
        {"case_delay": -1},
        {"case_delay": 61},
        {"case_delay": True},
        {"expected_model": ""},
        {"expected_model": "   "},
    ],
)
def test_invalid_limits_fail_before_provider(model_harness, options):
    state = model_harness
    with pytest.raises(ValueError):
        run_cases(state, **options)
    assert state.client_calls == 0


def test_missing_training_artifact_fails_before_provider(model_harness):
    state = model_harness
    (state.root / state.suite["cases"][1]["candidate_run"] / "metrics.jsonl").unlink()
    with pytest.raises(ValueError):
        run_cases(state)
    assert state.client_calls == 0


def test_unfinished_training_fails_before_provider(model_harness):
    state = model_harness
    state.original["progress"]["cases"][1]["training_status"] = "error"
    write_json(state.root / state.summary_path, state.original)
    with pytest.raises(ValueError):
        run_cases(state)
    assert state.client_calls == 0


def test_case_set_mismatch_fails_before_provider(model_harness):
    state = model_harness
    state.original["evaluation"]["cases"].pop()
    write_json(state.root / state.summary_path, state.original)
    with pytest.raises(ValueError):
        run_cases(state)
    assert state.client_calls == 0


@pytest.mark.parametrize("case_id", ["../outside", "nested/case", "nested\\case"])
def test_unsafe_case_id_fails_before_provider(model_harness, case_id):
    state = model_harness
    state.suite["cases"][0]["id"] = case_id
    state.original["progress"]["cases"][0]["id"] = case_id
    state.original["evaluation"]["cases"][0]["id"] = case_id
    write_json(state.root / state.original["manifest_path"], state.suite)
    write_json(state.root / state.summary_path, state.original)
    with pytest.raises(ValueError):
        run_cases(state)
    assert state.client_calls == 0
