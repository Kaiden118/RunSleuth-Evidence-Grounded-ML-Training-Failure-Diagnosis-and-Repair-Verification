"""Batch orchestration checks without GPU training or network requests."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import runsleuth.batch_evaluate as batch


@pytest.fixture
def batch_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "src/runsleuth/train.py"
    source.parent.mkdir(parents=True)
    source.write_text("def train_one_epoch():\n    pass\n", encoding="utf-8")

    state = SimpleNamespace(
        root=tmp_path,
        training_configs=[],
        agent_requests=[],
        repair_requests=[],
        evaluated_suites=[],
        initial_manifests=[],
        training_failure=lambda config: False,
        api_failure=lambda config: False,
        repair_failure=lambda config: False,
        false_positive=False,
        startup_failure=False,
        client_closed=False,
    )

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            state.client_closed = True

    def create_client():
        assert not state.training_configs
        assert list(tmp_path.rglob("progress.json"))
        for path in tmp_path.rglob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and "suite" in payload and "cases" in payload:
                state.initial_manifests.append(payload)
        if state.startup_failure:
            raise RuntimeError("Provider configuration is unavailable")
        return FakeClient(), "test-model"

    def train(config):
        state.training_configs.append(config)
        if state.training_failure(config):
            raise RuntimeError("Injected training exception")
        directory = Path(config.output_dir) / config.run_name
        directory.mkdir(parents=True, exist_ok=False)
        config.save(directory / "config.json")
        return directory

    def diagnose(client, model, tools, **kwargs):
        state.agent_requests.append(kwargs)
        config = json.loads(
            (Path(kwargs["candidate_run"]) / "config.json").read_text(encoding="utf-8")
        )
        if not config["optimizer_step_enabled"]:
            cause, field, old_value, new_value = (
                "missing_optimizer_step",
                "optimizer_step_enabled",
                False,
                True,
            )
        elif config["learning_rate"] > 0.001:
            cause, field, old_value, new_value = (
                "high_learning_rate",
                "learning_rate",
                config["learning_rate"],
                0.001,
            )
        elif state.false_positive:
            cause, field, old_value, new_value = (
                "high_learning_rate",
                "learning_rate",
                0.001,
                0.0001,
            )
        else:
            cause = None

        diagnosis = {
            "status": "failure_detected" if cause else "no_supported_failure",
            "summary": "Controlled orchestration test response",
            "ranked_causes": [{"root_cause": cause}] if cause else [],
            "uncertainties": [],
            "proposed_patch": {
                "config_call_id": "config-call",
                "field": field,
                "old_value": old_value,
                "new_value": new_value,
            }
            if cause
            else None,
        }
        failed = state.api_failure(SimpleNamespace(**config))
        payload = {
            "model": model,
            "status": "api_error" if failed else "completed",
            "diagnosis": None if failed else diagnosis,
            "error": "Injected provider failure" if failed else None,
            "model_calls": 1 if failed else 2,
            "tool_calls": 0 if failed else 4,
            "input_tokens": 0 if failed else 20,
            "output_tokens": 0 if failed else 10,
            "usage_complete": not failed,
        }
        return SimpleNamespace(**payload, to_json=lambda: json.dumps(payload))

    def repair(agent_report_path, **kwargs):
        state.repair_requests.append({"agent_report_path": agent_report_path, **kwargs})
        config = json.loads(
            (Path(kwargs["failed_run"]) / "config.json").read_text(encoding="utf-8")
        )
        if state.repair_failure(SimpleNamespace(**config)):
            raise RuntimeError("Injected repair exception")
        directory = Path(config["output_dir"]) / f"repair-{len(state.repair_requests)}"
        directory.mkdir(parents=True)
        payload = {
            "agent_report": str(agent_report_path),
            "repaired_run": directory.as_posix(),
            "verification": {"decision": "accepted"},
        }
        return SimpleNamespace(
            repaired_run=directory.as_posix(),
            verification=SimpleNamespace(decision="accepted"),
            to_json=lambda: json.dumps(payload),
        )

    def evaluate(suite, workspace_root):
        state.evaluated_suites.append(suite)
        outcomes = []
        for case in suite.cases:
            report_path = workspace_root / case.agent_report
            if not report_path.is_file():
                outcomes.append("error")
                continue
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report["status"] != "completed":
                outcomes.append("error")
            elif case.kind == "clean_self_comparison":
                outcomes.append("passed" if case.repair_report is None else "failed")
            elif case.repair_report is None or not (workspace_root / case.repair_report).is_file():
                outcomes.append("error")
            else:
                outcomes.append("passed")
        return {
            "summary": {
                "total_cases": len(outcomes),
                "passed_cases": outcomes.count("passed"),
                "failed_cases": outcomes.count("failed"),
                "error_cases": outcomes.count("error"),
            },
            "selected_report_usage": {},
            "cases": [],
        }

    monkeypatch.setattr(batch, "create_llm_client", create_client)
    monkeypatch.setattr(batch, "DiagnosticTools", lambda root: object())
    monkeypatch.setattr(batch, "run_experiment", train)
    monkeypatch.setattr(batch, "run_diagnostic_agent", diagnose)
    monkeypatch.setattr(batch, "run_bounded_agent_repair", repair)
    monkeypatch.setattr(batch, "evaluate_suite", evaluate)
    return state


def read_manifest(result: dict) -> dict:
    return json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "overrides",
    [
        {"seeds": ()},
        {"seeds": (7, 7)},
        {"seeds": (True,)},
        {"seeds": (-1,)},
        {"seeds": (2**32,)},
        {"epochs": 0},
        {"max_repair_epochs": True},
        {"max_model_calls": 0},
        {"max_tool_calls": -1},
        {"max_output_tokens": 1.5},
    ],
)
def test_settings_reject_invalid_budgets_and_seeds(overrides):
    with pytest.raises(ValueError):
        batch.BatchSettings(**overrides)


def test_budget_includes_possible_repairs_for_healthy_false_positives():
    settings = batch.BatchSettings(seeds=(7, 123), epochs=3, max_repair_epochs=2)
    budget = settings.describe_budget()
    assert budget["case_count"] == 6
    assert budget["training_epoch_upper_bound"] == 30
    assert budget["model_call_upper_bound"] == 36
    assert budget["tool_call_upper_bound"] == 30


def test_dry_run_does_not_initialize_provider_or_start_work(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("Dry run attempted provider initialization or batch execution")

    monkeypatch.setattr(batch, "create_llm_client", forbidden)
    monkeypatch.setattr(batch, "run_batch", forbidden)
    monkeypatch.setattr(sys, "argv", ["batch_evaluate", "--dry-run"])
    batch.main()
    budget = json.loads(capsys.readouterr().out)["budget"]
    assert budget["case_count"] == 9
    assert budget["training_epoch_upper_bound"] == 45


def test_batch_keeps_seeds_paired_and_declares_cases_before_api(batch_harness):
    state = batch_harness
    settings = batch.BatchSettings(seeds=(7, 123))
    result = batch.run_batch(settings, workspace_root=state.root)

    assert len(state.initial_manifests) == 1
    assert len(state.initial_manifests[0]["cases"]) == 6
    assert len(state.training_configs) == 6
    for seed in settings.seeds:
        configs = [config for config in state.training_configs if config.seed == seed]
        assert len(configs) == 3
        assert sum(config.learning_rate == 0.1 for config in configs) == 1
        assert sum(config.optimizer_step_enabled is False for config in configs) == 1
        for config in configs:
            assert config.epochs == settings.epochs
    assert len(state.agent_requests) == 6
    assert len(state.repair_requests) == 4
    assert state.client_closed
    for request in state.agent_requests:
        reference = json.loads(
            (Path(request["reference_run"]) / "config.json").read_text(encoding="utf-8")
        )
        candidate = json.loads(
            (Path(request["candidate_run"]) / "config.json").read_text(encoding="utf-8")
        )
        assert reference["seed"] == candidate["seed"]
        assert request["max_model_calls"] == settings.max_model_calls
        assert request["max_tool_calls"] == settings.max_tool_calls
        assert request["max_output_tokens"] == settings.max_output_tokens
    assert all(request["max_epochs"] == 2 for request in state.repair_requests)
    assert result["evaluation"]["summary"]["passed_cases"] == 6
    assert len(read_manifest(result)["cases"]) == 6


def test_reference_training_failure_retains_cases_and_continues_next_seed(batch_harness):
    state = batch_harness
    state.training_failure = lambda config: config.seed == 7
    result = batch.run_batch(batch.BatchSettings(seeds=(7, 123)), workspace_root=state.root)

    blocked = [case for case in result["progress"]["cases"] if case["seed"] == 7]
    assert len(blocked) == 3
    assert all(case["training_status"] == "blocked" for case in blocked)
    reference = next(item for item in result["progress"]["reference_runs"] if item["seed"] == 7)
    assert reference["status"] == "error"
    assert reference["error"]
    assert len(state.agent_requests) == 3
    assert len(read_manifest(result)["cases"]) == 6
    assert result["evaluation"]["summary"]["total_cases"] == 6
    assert result["evaluation"]["summary"]["error_cases"] == 3


def test_fault_training_failure_does_not_stop_other_cases(batch_harness):
    state = batch_harness
    state.training_failure = lambda config: config.learning_rate == 0.1
    result = batch.run_batch(batch.BatchSettings(seeds=(7,)), workspace_root=state.root)

    failed = [case for case in result["progress"]["cases"] if case["training_status"] == "error"]
    assert len(failed) == 1
    assert failed[0]["errors"]
    assert len(state.agent_requests) == 2
    assert len(state.repair_requests) == 1
    assert len(read_manifest(result)["cases"]) == 3
    assert result["evaluation"]["summary"]["error_cases"] == 1


def test_api_error_report_is_saved_and_counted_without_retry(batch_harness):
    state = batch_harness
    state.api_failure = lambda config: config.learning_rate == 0.1
    result = batch.run_batch(batch.BatchSettings(seeds=(7,)), workspace_root=state.root)

    manifest = read_manifest(result)
    case = next(
        case for case in manifest["cases"] if case["expected_root_cause"] == "high_learning_rate"
    )
    saved = json.loads(Path(case["agent_report"]).read_text(encoding="utf-8"))
    assert saved["status"] == "api_error"
    assert saved["usage_complete"] is False
    assert saved["model_calls"] == 1
    assert len(state.agent_requests) == 3
    assert len(state.repair_requests) == 1
    assert result["evaluation"]["summary"]["total_cases"] == 3
    assert result["evaluation"]["summary"]["error_cases"] == 1


def test_repair_exception_preserves_diagnosis_and_continues(batch_harness):
    state = batch_harness
    state.repair_failure = lambda config: config.learning_rate == 0.1
    result = batch.run_batch(batch.BatchSettings(seeds=(7,)), workspace_root=state.root)

    failed = [case for case in result["progress"]["cases"] if case["repair_status"] == "error"]
    assert len(failed) == 1
    assert failed[0]["errors"]
    manifest = read_manifest(result)
    case = next(
        case for case in manifest["cases"] if case["expected_root_cause"] == "high_learning_rate"
    )
    assert Path(case["agent_report"]).is_file()
    assert len(state.agent_requests) == 3
    assert len(state.repair_requests) == 2
    assert result["evaluation"]["summary"]["passed_cases"] == 2
    assert result["evaluation"]["summary"]["error_cases"] == 1


def test_healthy_false_positive_repair_is_not_suppressed_using_ground_truth(batch_harness):
    state = batch_harness
    state.false_positive = True
    result = batch.run_batch(batch.BatchSettings(seeds=(7,)), workspace_root=state.root)

    manifest = read_manifest(result)
    healthy = next(case for case in manifest["cases"] if case["kind"] == "clean_self_comparison")
    assert len(state.repair_requests) == 3
    assert healthy["repair_report"] is not None
    assert Path(healthy["repair_report"]).is_file()
    assert result["evaluation"]["summary"]["failed_cases"] == 1


def test_provider_initialization_failure_still_produces_full_evaluation(batch_harness):
    state = batch_harness
    state.startup_failure = True
    result = batch.run_batch(batch.BatchSettings(seeds=(7,)), workspace_root=state.root)

    assert len(state.initial_manifests) == 1
    assert len(state.initial_manifests[0]["cases"]) == 3
    assert result["progress"]["startup_error"]
    assert not state.training_configs
    assert not state.agent_requests
    assert not state.repair_requests
    assert len(read_manifest(result)["cases"]) == 3
    assert result["evaluation"]["summary"]["error_cases"] == 3


def test_cli_returns_nonzero_when_any_planned_case_did_not_pass(monkeypatch):
    result = {
        "evaluation": {
            "summary": {"total_cases": 3, "passed_cases": 2, "failed_cases": 0, "error_cases": 1},
            "selected_report_usage": {},
        },
        "progress": {"cases": [], "reference_runs": []},
        "manifest_path": "artifacts/batches/test/manifest.json",
        "evaluation_path": "artifacts/batches/test/evaluation.json",
        "batch_directory": "artifacts/batches/test",
    }
    monkeypatch.setattr(batch, "run_batch", lambda *args, **kwargs: result)
    monkeypatch.setattr(sys, "argv", ["batch_evaluate"])
    with pytest.raises(SystemExit) as raised:
        batch.main()
    assert raised.value.code == 1
