"""Run paired multi-seed training, bounded Agent diagnosis, and repair evaluation."""

import argparse
import json
import os
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from runsleuth.agent import run_diagnostic_agent
from runsleuth.agent_repair import run_bounded_agent_repair
from runsleuth.config import TrainingConfig
from runsleuth.diagnostic_tools import DiagnosticTools
from runsleuth.evaluate import _workspace_path, evaluate_suite
from runsleuth.evaluation import SmokeCase, SmokeSuite
from runsleuth.experiment import run_experiment
from runsleuth.llm_client import create_llm_client


@dataclass(frozen=True)
class BatchSettings:
    """Explicit limits for one sequential batch; every case gets one Agent attempt."""

    seeds: tuple[int, ...] = (7, 123, 2026)
    epochs: int = 3
    max_repair_epochs: int = 2
    max_model_calls: int = 6
    max_tool_calls: int = 5
    max_output_tokens: int = 2000
    device: str = "cuda"
    source_path: str = "src/runsleuth/train.py"
    output_dir: str = "artifacts/batches"

    def __post_init__(self) -> None:
        if not self.seeds or any(
            type(seed) is not int or not 0 <= seed <= 2**32 - 1 for seed in self.seeds
        ):
            raise ValueError("Seeds must be nonempty integers between 0 and 2**32 - 1")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Seeds must be unique")
        for field in (
            "epochs",
            "max_repair_epochs",
            "max_model_calls",
            "max_tool_calls",
            "max_output_tokens",
        ):
            value = getattr(self, field)
            if type(value) is not int or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("Device must be auto, cpu, or cuda")
        for field in ("source_path", "output_dir"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a nonempty relative path")
            object.__setattr__(self, field, value.replace("\\", "/"))

    def describe_budget(self) -> dict[str, int]:
        cases = len(self.seeds) * 3
        # All three predictions can request a repair, including a false positive.
        return {
            "case_count": cases,
            "initial_training_runs": len(self.seeds) * 3,
            "agent_attempt_upper_bound": cases,
            "repair_attempt_upper_bound": cases,
            "training_epoch_upper_bound": len(self.seeds)
            * (3 * self.epochs + 3 * min(self.epochs, self.max_repair_epochs)),
            "model_call_upper_bound": cases * self.max_model_calls,
            "tool_call_upper_bound": cases * self.max_tool_calls,
        }


def _save_text(path: Path, text: str) -> None:
    """Replace one progress file atomically; batch directories are always new."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text.rstrip("\n") + "\n", encoding="utf-8")
    temporary.replace(path)


def _save_json(path: Path, value: object) -> None:
    _save_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def _error_record(error: Exception) -> dict[str, str]:
    message = str(error)
    for name in ("GEMINI_API_KEY", "OPENAI_API_KEY"):
        secret = os.environ.get(name)
        if secret:
            message = message.replace(secret, "<redacted>")
    return {"type": type(error).__name__, "message": message[:1000]}


def _relative_run(path: Path, root: Path, batch_directory: Path) -> str:
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(batch_directory / "runs") or not resolved.is_dir():
        raise ValueError("Training must return a run directory inside this batch")
    return resolved.relative_to(root).as_posix()


def _make_plan(settings: BatchSettings, batch_path: str) -> tuple[SmokeSuite, dict[str, object]]:
    cases, records, references = [], [], []
    for seed in settings.seeds:
        pending_reference = f"{batch_path}/runs/pending-reference-{seed}"
        references.append(
            {"seed": seed, "status": "planned", "run": pending_reference, "error": None}
        )
        for cause in (None, "high_learning_rate", "missing_optimizer_step"):
            case_id = f"case-{len(cases) + 1:03d}"
            healthy = cause is None
            kind = "clean_self_comparison" if healthy else "injected_fault"
            cases.append(
                SmokeCase(
                    id=case_id,
                    kind=kind,
                    reference_run=pending_reference,
                    candidate_run=(
                        pending_reference if healthy else f"{batch_path}/runs/pending-{case_id}"
                    ),
                    expected_diagnosis_status=(
                        "no_supported_failure" if healthy else "failure_detected"
                    ),
                    expected_root_cause=cause,
                    agent_report=f"{batch_path}/agent_reports/{case_id}.json",
                    repair_report=None,
                )
            )
            records.append(
                {
                    "id": case_id,
                    "seed": seed,
                    "kind": kind,
                    "training_status": "planned",
                    "diagnosis_status": "planned",
                    "repair_status": "planned",
                    "errors": [],
                }
            )
    suite = SmokeSuite(
        suite="multi_seed_paired_smoke",
        max_repair_epochs=settings.max_repair_epochs,
        source_path=settings.source_path,
        notes=[
            "Reference, faults, and repairs are paired by seed within each group.",
            "Changing seed changes the split, model initialization, and training order together.",
            "Healthy cases are self-comparisons, not independent healthy controls.",
            "All planned cases remain in the denominator, including API and training failures.",
            "Each case gets one Agent invocation; the batch runner does not retry cases.",
            "Expected labels are used for generation and scoring, not passed to the Agent.",
            "Configuration exposes the injected changes; this is not a blinded benchmark.",
        ],
        cases=cases,
    )
    return suite, {
        "model": None,
        "startup_error": None,
        "reference_runs": references,
        "cases": records,
    }


def run_batch(settings: BatchSettings, workspace_root: Path) -> dict[str, object]:
    """Execute a fresh batch, saving every attempt before proceeding to the next."""
    root = workspace_root.resolve()
    if root != Path.cwd().resolve():
        raise ValueError("Run the batch from the workspace root")
    if not _workspace_path(settings.source_path, root).is_file():
        raise ValueError("Training source file does not exist")
    output_dir = _workspace_path(settings.output_dir, root)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    batch_directory = output_dir / timestamp
    batch_directory.mkdir(parents=True)
    batch_path = batch_directory.relative_to(root).as_posix()
    suite, progress = _make_plan(settings, batch_path)
    manifest_path = batch_directory / "manifest.json"

    def persist() -> None:
        _save_json(manifest_path, suite.model_dump(mode="json"))
        _save_json(batch_directory / "progress.json", progress)

    _save_json(
        batch_directory / "settings.json",
        {"settings": asdict(settings), "budget": settings.describe_budget()},
    )
    persist()
    print(f"batch_directory={batch_path}", flush=True)
    try:
        client, model = create_llm_client()
    except Exception as error:
        progress["startup_error"] = _error_record(error)
        for reference in progress["reference_runs"]:
            reference["status"] = "blocked"
        for record in progress["cases"]:
            record.update(
                training_status="blocked", diagnosis_status="blocked", repair_status="blocked"
            )
        persist()
    else:
        progress["model"] = model
        with client:
            tools = DiagnosticTools(root)
            for seed_index, seed in enumerate(settings.seeds):
                reference = progress["reference_runs"][seed_index]
                reference_config = TrainingConfig(
                    run_name=f"seed-{seed}-reference",
                    seed=seed,
                    epochs=settings.epochs,
                    device=settings.device,
                    output_dir=f"{batch_path}/runs",
                )
                start = seed_index * 3
                group = list(
                    zip(
                        suite.cases[start : start + 3],
                        progress["cases"][start : start + 3],
                        strict=True,
                    )
                )
                reference["status"] = "running"
                persist()
                try:
                    reference_path = _relative_run(
                        run_experiment(reference_config), root, batch_directory
                    )
                except Exception as error:
                    reference.update(status="error", error=_error_record(error))
                    for _, record in group:
                        record.update(
                            training_status="blocked",
                            diagnosis_status="blocked",
                            repair_status="blocked",
                        )
                    persist()
                    continue
                reference.update(status="completed", run=reference_path)
                for case, _record in group:
                    case.reference_run = reference_path
                    if case.kind == "clean_self_comparison":
                        case.candidate_run = reference_path
                persist()
                for case, record in group:
                    print(f"case={case.id}, seed={seed}", flush=True)
                    record["training_status"] = "running"
                    persist()
                    try:
                        if case.kind == "injected_fault":
                            candidate_config = replace(reference_config, run_name=case.id)
                            if case.expected_root_cause == "high_learning_rate":
                                candidate_config = replace(
                                    candidate_config,
                                    learning_rate=reference_config.learning_rate * 100.0,
                                )
                            else:
                                candidate_config = replace(
                                    candidate_config, optimizer_step_enabled=False
                                )
                            case.candidate_run = _relative_run(
                                run_experiment(candidate_config), root, batch_directory
                            )
                        record["training_status"] = "completed"
                    except Exception as error:
                        record.update(
                            training_status="error",
                            diagnosis_status="blocked",
                            repair_status="blocked",
                        )
                        record["errors"].append({"stage": "training", **_error_record(error)})
                        persist()
                        continue
                    record["diagnosis_status"] = "running"
                    persist()
                    try:
                        agent_report = run_diagnostic_agent(
                            client,
                            model,
                            tools,
                            source_path=settings.source_path,
                            reference_run=case.reference_run,
                            candidate_run=case.candidate_run,
                            max_model_calls=settings.max_model_calls,
                            max_tool_calls=settings.max_tool_calls,
                            max_output_tokens=settings.max_output_tokens,
                        )
                        text = agent_report.to_json()
                        _save_text(root / case.agent_report, text)
                        payload = json.loads(text)
                        if not isinstance(payload, dict) or not isinstance(
                            payload.get("status"), str
                        ):
                            raise ValueError("Agent returned an invalid report")
                        record["diagnosis_status"] = payload["status"]
                    except Exception as error:
                        record.update(diagnosis_status="error", repair_status="blocked")
                        record["errors"].append({"stage": "diagnosis", **_error_record(error)})
                        persist()
                        continue
                    persist()
                    diagnosis = payload.get("diagnosis")
                    wants_repair = (
                        payload["status"] == "completed"
                        and isinstance(diagnosis, dict)
                        and diagnosis.get("status") == "failure_detected"
                        and diagnosis.get("proposed_patch") is not None
                    )
                    if not wants_repair:
                        record["repair_status"] = "not_requested"
                        persist()
                        continue
                    # The model's proposal controls this gate; expected labels do not.
                    record["repair_status"] = "running"
                    persist()
                    try:
                        repair = run_bounded_agent_repair(
                            Path(case.agent_report),
                            source_path=Path(settings.source_path),
                            reference_run=Path(case.reference_run),
                            failed_run=Path(case.candidate_run),
                            max_epochs=settings.max_repair_epochs,
                        )
                        repaired_run = _relative_run(
                            Path(repair.repaired_run), root, batch_directory
                        )
                        repair_path = f"{repaired_run}/agent_repair_report.json"
                        _save_text(root / repair_path, repair.to_json())
                        case.repair_report = repair_path
                        record["repair_status"] = "completed"
                    except Exception as error:
                        record["repair_status"] = "error"
                        record["errors"].append({"stage": "repair", **_error_record(error)})
                    persist()
    evaluation = evaluate_suite(suite, root)
    evaluation["evaluation_type"] = "multi_seed_paired_smoke"
    evaluation_path = batch_directory / "evaluation.json"
    _save_json(evaluation_path, evaluation)
    summary = {
        "batch_directory": batch_path,
        "manifest_path": manifest_path.relative_to(root).as_posix(),
        "evaluation_path": evaluation_path.relative_to(root).as_posix(),
        "settings": asdict(settings),
        "budget": settings.describe_budget(),
        "progress": progress,
        "evaluation": evaluation,
    }
    _save_json(batch_directory / "batch_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 123, 2026])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-repair-epochs", type=int, default=2)
    parser.add_argument("--max-model-calls", type=int, default=6)
    parser.add_argument("--max-tool-calls", type=int, default=5)
    parser.add_argument("--max-output-tokens", type=int, default=2000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--source", default="src/runsleuth/train.py")
    parser.add_argument("--output-dir", default="artifacts/batches")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show limits without training or API calls."
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        settings = BatchSettings(
            seeds=tuple(args.seeds),
            epochs=args.epochs,
            max_repair_epochs=args.max_repair_epochs,
            max_model_calls=args.max_model_calls,
            max_tool_calls=args.max_tool_calls,
            max_output_tokens=args.max_output_tokens,
            device=args.device,
            source_path=args.source,
            output_dir=args.output_dir,
        )
    except ValueError as error:
        parser.error(str(error))
    print(
        json.dumps({"settings": asdict(settings), "budget": settings.describe_budget()}, indent=2)
    )
    if args.dry_run:
        return
    result = run_batch(settings, Path.cwd())
    print(f"batch_summary={result['batch_directory']}/batch_summary.json")
    print(f"evaluation_report={result['evaluation_path']}")
    rows = {row["id"]: row for row in result["evaluation"].get("cases", [])}
    for record in result["progress"]["cases"]:
        row = rows.get(record["id"], {})
        print(
            f"case={record['id']}, seed={record['seed']}, "
            f"outcome={row.get('outcome', 'unknown')}, "
            f"diagnosis={record['diagnosis_status']}, repair={record['repair_status']}"
        )
    print(json.dumps(result["evaluation"]["summary"], indent=2))
    usage = result["evaluation"].get("selected_report_usage")
    if usage is not None:
        print(json.dumps(usage, indent=2))
    summary = result["evaluation"]["summary"]
    if summary["passed_cases"] != summary["total_cases"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
