"""Prepare an evidence-bound controller repair; execute only with an explicit flag."""

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath

from runsleuth.camelyon_optimizer_probe import file_sha256, load_reference
from runsleuth.camelyon_optimizer_training import (
    REPAIR_POLICY,
    assess_rebind_training,
    run_optimizer_training,
)
from runsleuth.optimizer_agent_diagnosis import (
    OptimizerDiagnosis,
    validate_optimizer_diagnosis_evidence,
)
from runsleuth.optimizer_evidence import (
    MAX_FILE_BYTES,
    _data_identity,
    _parse,
    _same,
    inspect_optimizer_run,
)

VARIANTS = ("clean", "stale_head", "stale_head_repaired")


def _scoped(root: Path, path: Path, *, file: bool = False) -> Path:
    if PureWindowsPath(str(path)).drive and not path.is_absolute():
        raise ValueError("Foreign or drive-relative Windows paths are not supported")
    resolved = (path if path.is_absolute() else root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Path must stay inside the workspace boundary")
    if file and not resolved.is_file():
        raise ValueError(f"Required file is missing: {path}")
    return resolved


def _read(path: Path) -> dict:
    with path.open("rb") as stream:
        payload = stream.read(MAX_FILE_BYTES + 1)
    if len(payload) > MAX_FILE_BYTES:
        raise ValueError("JSON artifact exceeds the input size limit")
    result = _parse(payload.decode("utf-8-sig"))
    json.dumps(result, allow_nan=False)  # Also rejects overflow such as 1e999.
    return result


def _write(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _initial_state_hash(checkpoint: Path) -> str:
    import torch

    from runsleuth.optimizer_probe import state_dict_sha256

    return state_dict_sha256(torch.load(checkpoint, map_location="cpu", weights_only=True))


def _replay_trace(trace: object, root: Path, allowed: dict[str, Path]) -> dict[str, dict]:
    if not isinstance(trace, list) or not trace:
        raise ValueError("A nonempty tool trace is required")
    runs, call_ids = {}, set()
    for entry in trace:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("tool_name"), str)
            or not isinstance(entry.get("raw_arguments"), str)
        ):
            raise ValueError("Malformed tool trace entry")
        call_id, result = entry.get("call_id"), entry.get("result")
        if not isinstance(call_id, str) or not call_id or call_id in call_ids:
            raise ValueError("Tool call IDs must be nonempty and unique")
        call_ids.add(call_id)
        if (
            not isinstance(result, dict)
            or type(result.get("ok")) is not bool
            or result.get("tool_name") != entry["tool_name"]
        ):
            raise ValueError("Tool result must identify its request and contain a boolean ok field")
        if not result["ok"]:
            continue
        if entry.get("tool_name") != "inspect_optimizer_run":
            raise ValueError("Successful tools must be scoped optimizer inspections")
        raw = entry.get("raw_arguments")
        if not isinstance(raw, str):
            raise ValueError("Tool arguments must be JSON text")
        arguments = _parse(raw)
        name = arguments.get("run_directory")
        if set(arguments) != {"run_directory"} or not isinstance(name, str) or name not in allowed:
            raise ValueError("Optimizer inspection is outside the fixed task scope")
        current = inspect_optimizer_run(
            allowed[name], require_file=lambda path: _scoped(root, path, file=True)
        )
        current["run_directory"] = name
        expected = {
            "tool_name": "inspect_optimizer_run",
            "ok": True,
            "data": current,
            "error_code": None,
            "error_message": None,
        }
        _same(result, expected, "Current optimizer evidence")
        runs[name] = current
    if set(runs) != set(allowed):
        raise ValueError("Reference and candidate both require successful inspections")
    return runs


def _require_supported_pair(reference: dict, candidate: dict) -> None:
    for key in ("config", "initial_state_sha256", "data_identity"):
        _same(reference[key], candidate[key], f"Paired {key}")
    for run, stale in ((reference, False), (candidate, True)):
        audit = run["optimizer_audit"]
        if (
            set(audit["missing_trainable_names"]) != ({"fc.bias", "fc.weight"} if stale else set())
            or audit["foreign_parameter_tensors"] != (2 if stale else 0)
            or audit["duplicate_parameter_occurrences"] != 0
            or audit["model_parameter_tensors"] != audit["trainable_parameter_tensors"]
            or audit["optimizer_unique_parameter_tensors"] != audit["model_parameter_tensors"]
        ):
            raise ValueError(
                "Only the full-finetuning stale classification-head defect is supported"
            )
        for row in run["epochs"]:
            head, backbone = row["head"], row["backbone"]
            if (
                head["mean_gradient_l2_norm"] <= 0
                or (
                    head["mean_parameter_update_l2_norm"] != 0
                    if stale
                    else head["mean_parameter_update_l2_norm"] <= 0
                )
                or backbone["mean_gradient_l2_norm"] <= 0
                or backbone["mean_parameter_update_l2_norm"] <= 0
            ):
                raise ValueError("Every epoch must support the expected optimizer mechanism")


def prepare_optimizer_repair(
    agent_report_path: Path,
    *,
    workspace_root: Path,
    reference_run: Path,
    candidate_run: Path,
    baseline_run: Path,
    epochs: int = 3,
) -> dict:
    """Revalidate an Agent finding and prepare a controller-owned action without training."""
    if type(epochs) is not int or not 1 <= epochs <= 3:
        raise ValueError("epochs must be an integer between 1 and 3")
    root = workspace_root.resolve()
    paths = {
        key: _scoped(root, value)
        for key, value in (
            ("reference_run", reference_run),
            ("candidate_run", candidate_run),
            ("baseline_run", baseline_run),
        )
    }
    names = {key: value.relative_to(root).as_posix() for key, value in paths.items()}
    if paths["reference_run"] == paths["candidate_run"]:
        raise ValueError("A repair requires distinct reference and candidate runs")
    agent_path = _scoped(root, agent_report_path, file=True)
    agent = _read(agent_path)
    if (
        agent.get("status") != "completed"
        or agent.get("profile") != "optimizer_binding"
        or agent.get("pending_evidence") != []
        or agent.get("error") is not None
        or not isinstance(agent.get("model"), str)
        or not agent["model"].strip()
    ):
        raise ValueError(
            "A completed optimizer-binding Agent report with no pending evidence is required"
        )
    if (
        type(agent.get("first_pass_valid")) is not bool
        or type(agent.get("validation_retries")) is not int
    ):
        raise ValueError("Agent validation provenance is required")
    if agent["validation_retries"] not in (0, 1):
        raise ValueError("Unsupported Agent correction budget")
    diagnosis = OptimizerDiagnosis.model_validate(agent.get("diagnosis"))
    if diagnosis.implementation_status != "optimizer_binding_defect":
        raise ValueError("Repair requires an evidence-supported optimizer binding defect")
    allowed = {names[key]: paths[key] for key in ("reference_run", "candidate_run")}
    runs = _replay_trace(agent.get("tool_trace"), root, allowed)
    validate_optimizer_diagnosis_evidence(
        diagnosis,
        agent["tool_trace"],
        reference_run=names["reference_run"],
        candidate_run=names["candidate_run"],
    )
    reference, candidate = (runs[names[key]] for key in ("reference_run", "candidate_run"))
    _require_supported_pair(reference, candidate)
    baseline = paths["baseline_run"]
    files = ("config.json", "run_report.json", "data_manifest.json", "initial_state_dict.pt")
    baseline_files = {name: _scoped(root, baseline / name, file=True) for name in files}
    baseline_json = {name: _read(baseline_files[name]) for name in files[:-1]}
    if baseline_json["run_report.json"].get("error") is not None:
        raise ValueError("Baseline must not contain a recorded training error")
    config, manifest, checkpoint = load_reference(baseline)
    _same(
        baseline_json["run_report.json"].get("completed_epochs"),
        config.epochs,
        "Baseline completed epochs",
    )
    scientific = {
        key: value
        for key, value in asdict(config).items()
        if key not in ("run_name", "data_dir", "output_dir")
    }
    _same(scientific, reference["config"], "Baseline scientific configuration")
    identity, _ = _data_identity(manifest, config)
    _same(identity, reference["data_identity"], "Baseline selected data")
    if epochs > config.epochs or epochs > reference["config"]["epochs"]:
        raise ValueError("Repair epochs must not exceed the source training budgets")
    initial_hash = _initial_state_hash(checkpoint)
    _same(initial_hash, reference["initial_state_sha256"], "Actual baseline initial state")
    source_names = (
        "optimizer_agent_repair.py",
        "camelyon_optimizer_training.py",
        "optimizer_probe.py",
    )
    return {
        "schema_version": 1,
        "task": "optimizer_agent_repair",
        "status": "prepared",
        "execution_started": False,
        "decision": None,
        "agent": {
            "path": agent_path.relative_to(root).as_posix(),
            "sha256": file_sha256(agent_path),
            "model": agent["model"],
            "diagnosis": diagnosis.model_dump(mode="json"),
            "first_pass_valid": agent["first_pass_valid"],
            "validation_retries": agent["validation_retries"],
            "prior_usage": {
                key: agent.get(key)
                for key in (
                    "model_calls",
                    "tool_calls",
                    "input_tokens",
                    "output_tokens",
                    "usage_complete",
                )
            },
        },
        "inputs": names,
        "action": {
            "name": "rebuild_optimizer_before_first_step",
            "owner": "controller",
            "agent_proposed_patch": None,
        },
        "budget": {
            "epochs_per_variant": epochs,
            "variants": list(VARIANTS),
            "training_epoch_upper_bound": 3 * epochs,
            "model_calls": 0,
        },
        "policy": dict(REPAIR_POLICY),
        "baseline": {
            "scientific_config": scientific,
            "files_sha256": {key: file_sha256(value) for key, value in baseline_files.items()},
            "initial_state_sha256": initial_hash,
            "data_identity": identity,
        },
        "evidence": {
            f"{key}_files_sha256": runs[names[key]]["provenance"]["files_sha256"]
            for key in ("reference_run", "candidate_run")
        },
        "implementation_sha256": {
            name: file_sha256(Path(__file__).with_name(name)) for name in source_names
        },
        "training_mode": "from_reference_initial_state",
        "checkpoint_selection": "fixed_final_epoch",
        "test_evaluated": False,
    }


def _verify_training_result(path: Path, training_root: Path, plan: dict, root: Path) -> dict:
    path = _scoped(training_root.resolve(), path, file=True)
    if path.name != "optimizer_training_report.json":
        raise ValueError("Unexpected training report filename")
    report = _read(path)
    expected = {
        "schema_version": 1,
        "task": "camelyon17_optimizer_rebind_verification",
        "status": "completed",
        "error": None,
        "epochs_per_variant": plan["budget"]["epochs_per_variant"],
        "training_epoch_upper_bound": plan["budget"]["training_epoch_upper_bound"],
        "training_mode": "from_reference_initial_state",
        "checkpoint_selection": "fixed_final_epoch",
        "test_evaluated": False,
        "repair_acceptance_requested": True,
        "repair_acceptance_evaluated": True,
        "reference_initial_state_sha256": plan["baseline"]["initial_state_sha256"],
    }
    for key, value in expected.items():
        if key not in report:
            raise ValueError(f"Training report is missing {key}")
        _same(report[key], value, key)
    _same(
        _scoped(root, Path(report["reference_run"])).as_posix(),
        (root / plan["inputs"]["baseline_run"]).as_posix(),
        "Training baseline",
    )
    for report_key, filename in (
        ("reference_config_sha256", "config.json"),
        ("reference_initial_checkpoint_sha256", "initial_state_dict.pt"),
        ("reference_manifest_sha256", "data_manifest.json"),
    ):
        _same(report.get(report_key), plan["baseline"]["files_sha256"][filename], report_key)
    _same(
        report.get("controls", {}).get("performance_threshold"), plan["policy"], "Training policy"
    )
    policy_record = report["repair_verification_policy"]
    policy_path = _scoped(training_root.resolve(), Path(policy_record["path"]), file=True)
    _same(file_sha256(policy_path), policy_record["sha256"], "Saved policy digest")
    policy = _read(policy_path)
    _same(policy.get("policy"), plan["policy"], "Saved policy")
    _same(policy.get("epochs_per_variant"), expected["epochs_per_variant"], "Saved policy budget")
    if not isinstance(report.get("variants"), dict) or set(report["variants"]) != set(VARIANTS):
        raise ValueError("Training must contain exactly the three planned variants")
    expected_config = {
        **plan["baseline"]["scientific_config"],
        "epochs": expected["epochs_per_variant"],
    }
    for name in VARIANTS:
        directory = _scoped(training_root.resolve(), path.parent / name)
        child_path = _scoped(training_root.resolve(), directory / "run_report.json", file=True)
        child = _read(child_path)
        _same(child, report["variants"][name], f"Recorded {name} report")
        _same(child.get("variant"), name, "Training variant")
        recorded_directory = _scoped(training_root.resolve(), Path(child["run_directory"]))
        _same(recorded_directory.as_posix(), directory.as_posix(), "Variant output directory")
        evidence = inspect_optimizer_run(
            directory,
            require_file=lambda item: _scoped(training_root.resolve(), item, file=True),
        )
        _same(evidence["config"], expected_config, f"{name} scientific configuration")
        _same(
            evidence["initial_state_sha256"],
            plan["baseline"]["initial_state_sha256"],
            f"{name} initialization",
        )
        _same(evidence["data_identity"], plan["baseline"]["data_identity"], f"{name} selected data")
    verification = assess_rebind_training(
        report["variants"],
        expected["reference_initial_state_sha256"],
        expected["epochs_per_variant"],
    )
    _same(verification, report.get("repair_verification"), "Recomputed repair verification")
    _same(verification["policy"], plan["policy"], "Verification policy")
    return verification


def run_optimizer_agent_repair(
    agent_report_path: Path,
    *,
    workspace_root: Path,
    reference_run: Path,
    candidate_run: Path,
    baseline_run: Path,
    epochs: int = 3,
    execute: bool = False,
) -> Path:
    """Persist a new plan, then optionally run the existing bounded training controller."""
    if type(execute) is not bool:
        raise ValueError("execute must be an explicit boolean")
    root = workspace_root.resolve()
    if execute and Path.cwd().resolve() != root:
        raise ValueError(
            "Execute repairs from the workspace root so relative data paths remain unchanged"
        )
    plan = prepare_optimizer_repair(
        agent_report_path,
        workspace_root=root,
        reference_run=reference_run,
        candidate_run=candidate_run,
        baseline_run=baseline_run,
        epochs=epochs,
    )
    parent = _scoped(root, Path("artifacts/optimizer_agent_repairs"))
    directory = parent / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    directory.mkdir(parents=True)
    plan_path = directory / "repair_plan.json"
    _write(plan_path, plan)
    outcome = {
        **plan,
        "repair_plan": {
            "path": plan_path.relative_to(root).as_posix(),
            "sha256": file_sha256(plan_path),
        },
        "error": None,
    }
    if execute:
        outcome["execution_started"] = True
        training_root = directory / "training"
        try:
            report_path = run_optimizer_training(
                root / plan["inputs"]["baseline_run"], training_root, epochs, verify_rebind=True
            )
            _scoped(root, training_root)
            report_path = _scoped(training_root.resolve(), Path(report_path), file=True)
            verification = _verify_training_result(report_path, training_root, plan, root)
            outcome.update(
                status="completed",
                decision=verification["decision"],
                repair_verification=verification,
                training_report={
                    "path": report_path.relative_to(root).as_posix(),
                    "sha256": file_sha256(report_path),
                },
            )
        except (Exception, KeyboardInterrupt) as error:
            outcome.update(
                status="failed",
                decision=None,
                error={"type": type(error).__name__, "message": str(error)},
            )
            outcome["partial_reports"] = [
                path.relative_to(root).as_posix()
                for path in training_root.rglob("*report.json")
                if path.is_file() and path.resolve().is_relative_to(training_root.resolve())
            ]
    path = directory / "optimizer_agent_repair_report.json"
    _write(path, outcome)
    return path


def _relative_argument(value: str) -> Path:
    windows = PureWindowsPath(value)
    path = Path(value.replace("\\", "/"))
    if path.is_absolute() or windows.drive or windows.root or ".." in path.parts:
        raise argparse.ArgumentTypeError("Use a project-relative path without parent traversal")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("agent-report", "reference-run", "candidate-run", "baseline-run"):
        parser.add_argument(f"--{name}", type=_relative_argument, required=True)
    parser.add_argument("--epochs", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument(
        "--execute", action="store_true", help="Run three bounded training variants"
    )
    args = parser.parse_args()
    try:
        path = run_optimizer_agent_repair(
            args.agent_report,
            workspace_root=Path.cwd(),
            reference_run=args.reference_run,
            candidate_run=args.candidate_run,
            baseline_run=args.baseline_run,
            epochs=args.epochs,
            execute=args.execute,
        )
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    report = _read(path)
    print(f"optimizer_agent_repair_report={path}")
    for key in ("status", "execution_started", "decision"):
        print(f"{key}={report[key]}")
    print("action=" + json.dumps(report["action"]))
    print("budget=" + json.dumps(report["budget"]))
    if report["error"]:
        print("error=" + json.dumps(report["error"]))
    if report["status"] == "failed" or report["decision"] == "rejected":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
