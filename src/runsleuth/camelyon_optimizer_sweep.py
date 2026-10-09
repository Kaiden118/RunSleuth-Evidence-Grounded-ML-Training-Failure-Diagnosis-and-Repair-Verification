"""Evaluate optimizer rebinding on new training seeds with fixed data and budgets."""

import argparse
import json
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath

from runsleuth.camelyon_config import CamelyonConfig
from runsleuth.camelyon_optimizer_probe import file_sha256, load_reference
from runsleuth.camelyon_optimizer_training import (
    REPAIR_POLICY,
    assess_rebind_training,
    compare_performance,
    run_optimizer_training,
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


def _initial_state_hash(checkpoint: Path) -> str:
    import torch

    from runsleuth.optimizer_probe import state_dict_sha256

    return state_dict_sha256(torch.load(checkpoint, map_location="cpu", weights_only=True))


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


def _scientific(config: CamelyonConfig) -> dict:
    return {
        key: value
        for key, value in asdict(config).items()
        if key not in ("run_name", "data_dir", "output_dir")
    }


def _baseline(directory: Path, root: Path) -> tuple[CamelyonConfig, dict]:
    """Check the saved baseline before trusting its initialization or sample selection."""
    filenames = ("config.json", "run_report.json", "data_manifest.json", "initial_state_dict.pt")
    files = {name: _scoped(root, directory / name, file=True) for name in filenames}
    values = {name: _read(files[name]) for name in filenames[:-1]}
    config, manifest, checkpoint = load_reference(directory)
    report = values["run_report.json"]
    _same(report.get("completed_epochs"), config.epochs, "Baseline completed epochs")
    if report.get("error") is not None or report.get("test_evaluated") is not False:
        raise ValueError("Baseline must complete without errors or official test evaluation")
    identity, _ = _data_identity(manifest, config)
    return config, {
        "scientific_config": _scientific(config),
        "files_sha256": {name: file_sha256(path) for name, path in files.items()},
        "initial_state_sha256": _initial_state_hash(checkpoint),
        "data_identity": identity,
    }


def _run_baseline(config: CamelyonConfig) -> Path:
    from runsleuth.camelyon import run_camelyon_experiment

    return run_camelyon_experiment(config, download=False)


def _save(path: Path, value: dict, *, initial: bool = False) -> None:
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    with (path if initial else temporary).open("x", encoding="utf-8") as stream:
        stream.write(payload)
    if not initial:
        temporary.replace(path)


def _counts(cases: list[dict]) -> dict:
    completed = [case for case in cases if case["status"] == "completed"]
    return {
        "planned_seeds": len(cases),
        "completed_seeds": len(completed),
        "accepted": sum(case["decision"] == "accepted" for case in completed),
        "rejected": sum(case["decision"] == "rejected" for case in completed),
        "error_cases": sum(case["status"] in ("error", "interrupted") for case in cases),
        "not_run": sum(case["status"] == "not_run" for case in cases),
        "running": sum(case["status"] == "running" for case in cases),
        "structural_verified": {
            "numerator": sum(
                case["repair_verification"]["structure_verified"] for case in completed
            ),
            "denominator": len(cases),
        },
    }


def _execute_seed(
    case: dict,
    intended: CamelyonConfig,
    directory: Path,
    plan: dict,
    root: Path,
    persist,
) -> None:
    baseline_root = directory / "baseline"
    baseline_run = _scoped(baseline_root, Path(_run_baseline(intended)))
    _scoped(root, baseline_run)
    case["baseline_run"] = baseline_run.relative_to(root).as_posix()
    actual, baseline = _baseline(baseline_run, root)
    _same(asdict(actual), asdict(intended), "New baseline configuration")
    _same(
        baseline["data_identity"], plan["source_baseline"]["data_identity"], "Fixed selected data"
    )
    case["baseline"] = baseline
    case["phase"] = "paired_training"
    persist()
    training_root = directory / "paired"
    context = {
        "inputs": {"baseline_run": case["baseline_run"]},
        "baseline": baseline,
        "policy": plan["policy"],
        "budget": {
            "epochs_per_variant": intended.epochs,
            "training_epoch_upper_bound": 3 * intended.epochs,
        },
    }
    report_path = run_optimizer_training(
        baseline_run,
        training_root,
        intended.epochs,
        verify_rebind=True,
    )
    _scoped(root, training_root)
    report_path = _scoped(training_root.resolve(), Path(report_path), file=True)
    verification = _verify_training_result(report_path, training_root, context, root)
    report = _read(report_path)
    case.update(
        status="completed",
        phase="verified",
        decision=verification["decision"],
        repair_verification=verification,
        training_report={
            "path": report_path.relative_to(root).as_posix(),
            "sha256": file_sha256(report_path),
        },
        final_metrics={
            name: {
                f"{domain}_validation_{metric}": report["variants"][name]["final_metrics"][
                    f"{domain}_validation_{metric}"
                ]
                for domain in ("id", "ood")
                for metric in ("accuracy", "loss")
            }
            for name in VARIANTS
        },
        performance_comparison=compare_performance(report["variants"]),
    )


def run_optimizer_sweep(
    reference_run: Path,
    *,
    workspace_root: Path,
    seeds: tuple[int, ...] = (7, 2026),
    epochs: int = 3,
    execute: bool = False,
) -> Path:
    """Plan or execute a new-seed development sweep; preserve rejected and failed cases."""
    if type(execute) is not bool:
        raise ValueError("execute must be an explicit boolean")
    if type(epochs) is not int or not 1 <= epochs <= 3:
        raise ValueError("epochs must be an integer between 1 and 3")
    if (
        not isinstance(seeds, (list, tuple))
        or not 1 <= len(seeds) <= 3
        or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Provide one to three unique integer seeds in [0, 2**32)")
    root = workspace_root.resolve()
    if execute and Path.cwd().resolve() != root:
        raise ValueError("Execute from the workspace root to preserve relative data paths")
    reference = _scoped(root, reference_run)
    config, source = _baseline(reference, root)
    if config.device == "auto":
        raise ValueError(
            "Seed sweeps require a source baseline with an explicit cpu or cuda device"
        )
    if config.seed in seeds:
        raise ValueError("Sweep seeds must exclude the historical reference seed")
    if epochs > config.epochs:
        raise ValueError("epochs must not exceed the source baseline budget")
    parent = _scoped(root, Path("artifacts/optimizer_sweeps"))
    directory = parent / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    directory.mkdir(parents=True)
    source_names = (
        "camelyon_optimizer_sweep.py",
        "camelyon_optimizer_training.py",
        "optimizer_evidence.py",
    )
    plan = {
        "schema_version": 1,
        "task": "camelyon17_optimizer_seed_sweep",
        "reference_run": reference.relative_to(root).as_posix(),
        "source_baseline": source,
        "seeds": list(seeds),
        "subset_seed": config.subset_seed,
        "scientific_template": {
            key: value
            for key, value in _scientific(replace(config, epochs=epochs)).items()
            if key != "seed"
        },
        "policy": dict(REPAIR_POLICY),
        "budget": {
            "planned_seeds": len(seeds),
            "epochs_per_variant": epochs,
            "baseline_runs": len(seeds),
            "paired_training_runs": 3 * len(seeds),
            "variants": list(VARIANTS),
            "training_epoch_upper_bound": 4 * epochs * len(seeds),
            "model_calls": 0,
        },
        "scope": "New-seed development experiments; historical reference results are excluded",
        "baseline_training_mode": "full_finetuning_from_imagenet_with_new_seed",
        "paired_training_mode": "from_seed_specific_reference_initial_state",
        "checkpoint_selection": "fixed_final_epoch",
        "test_evaluated": False,
        "ood_used_for_repair_acceptance": True,
        "implementation_sha256": {
            name: file_sha256(Path(__file__).with_name(name)) for name in source_names
        },
    }
    plan_path = directory / "sweep_plan.json"
    _save(plan_path, plan, initial=True)
    cases = [{"seed": seed, "status": "not_run", "decision": None, "error": None} for seed in seeds]
    summary = {
        "schema_version": 1,
        "task": plan["task"],
        "status": "prepared",
        "execution_started": execute,
        "plan": {
            "path": plan_path.relative_to(root).as_posix(),
            "sha256": file_sha256(plan_path),
        },
        "budget": plan["budget"],
        "policy": plan["policy"],
        "cases": cases,
        "error": None,
    }
    summary_path = directory / "sweep_summary.json"

    def persist() -> None:
        summary["summary"] = _counts(cases)
        _save(summary_path, summary)

    persist()
    if not execute:
        return summary_path
    print(f"sweep_directory={directory}", flush=True)
    print("budget=" + json.dumps(plan["budget"]), flush=True)
    for index, case in enumerate(cases, start=1):
        seed = case["seed"]
        case.update(status="running", phase="baseline_training")
        summary["status"] = "running"
        persist()
        try:
            # Short names keep Windows paths under 260 characters for any seed; the
            # case records which seed each directory holds.
            seed_directory = _scoped(root, directory / f"s{index}")
            case["output_directory"] = seed_directory.relative_to(root).as_posix()
            seed_directory.mkdir()
            intended = replace(
                config,
                seed=seed,
                epochs=epochs,
                run_name="baseline",
                output_dir=str(seed_directory / "baseline"),
            )
            _execute_seed(case, intended, seed_directory, plan, root, persist)
        except (Exception, KeyboardInterrupt) as error:
            case.update(
                status="interrupted" if isinstance(error, KeyboardInterrupt) else "error",
                error={"type": type(error).__name__, "message": str(error)},
            )
            if isinstance(error, KeyboardInterrupt):
                summary.update(status="interrupted", error=case["error"])
                persist()
                break
        persist()
        print(f"seed={seed}, status={case['status']}, decision={case['decision']}", flush=True)
        if case["error"]:
            print("error=" + json.dumps(case["error"]), flush=True)
    else:
        summary["status"] = (
            "completed_with_errors" if _counts(cases)["error_cases"] else "completed"
        )
        persist()
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 2026])
    parser.add_argument("--epochs", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        path = run_optimizer_sweep(
            args.reference_run,
            workspace_root=Path.cwd(),
            seeds=tuple(args.seeds),
            epochs=args.epochs,
            execute=args.execute,
        )
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    report = _read(path)
    print(f"sweep_summary={path}")
    print(f"status={report['status']}")
    print("summary=" + json.dumps(report["summary"]))
    if report["status"] in ("completed_with_errors", "interrupted"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
