"""Evaluate different optimizer-factory structures on recorded Camelyon17 inputs."""

import argparse
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

from runsleuth.camelyon_optimizer_probe import file_sha256, load_reference
from runsleuth.evaluate_optimizer import _read, _scoped
from runsleuth.optimizer_evidence import _data_identity, _same
from runsleuth.optimizer_factory_analysis import analyze_optimizer_factory, bind_validated_factory
from runsleuth.optimizer_factory_cases import ORACLE_SOURCE, SUITE_VERSION, factory_cases
from runsleuth.optimizer_source_repair import _sha
from runsleuth.optimizer_source_runtime import _checks, _probe_step
from runsleuth.optimizer_source_verify import _write_report


def prepare_factory_suite() -> dict:
    """Analyze source without passing its expected label to the analyzer."""
    cases = []
    for fixture in factory_cases():
        analysis = analyze_optimizer_factory(fixture["source"])
        cases.append(
            {
                **fixture,
                "source_sha256": _sha(fixture["source"].encode("utf-8")),
                "analysis": analysis,
                "classification_correct": analysis["decision"] == fixture["expected_decision"],
                "runtime": None,
                "error": None,
            }
        )
    return {
        "schema_version": 1,
        "task": "optimizer_factory_development_suite",
        "suite_version": SUITE_VERSION,
        "status": "prepared",
        "error": None,
        "method": "controller_ast_data_flow",
        "execution_requested": False,
        "scope": "hand-authored development fixtures; not held-out generalization or LLM evaluation",
        "performance_recovery_evaluated": False,
        "repair_accepted": None,
        "test_evaluated": False,
        "budget": {"api_calls": 0, "training_epochs": 0, "optimizer_step_upper_bound": 20},
        "cases": cases,
    }


def _load_context(reference_run: Path, requested_device: str | None) -> dict:
    config, manifest, checkpoint = load_reference(reference_run)
    report, _ = _read(reference_run / "run_report.json")
    if report.get("error") is not None:
        raise ValueError("Baseline contains an execution error")
    import torch
    from torch.utils.data import default_collate

    from runsleuth.camelyon import build_camelyon_model
    from runsleuth.camelyon_data import build_camelyon_data
    from runsleuth.optimizer_probe import state_dict_sha256
    from runsleuth.train import resolve_device, seed_everything

    device = resolve_device(requested_device or config.device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if any(not torch.isfinite(tensor).all().item() for tensor in state.values()):
        raise ValueError("Reference initialization must be finite")
    seed_everything(config.seed)
    print("phase=load_cached_camelyon_data", flush=True)
    data = build_camelyon_data(config, device=device, download=False)
    _same(
        _data_identity(data.manifest, config)[0],
        _data_identity(manifest, config)[0],
        "Selected Camelyon17 data and preprocessing",
    )
    positions = list(next(iter(data.train_loader.batch_sampler)))
    inputs, targets = default_collate([data.train_loader.dataset[index] for index in positions])
    if not positions or len(positions) != len(targets):
        raise ValueError("The controlled batch is empty or inconsistent")
    selected_ids = manifest["splits"]["train"]["selected_sample_ids"]
    return {
        "torch": torch,
        "config": config,
        "device": device,
        "state": state,
        "state_sha256": state_dict_sha256(state),
        "inputs": inputs.to(device),
        "targets": targets.to(device),
        "model_factory": lambda: build_camelyon_model(pretrained=False),
        "provenance": {
            "reference_run": str(reference_run),
            "reference_files_sha256": {
                name: file_sha256(reference_run / name)
                for name in (
                    "config.json",
                    "data_manifest.json",
                    "initial_state_dict.pt",
                    "run_report.json",
                )
            },
            "initial_state_sha256": state_dict_sha256(state),
            "environment": {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "device": str(device),
                "cuda_runtime": torch.version.cuda,
            },
            "batch": {
                "selected_sample_ids": [selected_ids[index] for index in positions],
                "input_shape": list(inputs.shape),
                "targets": targets.tolist(),
                "tensor_sha256": state_dict_sha256({"inputs": inputs, "targets": targets}),
            },
        },
    }


def _healthy_checks(variants: dict, expected_hash: str) -> dict:
    clean, original = variants["clean"], variants["original"]
    return {
        "same_recorded_initialization": all(
            row["initial_state_sha256"] == expected_hash for row in variants.values()
        ),
        "fresh_optimizers": all(
            row["optimizer_state_entries_before"] == 0 for row in variants.values()
        ),
        "same_pre_update_loss_and_accuracy": all(
            original[key] == clean[key] for key in ("pre_update_loss", "pre_update_accuracy")
        ),
        "full_optimizer_coverage": all(
            not row["optimizer_audit"]["missing_trainable_names"]
            and row["optimizer_audit"]["foreign_parameter_tensors"] == 0
            and row["optimizer_audit"]["duplicate_parameter_occurrences"] == 0
            and row["optimizer_audit"]["model_parameter_tensors"]
            == row["optimizer_audit"]["trainable_parameter_tensors"]
            == row["optimizer_audit"]["optimizer_unique_parameter_tensors"]
            for row in variants.values()
        ),
        "head_and_backbone_have_gradients_and_updates": all(
            row[group][metric] > 0
            for row in variants.values()
            for group in ("head", "backbone")
            for metric in ("gradient_l2_norm", "parameter_update_l2_norm")
        ),
        "same_initial_gradients": all(
            original[group]["gradient_l2_norm"] == clean[group]["gradient_l2_norm"]
            for group in ("head", "backbone")
        ),
        "original_matches_clean_including_buffers": original["post_step_state_sha256"]
        == clean["post_step_state_sha256"],
    }


def probe_factory_case(case: dict, context: dict, runtime: dict) -> None:
    """Execute original/proposed source under a shared initialization and batch."""
    from runsleuth.train import seed_everything

    analysis = analyze_optimizer_factory(case["source"])
    _same(analysis, case["analysis"], "Factory analysis before execution")
    if analysis["decision"] == "unsupported":
        raise ValueError("Unsupported sources must never be executed")
    sources = {"clean": ORACLE_SOURCE, "original": case["source"]}
    if analysis["decision"] == "patch_proposed":
        sources["source_patched"] = analysis["proposed_source"]
    runtime.update(
        status="running",
        phase="prepare",
        variants={},
        optimizer_steps_recorded=0,
        step_accounting_complete=True,
        mechanism_verified=None,
        checks=None,
    )
    for name, source in sources.items():
        constructor = bind_validated_factory(source, context["torch"])
        seed_everything(context["config"].seed)
        model = context["model_factory"]()
        model.load_state_dict(context["state"], strict=True)
        model.to(context["device"])
        seed_everything(context["config"].seed)
        runtime.update(phase=f"optimizer_step:{name}", step_accounting_complete=False)
        row = _probe_step(
            model,
            context["inputs"],
            context["targets"],
            constructor=constructor,
            variant=name,
            config=context["config"],
        )
        runtime["variants"][name] = row
        runtime["optimizer_steps_recorded"] += 1
        runtime.update(phase="between_steps", step_accounting_complete=True)
        del model
    if analysis["decision"] == "patch_proposed":
        rows = runtime["variants"]
        checks = _checks(
            {
                "clean": rows["clean"],
                "stale_head": rows["original"],
                "source_patched": rows["source_patched"],
            },
            context["state_sha256"],
        )
    else:
        checks = _healthy_checks(runtime["variants"], context["state_sha256"])
    runtime.update(
        status="verified" if all(checks.values()) else "not_verified",
        phase="finished",
        checks=checks,
        mechanism_verified=all(checks.values()),
    )


def _rate(numerator: int, denominator: int) -> dict:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def _summarize(report: dict) -> None:
    cases = report["cases"]
    faults = [case for case in cases if case["expected_decision"] == "patch_proposed"]
    healthy = [case for case in cases if case["expected_decision"] == "no_change"]
    unsupported = [case for case in cases if case["expected_decision"] == "unsupported"]
    runtime = [case["runtime"] for case in cases if case["runtime"] is not None]

    def verified(case):
        return (
            case["classification_correct"]
            and (case["runtime"] or {}).get("mechanism_verified") is True
        )

    report["summary"] = {
        "total_cases": len(cases),
        "source_decisions_correct": _rate(
            sum(case["classification_correct"] for case in cases), len(cases)
        ),
        "healthy_false_positives": _rate(
            sum(case["analysis"]["decision"] == "patch_proposed" for case in healthy), len(healthy)
        ),
        "unsupported_refused": _rate(
            sum(case["analysis"]["decision"] == "unsupported" for case in unsupported),
            len(unsupported),
        ),
        "fault_repairs_mechanism_verified": _rate(
            sum(verified(case) for case in faults),
            len(faults) if report["execution_requested"] else 0,
        ),
        "healthy_controls_mechanism_verified": _rate(
            sum(verified(case) for case in healthy),
            len(healthy) if report["execution_requested"] else 0,
        ),
        "error_cases": sum(case["error"] is not None for case in cases),
        "optimizer_steps_recorded": sum(row["optimizer_steps_recorded"] for row in runtime),
        "step_accounting_complete": all(row["step_accounting_complete"] for row in runtime),
    }


def run_factory_suite(
    *,
    workspace_root: Path,
    reference_run: Path | None = None,
    verify: bool = False,
    requested_device: str | None = None,
) -> Path:
    if type(verify) is not bool or requested_device not in (None, "auto", "cpu", "cuda"):
        raise ValueError("Invalid execution mode or device")
    if verify and reference_run is None:
        raise ValueError("--verify requires --reference-run")
    root = workspace_root.resolve()
    reference = _scoped(root, reference_run) if reference_run is not None else None
    report = prepare_factory_suite()
    report["execution_requested"] = verify
    planned_steps = sum(
        {"patch_proposed": 3, "no_change": 2, "unsupported": 0}[case["analysis"]["decision"]]
        for case in report["cases"]
    )
    if planned_steps > report["budget"]["optimizer_step_upper_bound"]:
        raise ValueError("Factory suite exceeds its fixed twenty-step budget")
    report["budget"]["planned_optimizer_steps"] = planned_steps if verify else 0
    directory = _scoped(root, "artifacts/optimizer_factory_suites") / datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "report.json"
    snapshot = directory / "source_snapshot"
    snapshot.mkdir()
    report["runtime_source_sha256"] = {}
    for name in (
        "optimizer_factory_analysis.py",
        "optimizer_factory_cases.py",
        "optimizer_factory_suite.py",
        "optimizer_source_patch.py",
        "optimizer_source_runtime.py",
        "optimizer_probe.py",
        "optimizer_audit.py",
        "camelyon.py",
        "camelyon_data.py",
        "camelyon_config.py",
        "camelyon_mirror.py",
        "train.py",
    ):
        (snapshot / name).write_bytes((Path(__file__).parent / name).read_bytes())
        report["runtime_source_sha256"][name] = file_sha256(snapshot / name)
    for index, case in enumerate(report["cases"], 1):
        folder = directory / "cases" / f"case-{index:03d}"
        (folder / "original").mkdir(parents=True)
        original = folder / "original/optimizer_probe.py"
        original.write_bytes(case["source"].encode("utf-8"))
        case["original_file"] = original.relative_to(root).as_posix()
        case["proposed_file"] = case["patch_file"] = None
        if case["analysis"]["proposed_source"] is not None:
            (folder / "proposed").mkdir()
            proposed = folder / "proposed/optimizer_probe.py"
            proposed.write_bytes(case["analysis"]["proposed_source"].encode("utf-8"))
            patch = folder / "repair.patch"
            patch.write_bytes(case["analysis"]["unified_diff"].encode("utf-8"))
            case.update(
                proposed_file=proposed.relative_to(root).as_posix(),
                patch_file=patch.relative_to(root).as_posix(),
            )
    _summarize(report)
    _write_report(path, report)
    print(f"factory_suite_directory={directory}", flush=True)
    print(json.dumps(report["budget"]), flush=True)
    if not all(case["classification_correct"] for case in report["cases"]):
        report["status"] = "classification_failed"
    elif not verify:
        report["status"] = "static_completed"
    else:
        try:
            context = _load_context(reference, requested_device)
            report["provenance"] = context["provenance"]
            report["status"] = "running"
            _write_report(path, report)
            for case in report["cases"]:
                if case["analysis"]["decision"] == "unsupported":
                    continue
                case["runtime"] = {
                    "status": "starting",
                    "optimizer_steps_recorded": 0,
                    "step_accounting_complete": True,
                    "mechanism_verified": None,
                }
                try:
                    print(f"phase=factory_probe case={case['id']}", flush=True)
                    probe_factory_case(case, context, case["runtime"])
                except (Exception, KeyboardInterrupt) as error:
                    case["runtime"]["status"] = "error"
                    case["error"] = {"type": type(error).__name__, "message": str(error)}
                    raise
                finally:
                    _summarize(report)
                    _write_report(path, report)
            verified = [
                case["runtime"]["mechanism_verified"]
                for case in report["cases"]
                if case["runtime"] is not None
            ]
            report["status"] = "completed" if all(verified) else "mechanism_failed"
        except (Exception, KeyboardInterrupt) as error:
            report.update(
                status="execution_error",
                error={"type": type(error).__name__, "message": str(error)},
            )
    _summarize(report)
    _write_report(path, report)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path)
    parser.add_argument(
        "--verify", action="store_true", help="Run the fixed single-step mechanism budget."
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    try:
        path = run_factory_suite(
            workspace_root=Path.cwd(),
            reference_run=args.reference_run,
            verify=args.verify,
            requested_device=args.device,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    report, _ = _read(path)
    print(f"factory_suite_report={path}")
    print(f"status={report['status']}")
    print(json.dumps(report["summary"]))
    for case in report["cases"]:
        print(
            json.dumps(
                {
                    "case": case["id"],
                    "decision": case["analysis"]["decision"],
                    "classification_correct": case["classification_correct"],
                    "mechanism_verified": (case["runtime"] or {}).get("mechanism_verified"),
                    "error": case["error"],
                }
            )
        )
    if report["error"]:
        print(json.dumps(report["error"]))
    if report["status"] not in ("static_completed", "completed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
