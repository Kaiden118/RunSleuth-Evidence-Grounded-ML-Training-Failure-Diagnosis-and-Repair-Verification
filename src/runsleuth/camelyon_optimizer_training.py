"""Measure stale optimizer bindings over a fixed, paired Camelyon17 training budget."""

import argparse
import json
import platform
import shutil
from dataclasses import asdict, replace
from datetime import UTC, datetime
from math import isclose
from pathlib import Path
from time import perf_counter

from runsleuth.camelyon_optimizer_probe import (
    check_matching_data,
    file_sha256,
    load_reference,
    read_json,
)


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def append_json(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, allow_nan=False) + "\n")


def snapshot_sources(directory: Path) -> dict[str, str]:
    destination = directory / "source_snapshot"
    destination.mkdir()
    hashes = {}
    for name in (
        "camelyon_optimizer_training.py",
        "parameter_group_monitor.py",
        "camelyon_optimizer_probe.py",
        "optimizer_probe.py",
        "optimizer_audit.py",
        "camelyon.py",
        "camelyon_config.py",
        "camelyon_data.py",
        "camelyon_mirror.py",
        "train.py",
        "telemetry.py",
    ):
        shutil.copyfile(Path(__file__).parent / name, destination / name)
        hashes[name] = file_sha256(destination / name)
    return hashes


def assess_training(variants: dict, expected_hash: str, epochs: int) -> dict[str, bool]:
    """Verify the optimization mechanism separately from validation performance."""
    clean, stale = variants["clean"], variants["stale_head"]
    clean_audit, stale_audit = clean["optimizer_audit"], stale["optimizer_audit"]
    clean_rows = clean["parameter_group_epochs"]
    stale_rows = stale["parameter_group_epochs"]
    return {
        "same_reference_initial_state": (
            clean["initial_state_sha256"] == stale["initial_state_sha256"] == expected_hash
        ),
        "same_initial_validation_metrics": all(
            isclose(value, stale["initialization_metrics"][key], rel_tol=1e-6, abs_tol=1e-6)
            for key, value in clean["initialization_metrics"].items()
        ),
        "both_completed_fixed_budget": (
            clean["completed_epochs"] == stale["completed_epochs"] == epochs
            and len(clean_rows) == len(stale_rows) == epochs
        ),
        "same_optimizer_step_counts": (
            len(clean_rows) == len(stale_rows) == epochs
            and all(
                a["optimizer_steps"] == b["optimizer_steps"] > 0
                for a, b in zip(clean_rows, stale_rows, strict=True)
            )
        ),
        "clean_optimizer_covers_current_parameters": (
            not clean_audit["missing_trainable_names"]
            and clean_audit["foreign_parameter_tensors"] == 0
            and clean_audit["duplicate_parameter_occurrences"] == 0
        ),
        "stale_optimizer_misses_exactly_current_head": (
            set(stale_audit["missing_trainable_names"]) == {"fc.weight", "fc.bias"}
            and stale_audit["foreign_parameter_tensors"] == 2
            and stale_audit["duplicate_parameter_occurrences"] == 0
        ),
        "clean_head_updates_every_epoch": bool(clean_rows)
        and all(
            row["head"]["mean_gradient_l2_norm"] > 0
            and row["head"]["mean_parameter_update_l2_norm"] > 0
            for row in clean_rows
        ),
        "stale_head_has_gradients_without_updates_every_epoch": bool(stale_rows)
        and all(
            row["head"]["mean_gradient_l2_norm"] > 0
            and row["head"]["mean_parameter_update_l2_norm"] == 0
            for row in stale_rows
        ),
        "both_backbones_update_every_epoch": bool(clean_rows and stale_rows)
        and all(
            row["backbone"]["mean_parameter_update_l2_norm"] > 0 for row in clean_rows + stale_rows
        ),
    }


def compare_performance(variants: dict) -> dict:
    """Signed clean-minus-stale differences; no post-hoc acceptance threshold."""
    clean = variants["clean"]["final_metrics"]
    stale = variants["stale_head"]["final_metrics"]
    return {
        domain: {
            "clean_accuracy": clean[f"{domain}_validation_accuracy"],
            "stale_accuracy": stale[f"{domain}_validation_accuracy"],
            "accuracy_gap_percentage_points": 100
            * (clean[f"{domain}_validation_accuracy"] - stale[f"{domain}_validation_accuracy"]),
            "clean_loss": clean[f"{domain}_validation_loss"],
            "stale_loss": stale[f"{domain}_validation_loss"],
        }
        for domain in ("id", "ood")
    }


def _run_variant(variant, config, reference_manifest, state, expected_hash, directory, device):
    import torch
    from torch import nn

    from runsleuth.camelyon import (
        _check_training_result,
        _evaluate_domains,
        build_camelyon_model,
    )
    from runsleuth.camelyon_data import build_camelyon_data
    from runsleuth.optimizer_audit import audit_optimizer_parameters
    from runsleuth.optimizer_probe import make_probe_optimizer, state_dict_sha256
    from runsleuth.parameter_group_monitor import ParameterGroupMonitor
    from runsleuth.telemetry import EpochMetrics, append_epoch_metrics
    from runsleuth.train import seed_everything, train_one_epoch

    directory.mkdir()
    config.save(directory / "config.json")
    report = {
        "variant": variant,
        "status": "running",
        "completed_epochs": 0,
        "task": "camelyon17_controlled_optimizer_training",
        "error": None,
        "run_directory": str(directory),
        "parameter_group_epochs": [],
        "test_evaluated": False,
    }
    started = perf_counter()
    try:
        seed_everything(config.seed)
        print(f"phase=load_cached_data variant={variant}", flush=True)
        data = build_camelyon_data(config, device=device, download=False)
        check_matching_data(reference_manifest, data.manifest)
        write_json(directory / "data_manifest.json", data.manifest)
        report["data_manifest_sha256"] = file_sha256(directory / "data_manifest.json")
        model = build_camelyon_model(pretrained=False)
        model.load_state_dict(state, strict=True)
        model.to(device)
        optimizer = make_probe_optimizer(
            model,
            variant=variant,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        report["initial_state_sha256"] = state_dict_sha256(model.state_dict())
        if report["initial_state_sha256"] != expected_hash:
            raise ValueError("Variant does not match the reference initialization")
        report["optimizer_audit"] = asdict(audit_optimizer_parameters(model, optimizer))
        loss_function = nn.CrossEntropyLoss()
        baseline = _evaluate_domains(model, data, loss_function, device)
        report["initialization_metrics"] = baseline
        write_json(directory / "baseline_metrics.json", {"epoch": 0, **baseline})
        print(json.dumps({"variant": variant, "epoch": 0, **baseline}), flush=True)
        for epoch in range(1, config.epochs + 1):
            print(f"phase=train variant={variant} epoch={epoch}/{config.epochs}", flush=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            epoch_started = perf_counter()
            with ParameterGroupMonitor(model, optimizer) as monitor:
                training = train_one_epoch(
                    model, data.train_loader, optimizer, loss_function, device
                )
            groups = {"epoch": epoch, **monitor.summary()}
            _check_training_result(training)
            domains = _evaluate_domains(model, data, loss_function, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            metrics = EpochMetrics(
                epoch=epoch,
                train_loss=training.loss,
                train_accuracy=training.accuracy,
                validation_loss=domains["id_validation_loss"],
                validation_accuracy=domains["id_validation_accuracy"],
                mean_gradient_norm=training.mean_gradient_norm,
                mean_parameter_update_norm=training.mean_parameter_update_norm,
            )
            domain_row = {
                "epoch": epoch,
                **domains,
                "epoch_seconds": perf_counter() - epoch_started,
                "peak_cuda_memory_mb": (
                    torch.cuda.max_memory_allocated(device) / 1024**2
                    if device.type == "cuda"
                    else None
                ),
            }
            append_epoch_metrics(directory / "metrics.jsonl", metrics)
            append_json(directory / "domain_metrics.jsonl", domain_row)
            append_json(directory / "parameter_group_metrics.jsonl", groups)
            report["parameter_group_epochs"].append(groups)
            report["completed_epochs"] = epoch
            report["final_metrics"] = {**asdict(metrics), **domain_row}
            print(
                json.dumps(
                    {"variant": variant, **report["final_metrics"], "parameter_groups": groups},
                    allow_nan=False,
                ),
                flush=True,
            )
        checkpoint = directory / "model_state_dict.pt"
        torch.save(model.state_dict(), checkpoint)
        report["model_checkpoint"] = str(checkpoint)
        report["checkpoint_sha256"] = file_sha256(checkpoint)
        report["status"] = "completed"
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        report["elapsed_seconds"] = perf_counter() - started
        write_json(directory / "run_report.json", report)
    return report


def run_optimizer_training(
    reference_run: Path,
    output_dir: Path,
    epochs: int = 3,
    requested_device: str | None = None,
) -> Path:
    """Train two variants from the saved initialization; never load final reference weights."""
    if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 3:
        raise ValueError("epochs must be an integer between 1 and 3")
    import torch

    from runsleuth.optimizer_probe import state_dict_sha256
    from runsleuth.train import resolve_device

    reference_run = reference_run.expanduser().resolve()
    config, reference_manifest, checkpoint = load_reference(reference_run)
    if epochs > config.epochs:
        raise ValueError("epochs must not exceed the reference training budget")
    device = resolve_device(requested_device or config.device)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    directory = output_dir / f"pair-{timestamp}"
    directory.mkdir(parents=True)
    report_path = directory / "optimizer_training_report.json"
    report = {
        "schema_version": 1,
        "status": "running",
        "error": None,
        "task": "camelyon17_paired_optimizer_binding_experiment",
        "reference_run": str(reference_run),
        "epochs_per_variant": epochs,
        "training_mode": "from_reference_initial_state",
        "device": str(device),
        "checkpoint_selection": "fixed_final_epoch",
        "test_evaluated": False,
        "repair_acceptance_evaluated": False,
        "variants": {},
        "controls": {
            "same_initial_parameters_and_buffers": True,
            "same_selected_sample_ids": True,
            "same_loader_seed": config.seed,
            "same_subset_seed": config.subset_seed,
            "gradient_clear": "model.zero_grad before every training forward in both variants",
            "variant_difference": "optimizer constructed before versus after replacing fc",
            "performance_threshold": None,
            "ood_used_for_selection": False,
        },
    }
    print(f"experiment_directory={directory}", flush=True)
    started = perf_counter()
    try:
        write_json(
            directory / "environment.json",
            {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "cuda_runtime": torch.version.cuda,
                "device_name": torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else "CPU",
                "precision": "float32",
                "optimizer": "AdamW",
                "source_sha256": snapshot_sources(directory),
            },
        )
        report["reference_config_sha256"] = file_sha256(reference_run / "config.json")
        report["reference_initial_checkpoint_sha256"] = file_sha256(checkpoint)
        report["reference_manifest_sha256"] = file_sha256(reference_run / "data_manifest.json")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        expected_hash = state_dict_sha256(state)
        report["reference_initial_state_sha256"] = expected_hash
        for variant in ("clean", "stale_head"):
            variant_config = replace(
                config,
                epochs=epochs,
                run_name=variant,
                output_dir=str(directory),
                device=str(device),
            )
            report["variants"][variant] = _run_variant(
                variant,
                variant_config,
                reference_manifest,
                state,
                expected_hash,
                directory / variant,
                device,
            )
        report["checks"] = assess_training(report["variants"], expected_hash, epochs)
        report["mechanism_reproduced"] = all(report["checks"].values())
        report["performance_comparison"] = compare_performance(report["variants"])
        report["status"] = "completed" if report["mechanism_reproduced"] else "inconclusive"
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        # Each entered variant retains its own partial report, including failed epochs.
        for variant in ("clean", "stale_head"):
            partial = directory / variant / "run_report.json"
            if variant not in report["variants"] and partial.is_file():
                report["variants"][variant] = read_json(partial)
        raise
    finally:
        report["elapsed_seconds"] = perf_counter() - started
        write_json(report_path, report)
        print(f"optimizer_training_report={report_path}", flush=True)
    print(f"status={report['status']}", flush=True)
    print("checks=" + json.dumps(report["checks"]), flush=True)
    print("performance_comparison=" + json.dumps(report["performance_comparison"]), flush=True)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/optimizer_experiments"))
    parser.add_argument("--epochs", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    report_path = run_optimizer_training(
        args.reference_run, args.output_dir, args.epochs, args.device
    )
    if read_json(report_path)["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
