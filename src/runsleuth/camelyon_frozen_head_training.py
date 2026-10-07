"""Stage 2 of the frozen-head experiment: bounded training from a saved initialization.

Design and predictions: docs/camelyon17_frozen_head.md. Trains clean, stale_head,
frozen_head and frozen_head_repaired, then applies structural checks, the unchanged
development performance policy, and bitwise checkpoint comparisons for the
observed predictions.

camelyon_optimizer_training.py is a recorded fixture for the source-localization
experiments, which parse its optimizer dispatch, so it is imported but never
modified. The variant loop and performance gate below mirror it; tests require
identical results on the shared clean and stale_head variants and identical gate
outputs.
"""

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
from runsleuth.camelyon_optimizer_training import (
    REPAIR_POLICY,
    _complete_coverage,
    _finite_number,
    _valid_domain_metrics,
    append_json,
    snapshot_sources,
    write_json,
)

VARIANTS = ("clean", "stale_head", "frozen_head", "frozen_head_repaired")
COMPARATORS = ("clean", "frozen_head")
FROZEN_HEAD_NAMES = ["fc.bias", "fc.weight"]
EXTRA_SOURCES = ("camelyon_frozen_head_training.py", "frozen_head_probe.py")
NORMS = ("mean_gradient_l2_norm", "mean_parameter_update_l2_norm")


def _run_frozen_variant(
    variant, config, reference_manifest, state, expected_hash, directory, device
):
    """Mirror camelyon_optimizer_training._run_variant except for the optimizer factory."""
    import torch
    from torch import nn

    from runsleuth.camelyon import (
        _check_training_result,
        _evaluate_domains,
        build_camelyon_model,
    )
    from runsleuth.camelyon_data import build_camelyon_data
    from runsleuth.frozen_head_probe import make_variant_optimizer
    from runsleuth.optimizer_audit import audit_optimizer_parameters
    from runsleuth.optimizer_probe import state_dict_sha256
    from runsleuth.parameter_group_monitor import ParameterGroupMonitor
    from runsleuth.telemetry import EpochMetrics, append_epoch_metrics
    from runsleuth.train import seed_everything, train_one_epoch

    directory.mkdir()
    config.save(directory / "config.json")
    report = {
        "variant": variant,
        "status": "running",
        "completed_epochs": 0,
        "task": "camelyon17_frozen_head_training_variant",
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
        optimizer, repair = make_variant_optimizer(
            model,
            variant=variant,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        if repair is not None:
            report["repair"] = repair
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
        report["final_optimizer_audit"] = asdict(audit_optimizer_parameters(model, optimizer))
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


def _performance_checks(variants: dict, candidate: str, comparators: tuple[str, ...]) -> list:
    """Mirror the stale-binding experiment's development gate for one candidate."""
    performance = []
    repaired_metrics = variants[candidate]["final_metrics"]
    for comparator in comparators:
        reference_metrics = variants[comparator]["final_metrics"]
        valid = _valid_domain_metrics(repaired_metrics) and _valid_domain_metrics(reference_metrics)
        for domain in ("id", "ood"):
            for metric in ("accuracy", "loss"):
                actual = repaired_metrics[f"{domain}_validation_{metric}"]
                reference = reference_metrics[f"{domain}_validation_{metric}"]
                accuracy = metric == "accuracy"
                threshold = REPAIR_POLICY["max_accuracy_drop" if accuracy else "max_loss_ratio"]
                observed = None
                if valid:
                    observed = (
                        reference - actual
                        if accuracy
                        else (actual / reference if reference > 0 else None)
                    )
                passed = valid and (
                    actual >= reference - threshold if accuracy else actual <= reference * threshold
                )
                performance.append(
                    {
                        "comparator": comparator,
                        "domain": domain,
                        "metric": "accuracy_drop" if accuracy else "loss_ratio",
                        "repaired_value": actual if _finite_number(actual) else None,
                        "reference_value": reference if _finite_number(reference) else None,
                        "observed_value": observed if _finite_number(observed) else None,
                        "operator": "<=",
                        "threshold": threshold,
                        "passed": passed,
                    }
                )
    return performance


def _snapshot_sources(directory: Path) -> dict[str, str]:
    hashes = snapshot_sources(directory)
    destination = directory / "source_snapshot"
    destination.mkdir(exist_ok=True)
    for name in EXTRA_SOURCES:
        shutil.copyfile(Path(__file__).parent / name, destination / name)
        hashes[name] = file_sha256(destination / name)
    return hashes


def _all_tensors_trainable_with_gradients(row: dict, group: str) -> bool:
    counts = row[group]
    return (
        counts["min_trainable_tensors"] == counts["parameter_tensors"]
        and counts["min_tensors_with_gradient"] == counts["parameter_tensors"]
    )


def _learns(row: dict, group: str) -> bool:
    return _all_tensors_trainable_with_gradients(row, group) and all(
        row[group][metric] > 0 for metric in NORMS
    )


def _frozen(row: dict, group: str) -> bool:
    counts = row[group]
    return (
        counts["max_trainable_tensors"] == 0
        and counts["max_tensors_with_gradient"] == 0
        and counts["mean_parameter_update_l2_norm"] == 0
    )


def _metric_differences(left: dict, right: dict) -> dict:
    differences = {}
    for domain in ("id", "ood"):
        for metric in ("accuracy", "loss"):
            key = f"{domain}_validation_{metric}"
            values = (left["final_metrics"].get(key), right["final_metrics"].get(key))
            valid = all(_finite_number(value) for value in values)
            differences[key] = values[0] - values[1] if valid else None
    return differences


def _match_outcome(bitwise_equal: bool, differences: dict) -> str:
    if bitwise_equal:
        return "bitwise_equal"
    if all(value == 0 for value in differences.values()):
        return "metrics_equal_not_bitwise"
    return "differs"


def compare_checkpoints(variants: dict, initial_state: dict) -> dict:
    """Compare fixed final checkpoints bitwise; no tolerance is applied."""
    import torch

    states = {
        name: torch.load(variant["model_checkpoint"], map_location="cpu", weights_only=True)
        for name, variant in variants.items()
    }
    head = [key for key in initial_state if key.startswith("fc.")]
    rest = [key for key in initial_state if not key.startswith("fc.")]

    def equal(left: dict, right: dict, keys: list[str]) -> bool:
        return all(torch.equal(left[key], right[key]) for key in keys)

    return {
        "head_unchanged_from_initial": {
            name: equal(state, initial_state, head) for name, state in states.items()
        },
        "frozen_vs_stale_non_head_state_bitwise_equal": equal(
            states["frozen_head"], states["stale_head"], rest
        ),
        "repaired_vs_clean_state_bitwise_equal": equal(
            states["frozen_head_repaired"], states["clean"], list(initial_state)
        ),
    }


def assess_frozen_head_training(
    variants: dict, expected_hash: str, epochs: int, comparisons: dict
) -> dict:
    """Structural checks, the development performance gate and observed predictions."""
    clean, stale, frozen, repaired = (variants[name] for name in VARIANTS)
    rows = {name: variant["parameter_group_epochs"] for name, variant in variants.items()}
    stale_audit = stale["optimizer_audit"]
    repair = repaired.get("repair", {})
    before = repair.get("trainability_before", {})
    checks = {
        "all_variants_completed_fixed_budget": all(
            variant["status"] == "completed"
            and variant["completed_epochs"] == epochs
            and variant["final_metrics"].get("epoch") == epochs
            and [row["epoch"] for row in variant["parameter_group_epochs"]]
            == list(range(1, epochs + 1))
            for variant in variants.values()
        ),
        "same_reference_initial_state": all(
            variant["initial_state_sha256"] == expected_hash for variant in variants.values()
        ),
        "same_initial_validation_metrics": all(
            _valid_domain_metrics(variant["initialization_metrics"])
            and all(
                isclose(value, variant["initialization_metrics"][key], rel_tol=1e-6, abs_tol=1e-6)
                for key, value in clean["initialization_metrics"].items()
            )
            for variant in variants.values()
        ),
        "same_optimizer_step_counts": all(
            [row["optimizer_steps"] for row in variant_rows]
            == [row["optimizer_steps"] for row in rows["clean"]]
            and all(
                type(row["optimizer_steps"]) is int and row["optimizer_steps"] > 0
                for row in variant_rows
            )
            for variant_rows in rows.values()
        ),
        "all_recorded_group_norms_finite_nonnegative": all(
            _finite_number(row[group][metric]) and row[group][metric] >= 0
            for variant_rows in rows.values()
            for row in variant_rows
            for group in ("head", "backbone")
            for metric in NORMS
        ),
        "clean_head_and_backbone_learn_every_epoch": bool(rows["clean"])
        and all(_learns(row, group) for row in rows["clean"] for group in ("head", "backbone")),
        "stale_head_has_gradients_without_updates_every_epoch": (
            set(stale_audit["missing_trainable_names"]) == {"fc.weight", "fc.bias"}
            and stale_audit["foreign_parameter_tensors"] == 2
            and bool(rows["stale_head"])
            and all(
                _all_tensors_trainable_with_gradients(row, "head")
                and row["head"]["mean_gradient_l2_norm"] > 0
                and row["head"]["mean_parameter_update_l2_norm"] == 0
                for row in rows["stale_head"]
            )
        ),
        "frozen_head_has_no_gradients_or_updates_every_epoch": bool(rows["frozen_head"])
        and all(_frozen(row, "head") for row in rows["frozen_head"]),
        "frozen_head_passes_existing_coverage_audit": (
            _complete_coverage(frozen["optimizer_audit"])
            and _complete_coverage(frozen.get("final_optimizer_audit", {}))
        ),
        "all_backbones_learn_every_epoch": all(
            bool(variant_rows) and all(_learns(row, "backbone") for row in variant_rows)
            for variant_rows in rows.values()
        ),
    }
    repair_checks = {
        "reproduced_frozen_head_before_repair": (
            before.get("head", {}).get("trainable_tensors") == 0
            and before.get("head", {}).get("frozen_names") == FROZEN_HEAD_NAMES
            and before.get("backbone", {}).get("frozen_names") == []
        ),
        "repair_before_first_step_with_empty_state": (
            repair.get("action") == "restore_head_requires_grad"
            and repair.get("timing") == "before_first_optimizer_step"
            and type(repair.get("optimizer_state_entries_before")) is int
            and repair["optimizer_state_entries_before"] == 0
        ),
        "repair_preserves_model_state": (
            repair.get("model_state_sha256_before")
            == repair.get("model_state_sha256_after")
            == expected_hash
        ),
        "repaired_optimizer_covers_current_parameters_initially_and_finally": (
            _complete_coverage(repaired["optimizer_audit"])
            and _complete_coverage(repaired.get("final_optimizer_audit", {}))
        ),
        "repaired_all_tensors_trainable_with_gradients_and_updates_every_epoch": bool(
            rows["frozen_head_repaired"]
        )
        and all(
            _learns(row, group)
            for row in rows["frozen_head_repaired"]
            for group in ("head", "backbone")
        ),
    }
    structural = {**checks, **repair_checks}
    performance = _performance_checks(variants, "frozen_head_repaired", COMPARATORS)
    structure_verified = all(structural.values())
    performance_nonregression = all(check["passed"] for check in performance)
    frozen_vs_stale = _metric_differences(frozen, stale)
    repaired_vs_clean = _metric_differences(repaired, clean)
    predictions = {
        "P1_frozen_head_bitwise_unchanged": (
            comparisons["head_unchanged_from_initial"]["frozen_head"]
            and bool(rows["frozen_head"])
            and all(
                row["head"]["mean_parameter_update_l2_norm"] == 0 for row in rows["frozen_head"]
            )
        ),
        "P2_frozen_vs_stale": {
            "outcome": _match_outcome(
                comparisons["frozen_vs_stale_non_head_state_bitwise_equal"], frozen_vs_stale
            ),
            "final_metric_differences": frozen_vs_stale,
        },
        "P3_repaired_vs_clean": {
            "outcome": _match_outcome(
                comparisons["repaired_vs_clean_state_bitwise_equal"], repaired_vs_clean
            ),
            "final_metric_differences": repaired_vs_clean,
        },
    }
    return {
        "checks": checks,
        "mechanism_reproduced": all(checks.values()),
        "repair_verification": {
            "decision": (
                "accepted" if structure_verified and performance_nonregression else "rejected"
            ),
            "structural_checks": structural,
            "structure_verified": structure_verified,
            "policy": dict(REPAIR_POLICY),
            "comparators": list(COMPARATORS),
            "performance_checks": performance,
            "performance_nonregression": performance_nonregression,
            "scope": "development policy; no statistical or performance-recovery claim",
        },
        "predictions": predictions,
    }


def run_frozen_head_training(
    reference_run: Path,
    output_dir: Path,
    epochs: int = 3,
    requested_device: str | None = None,
) -> Path:
    """Train the four stage-2 variants from the reference initialization under a fixed budget."""
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
    directory = output_dir / f"run-{timestamp}"
    directory.mkdir(parents=True)
    report_path = directory / "frozen_head_training_report.json"
    report = {
        "schema_version": 1,
        "status": "running",
        "error": None,
        "task": "camelyon17_frozen_head_training",
        "design": "docs/camelyon17_frozen_head.md",
        "reference_run": str(reference_run),
        "seed": config.seed,
        "epochs_per_variant": epochs,
        "training_mode": "from_reference_initial_state",
        "device": str(device),
        "checkpoint_selection": "fixed_final_epoch",
        "test_evaluated": False,
        "budget": {"training_epoch_upper_bound": len(VARIANTS) * epochs, "model_calls": 0},
        "variants": {},
        "controls": {
            "same_initial_parameters_and_buffers": True,
            "same_selected_sample_ids": True,
            "same_loader_seed": config.seed,
            "same_subset_seed": config.subset_seed,
            "gradient_clear": "model.zero_grad before every training forward in all variants",
            "variant_difference": (
                "clean, stale head, frozen head, and frozen head with requires_grad "
                "restored before training"
            ),
            "performance_threshold": dict(REPAIR_POLICY),
            "ood_used_for_repair_acceptance": True,
            "ood_used_for_checkpoint_selection": False,
        },
    }
    print(f"experiment_directory={directory}", flush=True)
    started = perf_counter()
    try:
        policy_path = directory / "repair_verification_policy.json"
        write_json(
            policy_path,
            {
                "policy": dict(REPAIR_POLICY),
                "scope": "development experiment; not a held-out benchmark",
                "epochs_per_variant": epochs,
                "candidate": "frozen_head_repaired",
                "comparators": list(COMPARATORS),
                "domains": ["id", "ood"],
                "checkpoint_selection": "fixed_final_epoch",
                "ood_used_for_repair_acceptance": True,
                "ood_used_for_checkpoint_selection": False,
            },
        )
        report["repair_verification_policy"] = {
            "path": str(policy_path),
            "sha256": file_sha256(policy_path),
        }
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
                "source_sha256": _snapshot_sources(directory),
            },
        )
        report["reference_config_sha256"] = file_sha256(reference_run / "config.json")
        report["reference_initial_checkpoint_sha256"] = file_sha256(checkpoint)
        report["reference_manifest_sha256"] = file_sha256(reference_run / "data_manifest.json")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        expected_hash = state_dict_sha256(state)
        report["reference_initial_state_sha256"] = expected_hash
        for variant in VARIANTS:
            variant_config = replace(
                config,
                epochs=epochs,
                run_name=variant,
                output_dir=str(directory),
                device=str(device),
            )
            report["variants"][variant] = _run_frozen_variant(
                variant,
                variant_config,
                reference_manifest,
                state,
                expected_hash,
                directory / variant,
                device,
            )
        comparisons = compare_checkpoints(report["variants"], state)
        report["checkpoint_comparisons"] = comparisons
        report.update(
            assess_frozen_head_training(report["variants"], expected_hash, epochs, comparisons)
        )
        report["status"] = "completed" if report["mechanism_reproduced"] else "inconclusive"
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        # Each entered variant retains its own partial report, including failed epochs.
        for variant in VARIANTS:
            partial = directory / variant / "run_report.json"
            if variant not in report["variants"] and partial.is_file():
                report["variants"][variant] = read_json(partial)
        raise
    finally:
        report["elapsed_seconds"] = perf_counter() - started
        write_json(report_path, report)
        print(f"frozen_head_training_report={report_path}", flush=True)
    verification = report["repair_verification"]
    print(f"status={report['status']}", flush=True)
    print("checks=" + json.dumps(report["checks"]), flush=True)
    print(
        f"decision={verification['decision']} "
        f"structure_verified={verification['structure_verified']} "
        f"performance_nonregression={verification['performance_nonregression']}",
        flush=True,
    )
    print("predictions=" + json.dumps(report["predictions"]), flush=True)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/frozen_head_training"))
    parser.add_argument("--epochs", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    report_path = run_frozen_head_training(
        args.reference_run, args.output_dir, args.epochs, args.device
    )
    report = read_json(report_path)
    if (
        report["status"] != "completed"
        or report.get("repair_verification", {}).get("decision") == "rejected"
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
