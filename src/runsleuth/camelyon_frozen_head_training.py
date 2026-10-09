"""Stage 2 of the frozen-head experiment: bounded training from a saved initialization.

Design and predictions: docs/camelyon17_frozen_head.md. Trains clean, stale_head,
frozen_head and frozen_head_repaired, then applies structural checks, development
gate v2 (camelyon_variant_runner.REPAIR_GATE), and bitwise checkpoint comparisons
for the observed predictions.

The training loop and development gate come from camelyon_variant_runner, which
mirrors the stale-binding experiment.
"""

import argparse
import json
import platform
from dataclasses import replace
from datetime import UTC, datetime
from math import isclose
from pathlib import Path
from time import perf_counter

from runsleuth.camelyon_optimizer_probe import file_sha256, load_reference, read_json
from runsleuth.camelyon_optimizer_training import (
    REPAIR_POLICY,
    _complete_coverage,
    _finite_number,
    _valid_domain_metrics,
    write_json,
)
from runsleuth.camelyon_variant_runner import (
    REPAIR_GATE,
    match_outcome,
    metric_differences,
    run_training_variant,
    snapshot_sources_with,
    verify_repair,
)

VARIANTS = ("clean", "stale_head", "frozen_head", "frozen_head_repaired")
FROZEN_HEAD_NAMES = ["fc.bias", "fc.weight"]
EXTRA_SOURCES = (
    "camelyon_frozen_head_training.py",
    "camelyon_variant_runner.py",
    "frozen_head_probe.py",
)
VARIANT_TASK = "camelyon17_frozen_head_training_variant"
NORMS = ("mean_gradient_l2_norm", "mean_parameter_update_l2_norm")


def _frozen_head_optimizer(model, variant, config):
    from runsleuth.frozen_head_probe import make_variant_optimizer

    return make_variant_optimizer(
        model, variant=variant, learning_rate=config.learning_rate, weight_decay=config.weight_decay
    )


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
    gate = verify_repair(variants, "frozen_head_repaired", reference="clean", faulty="frozen_head")
    structure_verified = all(structural.values())
    frozen_vs_stale = metric_differences(frozen, stale)
    repaired_vs_clean = metric_differences(repaired, clean)
    predictions = {
        "P1_frozen_head_bitwise_unchanged": (
            comparisons["head_unchanged_from_initial"]["frozen_head"]
            and bool(rows["frozen_head"])
            and all(
                row["head"]["mean_parameter_update_l2_norm"] == 0 for row in rows["frozen_head"]
            )
        ),
        "P2_frozen_vs_stale": {
            "outcome": match_outcome(
                comparisons["frozen_vs_stale_non_head_state_bitwise_equal"], frozen_vs_stale
            ),
            "final_metric_differences": frozen_vs_stale,
        },
        "P3_repaired_vs_clean": {
            "outcome": match_outcome(
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
                "accepted"
                if structure_verified and gate["performance_nonregression"]
                else "rejected"
            ),
            "structural_checks": structural,
            "structure_verified": structure_verified,
            **gate,
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
                "gate": dict(REPAIR_GATE),
                "comparators": ["clean"],
                "informational_comparators": ["frozen_head"],
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
                "source_sha256": snapshot_sources_with(directory, EXTRA_SOURCES),
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
            report["variants"][variant] = run_training_variant(
                variant,
                variant_config,
                reference_manifest,
                state,
                expected_hash,
                directory / variant,
                device,
                make_optimizer=_frozen_head_optimizer,
                task=VARIANT_TASK,
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
