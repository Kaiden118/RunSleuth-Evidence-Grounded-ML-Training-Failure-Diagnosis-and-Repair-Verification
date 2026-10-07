"""Configuration faults on Camelyon17: a 100x learning rate and a missing optimizer step.

Both faults are global: they affect every parameter group, unlike the head-only
stale binding and frozen head. Each variant trains from the saved reference
initialization with the shared variant loop. A repair restores the faulty field
to the reference configuration before training and is verified by bounded
retraining under the unchanged development performance policy.
"""

import argparse
import json
import math
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
    _valid_domain_metrics,
    write_json,
)
from runsleuth.camelyon_variant_runner import (
    match_outcome,
    metric_differences,
    performance_checks,
    run_training_variant,
    snapshot_sources_with,
)

HIGH_LEARNING_RATE_FACTOR = 100
FAULTS = {
    "high_learning_rate": "high_learning_rate_repaired",
    "missing_optimizer_step": "missing_optimizer_step_repaired",
}
VARIANTS = ("clean", *(name for pair in FAULTS.items() for name in pair))
LEARNING_VARIANTS = ("clean", *FAULTS.values())
GROUPS = ("head", "backbone")
BUFFER_SUFFIXES = ("running_mean", "running_var", "num_batches_tracked")
EXTRA_SOURCES = ("camelyon_config_faults.py", "camelyon_variant_runner.py")
VARIANT_TASK = "camelyon17_config_fault_training_variant"


def variant_plan(reference_learning_rate: float) -> dict[str, dict]:
    """Learning rate, optimizer-step switch and repair record for every variant."""
    high = reference_learning_rate * HIGH_LEARNING_RATE_FACTOR

    def plan(learning_rate=reference_learning_rate, step=True, divergence=False, repair=None):
        return {
            "learning_rate": learning_rate,
            "optimizer_step_enabled": step,
            "record_divergence": divergence,
            "repair": repair,
        }

    def repair(field, faulty, repaired):
        return {
            "action": "restore_config_field",
            "field": field,
            "faulty_value": faulty,
            "repaired_value": repaired,
            "value_source": "reference_config",
            "timing": "before_training",
        }

    return {
        "clean": plan(),
        "high_learning_rate": plan(learning_rate=high, divergence=True),
        "high_learning_rate_repaired": plan(
            repair=repair("learning_rate", high, reference_learning_rate)
        ),
        "missing_optimizer_step": plan(step=False),
        "missing_optimizer_step_repaired": plan(
            repair=repair("optimizer_step_enabled", False, True)
        ),
    }


def _optimizer_factory(repair):
    def make_optimizer(model, variant, config):
        from runsleuth.optimizer_probe import make_probe_optimizer

        optimizer = make_probe_optimizer(
            model,
            variant="clean",
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        return optimizer, repair

    return make_optimizer


def compare_checkpoints(variants: dict, initial_state: dict) -> dict:
    """Bitwise checkpoint comparisons; BatchNorm buffers are compared separately."""
    import torch

    states = {
        name: torch.load(variant["model_checkpoint"], map_location="cpu", weights_only=True)
        for name, variant in variants.items()
        if variant.get("model_checkpoint")
    }
    parameters = [key for key in initial_state if not key.endswith(BUFFER_SUFFIXES)]
    buffers = [key for key in initial_state if key.endswith(BUFFER_SUFFIXES)]

    def equal(left: dict, right: dict, keys: list[str]) -> bool:
        return all(torch.equal(left[key], right[key]) for key in keys)

    return {
        "parameters_unchanged_from_initial": {
            name: equal(state, initial_state, parameters) for name, state in states.items()
        },
        "buffers_unchanged_from_initial": {
            name: equal(state, initial_state, buffers) for name, state in states.items()
        },
        "repaired_vs_clean_state_bitwise_equal": {
            fault: repaired in states
            and "clean" in states
            and equal(states[repaired], states["clean"], list(initial_state))
            for fault, repaired in FAULTS.items()
        },
    }


def _completed(variant: dict, epochs: int) -> bool:
    return (
        variant["status"] == "completed"
        and variant["completed_epochs"] == epochs
        and variant.get("final_metrics", {}).get("epoch") == epochs
        and [row["epoch"] for row in variant["parameter_group_epochs"]]
        == list(range(1, epochs + 1))
    )


def _learns_with_a_step_per_forward(variant: dict) -> bool:
    rows = variant["parameter_group_epochs"]
    return bool(rows) and all(
        row["optimizer_steps"] == row["training_forwards"] > 0
        and all(
            row[group]["min_trainable_tensors"] == row[group]["parameter_tensors"]
            and row[group]["min_tensors_with_gradient"] == row[group]["parameter_tensors"]
            and row[group]["mean_gradient_l2_norm"] > 0
            and row[group]["mean_parameter_update_l2_norm"] > 0
            for group in GROUPS
        )
        for row in rows
    )


def _first_epoch(variant: dict) -> dict | None:
    rows = variant.get("parameter_group_epochs", [])
    return rows[0] if rows and rows[0].get("head") is not None else None


def _inferred_learning_rate(row: dict | None) -> dict | None:
    """AdamW's first update moves each element by about lr, so norm / sqrt(n) ~ lr."""
    if row is None:
        return None
    return {
        group: row[group]["first_step_parameter_update_l2_norm"]
        / math.sqrt(row[group]["parameter_elements"])
        for group in GROUPS
    }


def assess_config_fault_training(
    variants: dict, expected_hash: str, epochs: int, comparisons: dict, plan: dict
) -> dict:
    """Mechanism checks, one repair verification per fault, and observed predictions."""
    clean, high, missing = (
        variants[name] for name in ("clean", "high_learning_rate", "missing_optimizer_step")
    )
    clean_steps = [row["optimizer_steps"] for row in clean["parameter_group_epochs"]]
    common = {
        "learning_variants_and_missing_step_completed_fixed_budget": all(
            _completed(variants[name], epochs)
            for name in (*LEARNING_VARIANTS, "missing_optimizer_step")
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
        "clean_learns_with_a_step_per_forward": _learns_with_a_step_per_forward(clean),
        "all_optimizer_audits_complete": all(
            _complete_coverage(variant["optimizer_audit"])
            and _complete_coverage(variant.get("final_optimizer_audit", {}))
            for variant in variants.values()
        ),
    }
    clean_first, high_first = _first_epoch(clean), _first_epoch(high)
    fault_checks = {
        "high_learning_rate": {
            "high_learning_rate_configured_at_factor": isclose(
                plan["high_learning_rate"]["learning_rate"],
                plan["clean"]["learning_rate"] * HIGH_LEARNING_RATE_FACTOR,
            ),
            "high_learning_rate_completed_or_recorded_divergence": _completed(high, epochs)
            or (high["status"] == "diverged" and "divergence" in high),
            "high_learning_rate_steps_every_forward": all(
                row["optimizer_steps"] == row["training_forwards"]
                for row in high["parameter_group_epochs"]
            ),
            "high_learning_rate_larger_first_updates_or_divergence": (
                high["status"] == "diverged" and high_first is None
            )
            or (
                high_first is not None
                and clean_first is not None
                and all(
                    high_first[group]["first_step_parameter_update_l2_norm"]
                    > clean_first[group]["first_step_parameter_update_l2_norm"]
                    for group in GROUPS
                )
            ),
        },
        "missing_optimizer_step": {
            "missing_step_forwards_without_steps_every_epoch": bool(
                missing["parameter_group_epochs"]
            )
            and [row["training_forwards"] for row in missing["parameter_group_epochs"]]
            == clean_steps
            and all(
                row["optimizer_steps"] == 0 and row["head"] is None and row["backbone"] is None
                for row in missing["parameter_group_epochs"]
            ),
            "missing_step_gradients_without_updates": (
                missing.get("final_metrics", {}).get("mean_gradient_norm", 0) > 0
                and missing.get("final_metrics", {}).get("mean_parameter_update_norm") == 0
            ),
            "missing_step_parameters_bitwise_unchanged": comparisons[
                "parameters_unchanged_from_initial"
            ].get("missing_optimizer_step", False),
        },
    }
    verification = {}
    for fault, repaired_name in FAULTS.items():
        repaired, faulty = variants[repaired_name], variants[fault]
        repair_checks = {
            "repair_restores_reference_value": (
                repaired.get("repair") == plan[repaired_name]["repair"]
                and plan[repaired_name]["learning_rate"] == plan["clean"]["learning_rate"]
                and plan[repaired_name]["optimizer_step_enabled"]
            ),
            "repaired_learns_with_a_step_per_forward": _learns_with_a_step_per_forward(repaired),
            "repaired_step_counts_match_clean": [
                row["optimizer_steps"] for row in repaired["parameter_group_epochs"]
            ]
            == clean_steps,
        }
        structural = {**common, **fault_checks[fault], **repair_checks}
        comparators = ("clean", fault) if _completed(faulty, epochs) else ("clean",)
        performance = performance_checks(variants, repaired_name, comparators)
        structure_verified = all(structural.values())
        performance_nonregression = all(check["passed"] for check in performance)
        verification[fault] = {
            "candidate": repaired_name,
            "decision": (
                "accepted" if structure_verified and performance_nonregression else "rejected"
            ),
            "structural_checks": structural,
            "structure_verified": structure_verified,
            "policy": dict(REPAIR_POLICY),
            "comparators": list(comparators),
            "skipped_comparators": {}
            if fault in comparators
            else {fault: f"faulty run status {faulty['status']}; no valid final metrics"},
            "performance_checks": performance,
            "performance_nonregression": performance_nonregression,
            "scope": "development policy; no statistical or performance-recovery claim",
        }
    predictions = {
        "high_learning_rate_first_update_ratio_to_clean": None
        if high_first is None or clean_first is None
        else {
            group: high_first[group]["first_step_parameter_update_l2_norm"]
            / clean_first[group]["first_step_parameter_update_l2_norm"]
            for group in GROUPS
        },
        "inferred_first_step_learning_rate": {
            "clean": _inferred_learning_rate(clean_first),
            "high_learning_rate": _inferred_learning_rate(high_first),
        },
        "high_learning_rate_outcome": {
            "status": high["status"],
            "divergence": high.get("divergence"),
            "final_metric_differences_vs_clean": metric_differences(high, clean),
        },
        "missing_step_parameters_bitwise_unchanged": fault_checks["missing_optimizer_step"][
            "missing_step_parameters_bitwise_unchanged"
        ],
        "missing_step_buffers_unchanged": comparisons["buffers_unchanged_from_initial"].get(
            "missing_optimizer_step"
        ),
        "repaired_vs_clean": {
            fault: match_outcome(
                comparisons["repaired_vs_clean_state_bitwise_equal"][fault],
                metric_differences(variants[repaired], clean),
            )
            for fault, repaired in FAULTS.items()
        },
    }
    mechanism = all(common.values()) and all(
        all(checks.values()) for checks in fault_checks.values()
    )
    return {
        "checks": {"common": common, **fault_checks},
        "mechanism_reproduced": mechanism,
        "repair_verification": verification,
        "predictions": predictions,
    }


def run_config_fault_training(
    reference_run: Path,
    output_dir: Path,
    epochs: int = 3,
    requested_device: str | None = None,
) -> Path:
    """Train all five variants from the reference initialization under a fixed budget."""
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
    plan = variant_plan(config.learning_rate)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    directory = output_dir / f"run-{timestamp}"
    directory.mkdir(parents=True)
    report_path = directory / "config_fault_training_report.json"
    report = {
        "schema_version": 1,
        "status": "running",
        "error": None,
        "task": "camelyon17_config_fault_training",
        "reference_run": str(reference_run),
        "seed": config.seed,
        "epochs_per_variant": epochs,
        "training_mode": "from_reference_initial_state",
        "device": str(device),
        "checkpoint_selection": "fixed_final_epoch",
        "test_evaluated": False,
        "budget": {"training_epoch_upper_bound": len(VARIANTS) * epochs, "model_calls": 0},
        "variant_plan": plan,
        "variants": {},
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
                "candidates": dict(FAULTS),
                "comparators": "clean and the faulty variant; a faulty variant without valid "
                "final metrics (diverged) is skipped and recorded",
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
            options = plan[variant]
            variant_config = replace(
                config,
                epochs=epochs,
                run_name=variant,
                output_dir=str(directory),
                device=str(device),
                learning_rate=options["learning_rate"],
            )
            report["variants"][variant] = run_training_variant(
                variant,
                variant_config,
                reference_manifest,
                state,
                expected_hash,
                directory / variant,
                device,
                make_optimizer=_optimizer_factory(options["repair"]),
                task=VARIANT_TASK,
                optimizer_step_enabled=options["optimizer_step_enabled"],
                allow_missing_steps=not options["optimizer_step_enabled"],
                record_divergence=options["record_divergence"],
            )
        comparisons = compare_checkpoints(report["variants"], state)
        report["checkpoint_comparisons"] = comparisons
        report.update(
            assess_config_fault_training(
                report["variants"], expected_hash, epochs, comparisons, plan
            )
        )
        report["status"] = "completed" if report["mechanism_reproduced"] else "inconclusive"
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        for variant in VARIANTS:
            partial = directory / variant / "run_report.json"
            if variant not in report["variants"] and partial.is_file():
                report["variants"][variant] = read_json(partial)
        raise
    finally:
        report["elapsed_seconds"] = perf_counter() - started
        write_json(report_path, report)
        print(f"config_fault_training_report={report_path}", flush=True)
    decisions = {fault: item["decision"] for fault, item in report["repair_verification"].items()}
    print(f"status={report['status']}", flush=True)
    print("decisions=" + json.dumps(decisions), flush=True)
    print("predictions=" + json.dumps(report["predictions"]), flush=True)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/config_fault_training"))
    parser.add_argument("--epochs", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    report_path = run_config_fault_training(
        args.reference_run, args.output_dir, args.epochs, args.device
    )
    report = read_json(report_path)
    decisions = [item["decision"] for item in report.get("repair_verification", {}).values()]
    if report["status"] != "completed" or "rejected" in decisions or not decisions:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
