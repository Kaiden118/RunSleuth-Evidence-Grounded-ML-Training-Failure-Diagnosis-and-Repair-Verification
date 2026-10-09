"""Training-loop faults on Camelyon17: training in eval mode, a scheduler stepped per batch.

Every variant anneals the learning rate with cosine over its epochs. The clean loop
calls model.train() and steps the schedule once per epoch. train_in_eval_mode never
calls model.train(), so after the initial evaluation every epoch trains in eval mode:
BatchNorm normalizes with its running statistics and never updates them.
scheduler_stepped_per_batch steps the same schedule after every optimizer step, so the
rate cycles within an epoch. Each variant trains from the saved reference
initialization with the shared variant loop; a repair restores the clean loop setting
before training and is verified under development gate v2.
"""

import argparse
import json
import platform
from dataclasses import replace
from datetime import UTC, datetime
from math import isclose
from pathlib import Path
from time import perf_counter

from runsleuth.camelyon_config_faults import (
    BUFFER_SUFFIXES,
    _completed,
    _learns_with_a_step_per_forward,
    _optimizer_factory,
)
from runsleuth.camelyon_optimizer_probe import file_sha256, load_reference, read_json
from runsleuth.camelyon_optimizer_training import (
    _complete_coverage,
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

FAULTS = {
    "train_in_eval_mode": "train_in_eval_mode_repaired",
    "scheduler_stepped_per_batch": "scheduler_stepped_per_batch_repaired",
}
VARIANTS = ("clean", *(name for pair in FAULTS.items() for name in pair))
EXTRA_SOURCES = ("camelyon_loop_faults.py", "camelyon_variant_runner.py")
VARIANT_TASK = "camelyon17_loop_fault_training_variant"


def variant_plan() -> dict[str, dict]:
    """Training mode, learning-rate schedule and repair record for every variant."""

    def plan(train_mode=True, lr_schedule="cosine_per_epoch", repair=None):
        return {"train_mode": train_mode, "lr_schedule": lr_schedule, "repair": repair}

    def repair(field, faulty, repaired):
        return {
            "action": "restore_loop_setting",
            "field": field,
            "faulty_value": faulty,
            "repaired_value": repaired,
            "value_source": "clean_loop",
            "timing": "before_training",
        }

    return {
        "clean": plan(),
        "train_in_eval_mode": plan(train_mode=False),
        "train_in_eval_mode_repaired": plan(repair=repair("train_mode", False, True)),
        "scheduler_stepped_per_batch": plan(lr_schedule="cosine_per_batch"),
        "scheduler_stepped_per_batch_repaired": plan(
            repair=repair("lr_schedule", "cosine_per_batch", "cosine_per_epoch")
        ),
    }


def compare_checkpoints(variants: dict, initial_state: dict) -> dict:
    """Bitwise checkpoint comparisons, with BatchNorm buffers apart from parameters."""
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
        "batchnorm_buffers": len(buffers),
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


def _rows(variant: dict) -> list[dict]:
    return variant.get("parameter_group_epochs", [])


def _trains_in_train_mode_with_constant_epoch_rates(variant: dict) -> bool:
    return bool(_rows(variant)) and all(
        row["eval_mode_training_forwards"] == 0 and row["learning_rate"]["changes"] == 0
        for row in _rows(variant)
    )


def assess_loop_fault_training(
    variants: dict, expected_hash: str, epochs: int, comparisons: dict, plan: dict
) -> dict:
    """Mechanism checks, one repair verification per fault, and observed predictions."""
    clean = variants["clean"]
    eval_mode, per_batch = (variants[name] for name in FAULTS)
    clean_rates = [row["learning_rate"]["first"] for row in _rows(clean)]
    common = {
        "all_variants_completed_fixed_budget": all(
            _completed(variant, epochs) for variant in variants.values()
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
        "all_optimizer_audits_complete": all(
            _complete_coverage(variant["optimizer_audit"])
            and _complete_coverage(variant.get("final_optimizer_audit", {}))
            for variant in variants.values()
        ),
        "clean_learns_with_a_step_per_forward": _learns_with_a_step_per_forward(clean),
        "clean_trains_in_train_mode_with_constant_epoch_rates": (
            _trains_in_train_mode_with_constant_epoch_rates(clean)
        ),
        "clean_rate_decays_across_epochs": len(clean_rates) == epochs
        and all(
            later < earlier for earlier, later in zip(clean_rates, clean_rates[1:], strict=False)
        ),
    }
    fault_checks = {
        "train_in_eval_mode": {
            "eval_mode_variant_trains_every_forward_in_eval_mode": bool(_rows(eval_mode))
            and all(
                row["eval_mode_training_forwards"] == row["training_forwards"] > 0
                for row in _rows(eval_mode)
            ),
            "eval_mode_variant_still_updates_parameters": _learns_with_a_step_per_forward(
                eval_mode
            ),
            "eval_mode_variant_batchnorm_statistics_unchanged": comparisons[
                "buffers_unchanged_from_initial"
            ].get("train_in_eval_mode", False),
        },
        "scheduler_stepped_per_batch": {
            "per_batch_rate_changes_at_every_step": bool(_rows(per_batch))
            and all(
                row["learning_rate"]["changes"] == row["optimizer_steps"] - 1 > 0
                for row in _rows(per_batch)
            ),
            "per_batch_rate_rebounds_within_every_epoch": bool(_rows(per_batch))
            and all(row["learning_rate"]["rebounds"] > 0 for row in _rows(per_batch)),
            "per_batch_variant_trains_in_train_mode": all(
                row["eval_mode_training_forwards"] == 0 for row in _rows(per_batch)
            ),
        },
    }
    verification = {}
    for fault, repaired_name in FAULTS.items():
        repaired = variants[repaired_name]
        repair_checks = {
            "repair_restores_the_clean_loop": (
                repaired.get("repair") == plan[repaired_name]["repair"]
                and {key: plan[repaired_name][key] for key in ("train_mode", "lr_schedule")}
                == {key: plan["clean"][key] for key in ("train_mode", "lr_schedule")}
            ),
            "repaired_learns_with_a_step_per_forward": _learns_with_a_step_per_forward(repaired),
            "repaired_trains_in_train_mode_with_constant_epoch_rates": (
                _trains_in_train_mode_with_constant_epoch_rates(repaired)
            ),
        }
        structural = {**common, **fault_checks[fault], **repair_checks}
        gate = verify_repair(
            variants,
            repaired_name,
            reference="clean",
            faulty=fault if _completed(variants[fault], epochs) else None,
        )
        structure_verified = all(structural.values())
        passed = structure_verified and gate["performance_nonregression"]
        verification[fault] = {
            "candidate": repaired_name,
            "decision": "accepted" if passed else "rejected",
            "structural_checks": structural,
            "structure_verified": structure_verified,
            **gate,
            "scope": "development policy; no statistical or performance-recovery claim",
        }
    predictions = {
        "clean_epoch_learning_rates": clean_rates,
        "per_batch_learning_rate_range": [
            {key: row["learning_rate"][key] for key in ("min", "max", "changes", "rebounds")}
            for row in _rows(per_batch)
        ],
        "eval_mode_batchnorm_statistics_unchanged": comparisons[
            "buffers_unchanged_from_initial"
        ].get("train_in_eval_mode"),
        "final_metric_differences_vs_clean": {
            fault: metric_differences(variants[fault], clean) for fault in FAULTS
        },
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


def run_loop_fault_training(
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
    plan = variant_plan()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    directory = output_dir / f"run-{timestamp}"
    directory.mkdir(parents=True)
    report_path = directory / "loop_fault_training_report.json"
    report = {
        "schema_version": 1,
        "status": "running",
        "error": None,
        "task": "camelyon17_loop_fault_training",
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
                "gate": dict(REPAIR_GATE),
                "scope": "development experiment; not a held-out benchmark",
                "epochs_per_variant": epochs,
                "candidates": dict(FAULTS),
                "comparators": "clean gates; the faulty variant is informational",
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
                train_mode=options["train_mode"],
                lr_schedule=options["lr_schedule"],
            )
        comparisons = compare_checkpoints(report["variants"], state)
        report["checkpoint_comparisons"] = comparisons
        report.update(
            assess_loop_fault_training(report["variants"], expected_hash, epochs, comparisons, plan)
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
        print(f"loop_fault_training_report={report_path}", flush=True)
    decisions = {fault: item["decision"] for fault, item in report["repair_verification"].items()}
    print(f"status={report['status']}", flush=True)
    print("decisions=" + json.dumps(decisions), flush=True)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/loop_fault_training"))
    parser.add_argument("--epochs", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    report = read_json(
        run_loop_fault_training(args.reference_run, args.output_dir, args.epochs, args.device)
    )
    decisions = [item["decision"] for item in report.get("repair_verification", {}).values()]
    if report["status"] != "completed" or "rejected" in decisions or not decisions:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
