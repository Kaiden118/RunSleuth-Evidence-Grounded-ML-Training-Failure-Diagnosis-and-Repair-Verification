"""Shared bounded-training loop and development gate for Camelyon17 fault experiments.

camelyon_optimizer_training.py is a recorded fixture for the source-localization
experiments, which parse its optimizer dispatch, so it is imported but never
modified. run_training_variant mirrors its _run_variant and performance_checks
mirrors its gate; tests require identical results on shared variants and identical
gate outputs. Experiments plug in their own optimizer factory and loop options.
verify_repair applies development gate v2 on top of performance_checks.
"""

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from runsleuth.camelyon_optimizer_probe import check_matching_data, file_sha256
from runsleuth.camelyon_optimizer_training import (
    REPAIR_POLICY,
    _finite_number,
    _valid_domain_metrics,
    append_json,
    snapshot_sources,
    write_json,
)

# Gate v1 required no regression against both the healthy and the faulty run. It
# rejected repairs that restored the clean model bitwise, because those silent faults
# trained as well as or better than clean. How the faulty run trains measures the
# fault, not the repair, so v2 gates on the healthy reference and records the faulty
# comparison. Revised after seeing those rejections; development policy only.
REPAIR_GATE = {
    "version": 2,
    "gating_comparator": "healthy reference of the same setup; the faulty run without one",
    "faulty_run": "informational when a healthy reference gates",
    "no_valid_comparator": "rejected",
}


def _non_finite_state(model) -> bool:
    import torch

    return any(
        not torch.isfinite(tensor).all()
        for parameter in model.parameters()
        for tensor in (parameter, parameter.grad)
        if tensor is not None
    )


def run_training_variant(
    variant,
    config,
    reference_manifest,
    state,
    expected_hash,
    directory,
    device,
    *,
    make_optimizer,
    task,
    optimizer_step_enabled=True,
    allow_missing_steps=False,
    record_divergence=False,
):
    """Train one variant from the reference state under the stale-binding protocol.

    make_optimizer(model, variant, config) returns (optimizer, repair record or None).
    With record_divergence, a non-finite parameter or gradient ends the variant with
    status "diverged" instead of failing the experiment.
    """
    import torch
    from torch import nn

    from runsleuth.camelyon import (
        _check_training_result,
        _evaluate_domains,
        build_camelyon_model,
    )
    from runsleuth.camelyon_data import build_camelyon_data
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
        "task": task,
        "error": None,
        "run_directory": str(directory),
        "optimizer_step_enabled": optimizer_step_enabled,
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
        optimizer, repair = make_optimizer(model, variant, config)
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
        diverged = False
        for epoch in range(1, config.epochs + 1):
            print(f"phase=train variant={variant} epoch={epoch}/{config.epochs}", flush=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            epoch_started = perf_counter()
            try:
                with ParameterGroupMonitor(
                    model, optimizer, allow_missing_steps=allow_missing_steps
                ) as monitor:
                    training = train_one_epoch(
                        model,
                        data.train_loader,
                        optimizer,
                        loss_function,
                        device,
                        optimizer_step_enabled=optimizer_step_enabled,
                    )
                groups = {"epoch": epoch, **monitor.summary()}
                _check_training_result(training)
            except ValueError as error:
                if not (record_divergence and _non_finite_state(model)):
                    raise
                report["divergence"] = {"epoch": epoch, "message": str(error)}
                print(f"phase=diverged variant={variant} epoch={epoch}", flush=True)
                diverged = True
                break
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
        report["status"] = "diverged" if diverged else "completed"
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        report["elapsed_seconds"] = perf_counter() - started
        write_json(directory / "run_report.json", report)
    return report


def performance_checks(variants: dict, candidate: str, comparators: tuple[str, ...]) -> list:
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


def verify_repair(
    variants: dict, candidate: str, *, reference: str | None, faulty: str | None
) -> dict:
    """Gate v2 for one candidate; callers pass only comparators with valid final metrics.

    The healthy reference gates and the faulty run is informational. Without a
    reference the faulty run gates; with neither, the gate fails instead of passing
    on zero checks.
    """
    gating = reference or faulty
    informational = faulty if reference else None
    checks = performance_checks(variants, candidate, (gating,) if gating else ())
    # True when the faulty run itself passes the gate: the fault left no regression
    # that validation metrics alone could catch.
    faulty_passes = None
    if reference and faulty:
        faulty_passes = all(
            check["passed"] for check in performance_checks(variants, faulty, (reference,))
        )
    return {
        "gate": dict(REPAIR_GATE),
        "policy": dict(REPAIR_POLICY),
        "comparators": [gating] if gating else [],
        "performance_checks": checks,
        "performance_nonregression": bool(checks) and all(check["passed"] for check in checks),
        "informational_comparators": [informational] if informational else [],
        "informational_checks": performance_checks(
            variants, candidate, (informational,) if informational else ()
        ),
        "faulty_passes_gate_vs_reference": faulty_passes,
    }


def snapshot_sources_with(directory: Path, extra_sources: tuple[str, ...]) -> dict[str, str]:
    """Snapshot the stale-binding sources plus an experiment's own modules."""
    hashes = snapshot_sources(directory)
    destination = directory / "source_snapshot"
    destination.mkdir(exist_ok=True)
    for name in extra_sources:
        shutil.copyfile(Path(__file__).parent / name, destination / name)
        hashes[name] = file_sha256(destination / name)
    return hashes


def metric_differences(left: dict, right: dict) -> dict:
    """Signed final ID/OOD metric differences; None when either value is invalid."""
    differences = {}
    for domain in ("id", "ood"):
        for metric in ("accuracy", "loss"):
            key = f"{domain}_validation_{metric}"
            values = (
                left.get("final_metrics", {}).get(key),
                right.get("final_metrics", {}).get(key),
            )
            valid = all(_finite_number(value) for value in values)
            differences[key] = values[0] - values[1] if valid else None
    return differences


def match_outcome(bitwise_equal: bool, differences: dict) -> str:
    """Report a match category without a post hoc tolerance."""
    if bitwise_equal:
        return "bitwise_equal"
    if all(value == 0 for value in differences.values()):
        return "metrics_equal_not_bitwise"
    return "differs"
