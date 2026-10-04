"""Train an exact validated source proposal from a recorded initial checkpoint."""

import json
import math
import platform
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from runsleuth.camelyon_optimizer_probe import file_sha256
from runsleuth.camelyon_optimizer_training import append_json, write_json
from runsleuth.optimizer_evidence import _data_identity, _same
from runsleuth.optimizer_source_repair import _sha
from runsleuth.optimizer_source_runtime import _validated_constructor_code


def train_source_patch(
    job: dict,
    directory: Path,
    requested_device: str | None = None,
) -> dict:
    """Execute three final-epoch training passes; retain failed-attempt accounting.

    The controller supplies a provenance-checked job. No snapshot module is
    imported: only its allowlisted constructor is compiled, using trusted
    bindings. Healthy and faulty comparator runs are not rerun here.
    """
    config = job["config"]
    if (
        type(config.epochs) is not int
        or config.epochs != 3
        or type(job.get("epochs", 3)) is not int
        or job.get("epochs", 3) != 3
    ):
        raise ValueError("Source training requires exactly three epochs")
    if requested_device not in (None, "auto", "cpu", "cuda"):
        raise ValueError("Device must be auto, cpu, or cuda")
    _same(config.seed, job["seed"], "Training seed")
    _, proposed_code = _validated_constructor_code(job["original_source"], job["proposed_source"])
    directory.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "task": "camelyon17_source_patch_training",
        "variant": "source_patched",
        "status": "running",
        "error": None,
        "run_directory": str(directory),
        "completed_epochs": 0,
        "parameter_group_epochs": [],
        "optimizer_steps_recorded": 0,
        "step_accounting_complete": True,
        "partial_epoch_possible": False,
        "training_started": False,
        "source_patch_executed": False,
        "execution_scope": "extracted_make_probe_optimizer_only",
        "training_mode": "from_reference_initial_state",
        "checkpoint_selection": "fixed_final_epoch",
        "test_evaluated": False,
        "source_execution": {
            "original_source_sha256": _sha(job["original_source"].encode("utf-8")),
            "proposed_source_sha256": _sha(job["proposed_source"].encode("utf-8")),
            "constructor_variant": "stale_head",
        },
    }
    started = perf_counter()
    try:
        config.save(directory / "config.json")
        # Check checkpoint bytes again immediately before deserialization.
        _same(
            file_sha256(job["checkpoint"]),
            job["baseline"]["files_sha256"]["initial_state_dict.pt"],
            "Initial checkpoint before training load",
        )
        import torch
        from torch import nn

        from runsleuth.camelyon import (
            _check_training_result,
            _evaluate_domains,
            build_camelyon_model,
        )
        from runsleuth.camelyon_data import build_camelyon_data
        from runsleuth.optimizer_audit import audit_optimizer_parameters
        from runsleuth.optimizer_probe import _VARIANTS, _parameter_groups, state_dict_sha256
        from runsleuth.parameter_group_monitor import ParameterGroupMonitor
        from runsleuth.telemetry import EpochMetrics, append_epoch_metrics
        from runsleuth.train import resolve_device, seed_everything, train_one_epoch

        device = resolve_device(requested_device or config.device)
        expected_environment = job.get("expected_environment")
        if expected_environment is not None:
            _same(str(torch.__version__), expected_environment["torch"], "PyTorch training version")
            _same(str(device), expected_environment["device"], "Training device")
        report["environment"] = {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "device": str(device),
            "cuda_runtime": torch.version.cuda,
        }
        state = torch.load(job["checkpoint"], map_location="cpu", weights_only=True)
        _same(state_dict_sha256(state), job["initial_state_sha256"], "Loaded initial model state")
        if any(not torch.isfinite(value).all().item() for value in state.values()):
            raise ValueError("Initial model state must be finite")
        seed_everything(config.seed)
        print(f"phase=load_cached_data case={job['id']}", flush=True)
        data = build_camelyon_data(config, device=device, download=False)
        identity, _ = _data_identity(data.manifest, config)
        expected_identity, _ = _data_identity(job["manifest"], config)
        _same(identity, expected_identity, "Training selected data and transform")
        write_json(directory / "data_manifest.json", data.manifest)
        report["data_manifest_sha256"] = file_sha256(directory / "data_manifest.json")
        model = build_camelyon_model(pretrained=False)
        model.load_state_dict(state, strict=True)
        model.to(device)
        before = state_dict_sha256(model.state_dict())
        _same(before, job["initial_state_sha256"], "Model initialization before constructor")
        namespace = {
            "__builtins__": {"ValueError": ValueError},
            "torch": torch,
            "math": math,
            "deepcopy": deepcopy,
            "_VARIANTS": _VARIANTS,
            "_parameter_groups": _parameter_groups,
        }
        exec(proposed_code, namespace)
        optimizer = namespace["make_probe_optimizer"](
            model,
            variant="stale_head",
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        report["source_patch_executed"] = True
        after = state_dict_sha256(model.state_dict())
        report.update(
            initial_state_sha256=after,
            model_state_sha256_before_constructor=before,
            model_state_sha256_after_constructor=after,
            optimizer_state_entries_before=len(optimizer.state),
            optimizer_audit=asdict(audit_optimizer_parameters(model, optimizer)),
        )
        _same(after, before, "Constructor must preserve model initialization")
        if report["optimizer_state_entries_before"] != 0:
            raise ValueError("Source training requires a fresh optimizer")
        loss_function = nn.CrossEntropyLoss()
        baseline = _evaluate_domains(model, data, loss_function, device)
        report["initialization_metrics"] = baseline
        write_json(directory / "baseline_metrics.json", {"epoch": 0, **baseline})
        print(json.dumps({"case": job["id"], "epoch": 0, **baseline}), flush=True)
        for epoch in range(1, 4):
            print(f"phase=source_train case={job['id']} epoch={epoch}/3", flush=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            epoch_started = perf_counter()
            report.update(
                training_started=True, partial_epoch_possible=True, step_accounting_complete=False
            )
            with ParameterGroupMonitor(model, optimizer) as monitor:
                training = train_one_epoch(
                    model, data.train_loader, optimizer, loss_function, device
                )
            groups = {"epoch": epoch, **monitor.summary()}
            report["optimizer_steps_recorded"] += groups["optimizer_steps"]
            report.update(partial_epoch_possible=False, step_accounting_complete=True)
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
                "peak_cuda_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2
                if device.type == "cuda"
                else None,
            }
            append_epoch_metrics(directory / "metrics.jsonl", metrics)
            append_json(directory / "domain_metrics.jsonl", domain_row)
            append_json(directory / "parameter_group_metrics.jsonl", groups)
            report["parameter_group_epochs"].append(groups)
            report["completed_epochs"] = epoch
            report["final_metrics"] = {**asdict(metrics), **domain_row}
            print(
                json.dumps(
                    {"case": job["id"], **report["final_metrics"], "parameter_groups": groups},
                    allow_nan=False,
                ),
                flush=True,
            )
        report["final_optimizer_audit"] = asdict(audit_optimizer_parameters(model, optimizer))
        final_state = model.state_dict()
        if any(not torch.isfinite(value).all().item() for value in final_state.values()):
            raise ValueError("Final model state must be finite")
        report["final_state_sha256"] = state_dict_sha256(final_state)
        checkpoint = directory / "model_state_dict.pt"
        torch.save(final_state, checkpoint)
        report.update(
            model_checkpoint=str(checkpoint),
            checkpoint_sha256=file_sha256(checkpoint),
            status="completed",
        )
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        report["elapsed_seconds"] = perf_counter() - started
        write_json(directory / "run_report.json", report)
    return report
