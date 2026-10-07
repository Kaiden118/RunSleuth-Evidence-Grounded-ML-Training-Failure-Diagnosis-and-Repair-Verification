"""Run the single-step frozen-head probe on CPU from a completed Camelyon17 baseline.

Design and predictions: docs/camelyon17_frozen_head.md. The probe reuses cached
data and the baseline's initial checkpoint, downloads nothing and evaluates no
validation or test data. Every variant takes one optimizer step on the same
training batch, on CPU, so bitwise comparisons do not depend on CUDA kernels.
"""

import argparse
import json
import shutil
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from runsleuth.camelyon_optimizer_probe import (
    check_matching_data,
    file_sha256,
    load_reference,
    read_json,
)

SOURCE_FILES = (
    "camelyon_frozen_head_probe.py",
    "frozen_head_probe.py",
    "camelyon_optimizer_probe.py",
    "optimizer_probe.py",
    "optimizer_audit.py",
    "parameter_group_monitor.py",
    "camelyon.py",
    "camelyon_config.py",
    "camelyon_data.py",
    "camelyon_mirror.py",
    "train.py",
)


def run_frozen_head_probe(reference_run: Path, output_dir: Path) -> Path:
    """Run every variant for one CPU optimizer step and write one JSON report."""
    import torch
    from torch.utils.data import default_collate

    from runsleuth.camelyon import build_camelyon_model
    from runsleuth.camelyon_data import build_camelyon_data
    from runsleuth.frozen_head_probe import VARIANTS, assess_frozen_head_probe, run_variant_step
    from runsleuth.optimizer_probe import state_dict_sha256
    from runsleuth.train import seed_everything

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    directory = output_dir / f"probe-{timestamp}"
    directory.mkdir(parents=True)
    report_path = directory / "frozen_head_probe_report.json"
    reference_run = reference_run.expanduser().resolve()
    device = torch.device("cpu")
    report = {
        "schema_version": 1,
        "status": "running",
        "task": "camelyon17_frozen_head_probe",
        "design": "docs/camelyon17_frozen_head.md",
        "reference_run": str(reference_run),
        "device": str(device),
        "optimizer_steps_per_variant": 1,
        "budget": {"optimizer_steps": len(VARIANTS), "gpu_training_epochs": 0, "model_calls": 0},
        "validation_evaluated": False,
        "test_evaluated": False,
        "scope": "single CPU optimizer step per variant; mechanism evidence, no performance claim",
        "error": None,
    }
    print(f"probe_directory={directory}", flush=True)
    try:
        config, reference_manifest, checkpoint = load_reference(reference_run)
        report["config"] = asdict(config)
        report["torch_version"] = str(torch.__version__)
        report["reference_config_sha256"] = file_sha256(reference_run / "config.json")
        report["reference_initial_checkpoint_sha256"] = file_sha256(checkpoint)
        report["reference_manifest_sha256"] = file_sha256(reference_run / "data_manifest.json")
        source_directory = directory / "source_snapshot"
        source_directory.mkdir()
        report["source_sha256"] = {}
        for name in SOURCE_FILES:
            destination = source_directory / name
            shutil.copyfile(Path(__file__).parent / name, destination)
            report["source_sha256"][name] = file_sha256(destination)

        print("phase=load_cached_data", flush=True)
        data = build_camelyon_data(config, device=device, download=False)
        check_matching_data(reference_manifest, data.manifest)
        positions = list(next(iter(data.train_loader.batch_sampler)))
        inputs, targets = default_collate([data.train_loader.dataset[i] for i in positions])
        if len(positions) != len(targets) or not positions:
            raise ValueError("The selected batch is empty or inconsistent")
        sample_ids = data.manifest["splits"]["train"]["selected_sample_ids"]
        report["batch"] = {
            "sample_ids": [sample_ids[i] for i in positions],
            "input_shape": list(inputs.shape),
            "targets": targets.tolist(),
            "tensor_sha256": state_dict_sha256({"inputs": inputs, "targets": targets}),
        }
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        expected_state_hash = state_dict_sha256(state)
        report["reference_initial_state_sha256"] = expected_state_hash

        variants = report["variants"] = {}
        for variant in VARIANTS:
            print(f"phase=single_step variant={variant}", flush=True)
            seed_everything(config.seed)
            model = build_camelyon_model(pretrained=False)
            model.load_state_dict(state, strict=True)
            model.to(device)
            variants[variant] = run_variant_step(
                model,
                inputs,
                targets,
                variant=variant,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            del model
        report.update(assess_frozen_head_probe(variants, expected_state_hash))
        report["mechanism_reproduced"] = all(report["checks"].values())
        report["status"] = "completed" if report["mechanism_reproduced"] else "inconclusive"
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        with report_path.open("x", encoding="utf-8") as output:
            output.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"frozen_head_probe_report={report_path}", flush=True)

    decisions = {name: item["decision"] for name, item in report["candidates"].items()}
    print(f"status={report['status']}", flush=True)
    print("checks=" + json.dumps(report["checks"]), flush=True)
    print("candidates=" + json.dumps(decisions), flush=True)
    print("predictions=" + json.dumps(report["predictions"]), flush=True)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/frozen_head_probes"))
    args = parser.parse_args()
    path = run_frozen_head_probe(args.reference_run, args.output_dir)
    if read_json(path)["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
