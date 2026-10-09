"""Run a bounded Camelyon17 ResNet18 clean baseline with ID/OOD validation."""

import argparse
import hashlib
import json
import platform
import shutil
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from math import isfinite
from pathlib import Path
from time import perf_counter

import torch
import torchvision
from torch import nn
from torch.optim import AdamW
from torchvision.models import ResNet18_Weights, resnet18

from runsleuth.camelyon_config import CamelyonConfig
from runsleuth.camelyon_data import build_camelyon_data
from runsleuth.train import (
    EpochMetrics,
    append_epoch_metrics,
    evaluate,
    resolve_device,
    seed_everything,
    train_one_epoch,
)


def build_camelyon_model(*, pretrained: bool = True) -> nn.Module:
    """Fine-tune the whole ResNet18; replace the ImageNet head with two logits."""
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _snapshot_sources(directory: Path) -> dict[str, str]:
    destination = directory / "source_snapshot"
    destination.mkdir()
    hashes = {}
    for name in (
        "camelyon.py",
        "camelyon_config.py",
        "camelyon_data.py",
        "train.py",
    ):
        source = Path(__file__).parent / name
        shutil.copyfile(source, destination / name)
        hashes[name] = _sha256(destination / name)
    return hashes


def _evaluate_domains(model, data, loss_function, device) -> dict[str, float]:
    results = {}
    for label, loader in (("id", data.id_val_loader), ("ood", data.ood_val_loader)):
        value = evaluate(model, loader, loss_function, device)
        loss, accuracy = float(value.loss), float(value.accuracy)
        if not isfinite(loss) or loss < 0 or not isfinite(accuracy) or not 0 <= accuracy <= 1:
            raise ValueError(f"Invalid {label} validation metrics")
        results[f"{label}_validation_loss"] = loss
        results[f"{label}_validation_accuracy"] = accuracy
    return results


def _check_training_result(result) -> None:
    for name, value in asdict(result).items():
        if not isfinite(value) or value < 0 or (name == "accuracy" and value > 1):
            raise ValueError(f"Invalid training metric: {name}={value}")


def run_camelyon_experiment(
    config: CamelyonConfig, *, download: bool = False, prepare_only: bool = False
) -> Path:
    """Run a fixed budget; retain failures and never evaluate the official test split."""
    device = resolve_device(config.device)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    directory = Path(config.output_dir) / f"{config.run_name}-{timestamp}"
    directory.mkdir(parents=True)
    config.save(directory / "config.json")
    print(f"run_directory={directory}", flush=True)
    report = {
        "schema_version": 1,
        "status": "running",
        "task": "camelyon17_resnet18_development_baseline",
        "training_mode": "full_finetuning_from_imagenet",
        "checkpoint_selection": "fixed_final_epoch",
        "completed_epochs": 0,
        "test_evaluated": False,
        "device": str(device),
        "workspace_root": Path.cwd().resolve().as_posix(),
        "error": None,
    }
    started = perf_counter()
    try:
        _write_json(
            directory / "environment.json",
            {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "torchvision": str(torchvision.__version__),
                "datasets": _package_version("datasets"),
                "huggingface_hub": _package_version("huggingface-hub"),
                "cuda_runtime": torch.version.cuda,
                "device_name": torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else "CPU",
                "precision": "float32",
                "optimizer": "AdamW",
                "source_sha256": _snapshot_sources(directory),
            },
        )
        print("phase=prepare_data", flush=True)
        data = build_camelyon_data(config, device=device, download=download)
        _write_json(directory / "data_manifest.json", data.manifest)
        report["data_manifest_sha256"] = _sha256(directory / "data_manifest.json")
        report["selected_samples"] = {
            name: record["selected_samples"] for name, record in data.manifest["splits"].items()
        }
        print("selected_samples=" + json.dumps(report["selected_samples"]), flush=True)
        # Decode one image per used split before allocating model/optimizer memory.
        for name, loader in (
            ("train", data.train_loader),
            ("id_val", data.id_val_loader),
            ("val", data.ood_val_loader),
        ):
            inputs, target = loader.dataset[0]
            if tuple(inputs.shape) != (3, 96, 96) or not torch.isfinite(inputs).all():
                raise ValueError(f"{name} must provide finite RGB tensors of shape (3, 96, 96)")
            if int(target) not in (0, 1):
                raise ValueError(f"{name} must provide binary labels")
        if prepare_only:
            report["status"] = "prepared"
        else:
            seed_everything(config.seed)
            print("phase=load_pretrained_model", flush=True)
            model = build_camelyon_model().to(device)
            initial_checkpoint = directory / "initial_state_dict.pt"
            torch.save(model.state_dict(), initial_checkpoint)
            report["initial_checkpoint_sha256"] = _sha256(initial_checkpoint)
            loss_function = nn.CrossEntropyLoss()
            print("phase=initialization_check", flush=True)
            baseline = _evaluate_domains(model, data, loss_function, device)
            _write_json(directory / "baseline_metrics.json", {"epoch": 0, **baseline})
            report["initialization_metrics"] = baseline
            print(json.dumps({"epoch": 0, **baseline}), flush=True)
            optimizer = AdamW(
                model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
            )
            for epoch in range(1, config.epochs + 1):
                print(f"phase=train epoch={epoch}/{config.epochs}", flush=True)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                epoch_started = perf_counter()
                training = train_one_epoch(
                    model, data.train_loader, optimizer, loss_function, device
                )
                _check_training_result(training)
                domains = _evaluate_domains(model, data, loss_function, device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = perf_counter() - epoch_started
                metrics = EpochMetrics(
                    epoch=epoch,
                    train_loss=training.loss,
                    train_accuracy=training.accuracy,
                    validation_loss=domains["id_validation_loss"],
                    validation_accuracy=domains["id_validation_accuracy"],
                    mean_gradient_norm=training.mean_gradient_norm,
                    mean_parameter_update_norm=training.mean_parameter_update_norm,
                )
                append_epoch_metrics(directory / "metrics.jsonl", metrics)
                domain_row = {
                    "epoch": epoch,
                    **domains,
                    "epoch_seconds": elapsed,
                    "peak_cuda_memory_mb": (
                        torch.cuda.max_memory_allocated(device) / 1024**2
                        if device.type == "cuda"
                        else None
                    ),
                }
                with (directory / "domain_metrics.jsonl").open("a", encoding="utf-8") as output:
                    output.write(json.dumps(domain_row, allow_nan=False) + "\n")
                report["completed_epochs"] = epoch
                report["final_metrics"] = {**asdict(metrics), **domain_row}
                print(json.dumps(report["final_metrics"], allow_nan=False), flush=True)
            checkpoint = directory / "model_state_dict.pt"
            torch.save(model.state_dict(), checkpoint)
            report["model_checkpoint"] = str(checkpoint)
            report["checkpoint_sha256"] = _sha256(checkpoint)
            report["status"] = "completed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        report["elapsed_seconds"] = perf_counter() - started
        _write_json(directory / "run_report.json", report)
        print(f"run_report={directory / 'run_report.json'}", flush=True)
        raise
    report["elapsed_seconds"] = perf_counter() - started
    _write_json(directory / "run_report.json", report)
    print(f"run_report={directory / 'run_report.json'}", flush=True)
    print(f"status={report['status']}", flush=True)
    return directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/camelyon17_clean.json"))
    parser.add_argument(
        "--download", action="store_true", help="Allow fetching missing dataset shards."
    )
    parser.add_argument(
        "--prepare-only", action="store_true", help="Prepare data; skip model training."
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_camelyon_experiment(
        CamelyonConfig.load(args.config), download=args.download, prepare_only=args.prepare_only
    )


if __name__ == "__main__":
    main()
