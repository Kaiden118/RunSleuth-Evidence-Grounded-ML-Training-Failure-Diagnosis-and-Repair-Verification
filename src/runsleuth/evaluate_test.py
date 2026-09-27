"""Evaluate fixed FashionMNIST checkpoints on the official test split."""

import argparse
import hashlib
import io
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from runsleuth.config import TrainingConfig
from runsleuth.data import FASHION_MNIST_MEAN, FASHION_MNIST_STD
from runsleuth.model import FashionMNISTCNN
from runsleuth.train import evaluate, resolve_device, seed_everything


@dataclass(frozen=True)
class PreparedRun:
    """Capture exactly the configuration and weights selected before testing."""

    run_directory: Path
    config: TrainingConfig
    config_sha256: str
    checkpoint_sha256: str
    checkpoint_bytes: bytes

    def metadata(self) -> dict[str, object]:
        return {
            "run_directory": self.run_directory.as_posix(),
            "training_config": asdict(self.config),
            "config_sha256": self.config_sha256,
            "checkpoint": (self.run_directory / "model_state_dict.pt").as_posix(),
            "checkpoint_sha256": self.checkpoint_sha256,
        }


def prepare_runs(run_directories: list[Path]) -> list[PreparedRun]:
    """Snapshot all selections before any test data is evaluated."""
    if not run_directories:
        raise ValueError("At least one run must be selected")
    resolved = [path.resolve() for path in run_directories]
    if len(resolved) != len(set(resolved)):
        raise ValueError("Select each run directory only once")
    prepared = []
    for directory in run_directories:
        config_bytes = (directory / "config.json").read_bytes()
        values = json.loads(config_bytes)
        if not isinstance(values, dict):
            raise ValueError("Training config must be a JSON object")
        config = TrainingConfig(**values)
        checkpoint_bytes = (directory / "model_state_dict.pt").read_bytes()
        if not checkpoint_bytes:
            raise ValueError(f"Checkpoint is empty: {directory}")
        prepared.append(
            PreparedRun(
                run_directory=directory,
                config=config,
                config_sha256=hashlib.sha256(config_bytes).hexdigest(),
                checkpoint_sha256=hashlib.sha256(checkpoint_bytes).hexdigest(),
                checkpoint_bytes=checkpoint_bytes,
            )
        )
    return prepared


def evaluate_checkpoint(
    prepared: PreparedRun, *, device: torch.device, batch_size: int
) -> dict[str, object]:
    """Measure a saved model without training or selecting a repair."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    seed_everything(prepared.config.seed)
    state = torch.load(io.BytesIO(prepared.checkpoint_bytes), map_location="cpu", weights_only=True)
    model = FashionMNISTCNN()
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(FASHION_MNIST_MEAN, FASHION_MNIST_STD),
        ]
    )
    dataset = datasets.FashionMNIST(
        root=prepared.config.data_dir,
        train=False,
        download=True,
        transform=transform,
    )
    if len(dataset) < 1:
        raise ValueError("Test dataset is empty")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    with torch.inference_mode():
        result = evaluate(model, loader, nn.CrossEntropyLoss(), device)
    loss, accuracy = float(result.loss), float(result.accuracy)
    if not isfinite(loss) or loss < 0:
        raise ValueError("Test loss must be finite and nonnegative")
    if not isfinite(accuracy) or not 0 <= accuracy <= 1:
        raise ValueError("Test accuracy must be finite and in [0, 1]")
    return {
        "test_examples": len(dataset),
        "test_loss": loss,
        "test_accuracy": accuracy,
    }


def _save_json(path: Path, value: dict[str, object]) -> None:
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as destination:
        destination.write(text)


def run_test_evaluation(
    run_directories: list[Path],
    *,
    device: str = "auto",
    batch_size: int = 128,
    output_dir: Path = Path("artifacts/test_evaluations"),
) -> Path:
    """Record the fixed plan, then retain the result of every selected checkpoint."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    resolved_device = resolve_device(device)
    prepared = prepare_runs(run_directories)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    directory = output_dir / timestamp
    directory.mkdir(parents=True)
    plan = {
        "schema_version": 1,
        "evaluation_type": "held_out_classification_test",
        "dataset": "FashionMNIST",
        "split": "test",
        "workspace_root": Path.cwd().resolve().as_posix(),
        "model_class": "FashionMNISTCNN",
        "device": str(resolved_device),
        "batch_size": batch_size,
        "num_workers": 0,
        "preprocessing": {
            "to_tensor": True,
            "normalize_mean": list(FASHION_MNIST_MEAN),
            "normalize_std": list(FASHION_MNIST_STD),
        },
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
        "selection_policy": "All checkpoints selected before test inference; no repair selection.",
        "runs": [item.metadata() for item in prepared],
    }
    _save_json(directory / "test_plan.json", plan)
    print(f"test_directory={directory}")
    results = []
    for index, item in enumerate(prepared, start=1):
        row = {**item.metadata(), "status": "error", "error": None}
        try:
            row.update(evaluate_checkpoint(item, device=resolved_device, batch_size=batch_size))
            row["status"] = "completed"
        except Exception as error:
            # Preserve a failed attempt; never replace it with a successful retry.
            row["error"] = {"type": type(error).__name__, "message": str(error)}
        _save_json(directory / f"run-{index:03d}.json", row)
        results.append(row)
        print(json.dumps(row, allow_nan=False))
    report = {
        **plan,
        "summary": {
            "selected_runs": len(results),
            "completed_runs": sum(row["status"] == "completed" for row in results),
            "error_runs": sum(row["status"] == "error" for row in results),
        },
        "results": results,
    }
    report_path = directory / "test_report.json"
    _save_json(report_path, report)
    return report_path


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=positive_int, default=128)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/test_evaluations"))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        path = run_test_evaluation(
            args.run, device=args.device, batch_size=args.batch_size, output_dir=args.output_dir
        )
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        parser.error(str(error))
    report = json.loads(path.read_text(encoding="utf-8"))
    print(f"test_report={path}")
    print(json.dumps(report["summary"]))
    if report["summary"]["error_runs"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
