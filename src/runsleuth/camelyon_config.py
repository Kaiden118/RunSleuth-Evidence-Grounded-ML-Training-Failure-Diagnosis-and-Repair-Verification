"""Configuration for the Camelyon17 ResNet18 development baseline."""

import json
import re
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Self

CAMELYON_REVISION = "d784d5344ba6c967f83f9f3d9b2f1e2a4d6eb78f"


@dataclass(frozen=True, slots=True)
class CamelyonConfig:
    dataset: str = "camelyon17"
    dataset_version: str = "1.0"
    model: str = "resnet18"
    pretrained_weights: str = "IMAGENET1K_V1"
    hf_revision: str = CAMELYON_REVISION
    run_name: str = "camelyon17-clean"
    seed: int = 42
    subset_seed: int = 2026
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 0.0001
    weight_decay: float = 0.0001
    max_train_samples: int | None = 20000
    max_id_val_samples: int | None = 5000
    max_ood_val_samples: int | None = 5000
    num_workers: int = 0
    data_dir: str = "data/raw/camelyon17"
    output_dir: str = "artifacts/camelyon_runs"
    device: str = "cuda"

    def __post_init__(self) -> None:
        fixed = {
            "dataset": "camelyon17",
            "dataset_version": "1.0",
            "model": "resnet18",
            "pretrained_weights": "IMAGENET1K_V1",
            "hf_revision": CAMELYON_REVISION,
        }
        for name, expected in fixed.items():
            if getattr(self, name) != expected:
                raise ValueError(f"{name} must be {expected!r} for this baseline")
        if not isinstance(self.run_name, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", self.run_name
        ):
            raise ValueError("run_name must be a simple name, not a path")
        for name in ("epochs", "batch_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("seed", "subset_seed", "num_workers"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < 2**32:
                raise ValueError(f"{name} must be an integer in [0, 2**32)")
        for name in ("max_train_samples", "max_id_val_samples", "max_ood_val_samples"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer or null")
        for name in ("learning_rate", "weight_decay"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)):
                raise ValueError(f"{name} must be numeric")
            if not isfinite(value) or value < 0 or (name == "learning_rate" and value == 0):
                raise ValueError(f"{name} must be finite with a valid sign")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu, or cuda")
        for name in ("data_dir", "output_dir"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty path")

    @classmethod
    def load(cls, path: Path) -> Self:
        values = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(values, dict):
            raise ValueError("Config must be a JSON object")
        return cls(**values)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as output:
            output.write(json.dumps(asdict(self), indent=2, allow_nan=False) + "\n")
