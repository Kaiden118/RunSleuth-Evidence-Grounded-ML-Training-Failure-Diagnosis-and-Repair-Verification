"""Configuration for reproducible training experiments."""
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    run_name: str = "clean"
    seed: int = 42
    epochs: int = 3
    batch_size: int = 128
    learning_rate: float = 0.001
    validation_size: int = 5_000
    num_workers: int = 0
    data_dir: str = "data/raw"
    output_dir: str = "artifacts/runs"
    device: str = "auto"

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load a complete experiment configuration from JSON."""
        values = json.loads(path.read_text(encoding="utf-8"))
        return cls(**values)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(asdict(self), indent=2)
        path.write_text(f"{content}\n", encoding="utf-8")
