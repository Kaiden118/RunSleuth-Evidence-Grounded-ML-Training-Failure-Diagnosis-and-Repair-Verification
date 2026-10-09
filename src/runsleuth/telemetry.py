"""Telemetry primitives for measuring training behavior."""

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from torch import Tensor, nn

ParameterSnapshot = dict[str, Tensor]


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    validation_loss: float
    validation_accuracy: float
    mean_gradient_norm: float
    mean_parameter_update_norm: float


def snapshot_parameters(model: nn.Module) -> ParameterSnapshot:
    """Copy model parameters immediately before an optimizer step."""
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


def gradient_l2_norm(model: nn.Module) -> float:
    """Compute one global L2 norm over all available gradients."""
    squared_norm = sum(
        parameter.grad.detach().pow(2).sum().item()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    return math.sqrt(squared_norm)


def parameter_update_l2_norm(
    model: nn.Module,
    before_step: Mapping[str, Tensor],
) -> float:
    """Measure how far parameters moved during an optimizer step."""
    squared_norm = sum(
        (parameter.detach() - before_step[name]).pow(2).sum().item()
        for name, parameter in model.named_parameters()
    )
    return math.sqrt(squared_norm)


def append_epoch_metrics(path: Path, metrics: EpochMetrics) -> None:
    """Append one epoch record as a JSON line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(f"{json.dumps(asdict(metrics))}\n")
