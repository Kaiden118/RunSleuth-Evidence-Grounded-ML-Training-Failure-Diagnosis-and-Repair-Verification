"""Training and evaluation loops, and the per-step measurements they record."""

import json
import math
import random
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


@dataclass(frozen=True, slots=True)
class TrainingResult:
    loss: float
    accuracy: float
    mean_gradient_norm: float
    mean_parameter_update_norm: float


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    loss: float
    accuracy: float


def seed_everything(seed: int) -> None:
    """Seed Python and PyTorch for comparable experiment runs."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(requested_device: str) -> torch.device:
    """Resolve 'auto', while rejecting an unavailable requested GPU."""
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    loss_function: nn.Module,
    device: torch.device,
    *,
    optimizer_step_enabled: bool = True,
    train_mode: bool = True,
) -> TrainingResult:
    """Train for one epoch and collect optimization evidence.

    train_mode=False leaves the model's mode alone, as a loop without model.train()
    does; after an evaluation that means training in eval mode.
    """
    if train_mode:
        model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    total_gradient_norm = 0.0
    total_update_norm = 0.0
    batch_count = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(inputs)
        loss = loss_function(logits, targets)
        loss.backward()

        gradient_norm = gradient_l2_norm(model)
        parameters_before_step = snapshot_parameters(model)

        if optimizer_step_enabled:
            optimizer.step()

        update_norm = parameter_update_l2_norm(
            model,
            parameters_before_step,
        )

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_examples += batch_size
        total_gradient_norm += gradient_norm
        total_update_norm += update_norm
        batch_count += 1

    return TrainingResult(
        loss=total_loss / total_examples,
        accuracy=total_correct / total_examples,
        mean_gradient_norm=total_gradient_norm / batch_count,
        mean_parameter_update_norm=total_update_norm / batch_count,
    )


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> EvaluationResult:
    """Evaluate without creating gradients or changing parameters."""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(inputs)
        loss = loss_function(logits, targets)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_examples += batch_size

    return EvaluationResult(
        loss=total_loss / total_examples,
        accuracy=total_correct / total_examples,
    )


# Per-step measurements and per-epoch records.

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
