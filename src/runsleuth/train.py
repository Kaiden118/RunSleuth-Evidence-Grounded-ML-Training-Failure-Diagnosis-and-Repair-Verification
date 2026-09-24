"""Training and evaluation loops with telemetry collection."""

import random
from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from runsleuth.telemetry import (
    gradient_l2_norm,
    parameter_update_l2_norm,
    snapshot_parameters,
)


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
) -> TrainingResult:
    """Train for one epoch and collect optimization evidence."""
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
