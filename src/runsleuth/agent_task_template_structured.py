"""Fine-tune ResNet18 to tell tumor from normal tissue in Camelyon17 patches.

The harness imports this file, supplies the patches, calls the five entry points at the
end of it (use_for_training, prepare_training_batch, prepare_evaluation_batch, build
and train_one_epoch) and evaluates the model itself after every epoch.
"""

from typing import NamedTuple

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class TrainingConfig(NamedTuple):
    """Hyperparameters of a run."""

    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    num_classes: int = 2
    # "epoch" steps the schedule once per epoch, "step" once per batch.
    scheduler_interval: str = "epoch"
    warmup_epochs: int = 0
    grad_clip_norm: float | None = None
    freeze_backbone: bool = False
    freeze_head_after_epochs: int | None = None


CONFIG = TrainingConfig()

# What each hospital of Camelyon17 is used for.
HOSPITAL_ROLES = {0: "train", 1: "validation", 2: "test", 3: "train", 4: "train"}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Preprocess:
    """Turns RGB patches in [0, 1] into normalized model inputs."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def apply(self, images: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.mean, device=images.device).view(1, 3, 1, 1)
        std = torch.tensor(self.std, device=images.device).view(1, 3, 1, 1)
        return (images - mean) / std


TRAIN_PREPROCESS = Preprocess(IMAGENET_MEAN, IMAGENET_STD)
EVAL_PREPROCESS = Preprocess(IMAGENET_MEAN, IMAGENET_STD)


def build_backbone() -> nn.Module:
    """ResNet18 with its ImageNet weights."""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    return model


def replace_head(model: nn.Module, num_classes: int) -> None:
    """Swap the ImageNet classifier for a new one, on the device of the backbone."""
    device = next(model.parameters()).device
    model.fc = nn.Linear(model.fc.in_features, num_classes).to(device)


def set_trainable(model: nn.Module, freeze_backbone: bool) -> None:
    """Train everything, or only the head when the backbone is frozen."""
    for name, parameter in model.named_parameters():
        parameter.requires_grad = not (freeze_backbone and not name.startswith("fc."))


def parameter_groups(model: nn.Module, config: TrainingConfig) -> list:
    """The optimizer's parameter groups."""
    parameters = [p for p in model.parameters() if p.requires_grad]
    return [{"params": parameters, "weight_decay": config.weight_decay}]


def make_optimizer(groups: list, config: TrainingConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(groups, lr=config.learning_rate)


def make_scheduler(optimizer, config: TrainingConfig, epochs: int, steps_per_epoch: int):
    """A cosine from the learning rate down to zero over the run, after any warmup."""
    per_epoch = 1 if config.scheduler_interval == "epoch" else steps_per_epoch
    warmup = config.warmup_epochs * per_epoch
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(epochs - config.warmup_epochs) * per_epoch
    )
    if not warmup:
        return cosine
    ramp = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup)
    return torch.optim.lr_scheduler.SequentialLR(optimizer, [ramp, cosine], milestones=[warmup])


class EpochRunner:
    """Runs the training loop of one epoch, and counts the epochs it has run."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.epochs_done = 0

    def start(self) -> None:
        self.epochs_done = 0

    def run(self, model, batches, optimizer, scheduler, loss_function) -> None:
        freeze_after = self.config.freeze_head_after_epochs
        if freeze_after is not None and self.epochs_done >= freeze_after:
            model.fc.requires_grad_(False)
        model.train()
        for images, labels in batches:
            optimizer.zero_grad()
            loss = loss_function(model(TRAIN_PREPROCESS.apply(images)), labels)
            loss.backward()
            if self.config.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip_norm)
            optimizer.step()
            if self.config.scheduler_interval == "step":
                scheduler.step()
        if self.config.scheduler_interval == "epoch":
            scheduler.step()
        self.epochs_done += 1


RUNNER = EpochRunner(CONFIG)


def use_for_training(hospital: int) -> bool:
    """Whether patches from this hospital go into the training set."""
    return HOSPITAL_ROLES.get(hospital) == "train"


def prepare_training_batch(images: torch.Tensor) -> torch.Tensor:
    return TRAIN_PREPROCESS.apply(images)


def prepare_evaluation_batch(images: torch.Tensor) -> torch.Tensor:
    return EVAL_PREPROCESS.apply(images)


def build(device: torch.device, epochs: int, steps_per_epoch: int):
    """Create the model, its optimizer and the learning-rate schedule.

    The harness trains for epochs epochs of steps_per_epoch batches each.
    """
    model = build_backbone()
    replace_head(model, CONFIG.num_classes)
    set_trainable(model, CONFIG.freeze_backbone)
    model.to(device)
    optimizer = make_optimizer(parameter_groups(model, CONFIG), CONFIG)
    scheduler = make_scheduler(optimizer, CONFIG, epochs, steps_per_epoch)
    RUNNER.start()
    return model, optimizer, scheduler


def train_one_epoch(model, batches, optimizer, scheduler, loss_function) -> None:
    """Train for one epoch; batches yields (images, labels) already on the device."""
    RUNNER.run(model, batches, optimizer, scheduler, loss_function)
