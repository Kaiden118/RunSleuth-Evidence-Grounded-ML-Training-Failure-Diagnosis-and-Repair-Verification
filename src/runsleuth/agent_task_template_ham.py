"""Fine-tune ResNet18 to tell malignant from benign skin lesions in HAM10000 images.

The harness imports this file, supplies the images, calls the functions below and
evaluates the model itself after every epoch.
"""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
VALIDATION_SOURCE = 1
TEST_SOURCE = 2
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def use_for_training(source: int) -> bool:
    """Whether images from this source go into the training set."""
    return source not in (VALIDATION_SOURCE, TEST_SOURCE)


def held_out_key(image_id: str, lesion_id: str) -> str:
    """What an image shares with a validation image when it must be held out with it."""
    return lesion_id


def normalize(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


def prepare_training_batch(images: torch.Tensor) -> torch.Tensor:
    """Turn RGB images in [0, 1] into model inputs for training."""
    return normalize(images)


def prepare_evaluation_batch(images: torch.Tensor) -> torch.Tensor:
    """Turn RGB images in [0, 1] into model inputs for evaluation."""
    return normalize(images)


def build(device: torch.device, epochs: int, steps_per_epoch: int):
    """Create the model, its optimizer and the learning-rate schedule.

    The harness trains for epochs epochs of steps_per_epoch batches each.
    """
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    return model, optimizer, scheduler


def train_one_epoch(model, batches, optimizer, scheduler, loss_function) -> None:
    """Train for one epoch; batches yields (images, labels) already on the device."""
    model.train()
    for images, labels in batches:
        optimizer.zero_grad()
        loss = loss_function(model(prepare_training_batch(images)), labels)
        loss.backward()
        optimizer.step()
    scheduler.step()
