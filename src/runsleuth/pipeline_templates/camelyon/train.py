"""Fine-tune ResNet18 to tell tumor from normal tissue in Camelyon17 patches.

The harness calls main(run). It may stop the run between two epochs and call main
again, which then resumes from the last checkpoint.
"""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

import config
import data


def build_model():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def train_one_epoch(model, loader, optimizer, loss_function, device):
    model.train()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = loss_function(model(data.normalize(images)), labels)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def evaluate(model, loader, loss_function, device):
    model.eval()
    loss, correct, count = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(data.normalize(images))
        loss += float(loss_function(logits, labels)) * len(labels)
        correct += int((logits.argmax(dim=1) == labels).sum())
        count += len(labels)
    return {"validation_loss": loss / count, "validation_accuracy": correct / count}


def main(run):
    model = build_model().to(run.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=run.epochs)
    loss_function = nn.CrossEntropyLoss()
    start, best_accuracy = 1, -1.0
    if config.LAST_CHECKPOINT in run.checkpoints:
        state = run.checkpoints.read(config.LAST_CHECKPOINT)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start, best_accuracy = state["epoch"] + 1, state["best_accuracy"]
    run.track(model, optimizer)
    training = data.training_set(run.data.train)
    validation = data.validation_loader(run.data.validation)
    for epoch in range(start, run.epochs + 1):
        loader = data.training_loader(training, run.seed, epoch)
        with run.epoch(epoch):
            train_one_epoch(model, loader, optimizer, loss_function, run.device)
        scheduler.step()
        metrics = evaluate(model, validation, loss_function, run.device)
        run.log(**metrics)
        if metrics["validation_accuracy"] > best_accuracy:
            best_accuracy = metrics["validation_accuracy"]
            run.checkpoints.write(config.BEST_CHECKPOINT, model.state_dict())
        run.checkpoints.write(
            config.LAST_CHECKPOINT,
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_accuracy": best_accuracy,
            },
        )
    run.deliver(run.checkpoints.read(config.BEST_CHECKPOINT))
