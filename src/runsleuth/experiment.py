"""Orchestration for one bounded training experiment."""

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam

from runsleuth.config import TrainingConfig
from runsleuth.data import build_dataloaders
from runsleuth.model import FashionMNISTCNN
from runsleuth.telemetry import EpochMetrics, append_epoch_metrics
from runsleuth.train import evaluate, resolve_device, seed_everything, train_one_epoch


def create_run_directory(config: TrainingConfig) -> Path:
    """Create a unique artifact directory without overwriting an earlier run."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_directory = Path(config.output_dir) / f"{config.run_name}-{timestamp}"
    run_directory.mkdir(parents=True)
    return run_directory


def run_experiment(config: TrainingConfig) -> Path:
    """Run training, record evidence, and return the artifact directory."""
    seed_everything(config.seed)
    device = resolve_device(config.device)
    run_directory = create_run_directory(config)

    config.save(run_directory / "config.json")
    metrics_path = run_directory / "metrics.jsonl"

    training_loader, validation_loader = build_dataloaders(config)
    model = FashionMNISTCNN().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    print(f"device={device}")
    print(f"run_directory={run_directory}")

    for epoch in range(1, config.epochs + 1):
        training_result = train_one_epoch(
            model,
            training_loader,
            optimizer,
            loss_function,
            device,
            optimizer_step_enabled=config.optimizer_step_enabled,
        )
        validation_result = evaluate(
            model,
            validation_loader,
            loss_function,
            device,
        )

        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=training_result.loss,
            train_accuracy=training_result.accuracy,
            validation_loss=validation_result.loss,
            validation_accuracy=validation_result.accuracy,
            mean_gradient_norm=training_result.mean_gradient_norm,
            mean_parameter_update_norm=(training_result.mean_parameter_update_norm),
        )
        append_epoch_metrics(metrics_path, metrics)
        print(json.dumps(asdict(metrics), sort_keys=True))

    torch.save(
        model.state_dict(),
        run_directory / "model_state_dict.pt",
    )
    return run_directory
