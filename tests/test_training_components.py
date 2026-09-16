import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.optim import SGD

from runsleuth.config import TrainingConfig
from runsleuth.model import FashionMNISTCNN
from runsleuth.telemetry import (
    gradient_l2_norm,
    parameter_update_l2_norm,
    snapshot_parameters,
)


def test_model_produces_ten_class_logits() -> None:
    model = FashionMNISTCNN()
    inputs = torch.randn(4, 1, 28, 28)

    logits = model(inputs)

    assert logits.shape == (4, 10)


def test_gradient_and_parameter_update_norms() -> None:
    model = nn.Linear(2, 1, bias=False)
    optimizer = SGD(model.parameters(), lr=0.1)

    with torch.no_grad():
        model.weight.zero_()

    inputs = torch.tensor([[3.0, 4.0]])
    targets = torch.tensor([[1.0]])

    prediction = model(inputs)
    loss = nn.functional.mse_loss(prediction, targets)
    loss.backward()

    assert gradient_l2_norm(model) == pytest.approx(10.0)

    before_step = snapshot_parameters(model)
    optimizer.step()

    assert parameter_update_l2_norm(model, before_step) == pytest.approx(1.0)


def test_training_config_is_saved_as_json(tmp_path: Path) -> None:
    config = TrainingConfig(run_name="test-run")
    config_path = tmp_path / "config.json"

    config.save(config_path)

    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved_config["run_name"] == "test-run"
    assert saved_config["learning_rate"] == 0.001
    assert saved_config["epochs"] == 3
