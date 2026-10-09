import pytest
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from runsleuth.train import (
    gradient_l2_norm,
    parameter_update_l2_norm,
    snapshot_parameters,
    train_one_epoch,
)


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


@pytest.mark.parametrize("step_enabled", [True, False])
def test_optimizer_step_controls_parameter_updates(step_enabled: bool) -> None:
    model = nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()

    dataset = TensorDataset(
        torch.eye(2),
        torch.tensor([0, 1]),
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    optimizer = SGD(model.parameters(), lr=0.1)

    parameters_before = [parameter.detach().clone() for parameter in model.parameters()]

    result = train_one_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        loss_function=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        optimizer_step_enabled=step_enabled,
    )

    parameters_changed = any(
        not torch.equal(before, after.detach())
        for before, after in zip(
            parameters_before,
            model.parameters(),
            strict=True,
        )
    )

    assert result.mean_gradient_norm > 0.0
    assert parameters_changed is step_enabled

    if step_enabled:
        assert result.mean_parameter_update_norm > 0.0
    else:
        assert result.mean_parameter_update_norm == 0.0
