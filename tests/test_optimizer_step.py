import pytest
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from runsleuth.train import train_one_epoch


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
