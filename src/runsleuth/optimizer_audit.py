"""Read-only optimizer membership checks without importing torch."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class OptimizerAudit:
    """Identity-based tensor counts; omissions may reflect intentional training."""

    model_parameter_tensors: int
    trainable_parameter_tensors: int
    optimizer_unique_parameter_tensors: int
    missing_trainable_names: tuple[str, ...]
    foreign_parameter_tensors: int
    duplicate_parameter_occurrences: int

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible values without parameter objects or identities."""
        return asdict(self)


def audit_optimizer_parameters(model: Any, optimizer: Any) -> OptimizerAudit:
    """Compare named model parameters with every optimizer parameter group.

    Parameters are matched by object identity, never tensor equality. Frozen
    parameters count toward model membership but cannot be missing trainable
    parameters. Empty inputs are valid, and missing parameters are observations,
    not automatic failures: an optimizer may intentionally train only a subset.
    """
    named_parameters = tuple(model.named_parameters())
    model_ids = {id(parameter) for _, parameter in named_parameters}
    trainable_ids = {id(parameter) for _, parameter in named_parameters if parameter.requires_grad}
    optimizer_parameters = tuple(
        parameter for group in optimizer.param_groups for parameter in group["params"]
    )
    optimizer_ids = {id(parameter) for parameter in optimizer_parameters}

    return OptimizerAudit(
        model_parameter_tensors=len(model_ids),
        trainable_parameter_tensors=len(trainable_ids),
        optimizer_unique_parameter_tensors=len(optimizer_ids),
        missing_trainable_names=tuple(
            sorted(
                name
                for name, parameter in named_parameters
                if parameter.requires_grad and id(parameter) not in optimizer_ids
            )
        ),
        foreign_parameter_tensors=len(optimizer_ids - model_ids),
        duplicate_parameter_occurrences=len(optimizer_parameters) - len(optimizer_ids),
    )
