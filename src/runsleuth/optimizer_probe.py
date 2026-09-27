"""A controlled, single-step experiment for stale optimizer parameter references.

The clean and stale variants replace ``model.fc`` with an identical deep copy.
They differ in whether the optimizer is built before or after replacement.
The repaired variant rebuilds the stale optimizer before its first step.
The caller owns model construction, initialization, device placement and RNG
seeding; this module neither downloads nor saves anything.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict

import torch
from torch import nn

from runsleuth.optimizer_audit import audit_optimizer_parameters

_VARIANTS = ("clean", "stale_head")


def state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash names, dtypes, shapes and exact tensor bytes, including buffers.

    Length-prefixed fields prevent ambiguous concatenation.  Viewing a flat
    contiguous tensor as bytes also handles dtypes such as bfloat16 without
    asking NumPy to represent the original dtype.
    """
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"State entry {name!r} must be a tensor")
        if tensor.layout != torch.strided or tensor.is_quantized:
            raise TypeError(f"State entry {name!r} must be a dense, unquantized tensor")
        data = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        fields = (
            name.encode("utf-8"),
            str(tensor.dtype).encode("ascii"),
            json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"),
            data.numpy().tobytes(),
        )
        for field in fields:
            digest.update(len(field).to_bytes(8, byteorder="big"))
            digest.update(field)
    return digest.hexdigest()


def _parameter_groups(model: nn.Module) -> tuple[dict[str, nn.Parameter], dict[str, nn.Parameter]]:
    if not isinstance(getattr(model, "fc", None), nn.Module):
        raise ValueError("The probe requires a model.fc module")
    parameters = dict(model.named_parameters())
    head = {name: parameter for name, parameter in parameters.items() if name.startswith("fc.")}
    backbone = {
        name: parameter for name, parameter in parameters.items() if not name.startswith("fc.")
    }
    if not head:
        raise ValueError("The probe requires at least one model.fc parameter")
    if not backbone:
        raise ValueError("The probe requires at least one backbone parameter outside model.fc")
    return head, backbone


def make_probe_optimizer(
    model: nn.Module,
    *,
    variant: str,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    """Replace the head with equal weights and deliberately vary operation order.

    In ``stale_head``, AdamW retains references to the discarded head.  The
    replacement head participates in autograd but is absent from the optimizer.
    """
    if variant not in _VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; expected one of {_VARIANTS}")
    _parameter_groups(model)
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and greater than zero")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")

    new_head = deepcopy(model.fc)
    if variant == "clean":
        model.fc = new_head
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        model.fc = new_head
    return optimizer


def _l2_norm(tensors) -> float:
    squared = 0.0
    for tensor in tensors:
        values = tensor.detach().to(device="cpu", dtype=torch.float64)
        squared += values.square().sum().item()
    return math.sqrt(squared)


def make_repaired_probe_optimizer(
    model: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
) -> tuple[torch.optim.AdamW, dict]:
    """Reproduce a stale binding and rebuild it before any optimizer step.

    Both the single-step probe and the bounded training experiment use this
    same intervention. This is not a checkpoint-resume repair: a nonempty
    optimizer state is rejected rather than discarded.
    """
    optimizer = make_probe_optimizer(
        model,
        variant="stale_head",
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    audit_before = audit_optimizer_parameters(model, optimizer)
    hash_before = state_dict_sha256(model.state_dict())
    state_entries_before = len(optimizer.state)
    if state_entries_before != 0:
        raise ValueError("Cannot rebuild an optimizer with nonempty state")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optimizer, {
        "action": "rebuild_optimizer",
        "timing": "before_first_optimizer_step",
        "optimizer_state_entries_before": state_entries_before,
        "model_state_sha256_before": hash_before,
        "model_state_sha256_after": state_dict_sha256(model.state_dict()),
        "optimizer_audit_before": asdict(audit_before),
    }


def run_probe_step(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    variant: str,
    learning_rate: float,
    weight_decay: float,
) -> dict:
    """Run one training step and return JSON-serializable causal evidence.

    Inputs and targets must already be on the model's device.  Accuracy is a
    fraction in [0, 1].  Norms cover all named head/backbone parameter tensors;
    parameters without gradients contribute zero to their gradient norm.
    ``stale_head_repaired`` records a genuine stale binding, then rebuilds the
    optimizer before training.  A nonempty optimizer state is never discarded.
    """
    repair = None
    if variant == "stale_head_repaired":
        optimizer, repair = make_repaired_probe_optimizer(
            model, learning_rate=learning_rate, weight_decay=weight_decay
        )
    else:
        optimizer = make_probe_optimizer(
            model,
            variant=variant,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
    head, backbone = _parameter_groups(model)
    groups = {"head": head, "backbone": backbone}
    initial_state_sha256 = state_dict_sha256(model.state_dict())
    before = {
        name: parameter.detach().clone()
        for parameters in groups.values()
        for name, parameter in parameters.items()
    }
    audit = audit_optimizer_parameters(model, optimizer)

    model.train()
    # The model clear includes the new head, which stale_head's optimizer cannot
    # clear.  The optimizer clear also includes its discarded head references.
    model.zero_grad(set_to_none=True)
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = nn.functional.cross_entropy(logits, targets)
    pre_update_loss = float(loss.detach().item())
    pre_update_accuracy = float((logits.detach().argmax(dim=1) == targets).float().mean().item())
    if not math.isfinite(pre_update_loss) or not math.isfinite(pre_update_accuracy):
        raise ValueError("Probe produced non-finite pre-update loss or accuracy")
    loss.backward()

    measurements = {
        group: {
            "gradient_l2_norm": _l2_norm(
                parameter.grad for parameter in parameters.values() if parameter.grad is not None
            )
        }
        for group, parameters in groups.items()
    }
    if any(not math.isfinite(item["gradient_l2_norm"]) for item in measurements.values()):
        raise ValueError("Probe produced non-finite gradients")
    optimizer.step()
    for group, parameters in groups.items():
        measurements[group]["parameter_update_l2_norm"] = _l2_norm(
            parameter.detach() - before[name] for name, parameter in parameters.items()
        )
        if not math.isfinite(measurements[group]["parameter_update_l2_norm"]):
            raise ValueError("Probe produced non-finite parameter updates")

    result = {
        "variant": variant,
        "initial_state_sha256": initial_state_sha256,
        "pre_update_loss": pre_update_loss,
        "pre_update_accuracy": pre_update_accuracy,
        "optimizer_audit": asdict(audit),
        **measurements,
    }
    if repair is not None:
        result["repair"] = repair
    return result
