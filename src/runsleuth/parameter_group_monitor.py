"""Epoch-scoped head/backbone measurements around the existing training loop."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer


def _parameter_groups(model: nn.Module) -> dict[str, dict[str, nn.Parameter]]:
    groups: dict[str, dict[str, nn.Parameter]] = {"head": {}, "backbone": {}}
    for name, parameter in model.named_parameters():
        group = "head" if name.startswith("fc.") else "backbone"
        groups[group][name] = parameter
    if not groups["head"] or not groups["backbone"]:
        raise ValueError("Monitoring requires model.fc and backbone parameter tensors")
    devices = {parameter.device for group in groups.values() for parameter in group.values()}
    if len(devices) != 1:
        raise ValueError("Monitoring requires all model parameters on one device")
    return groups


def parameter_trainability(model: nn.Module) -> dict[str, dict[str, Any]]:
    """Count trainable tensors per group and name frozen ones, without parameter objects."""
    return {
        group: {
            "parameter_tensors": len(parameters),
            "trainable_tensors": sum(parameter.requires_grad for parameter in parameters.values()),
            "frozen_names": sorted(
                name for name, parameter in parameters.items() if not parameter.requires_grad
            ),
        }
        for group, parameters in _parameter_groups(model).items()
    }


def _l2_norm(tensors: Iterable[Tensor]) -> float:
    """Reduce on the model device, with one scalar transfer per group norm."""
    norms = []
    for tensor in tensors:
        values = tensor.detach()
        if values.is_sparse:
            values = values.coalesce().values()
        dtype = torch.float32 if values.dtype in (torch.float16, torch.bfloat16) else None
        norms.append(torch.linalg.vector_norm(values, dtype=dtype))
    if not norms:
        return 0.0
    result = float(torch.linalg.vector_norm(torch.stack(norms)).item())
    if not math.isfinite(result):
        raise ValueError("Parameter group monitoring produced a non-finite norm")
    return result


class ParameterGroupMonitor:
    """Collect per-step means for one epoch without changing train_one_epoch.

    Use a fresh instance for each epoch. This monitor expects one training
    forward followed by one closure-free optimizer step per batch, as in
    train_one_epoch.
    Its root forward hook clears MODEL gradients so an optimizer's missing
    head parameters cannot accumulate gradients across batches. Evaluation
    forwards are ignored. Optimizer hooks measure current model parameters,
    including a replacement head absent from optimizer parameter groups.
    Each step also counts trainable tensors and tensors with gradients per
    group, so an absent gradient (a frozen head) is not mistaken for a zero one.

    Hooks and temporary snapshots are removed on every context exit. A failed
    or incomplete epoch never yields a successful summary.

    With allow_missing_steps, a training forward that is never followed by an
    optimizer step is counted instead of rejected, so a missing step becomes
    evidence (training_forwards > optimizer_steps). Group statistics cover only
    the steps that happened and are None when no step happened.
    """

    def __init__(
        self, model: nn.Module, optimizer: Optimizer, *, allow_missing_steps: bool = False
    ) -> None:
        _parameter_groups(model)
        self.model = model
        self.optimizer = optimizer
        self._allow_missing_steps = allow_missing_steps
        self._training_forwards = 0
        self._handles: list[Any] = []
        self._used = False
        self._active = False
        self._failed = False
        self._forward_pending = False
        self._pending_step: dict[str, Any] | None = None
        self._optimizer_steps = 0
        self._totals = {group: {"gradient": 0.0, "update": 0.0} for group in ("head", "backbone")}
        self._counts: dict[str, list[tuple[int, int, int]]] = {group: [] for group in self._totals}

    def __enter__(self) -> ParameterGroupMonitor:
        if self._used:
            raise RuntimeError("Use a new ParameterGroupMonitor for each epoch")
        self._used = True
        try:
            self._handles.append(self.model.register_forward_pre_hook(self._before_forward))
            self._handles.append(self.optimizer.register_step_pre_hook(self._before_step))
            self._handles.append(self.optimizer.register_step_post_hook(self._after_step))
        except BaseException:
            self._failed = True
            self._cleanup()
            raise
        self._active = True
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        try:
            if exc_type is not None:
                self._failed = True
            else:
                try:
                    self.summary()
                except BaseException:
                    self._failed = True
                    raise
        finally:
            self._cleanup()
        return False

    def _cleanup(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        if self._pending_step is not None:
            self._pending_step["snapshots"].clear()
            self._pending_step.clear()
        self._pending_step = None
        self._forward_pending = False
        self._active = False

    def _before_forward(self, module: nn.Module, inputs: tuple[Any, ...]) -> None:
        if not module.training or not torch.is_grad_enabled():
            return
        if self._pending_step is not None or (
            self._forward_pending and not self._allow_missing_steps
        ):
            raise RuntimeError("Previous training forward has no completed optimizer step")
        module.zero_grad(set_to_none=True)
        self._forward_pending = True
        self._training_forwards += 1

    def _before_step(self, optimizer: Optimizer, args: tuple, kwargs: dict) -> None:
        if not self._forward_pending:
            raise RuntimeError("Optimizer step has no preceding training forward")
        if self._pending_step is not None:
            raise RuntimeError("Previous optimizer step is incomplete")
        groups = _parameter_groups(self.model)
        if not any(
            parameter.grad is not None
            for parameters in groups.values()
            for parameter in parameters.values()
        ):
            raise RuntimeError("Optimizer step has no model gradients from backward")
        gradients = {
            group: _l2_norm(
                parameter.grad for parameter in parameters.values() if parameter.grad is not None
            )
            for group, parameters in groups.items()
        }
        counts = {
            group: (
                len(parameters),
                sum(parameter.requires_grad for parameter in parameters.values()),
                sum(parameter.grad is not None for parameter in parameters.values()),
            )
            for group, parameters in groups.items()
        }
        snapshots = {
            name: parameter.detach().clone()
            for parameters in groups.values()
            for name, parameter in parameters.items()
        }
        self._pending_step = {
            "groups": groups,
            "gradients": gradients,
            "counts": counts,
            "snapshots": snapshots,
        }

    def _after_step(self, optimizer: Optimizer, args: tuple, kwargs: dict) -> None:
        if self._pending_step is None:
            raise RuntimeError("Optimizer step finished without a matching pre-step capture")
        pending = self._pending_step
        try:
            current = dict(self.model.named_parameters())
            captured = {
                name: parameter
                for parameters in pending["groups"].values()
                for name, parameter in parameters.items()
            }
            if current.keys() != captured.keys() or any(
                current[name] is not parameter for name, parameter in captured.items()
            ):
                raise RuntimeError("Model parameters changed identity during optimizer step")
            updates = {
                group: _l2_norm(
                    parameter.detach() - pending["snapshots"][name]
                    for name, parameter in parameters.items()
                )
                for group, parameters in pending["groups"].items()
            }
            for group in self._totals:
                self._totals[group]["gradient"] += pending["gradients"][group]
                self._totals[group]["update"] += updates[group]
                self._counts[group].append(pending["counts"][group])
            self._optimizer_steps += 1
            self._forward_pending = False
        finally:
            # Clear the dictionary as well as our reference: exception
            # tracebacks can retain this frame and its local ``pending``.
            pending["snapshots"].clear()
            pending.clear()
            self._pending_step = None

    def summary(self) -> dict[str, Any]:
        """Return JSON-compatible means and tensor counts; reject failed or unfinished epochs."""
        if not self._used:
            raise RuntimeError("ParameterGroupMonitor has not entered its context")
        if self._failed:
            raise RuntimeError("Cannot summarize a failed parameter group monitoring context")
        if self._pending_step is not None or (
            self._forward_pending and not self._allow_missing_steps
        ):
            raise RuntimeError("Cannot summarize an incomplete optimizer step")
        if self._optimizer_steps == 0 and not (
            self._allow_missing_steps and self._training_forwards > 0
        ):
            raise RuntimeError("Parameter group monitoring observed zero optimizer steps")
        result: dict[str, Any] = {
            "optimizer_steps": self._optimizer_steps,
            "training_forwards": self._training_forwards,
        }
        if self._optimizer_steps == 0:
            return {**result, "head": None, "backbone": None}
        for group, totals in self._totals.items():
            sizes, trainable, with_gradient = zip(*self._counts[group], strict=True)
            if len(set(sizes)) != 1:
                raise RuntimeError("Parameter group size changed within the epoch")
            result[group] = {
                "mean_gradient_l2_norm": totals["gradient"] / self._optimizer_steps,
                "mean_parameter_update_l2_norm": totals["update"] / self._optimizer_steps,
                "parameter_tensors": sizes[0],
                "min_trainable_tensors": min(trainable),
                "max_trainable_tensors": max(trainable),
                "min_tensors_with_gradient": min(with_gradient),
                "max_tensors_with_gradient": max(with_gradient),
            }
            if any(not math.isfinite(value) for value in result[group].values()):
                raise ValueError("Parameter group monitoring produced non-finite mean norms")
        return result
