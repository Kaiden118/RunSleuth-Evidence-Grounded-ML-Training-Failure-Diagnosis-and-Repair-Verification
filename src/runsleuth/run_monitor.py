"""Drop-in RunSleuth telemetry for any PyTorch training loop.

    monitor = RunMonitor(model, optimizer, "runs/my-run", head="fc")
    for epoch in range(epochs):
        with monitor.epoch():
            for inputs, targets in loader:  # your own loop, unchanged
                optimizer.zero_grad()
                loss_fn(model(inputs), targets).backward()
                optimizer.step()
        monitor.log(train_loss=..., validation_accuracy=...)
    monitor.close()

The monitor writes run_report.json after every epoch in the format the
Camelyon17 runners use, so signature_matching can diagnose the run. It observes
training forwards and optimizer steps through hooks and never changes
gradients except clearing model gradients before each training forward, as the
parameter group monitor documents. A training error is recorded (as diverged
when parameters or gradients are non-finite) and re-raised.
"""

from __future__ import annotations

import json
import math
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from runsleuth.optimizer_audit import audit_optimizer_parameters
from runsleuth.parameter_group_monitor import ParameterGroupMonitor, parameter_trainability


def _l2(tensors) -> float:
    return math.sqrt(sum(float(tensor.detach().double().square().sum()) for tensor in tensors))


def _non_finite(model: nn.Module) -> bool:
    return any(
        not torch.isfinite(tensor).all()
        for parameter in model.parameters()
        for tensor in (parameter, parameter.grad)
        if tensor is not None
    )


class RunMonitor:
    """Record per-epoch evidence for one training run and write a run report."""

    def __init__(
        self, model: nn.Module, optimizer: Optimizer, run_dir: str | Path, *, head: str = "fc"
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.head_prefix = f"{head}."
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.report_path = self.run_dir / "run_report.json"
        self._epoch = 0
        self._closed = False
        self.report: dict[str, Any] = {
            "schema_version": 1,
            "task": "runsleuth_monitored_run",
            "status": "running",
            "error": None,
            "optimizer": type(optimizer).__name__,
            "head_prefix": self.head_prefix,
            "completed_epochs": 0,
            "trainability": parameter_trainability(model, self.head_prefix),
            "optimizer_audit": asdict(audit_optimizer_parameters(model, optimizer)),
            "parameter_group_epochs": [],
            "final_metrics": {},
        }
        self._write()

    def _write(self) -> None:
        self.report_path.write_text(
            json.dumps(self.report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    @contextmanager
    def epoch(self):
        """Wrap one epoch of the caller's own training loop."""
        if self._closed:
            raise RuntimeError("RunMonitor is closed")
        self._epoch += 1
        before = [parameter.detach().clone() for parameter in self.model.parameters()]
        monitor = ParameterGroupMonitor(
            self.model, self.optimizer, allow_missing_steps=True, head_prefix=self.head_prefix
        )
        try:
            with monitor:
                yield self
        except BaseException as error:
            diverged = isinstance(error, ValueError) and _non_finite(self.model)
            self.report["status"] = "diverged" if diverged else "failed"
            self.report["error"] = {"type": type(error).__name__, "message": str(error)}
            if diverged:
                self.report["divergence"] = {"epoch": self._epoch, "message": str(error)}
            self._closed = True
            self._write()
            raise
        gradients = [p.grad for p in self.model.parameters() if p.grad is not None]
        row = {
            "epoch": self._epoch,
            **monitor.summary(),
            # Epoch-level measurements also cover epochs without any optimizer step.
            "epoch_parameter_change_l2": _l2(
                parameter.detach() - start
                for parameter, start in zip(self.model.parameters(), before, strict=True)
            ),
            "gradient_l2_at_epoch_end": _l2(gradients),
            "tensors_with_gradient_at_epoch_end": len(gradients),
        }
        self.report["parameter_group_epochs"].append(row)
        self.report["completed_epochs"] = self._epoch
        self._write()

    def log(self, **metrics: float) -> None:
        """Attach finite scalar metrics to the latest epoch and to final_metrics."""
        if not self.report["parameter_group_epochs"]:
            raise RuntimeError("Log metrics after a completed monitor.epoch()")
        values = {name: float(value) for name, value in metrics.items()}
        if any(not math.isfinite(value) for value in values.values()):
            raise ValueError("RunMonitor metrics must be finite")
        row = self.report["parameter_group_epochs"][-1]
        row["metrics"] = {**row.get("metrics", {}), **values}
        self.report["final_metrics"] = {**self.report["final_metrics"], "epoch": self._epoch}
        self.report["final_metrics"].update(values)
        self._write()

    def close(self) -> Path:
        """Record final optimizer membership and trainability; return the report path."""
        if not self._closed:
            self.report["final_optimizer_audit"] = asdict(
                audit_optimizer_parameters(self.model, self.optimizer)
            )
            self.report["final_trainability"] = parameter_trainability(self.model, self.head_prefix)
            if self.report["status"] == "running":
                self.report["status"] = "completed"
            self._closed = True
            self._write()
        return self.report_path

    def __enter__(self) -> RunMonitor:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if exc_type is not None and self.report["status"] == "running":
            self.report["status"] = "failed"
            self.report["error"] = {"type": exc_type.__name__, "message": str(exc_value)}
        self.close()
        return False
