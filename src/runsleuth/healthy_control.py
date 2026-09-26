"""Validate and compare two healthy runs with a changed learning rate."""

from __future__ import annotations

import json
import math
from decimal import Decimal
from pathlib import Path

_METRICS = (
    "validation_accuracy",
    "validation_loss",
    "mean_gradient_norm",
    "mean_parameter_update_norm",
)
_IGNORED_CONFIG_FIELDS = {"run_name", "output_dir", "learning_rate"}
_POLICY = {
    "max_validation_accuracy_shortfall": 0.03,
    "max_validation_loss_ratio": 1.25,
}


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _summary(run: Path, epochs: int) -> dict[str, float]:
    try:
        lines = (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Cannot read metrics for {run}") from exc
    if len(lines) != epochs:
        raise ValueError(f"{run}: expected exactly {epochs} metric rows")
    rows = []
    for expected_epoch, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except (ValueError, RecursionError) as exc:
            raise ValueError(f"{run}: invalid metrics JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{run}: each metrics row must be an object")
        epoch = row.get("epoch")
        if type(epoch) is not int or epoch != expected_epoch:
            raise ValueError(f"{run}: expected epoch {expected_epoch}")
        values = {name: _number(row.get(name), name) for name in _METRICS}
        if not 0 <= values["validation_accuracy"] <= 1:
            raise ValueError(f"{run}: validation_accuracy must be in [0, 1]")
        if any(values[name] < 0 for name in _METRICS[1:]):
            raise ValueError(f"{run}: losses and norms must be nonnegative")
        rows.append(values)
    return {
        "final_validation_accuracy": rows[-1]["validation_accuracy"],
        "final_validation_loss": rows[-1]["validation_loss"],
        "first_epoch_gradient_norm": rows[0]["mean_gradient_norm"],
        "first_epoch_parameter_update_norm": rows[0]["mean_parameter_update_norm"],
        "final_epoch_gradient_norm": rows[-1]["mean_gradient_norm"],
        "final_epoch_parameter_update_norm": rows[-1]["mean_parameter_update_norm"],
    }


def inspect_healthy_control(
    reference_run: Path,
    candidate_run: Path,
    reference_config: dict[str, object],
    candidate_config: dict[str, object],
) -> dict[str, object]:
    """Return metric checks, raising ValueError for invalid study inputs."""
    try:
        reference_run, candidate_run = reference_run.resolve(), candidate_run.resolve()
        if not reference_run.is_dir() or not candidate_run.is_dir():
            raise ValueError("Both runs must be existing directories")
        if reference_run == candidate_run or reference_run.samefile(candidate_run):
            raise ValueError("Reference and candidate must be distinct run directories")
    except (OSError, RuntimeError) as exc:
        raise ValueError("Cannot resolve run directories") from exc
    configs = (reference_config, candidate_config)
    if any(config.get("optimizer_step_enabled") is not True for config in configs):
        raise ValueError("Both runs must enable the optimizer step")
    try:
        controlled = [
            json.dumps(
                {k: v for k, v in config.items() if k not in _IGNORED_CONFIG_FIELDS},
                sort_keys=True,
                allow_nan=False,
            )
            for config in configs
        ]
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("Controlled configs must contain finite JSON values") from exc
    if controlled[0] != controlled[1]:
        raise ValueError("Configs may differ only in run_name, output_dir, and learning_rate")
    epochs = reference_config.get("epochs")
    if type(epochs) is not int or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if type(candidate_config.get("epochs")) is not int:
        raise ValueError("epochs must be a positive integer")
    rates = [_number(config.get("learning_rate"), "learning_rate") for config in configs]
    if any(rate <= 0 for rate in rates) or rates[0] == rates[1]:
        raise ValueError("Learning rates must be positive and different")
    reference, candidate = (_summary(run, epochs) for run in (reference_run, candidate_run))
    for name in (
        "final_validation_loss",
        "first_epoch_parameter_update_norm",
        "final_epoch_gradient_norm",
    ):
        if reference[name] <= 0:
            raise ValueError(f"Reference {name} must be positive")
    # Decimal arithmetic keeps exact recorded decimal boundaries inclusive.
    observations = (
        (
            "validation_accuracy_shortfall",
            Decimal(str(reference["final_validation_accuracy"]))
            - Decimal(str(candidate["final_validation_accuracy"])),
            _POLICY["max_validation_accuracy_shortfall"],
        ),
        (
            "validation_loss_ratio",
            Decimal(str(candidate["final_validation_loss"]))
            / Decimal(str(reference["final_validation_loss"])),
            _POLICY["max_validation_loss_ratio"],
        ),
    )
    checks = []
    for metric, value, limit in observations:
        passed = value <= Decimal(str(limit))
        checks.append(
            dict(
                metric=metric,
                observed_value=_number(float(value), metric),
                operator="<=",
                threshold=limit,
                passed=passed,
            )
        )
    return {
        "passed": all(check["passed"] for check in checks),
        "policy": dict(_POLICY),
        "reference": reference,
        "candidate": candidate,
        "config_changes": {"learning_rate": {"reference": rates[0], "candidate": rates[1]}},
        "checks": checks,
    }
