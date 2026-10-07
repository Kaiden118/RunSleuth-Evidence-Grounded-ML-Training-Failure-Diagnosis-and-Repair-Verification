"""A controlled, single-step experiment for an unintended frozen classification head.

Design, evidence and predictions: docs/camelyon17_frozen_head.md. The frozen
variant follows the clean operation order and also sets requires_grad=False on
the replacement head before AdamW is constructed. Two candidate repairs start
from that fault: restoring requires_grad, and rebuilding the optimizer (the
stale-binding repair, applied here as a negative control).
The caller owns model construction, initialization, device placement and RNG
seeding; this module neither downloads nor saves anything.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import asdict

import torch
from torch import nn

from runsleuth.optimizer_audit import audit_optimizer_parameters
from runsleuth.optimizer_probe import (
    _l2_norm,
    _parameter_groups,
    make_probe_optimizer,
    state_dict_sha256,
)
from runsleuth.parameter_group_monitor import parameter_trainability

VARIANTS = (
    "clean",
    "stale_head",
    "frozen_head",
    "frozen_head_repaired",
    "frozen_head_optimizer_rebuilt",
)
CANDIDATE_ACTIONS = {
    "frozen_head_repaired": "restore_head_requires_grad",
    "frozen_head_optimizer_rebuilt": "rebuild_optimizer",
}
FROZEN_HEAD_NAMES = ["fc.bias", "fc.weight"]


def make_variant_optimizer(
    model: nn.Module, *, variant: str, learning_rate: float, weight_decay: float
) -> tuple[torch.optim.AdamW, dict | None]:
    """Build one variant's optimizer; candidate repairs also return an intervention record."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; expected one of {VARIANTS}")
    if variant in ("clean", "stale_head"):
        optimizer = make_probe_optimizer(
            model, variant=variant, learning_rate=learning_rate, weight_decay=weight_decay
        )
        return optimizer, None
    _parameter_groups(model)
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and greater than zero")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")

    new_head = deepcopy(model.fc)
    model.fc = new_head
    model.fc.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if variant == "frozen_head":
        return optimizer, None

    state_entries_before = len(optimizer.state)
    if state_entries_before != 0:
        raise ValueError("Cannot intervene on an optimizer with nonempty state")
    before = {
        "trainability_before": parameter_trainability(model),
        "optimizer_audit_before": asdict(audit_optimizer_parameters(model, optimizer)),
        "model_state_sha256_before": state_dict_sha256(model.state_dict()),
    }
    if variant == "frozen_head_repaired":
        model.fc.requires_grad_(True)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    return optimizer, {
        "action": CANDIDATE_ACTIONS[variant],
        "timing": "before_first_optimizer_step",
        "optimizer_state_entries_before": state_entries_before,
        **before,
        "model_state_sha256_after": state_dict_sha256(model.state_dict()),
    }


def _group_sha256(tensors: dict[str, torch.Tensor]) -> str:
    return state_dict_sha256({name: tensor.detach() for name, tensor in tensors.items()})


def run_variant_step(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    variant: str,
    learning_rate: float,
    weight_decay: float,
) -> dict:
    """Run one training step and return JSON-serializable evidence with bitwise hashes.

    Hashes cover names, dtypes, shapes and exact bytes. A group's gradient hash
    covers only tensors that have a gradient; ``tensors_with_gradient`` counts them.
    """
    optimizer, intervention = make_variant_optimizer(
        model, variant=variant, learning_rate=learning_rate, weight_decay=weight_decay
    )
    head, backbone = _parameter_groups(model)
    groups = {"head": head, "backbone": backbone}
    before = {
        name: parameter.detach().clone()
        for parameters in groups.values()
        for name, parameter in parameters.items()
    }
    result = {
        "variant": variant,
        "initial_state_sha256": state_dict_sha256(model.state_dict()),
        "trainability": parameter_trainability(model),
        "optimizer_audit": asdict(audit_optimizer_parameters(model, optimizer)),
    }

    model.train()
    model.zero_grad(set_to_none=True)
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = nn.functional.cross_entropy(logits, targets)
    result["pre_update_loss"] = float(loss.detach().item())
    result["pre_update_accuracy"] = float(
        (logits.detach().argmax(dim=1) == targets).float().mean().item()
    )
    if not math.isfinite(result["pre_update_loss"]):
        raise ValueError("Probe produced a non-finite pre-update loss")
    loss.backward()
    for group, parameters in groups.items():
        gradients = {
            name: parameter.grad
            for name, parameter in parameters.items()
            if parameter.grad is not None
        }
        result[group] = {
            "tensors": len(parameters),
            "tensors_with_gradient": len(gradients),
            "gradient_l2_norm": _l2_norm(gradients.values()),
            "gradient_sha256": _group_sha256(gradients),
            "initial_sha256": _group_sha256({name: before[name] for name in parameters}),
        }
    optimizer.step()
    for group, parameters in groups.items():
        result[group]["parameter_update_l2_norm"] = _l2_norm(
            parameter.detach() - before[name] for name, parameter in parameters.items()
        )
        result[group]["post_step_sha256"] = _group_sha256(parameters)
        if not all(
            math.isfinite(result[group][key])
            for key in ("gradient_l2_norm", "parameter_update_l2_norm")
        ):
            raise ValueError("Probe produced non-finite gradients or updates")
    if intervention is not None:
        result["intervention"] = intervention
    return result


def _complete_coverage(audit: dict) -> bool:
    return (
        not audit["missing_trainable_names"]
        and audit["foreign_parameter_tensors"] == 0
        and audit["duplicate_parameter_occurrences"] == 0
    )


def _fully_trainable(trainability: dict) -> bool:
    return all(
        group["trainable_tensors"] == group["parameter_tensors"] and not group["frozen_names"]
        for group in trainability.values()
    )


def _frozen_head_only(trainability: dict) -> bool:
    head, backbone = trainability["head"], trainability["backbone"]
    return (
        head["trainable_tensors"] == 0
        and head["frozen_names"] == FROZEN_HEAD_NAMES
        and backbone["trainable_tensors"] == backbone["parameter_tensors"]
    )


def _has_gradients_and_updates(group: dict) -> bool:
    return (
        group["tensors_with_gradient"] == group["tensors"]
        and group["gradient_l2_norm"] > 0
        and group["parameter_update_l2_norm"] > 0
    )


def assess_candidate(candidate: dict, frozen: dict, expected_state_hash: str) -> dict:
    """Single-step structural verdict for one candidate repair; not performance acceptance."""
    intervention = candidate["intervention"]
    checks = {
        "reproduced_frozen_head_before_intervention": (
            intervention["trainability_before"] == frozen["trainability"]
            and _frozen_head_only(intervention["trainability_before"])
        ),
        "intervention_before_first_step_with_empty_state": (
            intervention["timing"] == "before_first_optimizer_step"
            and intervention["optimizer_state_entries_before"] == 0
        ),
        "intervention_preserves_model_state": (
            intervention["model_state_sha256_before"]
            == intervention["model_state_sha256_after"]
            == expected_state_hash
        ),
        "optimizer_covers_current_parameters": _complete_coverage(candidate["optimizer_audit"]),
        "all_parameters_trainable": _fully_trainable(candidate["trainability"]),
        "head_and_backbone_have_gradients_and_updates": all(
            _has_gradients_and_updates(candidate[group]) for group in ("head", "backbone")
        ),
    }
    return {
        "action": intervention["action"],
        "checks": checks,
        "decision": "accepted" if all(checks.values()) else "rejected",
        "scope": "single-step structural verdict; bounded-training acceptance is stage 2",
    }


def assess_frozen_head_probe(variants: dict, expected_state_hash: str) -> dict:
    """Keep mechanism checks apart from pre-registered predictions P1-P4.

    Checks decide whether the specified variants were reproduced. Predictions
    are reported as observed and never change that outcome.
    """
    clean, stale, frozen = (variants[name] for name in ("clean", "stale_head", "frozen_head"))
    repaired = variants["frozen_head_repaired"]
    rebuilt = variants["frozen_head_optimizer_rebuilt"]
    stale_audit = stale["optimizer_audit"]
    checks = {
        "same_reference_initial_state": all(
            variant["initial_state_sha256"] == expected_state_hash for variant in variants.values()
        ),
        "same_pre_update_loss": all(
            math.isclose(
                variant["pre_update_loss"], clean["pre_update_loss"], rel_tol=1e-6, abs_tol=1e-6
            )
            for variant in variants.values()
        ),
        "clean_head_trainable_with_gradients_and_updates": (
            _fully_trainable(clean["trainability"]) and _has_gradients_and_updates(clean["head"])
        ),
        "stale_head_has_gradients_without_updates": (
            stale["head"]["tensors_with_gradient"] == stale["head"]["tensors"]
            and stale["head"]["gradient_l2_norm"] > 0
            and stale["head"]["parameter_update_l2_norm"] == 0
            and set(stale_audit["missing_trainable_names"]) == {"fc.weight", "fc.bias"}
            and stale_audit["foreign_parameter_tensors"] == 2
        ),
        "frozen_head_has_no_gradients_or_updates": (
            _frozen_head_only(frozen["trainability"])
            and frozen["head"]["tensors_with_gradient"] == 0
            and frozen["head"]["parameter_update_l2_norm"] == 0
        ),
        "frozen_head_passes_existing_coverage_audit": _complete_coverage(frozen["optimizer_audit"]),
        "all_backbones_have_gradients_and_updates": all(
            _has_gradients_and_updates(variant["backbone"]) for variant in variants.values()
        ),
    }
    candidates = {
        name: assess_candidate(variants[name], frozen, expected_state_hash)
        for name in CANDIDATE_ACTIONS
    }
    hashes = ("gradient_sha256", "post_step_sha256")
    predictions = {
        "P1_frozen_head_bitwise_unchanged": (
            frozen["head"]["post_step_sha256"] == frozen["head"]["initial_sha256"]
        ),
        "P2_frozen_and_stale_backbones_bitwise_equal": (
            frozen["pre_update_loss"] == stale["pre_update_loss"]
            and all(frozen["backbone"][key] == stale["backbone"][key] for key in hashes)
        ),
        "P3_repaired_matches_clean_bitwise": (
            repaired["pre_update_loss"] == clean["pre_update_loss"]
            and all(
                repaired[group][key] == clean[group][key]
                for group in ("head", "backbone")
                for key in hashes
            )
        ),
        "P4_rebuilt_optimizer_rejected": (
            candidates["frozen_head_optimizer_rebuilt"]["decision"] == "rejected"
            and rebuilt["head"]["parameter_update_l2_norm"] == 0
        ),
    }
    return {"checks": checks, "candidates": candidates, "predictions": predictions}
