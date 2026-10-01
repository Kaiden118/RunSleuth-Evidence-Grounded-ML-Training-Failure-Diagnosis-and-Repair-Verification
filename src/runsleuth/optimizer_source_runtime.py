"""Execute only the validated constructor from two isolated source versions.

This is a controlled one-step mechanism check. It neither imports the snapshot
as a module nor evaluates validation performance or accepts a training repair.
"""

import ast
import math
from copy import deepcopy
from dataclasses import asdict

from runsleuth.optimizer_source_patch import propose_optimizer_order_patch


def _validated_constructor_code(original_source: str, proposed_source: str) -> tuple:
    """Validate the exact two-statement change before compiling either function."""
    proposal = propose_optimizer_order_patch(original_source)
    if proposed_source != proposal["proposed_source"]:
        raise ValueError("Proposed source must equal the exact validated statement reorder")

    def extract(source: str, filename: str):
        function = next(
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name == "make_probe_optimizer"
        )
        # Annotations were ignored by the shape validator. Strip them so that
        # annotation expressions cannot execute when this function is defined.
        function.returns = None
        for argument in function.args.posonlyargs + function.args.args + function.args.kwonlyargs:
            argument.annotation = None
        module = ast.Module(body=[function], type_ignores=[])
        ast.fix_missing_locations(module)
        return compile(module, filename, "exec", dont_inherit=True)

    return (
        extract(original_source, "<validated-original-optimizer-constructor>"),
        extract(proposed_source, "<validated-proposed-optimizer-constructor>"),
    )


def _build_model():
    from runsleuth.camelyon import build_camelyon_model

    return build_camelyon_model(pretrained=False)


def _probe_step(model, inputs, targets, *, constructor, variant: str, config) -> dict:
    import torch

    from runsleuth.optimizer_audit import audit_optimizer_parameters
    from runsleuth.optimizer_probe import _l2_norm, _parameter_groups, state_dict_sha256

    optimizer = constructor(
        model,
        variant=variant,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    head, backbone = _parameter_groups(model)
    groups = {"head": head, "backbone": backbone}
    before = {
        name: parameter.detach().clone()
        for parameters in groups.values()
        for name, parameter in parameters.items()
    }
    audit = asdict(audit_optimizer_parameters(model, optimizer))
    initial_hash = state_dict_sha256(model.state_dict())
    state_entries_before = len(optimizer.state)
    model.train()
    model.zero_grad(set_to_none=True)
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = torch.nn.functional.cross_entropy(logits, targets)
    loss_value = float(loss.detach().item())
    accuracy = float((logits.detach().argmax(dim=1) == targets).float().mean().item())
    if not math.isfinite(loss_value) or not math.isfinite(accuracy):
        raise ValueError("Source probe produced non-finite loss or accuracy")
    loss.backward()
    measurements = {
        group: {
            "gradient_l2_norm": _l2_norm(
                parameter.grad for parameter in parameters.values() if parameter.grad is not None
            )
        }
        for group, parameters in groups.items()
    }
    if any(not math.isfinite(values["gradient_l2_norm"]) for values in measurements.values()):
        raise ValueError("Source probe produced non-finite gradients")
    optimizer.step()
    for group, parameters in groups.items():
        update_norm = _l2_norm(
            parameter.detach() - before[name] for name, parameter in parameters.items()
        )
        if not math.isfinite(update_norm):
            raise ValueError("Source probe produced non-finite parameter updates")
        measurements[group]["parameter_update_l2_norm"] = update_norm
    if any(not torch.isfinite(tensor).all().item() for tensor in model.state_dict().values()):
        raise ValueError("Source probe produced a non-finite model state")
    return {
        "constructor_variant": variant,
        "optimizer_state_entries_before": state_entries_before,
        "initial_state_sha256": initial_hash,
        "post_step_state_sha256": state_dict_sha256(model.state_dict()),
        "pre_update_loss": loss_value,
        "pre_update_accuracy": accuracy,
        "optimizer_audit": audit,
        **measurements,
    }


def _checks(variants: dict, expected_state_hash: str) -> dict:
    clean = variants["clean"]
    stale = variants["stale_head"]
    patched = variants["source_patched"]

    def full_binding(result: dict) -> bool:
        audit = result["optimizer_audit"]
        return (
            not audit["missing_trainable_names"]
            and audit["foreign_parameter_tensors"] == 0
            and audit["duplicate_parameter_occurrences"] == 0
            and audit["optimizer_unique_parameter_tensors"]
            == audit["trainable_parameter_tensors"]
            == audit["model_parameter_tensors"]
        )

    def updates(result: dict, group: str) -> bool:
        return (
            result[group]["gradient_l2_norm"] > 0.0
            and result[group]["parameter_update_l2_norm"] > 0.0
        )

    stale_audit = stale["optimizer_audit"]
    return {
        "all_use_reference_initial_state": all(
            result["initial_state_sha256"] == expected_state_hash for result in variants.values()
        ),
        "all_optimizers_are_fresh": all(
            result["optimizer_state_entries_before"] == 0 for result in variants.values()
        ),
        "same_pre_update_loss_and_accuracy": all(
            result["pre_update_loss"] == clean["pre_update_loss"]
            and result["pre_update_accuracy"] == clean["pre_update_accuracy"]
            for result in variants.values()
        ),
        "same_initial_head_and_backbone_gradients": all(
            result[group]["gradient_l2_norm"] == clean[group]["gradient_l2_norm"]
            for result in variants.values()
            for group in ("head", "backbone")
        ),
        "clean_optimizer_covers_current_parameters": full_binding(clean),
        "stale_optimizer_misses_exactly_current_head": (
            tuple(stale_audit["missing_trainable_names"]) == ("fc.bias", "fc.weight")
            and stale_audit["foreign_parameter_tensors"] == 2
            and stale_audit["duplicate_parameter_occurrences"] == 0
            and stale_audit["optimizer_unique_parameter_tensors"]
            == stale_audit["trainable_parameter_tensors"]
            == stale_audit["model_parameter_tensors"]
        ),
        "clean_head_has_gradients_and_updates": updates(clean, "head"),
        "stale_head_has_gradients_without_updates": (
            stale["head"]["gradient_l2_norm"] > 0.0
            and stale["head"]["parameter_update_l2_norm"] == 0.0
        ),
        "patched_optimizer_covers_current_parameters": full_binding(patched),
        "patched_head_has_gradients_and_updates": updates(patched, "head"),
        "all_backbones_have_gradients_and_updates": all(
            updates(result, "backbone") for result in variants.values()
        ),
        "patched_step_matches_clean_including_buffers": (
            patched["post_step_state_sha256"] == clean["post_step_state_sha256"]
        ),
        "stale_step_differs_from_clean": (
            stale["post_step_state_sha256"] != clean["post_step_state_sha256"]
        ),
    }


def run_source_step_probe(
    original_source: str,
    proposed_source: str,
    *,
    state,
    inputs,
    targets,
    config,
    device,
) -> dict:
    """Compare a fresh clean, stale and source-patched one-step execution.

    The caller supplies a hash-checked recorded initial checkpoint and identical batch.
    No code from the snapshot outside the allowlisted constructor is executed.
    """
    original_code, proposed_code = _validated_constructor_code(original_source, proposed_source)
    if type(config.seed) is not int or not 0 <= config.seed < 2**64:
        raise ValueError("Source probe seed must be an integer in [0, 2**64)")
    import torch

    from runsleuth.optimizer_probe import _VARIANTS, _parameter_groups, state_dict_sha256
    from runsleuth.train import seed_everything

    def bind(code):
        namespace = {
            "__builtins__": {"ValueError": ValueError},
            "torch": torch,
            "math": math,
            "deepcopy": deepcopy,
            "_VARIANTS": _VARIANTS,
            "_parameter_groups": _parameter_groups,
        }
        exec(code, namespace)
        return namespace["make_probe_optimizer"]

    original_constructor = bind(original_code)
    patched_constructor = bind(proposed_code)
    expected_state_hash = state_dict_sha256(state)
    if any(not torch.isfinite(tensor).all().item() for tensor in state.values()):
        raise ValueError("Source probe requires a finite initial model state")
    inputs, targets = inputs.to(device), targets.to(device)
    variants = {}
    for name, constructor, argument in (
        ("clean", original_constructor, "clean"),
        ("stale_head", original_constructor, "stale_head"),
        ("source_patched", patched_constructor, "stale_head"),
    ):
        seed_everything(config.seed)
        model = _build_model()
        model.load_state_dict(state, strict=True)
        model.to(device)
        # Reset before the forward pass as well: model initialization must not
        # influence any stochastic layer's sequence in the controlled step.
        seed_everything(config.seed)
        variants[name] = _probe_step(
            model, inputs, targets, constructor=constructor, variant=argument, config=config
        )
        del model
    checks = _checks(variants, expected_state_hash)
    return {
        "execution_scope": "extracted_make_probe_optimizer_only",
        "source_patch_executed": True,
        "training_mode": "from_reference_initial_state",
        "optimizer_steps": 3,
        "reference_initial_state_sha256": expected_state_hash,
        "variants": variants,
        "checks": checks,
        "mechanism_verified": all(checks.values()),
        "performance_recovery_evaluated": False,
    }
