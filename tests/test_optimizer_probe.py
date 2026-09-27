"""CPU-only tests for the controlled optimizer probe (skip without PyTorch)."""

import json
import math
import unittest
from copy import deepcopy
from unittest.mock import patch

try:
    import torch
    from torch import nn
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None
    nn = None

if torch is not None:
    from runsleuth.optimizer_probe import (
        make_probe_optimizer,
        run_probe_step,
        state_dict_sha256,
    )

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(3, 4)
            self.fc = nn.Linear(4, 2)
            self.register_buffer("reference_buffer", torch.tensor([1.0, 2.0]))
            with torch.no_grad():
                self.backbone.weight.copy_(torch.linspace(-0.5, 0.6, 12).reshape(4, 3))
                self.backbone.bias.copy_(torch.tensor([0.1, -0.2, 0.3, -0.1]))
                self.fc.weight.copy_(torch.tensor([[0.3, -0.4, 0.2, 0.5], [-0.2, 0.1, 0.4, -0.3]]))
                self.fc.bias.copy_(torch.tensor([0.1, -0.1]))

        def forward(self, inputs):
            return self.fc(torch.tanh(self.backbone(inputs)))


@unittest.skipIf(torch is None, "PyTorch is not installed; CPU probe tests require torch")
class OptimizerProbeTests(unittest.TestCase):
    def setUp(self):
        self.model = TinyModel()
        self.inputs = torch.tensor(
            [[0.2, 0.7, -0.4], [-0.5, 0.1, 0.8], [1.0, -0.2, 0.3], [-0.4, -0.6, 0.2]]
        )
        self.targets = torch.tensor([0, 1, 1, 0])
        self.options = {"learning_rate": 0.01, "weight_decay": 0.02}

    def run_variant(self, variant, model=None):
        return run_probe_step(
            deepcopy(self.model) if model is None else model,
            self.inputs,
            self.targets,
            variant=variant,
            **self.options,
        )

    def test_same_initial_state_and_loss_expose_missing_head_update(self):
        clean = self.run_variant("clean")
        stale = self.run_variant("stale_head")
        self.assertEqual(clean["initial_state_sha256"], stale["initial_state_sha256"])
        self.assertEqual(clean["initial_state_sha256"], state_dict_sha256(self.model.state_dict()))
        self.assertEqual(clean["pre_update_loss"], stale["pre_update_loss"])
        self.assertEqual(clean["pre_update_accuracy"], stale["pre_update_accuracy"])
        self.assertEqual(clean["head"]["gradient_l2_norm"], stale["head"]["gradient_l2_norm"])
        self.assertEqual(
            clean["backbone"]["gradient_l2_norm"], stale["backbone"]["gradient_l2_norm"]
        )

        self.assertEqual(clean["optimizer_audit"]["missing_trainable_names"], ())
        self.assertEqual(clean["optimizer_audit"]["foreign_parameter_tensors"], 0)
        self.assertEqual(
            stale["optimizer_audit"]["missing_trainable_names"], ("fc.bias", "fc.weight")
        )
        self.assertEqual(stale["optimizer_audit"]["foreign_parameter_tensors"], 2)
        for result in (clean, stale):
            audit = result["optimizer_audit"]
            self.assertEqual(audit["model_parameter_tensors"], 4)
            self.assertEqual(audit["trainable_parameter_tensors"], 4)
            self.assertEqual(audit["optimizer_unique_parameter_tensors"], 4)
            self.assertEqual(audit["duplicate_parameter_occurrences"], 0)
            self.assertGreater(result["head"]["gradient_l2_norm"], 0)
            self.assertGreater(result["backbone"]["gradient_l2_norm"], 0)
            self.assertGreater(result["backbone"]["parameter_update_l2_norm"], 0)
            self.assertTrue(math.isfinite(result["pre_update_loss"]))
            json.dumps(result, allow_nan=False)
        self.assertGreater(clean["head"]["parameter_update_l2_norm"], 0)
        self.assertEqual(stale["head"]["parameter_update_l2_norm"], 0.0)

    def test_rebuilt_optimizer_matches_clean_step_and_final_model_state(self):
        clean_model = deepcopy(self.model)
        repaired_model = deepcopy(self.model)
        clean = self.run_variant("clean", clean_model)
        repaired = self.run_variant("stale_head_repaired", repaired_model)
        self.assertNotIn("repair", clean)
        self.assertNotIn("repair", self.run_variant("stale_head"))
        for key in (
            "initial_state_sha256",
            "pre_update_loss",
            "pre_update_accuracy",
            "optimizer_audit",
            "head",
            "backbone",
        ):
            with self.subTest(key=key):
                self.assertEqual(clean[key], repaired[key])
        self.assertGreater(repaired["head"]["parameter_update_l2_norm"], 0)
        self.assertGreater(repaired["backbone"]["parameter_update_l2_norm"], 0)
        self.assertEqual(
            state_dict_sha256(clean_model.state_dict()),
            state_dict_sha256(repaired_model.state_dict()),
        )
        json.dumps(repaired, allow_nan=False)

    def test_repair_records_stale_binding_without_changing_model_state(self):
        initial_hash = state_dict_sha256(self.model.state_dict())
        result = self.run_variant("stale_head_repaired")
        repair = result["repair"]
        self.assertEqual(repair["action"], "rebuild_optimizer")
        self.assertEqual(repair["timing"], "before_first_optimizer_step")
        self.assertEqual(repair["optimizer_state_entries_before"], 0)
        self.assertEqual(repair["model_state_sha256_before"], initial_hash)
        self.assertEqual(repair["model_state_sha256_after"], initial_hash)
        self.assertEqual(result["initial_state_sha256"], initial_hash)
        self.assertEqual(
            repair["optimizer_audit_before"]["missing_trainable_names"],
            ("fc.bias", "fc.weight"),
        )
        self.assertEqual(repair["optimizer_audit_before"]["foreign_parameter_tensors"], 2)
        self.assertEqual(repair["optimizer_audit_before"]["duplicate_parameter_occurrences"], 0)
        self.assertEqual(result["optimizer_audit"]["missing_trainable_names"], ())
        self.assertEqual(result["optimizer_audit"]["foreign_parameter_tensors"], 0)

    def test_repair_rejects_existing_optimizer_state_before_rebuild_or_training(self):
        model = deepcopy(self.model)
        optimizer = make_probe_optimizer(model, variant="stale_head", **self.options)
        parameter = optimizer.param_groups[0]["params"][0]
        previous_state = {"step": torch.tensor(1.0), "exp_avg": torch.ones_like(parameter)}
        optimizer.state[parameter] = previous_state
        initial_hash = state_dict_sha256(model.state_dict())
        with (
            patch("runsleuth.optimizer_probe.make_probe_optimizer", return_value=optimizer),
            patch("runsleuth.optimizer_probe.torch.optim.AdamW") as rebuild,
            patch.object(model, "forward", wraps=model.forward) as forward,
            patch.object(optimizer, "step", wraps=optimizer.step) as step,
        ):
            with self.assertRaisesRegex(ValueError, "nonempty state"):
                self.run_variant("stale_head_repaired", model)
            rebuild.assert_not_called()
            forward.assert_not_called()
            step.assert_not_called()
        self.assertIs(optimizer.state[parameter], previous_state)
        self.assertEqual(initial_hash, state_dict_sha256(model.state_dict()))

    def test_both_variants_clone_weights_but_change_parameter_identity(self):
        for variant in ("clean", "stale_head"):
            with self.subTest(variant=variant):
                model = deepcopy(self.model)
                old_head = model.fc
                old_parameters = tuple(old_head.parameters())
                initial_hash = state_dict_sha256(model.state_dict())
                optimizer = make_probe_optimizer(model, variant=variant, **self.options)
                self.assertIsNot(model.fc, old_head)
                self.assertEqual(initial_hash, state_dict_sha256(model.state_dict()))
                optimizer_ids = {
                    id(parameter)
                    for group in optimizer.param_groups
                    for parameter in group["params"]
                }
                for old_parameter, new_parameter in zip(
                    old_parameters, model.fc.parameters(), strict=True
                ):
                    self.assertIsNot(old_parameter, new_parameter)
                    self.assertTrue(torch.equal(old_parameter, new_parameter))
                    self.assertTrue(new_parameter.requires_grad)
                    self.assertEqual(id(new_parameter) in optimizer_ids, variant == "clean")
                    self.assertEqual(id(old_parameter) in optimizer_ids, variant == "stale_head")

    def test_stale_optimizer_keeps_discarded_head_with_no_backward_gradient(self):
        model = deepcopy(self.model)
        old_head = model.fc
        old_values = [parameter.detach().clone() for parameter in old_head.parameters()]
        optimizer = make_probe_optimizer(model, variant="stale_head", **self.options)
        model.zero_grad(set_to_none=True)
        optimizer.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(model(self.inputs), self.targets).backward()
        for parameter in model.fc.parameters():
            self.assertIsNotNone(parameter.grad)
        optimizer.step()
        for parameter, old_value in zip(old_head.parameters(), old_values, strict=True):
            self.assertIsNone(parameter.grad)
            self.assertTrue(torch.equal(parameter, old_value))

    def test_preexisting_gradients_are_cleared(self):
        for variant in ("clean", "stale_head", "stale_head_repaired"):
            with self.subTest(variant=variant):
                dirty = deepcopy(self.model)
                for parameter in dirty.parameters():
                    parameter.grad = torch.full_like(parameter, 123.0)
                self.assertEqual(self.run_variant(variant), self.run_variant(variant, dirty))

    def test_frozen_parameter_stays_unchanged(self):
        for variant in ("clean", "stale_head"):
            with self.subTest(variant=variant):
                model = deepcopy(self.model)
                model.backbone.bias.requires_grad_(False)
                before = model.backbone.bias.detach().clone()
                result = self.run_variant(variant, model)
                self.assertTrue(torch.equal(before, model.backbone.bias))
                self.assertIsNone(model.backbone.bias.grad)
                self.assertEqual(result["optimizer_audit"]["trainable_parameter_tensors"], 3)

    def test_state_hash_covers_names_dtype_shape_values_and_buffers(self):
        state = self.model.state_dict()
        expected = state_dict_sha256(state)
        self.assertEqual(len(expected), 64)
        self.assertEqual(expected, state_dict_sha256(dict(reversed(list(state.items())))))
        changed = deepcopy(state)
        changed["reference_buffer"][0] += 1
        self.assertNotEqual(expected, state_dict_sha256(changed))
        value = torch.tensor([1, 2], dtype=torch.int32)
        base = state_dict_sha256({"value": value})
        self.assertNotEqual(base, state_dict_sha256({"renamed": value}))
        self.assertNotEqual(base, state_dict_sha256({"value": value.reshape(1, 2)}))
        self.assertNotEqual(base, state_dict_sha256({"value": value.float()}))
        self.assertEqual(
            len(state_dict_sha256({"scalar": torch.tensor(1.0, dtype=torch.bfloat16)})), 64
        )

    def test_hash_accepts_noncontiguous_tensor_and_scalar_buffer(self):
        tensor = torch.arange(12).reshape(3, 4).t()
        self.assertFalse(tensor.is_contiguous())
        state = {"weights": tensor, "batch_counter": torch.tensor(0)}
        contiguous = {name: value.contiguous() for name, value in state.items()}
        self.assertEqual(state_dict_sha256(state), state_dict_sha256(contiguous))

    def test_invalid_variant_is_rejected_without_mutating_head(self):
        original_head = self.model.fc
        with self.assertRaisesRegex(ValueError, "Unknown variant"):
            make_probe_optimizer(self.model, variant="unknown", **self.options)
        self.assertIs(original_head, self.model.fc)

    def test_requires_head_and_backbone_parameters(self):
        no_head = nn.Sequential(nn.Linear(3, 2))
        empty_head = nn.Module()
        empty_head.backbone = nn.Linear(3, 2)
        empty_head.fc = nn.Identity()
        no_backbone = nn.Module()
        no_backbone.fc = nn.Linear(3, 2)
        for model in (no_head, empty_head, no_backbone):
            with self.subTest(model=model):
                with self.assertRaisesRegex(ValueError, "requires"):
                    make_probe_optimizer(model, variant="clean", **self.options)

    def test_nonfinite_measurements_are_rejected(self):
        inputs = self.inputs.clone()
        inputs[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            run_probe_step(self.model, inputs, self.targets, variant="clean", **self.options)


if __name__ == "__main__":
    unittest.main()
