"""CPU-only tests for the single-step frozen-head probe (skip without PyTorch)."""

import json
import unittest
from copy import deepcopy

try:
    import torch
    from torch import nn
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None
    nn = None

if torch is not None:
    from runsleuth.frozen_head_probe import (
        VARIANTS,
        assess_frozen_head_probe,
        make_variant_optimizer,
        run_variant_step,
    )
    from runsleuth.optimizer_probe import state_dict_sha256

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(3, 4)
            self.fc = nn.Linear(4, 2)
            with torch.no_grad():
                self.backbone.weight.copy_(torch.linspace(-0.5, 0.6, 12).reshape(4, 3))
                self.backbone.bias.copy_(torch.tensor([0.1, -0.2, 0.3, -0.1]))
                self.fc.weight.copy_(torch.tensor([[0.3, -0.4, 0.2, 0.5], [-0.2, 0.1, 0.4, -0.3]]))
                self.fc.bias.copy_(torch.tensor([0.1, -0.1]))

        def forward(self, inputs):
            return self.fc(torch.tanh(self.backbone(inputs)))


@unittest.skipIf(torch is None, "PyTorch is not installed; CPU probe tests require torch")
class FrozenHeadProbeTests(unittest.TestCase):
    def setUp(self):
        self.model = TinyModel()
        self.inputs = torch.tensor(
            [[0.2, 0.7, -0.4], [-0.5, 0.1, 0.8], [1.0, -0.2, 0.3], [-0.4, -0.6, 0.2]]
        )
        self.targets = torch.tensor([0, 1, 1, 0])
        self.options = {"learning_rate": 0.01, "weight_decay": 0.02}
        self.expected_hash = state_dict_sha256(self.model.state_dict())

    def run_all(self):
        return {
            variant: run_variant_step(
                deepcopy(self.model), self.inputs, self.targets, variant=variant, **self.options
            )
            for variant in VARIANTS
        }

    def test_variants_reproduce_mechanisms_and_predictions(self):
        variants = self.run_all()
        assessment = assess_frozen_head_probe(variants, self.expected_hash)
        self.assertTrue(all(assessment["checks"].values()), assessment["checks"])
        self.assertEqual(assessment["candidates"]["frozen_head_repaired"]["decision"], "accepted")
        rebuilt = assessment["candidates"]["frozen_head_optimizer_rebuilt"]
        self.assertEqual(rebuilt["decision"], "rejected")
        self.assertFalse(rebuilt["checks"]["all_parameters_trainable"])
        self.assertFalse(rebuilt["checks"]["head_and_backbone_have_gradients_and_updates"])
        self.assertTrue(all(assessment["predictions"].values()), assessment["predictions"])
        json.dumps({"variants": variants, **assessment}, allow_nan=False)

    def test_frozen_variant_matches_documented_fault_definition(self):
        model = deepcopy(self.model)
        optimizer, intervention = make_variant_optimizer(
            model, variant="frozen_head", **self.options
        )
        self.assertIsNone(intervention)
        self.assertEqual(
            [id(parameter) for group in optimizer.param_groups for parameter in group["params"]],
            [id(parameter) for parameter in model.parameters()],
        )
        self.assertFalse(any(parameter.requires_grad for parameter in model.fc.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.backbone.parameters()))
        self.assertEqual(state_dict_sha256(model.state_dict()), self.expected_hash)

    def test_assessment_detects_falsified_predictions_and_broken_mechanisms(self):
        variants = self.run_all()
        changed = deepcopy(variants)
        changed["frozen_head"]["backbone"]["post_step_sha256"] = "0" * 64
        predictions = assess_frozen_head_probe(changed, self.expected_hash)["predictions"]
        self.assertFalse(predictions["P2_frozen_and_stale_backbones_bitwise_equal"])

        changed = deepcopy(variants)
        changed["frozen_head_optimizer_rebuilt"] = dict(
            variants["frozen_head_repaired"], variant="frozen_head_optimizer_rebuilt"
        )
        predictions = assess_frozen_head_probe(changed, self.expected_hash)["predictions"]
        self.assertFalse(predictions["P4_rebuilt_optimizer_rejected"])

        changed = deepcopy(variants)
        changed["frozen_head"]["head"]["tensors_with_gradient"] = 2
        checks = assess_frozen_head_probe(changed, self.expected_hash)["checks"]
        self.assertFalse(checks["frozen_head_has_no_gradients_or_updates"])

    def test_unknown_variant_and_invalid_hyperparameters_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown variant"):
            make_variant_optimizer(deepcopy(self.model), variant="frozen", **self.options)
        for options in (
            {"learning_rate": 0.0, "weight_decay": 0.0},
            {"learning_rate": 0.01, "weight_decay": -1.0},
        ):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    make_variant_optimizer(deepcopy(self.model), variant="frozen_head", **options)


if __name__ == "__main__":
    unittest.main()
