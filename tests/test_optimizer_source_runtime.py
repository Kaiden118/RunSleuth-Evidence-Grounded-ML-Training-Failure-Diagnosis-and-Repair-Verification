"""Actual source execution is restricted to a validated constructor function."""

import ast
import json
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from runsleuth.optimizer_source_patch import propose_optimizer_order_patch
from runsleuth.optimizer_source_runtime import (
    _checks,
    _validated_constructor_code,
    run_source_step_probe,
)

try:
    import torch
    from torch import nn
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None
    nn = None


SOURCE_PATH = Path(__file__).resolve().parents[1] / "src/runsleuth/optimizer_probe.py"


class OptimizerSourceRuntimeExtractionTests(unittest.TestCase):
    def setUp(self):
        self.source = SOURCE_PATH.read_text(encoding="utf-8")
        self.proposed = propose_optimizer_order_patch(self.source)["proposed_source"]

    def test_only_constructor_is_defined_without_importing_source_module(self):
        source = (
            "raise RuntimeError('Module body must never execute')\n"
            "import module_that_does_not_exist\n"
            + self.source.replace("from __future__ import annotations\n", "")
        )
        proposed = propose_optimizer_order_patch(source)["proposed_source"]
        for code in _validated_constructor_code(source, proposed):
            with self.subTest(filename=code.co_filename):
                namespace = {"__builtins__": {}}
                exec(code, namespace)
                self.assertEqual(set(namespace), {"__builtins__", "make_probe_optimizer"})
                function = namespace["make_probe_optimizer"]
                self.assertEqual(function.__annotations__, {})
                self.assertEqual(function.__kwdefaults__, None)

    def test_annotation_expressions_are_erased_before_function_definition(self):
        source = self.source.replace("model: nn.Module,", "model: forbidden(),", 1)
        source = source.replace("variant: str,", "variant: forbidden(),", 1)
        source = source.replace("-> torch.optim.AdamW:", "-> forbidden():", 1)
        proposed = propose_optimizer_order_patch(source)["proposed_source"]
        for code in _validated_constructor_code(source, proposed):
            namespace = {"__builtins__": {}}
            exec(code, namespace)
            self.assertEqual(namespace["make_probe_optimizer"].__annotations__, {})

    def test_changed_or_unchanged_proposal_is_rejected_before_model_execution(self):
        for proposed in (self.source, self.proposed + "\n", self.proposed.replace("AdamW", "SGD")):
            with self.subTest(proposed=proposed[-60:]):
                with patch("runsleuth.optimizer_source_runtime._build_model") as build:
                    with self.assertRaisesRegex(ValueError, "exact validated statement reorder"):
                        run_source_step_probe(
                            self.source,
                            proposed,
                            state={},
                            inputs=None,
                            targets=None,
                            config=SimpleNamespace(seed=42),
                            device="cpu",
                        )
                    build.assert_not_called()

    def test_nonallowlisted_constructor_statements_and_defaults_are_refused(self):
        originals = (
            self.source.replace("new_head = deepcopy(model.fc)", "new_head = forbidden()", 1),
            self.source.replace("variant: str,", "variant: str = forbidden(),", 1),
            self.source.replace(
                "def make_probe_optimizer(", "@forbidden()\ndef make_probe_optimizer(", 1
            ),
        )
        for source in originals:
            with self.subTest(source=source[-60:]):
                with self.assertRaises(ValueError):
                    _validated_constructor_code(source, self.proposed)

    def test_original_and_patched_code_retain_different_stale_operation_order(self):
        original, proposed = _validated_constructor_code(self.source, self.proposed)
        namespaces = []
        for code in (original, proposed):
            namespace = {"__builtins__": {}}
            exec(code, namespace)
            namespaces.append(namespace)
        self.assertNotEqual(
            namespaces[0]["make_probe_optimizer"].__code__.co_code,
            namespaces[1]["make_probe_optimizer"].__code__.co_code,
        )
        ast.parse(self.proposed, feature_version=(3, 11))

    def test_exact_post_step_comparison_includes_state_hash_not_just_norms(self):
        audit = {
            "model_parameter_tensors": 4,
            "trainable_parameter_tensors": 4,
            "optimizer_unique_parameter_tensors": 4,
            "missing_trainable_names": [],
            "foreign_parameter_tensors": 0,
            "duplicate_parameter_occurrences": 0,
        }
        clean = {
            "optimizer_audit": audit,
            "initial_state_sha256": "initial",
            "post_step_state_sha256": "updated",
            "optimizer_state_entries_before": 0,
            "pre_update_loss": 0.5,
            "pre_update_accuracy": 0.5,
            "head": {"gradient_l2_norm": 1.0, "parameter_update_l2_norm": 0.1},
            "backbone": {"gradient_l2_norm": 1.0, "parameter_update_l2_norm": 0.1},
        }
        stale = deepcopy(clean)
        stale["optimizer_audit"]["missing_trainable_names"] = ["fc.bias", "fc.weight"]
        stale["optimizer_audit"]["foreign_parameter_tensors"] = 2
        stale["head"]["parameter_update_l2_norm"] = 0.0
        stale["post_step_state_sha256"] = "stale"
        variants = {"clean": clean, "stale_head": stale, "source_patched": deepcopy(clean)}
        self.assertTrue(all(_checks(variants, "initial").values()))
        variants["source_patched"]["post_step_state_sha256"] = "different-buffer"
        self.assertFalse(
            _checks(variants, "initial")["patched_step_matches_clean_including_buffers"]
        )


if torch is not None:

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(3, 4)
            self.norm = nn.BatchNorm1d(4)
            self.dropout = nn.Dropout(0.2)
            self.fc = nn.Linear(4, 2)

        def forward(self, inputs):
            return self.fc(self.dropout(torch.tanh(self.norm(self.backbone(inputs)))))


@unittest.skipIf(torch is None, "PyTorch is required for actual source one-step CPU tests")
class OptimizerSourceRuntimeTensorTests(unittest.TestCase):
    def setUp(self):
        self.source = SOURCE_PATH.read_text(encoding="utf-8")
        self.proposed = propose_optimizer_order_patch(self.source)["proposed_source"]
        torch.manual_seed(42)
        self.state = deepcopy(TinyModel().state_dict())
        self.inputs = torch.tensor(
            [[0.2, 0.7, -0.4], [-0.5, 0.1, 0.8], [1.0, -0.2, 0.3], [-0.4, -0.6, 0.2]]
        )
        self.targets = torch.tensor([0, 1, 1, 0])
        self.config = SimpleNamespace(seed=42, learning_rate=0.01, weight_decay=0.02)

    def run_probe(self, source=None):
        source = self.source if source is None else source
        proposed = propose_optimizer_order_patch(source)["proposed_source"]
        with patch("runsleuth.optimizer_source_runtime._build_model", side_effect=TinyModel):
            return run_source_step_probe(
                source,
                proposed,
                state=self.state,
                inputs=self.inputs,
                targets=self.targets,
                config=self.config,
                device=torch.device("cpu"),
            )

    def test_actual_proposed_constructor_restores_updates_and_matches_clean_state(self):
        from runsleuth.optimizer_probe import state_dict_sha256

        expected_hash = state_dict_sha256(self.state)
        with (
            patch("runsleuth.optimizer_probe.make_probe_optimizer", side_effect=AssertionError),
            patch(
                "runsleuth.optimizer_probe.make_repaired_probe_optimizer",
                side_effect=AssertionError,
            ),
        ):
            report = self.run_probe()
        self.assertTrue(report["mechanism_verified"], report["checks"])
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(report["execution_scope"], "extracted_make_probe_optimizer_only")
        self.assertTrue(report["source_patch_executed"])
        self.assertFalse(report["performance_recovery_evaluated"])
        self.assertEqual(report["optimizer_steps"], 3)
        self.assertEqual(state_dict_sha256(self.state), expected_hash)
        variants = report["variants"]
        self.assertEqual(variants["stale_head"]["head"]["parameter_update_l2_norm"], 0.0)
        self.assertGreater(variants["source_patched"]["head"]["parameter_update_l2_norm"], 0.0)
        self.assertEqual(variants["source_patched"]["constructor_variant"], "stale_head")
        json.dumps(report, allow_nan=False)

    def test_module_side_effects_and_annotation_calls_are_never_executed(self):
        source = self.source.replace("from __future__ import annotations\n", "")
        source = "raise RuntimeError('Do not execute snapshot module')\n" + source
        source = source.replace("model: nn.Module,", "model: forbidden(),", 1)
        self.assertTrue(self.run_probe(source)["mechanism_verified"])

    def test_nonfinite_input_and_checkpoint_are_refused(self):
        self.inputs[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite loss"):
            self.run_probe()
        self.inputs[0, 0] = 0.2
        self.state["fc.weight"][0, 0] = float("inf")
        with self.assertRaisesRegex(ValueError, "finite initial model state"):
            self.run_probe()

    def test_checkpoint_is_loaded_strictly(self):
        self.state.pop("fc.bias")
        with self.assertRaises(RuntimeError):
            self.run_probe()
