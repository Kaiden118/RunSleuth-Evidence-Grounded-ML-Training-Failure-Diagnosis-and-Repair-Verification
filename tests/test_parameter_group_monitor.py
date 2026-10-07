"""CPU integration tests for epoch-scoped optimizer parameter group monitoring."""

import json
import math
import unittest
from copy import deepcopy
from unittest.mock import patch

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None
    nn = None

if torch is not None:
    from runsleuth.parameter_group_monitor import ParameterGroupMonitor, parameter_trainability
    from runsleuth.train import train_one_epoch

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


@unittest.skipIf(torch is None, "PyTorch is not installed; monitor CPU tests require torch")
class ParameterGroupMonitorTests(unittest.TestCase):
    def setUp(self):
        self.model = TinyModel()
        self.inputs = torch.tensor(
            [[0.2, 0.7, -0.4], [-0.5, 0.1, 0.8], [1.0, -0.2, 0.3], [-0.4, -0.6, 0.2]]
        )
        self.targets = torch.tensor([0, 1, 1, 0])
        self.loader = DataLoader(TensorDataset(self.inputs, self.targets), batch_size=2)

    def make_pair(self, variant="clean"):
        model = deepcopy(self.model)
        head = deepcopy(model.fc)
        if variant in ("clean", "frozen_head"):
            model.fc = head
        if variant == "frozen_head":
            model.fc.requires_grad_(False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.02)
        if variant == "stale_head":
            model.fc = head
        return model, optimizer

    def train(self, model, optimizer, **kwargs):
        return train_one_epoch(
            model,
            self.loader,
            optimizer,
            nn.CrossEntropyLoss(),
            torch.device("cpu"),
            **kwargs,
        )

    def assert_cleaned_up(self, monitor, model, optimizer):
        self.assertEqual(len(model._forward_pre_hooks), 0)
        self.assertEqual(len(optimizer._optimizer_step_pre_hooks), 0)
        self.assertEqual(len(optimizer._optimizer_step_post_hooks), 0)
        self.assertEqual(monitor._handles, [])
        self.assertIsNone(monitor._pending_step)

    @staticmethod
    def reference_norm(tensors):
        return math.sqrt(sum(tensor.detach().double().square().sum().item() for tensor in tensors))

    def reference_epoch(self, model, optimizer):
        """Independent explicit reset/step loop to verify per-step means."""
        records = {group: {"gradient": [], "update": []} for group in ("head", "backbone")}
        model.train()
        for inputs, targets in self.loader:
            model.zero_grad(set_to_none=True)
            optimizer.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(model(inputs), targets).backward()
            groups = {
                "head": dict(model.fc.named_parameters(prefix="fc")),
                "backbone": dict(model.backbone.named_parameters(prefix="backbone")),
            }
            before = {
                name: parameter.detach().clone() for name, parameter in model.named_parameters()
            }
            for group, parameters in groups.items():
                records[group]["gradient"].append(
                    self.reference_norm(parameter.grad for parameter in parameters.values())
                )
            optimizer.step()
            for group, parameters in groups.items():
                records[group]["update"].append(
                    self.reference_norm(
                        parameter.detach() - before[name] for name, parameter in parameters.items()
                    )
                )
        return records

    def test_existing_training_loop_matches_independent_resets_and_mean_norms(self):
        for variant in ("clean", "stale_head"):
            with self.subTest(variant=variant):
                model, optimizer = self.make_pair(variant)
                reference_model, reference_optimizer = self.make_pair(variant)
                for parameter in model.parameters():
                    parameter.grad = torch.full_like(parameter, 123.0)
                with ParameterGroupMonitor(model, optimizer) as monitor:
                    self.train(model, optimizer)
                actual = monitor.summary()
                expected = self.reference_epoch(reference_model, reference_optimizer)
                self.assertEqual(actual["optimizer_steps"], 2)
                for group in ("head", "backbone"):
                    for kind, key in (
                        ("gradient", "mean_gradient_l2_norm"),
                        ("update", "mean_parameter_update_l2_norm"),
                    ):
                        self.assertAlmostEqual(
                            actual[group][key], sum(expected[group][kind]) / 2, places=6
                        )
                    self.assertGreater(actual[group]["mean_gradient_l2_norm"], 0)
                self.assertGreater(actual["backbone"]["mean_parameter_update_l2_norm"], 0)
                if variant == "clean":
                    self.assertGreater(actual["head"]["mean_parameter_update_l2_norm"], 0)
                else:
                    self.assertEqual(actual["head"]["mean_parameter_update_l2_norm"], 0.0)
                for name, parameter in model.named_parameters():
                    self.assertTrue(
                        torch.equal(parameter, dict(reference_model.named_parameters())[name])
                    )
                json.dumps(actual, allow_nan=False)
                self.assert_cleaned_up(monitor, model, optimizer)

    def test_clean_and_stale_first_batch_gradients_match(self):
        summaries = []
        for variant in ("clean", "stale_head"):
            model, optimizer = self.make_pair(variant)
            with ParameterGroupMonitor(model, optimizer) as monitor:
                optimizer.zero_grad(set_to_none=True)
                nn.functional.cross_entropy(model(self.inputs), self.targets).backward()
                optimizer.step()
            summaries.append(monitor.summary())
        for group in ("head", "backbone"):
            self.assertEqual(
                summaries[0][group]["mean_gradient_l2_norm"],
                summaries[1][group]["mean_gradient_l2_norm"],
            )

    def test_tensor_counts_separate_frozen_head_from_stale_head(self):
        # variant: (head trainable tensors, head tensors with gradient)
        expected = {"clean": (2, 2), "stale_head": (2, 2), "frozen_head": (0, 0)}
        for variant, (trainable, with_gradient) in expected.items():
            with self.subTest(variant=variant):
                model, optimizer = self.make_pair(variant)
                initial_head = [parameter.detach().clone() for parameter in model.fc.parameters()]
                with ParameterGroupMonitor(model, optimizer) as monitor:
                    self.train(model, optimizer)
                summary = monitor.summary()
                head, backbone = summary["head"], summary["backbone"]
                self.assertEqual(head["parameter_tensors"], 2)
                self.assertEqual(head["min_trainable_tensors"], trainable)
                self.assertEqual(head["max_trainable_tensors"], trainable)
                self.assertEqual(head["min_tensors_with_gradient"], with_gradient)
                self.assertEqual(head["max_tensors_with_gradient"], with_gradient)
                for key in ("min_trainable_tensors", "min_tensors_with_gradient"):
                    self.assertEqual(backbone[key], backbone["parameter_tensors"])
                self.assertGreater(backbone["mean_parameter_update_l2_norm"], 0)
                head_unchanged = all(
                    torch.equal(parameter, initial)
                    for parameter, initial in zip(model.fc.parameters(), initial_head, strict=True)
                )
                self.assertEqual(head_unchanged, variant != "clean")
                if variant == "frozen_head":
                    self.assertEqual(head["mean_gradient_l2_norm"], 0.0)
                    self.assertEqual(head["mean_parameter_update_l2_norm"], 0.0)
                else:
                    self.assertGreater(head["mean_gradient_l2_norm"], 0)
                json.dumps(summary, allow_nan=False)

    def test_tensor_counts_report_minimum_and_maximum_across_steps(self):
        model, optimizer = self.make_pair("frozen_head")
        with ParameterGroupMonitor(model, optimizer) as monitor:
            for index, (inputs, targets) in enumerate(self.loader):
                if index == 1:
                    model.fc.requires_grad_(True)
                optimizer.zero_grad(set_to_none=True)
                nn.functional.cross_entropy(model(inputs), targets).backward()
                optimizer.step()
        summary = monitor.summary()
        head = summary["head"]
        self.assertEqual(summary["optimizer_steps"], 2)
        self.assertEqual((head["min_trainable_tensors"], head["max_trainable_tensors"]), (0, 2))
        self.assertEqual(
            (head["min_tensors_with_gradient"], head["max_tensors_with_gradient"]), (0, 2)
        )
        self.assertGreater(head["mean_parameter_update_l2_norm"], 0)

    def test_group_size_change_within_epoch_is_rejected(self):
        model, optimizer = self.make_pair()
        monitor = ParameterGroupMonitor(model, optimizer)
        with self.assertRaisesRegex(RuntimeError, "size changed"):
            with monitor:
                for index, (inputs, targets) in enumerate(self.loader):
                    if index == 1:
                        model.fc = nn.Linear(4, 2, bias=False)
                    optimizer.zero_grad(set_to_none=True)
                    nn.functional.cross_entropy(model(inputs), targets).backward()
                    optimizer.step()
        self.assert_cleaned_up(monitor, model, optimizer)

    def test_parameter_trainability_names_frozen_tensors(self):
        for variant, frozen in (("clean", []), ("frozen_head", ["fc.bias", "fc.weight"])):
            with self.subTest(variant=variant):
                model, _ = self.make_pair(variant)
                self.assertEqual(
                    parameter_trainability(model),
                    {
                        "head": {
                            "parameter_tensors": 2,
                            "trainable_tensors": 2 - len(frozen),
                            "frozen_names": frozen,
                        },
                        "backbone": {
                            "parameter_tensors": 2,
                            "trainable_tensors": 2,
                            "frozen_names": [],
                        },
                    },
                )

    def test_evaluation_and_no_grad_forwards_do_not_clear_gradients_or_count_steps(self):
        model, optimizer = self.make_pair()
        with ParameterGroupMonitor(model, optimizer) as monitor:
            self.train(model, optimizer)
            expected = model.fc.weight.grad.clone()
            model.eval()
            model(self.inputs)
            self.assertTrue(torch.equal(model.fc.weight.grad, expected))
            model.train()
            with torch.no_grad():
                model(self.inputs)
            self.assertTrue(torch.equal(model.fc.weight.grad, expected))
        self.assertEqual(monitor.summary()["optimizer_steps"], 2)
        self.assertEqual(monitor.summary()["training_forwards"], 2)

    def test_weight_norms_ratios_and_first_update_match_independent_loop(self):
        model, optimizer = self.make_pair()
        reference_model, reference_optimizer = self.make_pair()
        with ParameterGroupMonitor(model, optimizer) as monitor:
            self.train(model, optimizer)
        actual = monitor.summary()
        records = {group: {"norm": [], "update": []} for group in ("head", "backbone")}
        reference_model.train()
        for inputs, targets in self.loader:
            reference_model.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(reference_model(inputs), targets).backward()
            groups = {
                "head": dict(reference_model.fc.named_parameters(prefix="fc")),
                "backbone": dict(reference_model.backbone.named_parameters(prefix="backbone")),
            }
            before = {
                name: parameter.detach().clone()
                for name, parameter in reference_model.named_parameters()
            }
            for group, parameters in groups.items():
                records[group]["norm"].append(self.reference_norm(parameters.values()))
            reference_optimizer.step()
            for group, parameters in groups.items():
                records[group]["update"].append(
                    self.reference_norm(
                        parameter.detach() - before[name] for name, parameter in parameters.items()
                    )
                )
        for group, elements in (("head", 10), ("backbone", 16)):
            norms, updates = records[group]["norm"], records[group]["update"]
            self.assertEqual(actual[group]["parameter_elements"], elements)
            self.assertAlmostEqual(
                actual[group]["mean_parameter_l2_norm"], sum(norms) / 2, places=6
            )
            self.assertAlmostEqual(
                actual[group]["mean_update_to_weight_ratio"],
                sum(update / norm for update, norm in zip(updates, norms, strict=True)) / 2,
                places=6,
            )
            self.assertAlmostEqual(
                actual[group]["first_step_parameter_update_l2_norm"], updates[0], places=6
            )

    def test_missing_steps_are_counted_when_allowed(self):
        model, optimizer = self.make_pair()
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        with ParameterGroupMonitor(model, optimizer, allow_missing_steps=True) as monitor:
            self.train(model, optimizer, optimizer_step_enabled=False)
        self.assertEqual(
            monitor.summary(),
            {"optimizer_steps": 0, "training_forwards": 2, "head": None, "backbone": None},
        )
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.equal(parameter, before[name]), name)
        self.assert_cleaned_up(monitor, model, optimizer)

    def test_partial_steps_report_groups_over_completed_steps_only(self):
        model, optimizer = self.make_pair()
        with ParameterGroupMonitor(model, optimizer, allow_missing_steps=True) as monitor:
            for index, (inputs, targets) in enumerate(self.loader):
                optimizer.zero_grad(set_to_none=True)
                nn.functional.cross_entropy(model(inputs), targets).backward()
                if index == 1:
                    optimizer.step()
        summary = monitor.summary()
        self.assertEqual((summary["optimizer_steps"], summary["training_forwards"]), (1, 2))
        self.assertEqual(summary["head"]["min_tensors_with_gradient"], 2)
        self.assertGreater(summary["backbone"]["mean_parameter_update_l2_norm"], 0)

    def test_no_steps_and_skipped_steps_fail_and_remove_hooks(self):
        for skip_training in (True, False):
            with self.subTest(skip_training=skip_training):
                model, optimizer = self.make_pair()
                monitor = ParameterGroupMonitor(model, optimizer)
                with self.assertRaisesRegex(
                    RuntimeError, "zero optimizer steps|completed optimizer step"
                ):
                    with monitor:
                        if not skip_training:
                            self.train(model, optimizer, optimizer_step_enabled=False)
                self.assert_cleaned_up(monitor, model, optimizer)
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    monitor.summary()

    def test_forward_without_step_and_step_without_backward_are_rejected(self):
        for attempt_step in (False, True):
            with self.subTest(attempt_step=attempt_step):
                model, optimizer = self.make_pair()
                monitor = ParameterGroupMonitor(model, optimizer)
                with self.assertRaisesRegex(RuntimeError, "incomplete|no model gradients"):
                    with monitor:
                        model(self.inputs)
                        if attempt_step:
                            optimizer.step()
                self.assert_cleaned_up(monitor, model, optimizer)

    def test_step_without_forward_and_duplicate_step_are_rejected(self):
        for train_first in (False, True):
            with self.subTest(train_first=train_first):
                model, optimizer = self.make_pair()
                monitor = ParameterGroupMonitor(model, optimizer)
                with self.assertRaisesRegex(RuntimeError, "no preceding training forward"):
                    with monitor:
                        if train_first:
                            self.train(model, optimizer)
                        optimizer.step()
                self.assert_cleaned_up(monitor, model, optimizer)

    def test_forward_error_preserves_original_exception_and_removes_hooks(self):
        model, optimizer = self.make_pair()
        monitor = ParameterGroupMonitor(model, optimizer)
        with patch.object(model, "forward", side_effect=LookupError("forward failed")):
            with self.assertRaisesRegex(LookupError, "forward failed"):
                with monitor:
                    self.train(model, optimizer)
        self.assert_cleaned_up(monitor, model, optimizer)

    def test_optimizer_error_clears_captured_snapshots_and_preserves_exception(self):
        model, optimizer = self.make_pair()
        monitor = ParameterGroupMonitor(model, optimizer)

        def fail_after_capture(*args, **kwargs):
            self.assertIsNotNone(monitor._pending_step)
            raise LookupError("step failed")

        with self.assertRaisesRegex(LookupError, "step failed"):
            with monitor:
                failure_hook = optimizer.register_step_pre_hook(fail_after_capture)
                try:
                    self.train(model, optimizer)
                finally:
                    failure_hook.remove()
        self.assert_cleaned_up(monitor, model, optimizer)

    def test_backward_error_preserves_original_exception_and_removes_hooks(self):
        model, optimizer = self.make_pair()
        monitor = ParameterGroupMonitor(model, optimizer)
        with patch.object(torch.Tensor, "backward", side_effect=LookupError("backward failed")):
            with self.assertRaisesRegex(LookupError, "backward failed"):
                with monitor:
                    self.train(model, optimizer)
        self.assert_cleaned_up(monitor, model, optimizer)

    def test_nonfinite_gradient_or_update_is_rejected(self):
        for inject_update in (False, True):
            with self.subTest(inject_update=inject_update):
                model, optimizer = self.make_pair()
                monitor = ParameterGroupMonitor(model, optimizer)
                extra_hook = None
                if inject_update:

                    def corrupt_update(*args, model=model, **kwargs):
                        with torch.no_grad():
                            model.fc.weight[0, 0] = float("nan")

                    extra_hook = optimizer.register_step_post_hook(corrupt_update)
                try:
                    with self.assertRaisesRegex(ValueError, "non-finite"):
                        with monitor:
                            nn.functional.cross_entropy(model(self.inputs), self.targets).backward()
                            if not inject_update:
                                model.fc.weight.grad[0, 0] = float("inf")
                            optimizer.step()
                finally:
                    if extra_hook is not None:
                        extra_hook.remove()
                self.assert_cleaned_up(monitor, model, optimizer)

    def test_partial_registration_failure_removes_already_registered_hooks(self):
        model, optimizer = self.make_pair()
        monitor = ParameterGroupMonitor(model, optimizer)
        with patch.object(
            optimizer, "register_step_post_hook", side_effect=LookupError("register failed")
        ):
            with self.assertRaisesRegex(LookupError, "register failed"):
                with monitor:
                    self.fail("Context registration should fail")
        self.assert_cleaned_up(monitor, model, optimizer)

    def test_model_parameter_identity_change_during_step_is_rejected(self):
        model, optimizer = self.make_pair()
        monitor = ParameterGroupMonitor(model, optimizer)

        def replace_head(*args, **kwargs):
            model.fc = deepcopy(model.fc)

        with self.assertRaisesRegex(RuntimeError, "changed identity"):
            with monitor:
                replacement_hook = optimizer.register_step_pre_hook(replace_head)
                try:
                    self.train(model, optimizer)
                finally:
                    replacement_hook.remove()
        self.assert_cleaned_up(monitor, model, optimizer)

    def test_monitor_is_single_use_and_missing_parameter_groups_are_rejected(self):
        model, optimizer = self.make_pair()
        monitor = ParameterGroupMonitor(model, optimizer)
        with self.assertRaisesRegex(RuntimeError, "not entered"):
            monitor.summary()
        with monitor:
            self.train(model, optimizer)
        with self.assertRaisesRegex(RuntimeError, "new ParameterGroupMonitor"):
            with monitor:
                self.fail("A monitor must not span multiple epochs")
        no_head = nn.Linear(3, 2)
        with self.assertRaisesRegex(ValueError, "model.fc and backbone"):
            ParameterGroupMonitor(no_head, torch.optim.AdamW(no_head.parameters()))


if __name__ == "__main__":
    unittest.main()
