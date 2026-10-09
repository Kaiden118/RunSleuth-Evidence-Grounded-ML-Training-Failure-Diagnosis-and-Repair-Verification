"""Optimizer membership checks using torch-free, equality-hostile fakes."""

import json
import unittest
from dataclasses import FrozenInstanceError

from runsleuth.parameter_group_monitor import OptimizerAudit, audit_optimizer_parameters


class FakeParameter:
    def __init__(self, *, requires_grad: bool = True) -> None:
        self.requires_grad = requires_grad
        self.shape = (2, 2)
        self.value = 1.0

    def __eq__(self, other: object) -> bool:
        raise AssertionError("Parameters must never be compared by equality")

    __hash__ = None


class FakeModel:
    def __init__(self, parameters: list[tuple[str, FakeParameter]]) -> None:
        self.parameters = parameters

    def named_parameters(self):
        return iter(self.parameters)


class FakeOptimizer:
    def __init__(self, *groups: list[FakeParameter]) -> None:
        self.param_groups = [{"params": group} for group in groups]


class OptimizerAuditTests(unittest.TestCase):
    def test_complete_membership_is_frozen_and_json_serializable(self) -> None:
        weight, bias = FakeParameter(), FakeParameter()
        audit = audit_optimizer_parameters(
            FakeModel([("weight", weight), ("bias", bias)]),
            FakeOptimizer([weight, bias]),
        )

        self.assertEqual(audit, OptimizerAudit(2, 2, 2, (), 0, 0))
        self.assertEqual(
            json.loads(json.dumps(audit.to_dict())),
            {
                "model_parameter_tensors": 2,
                "trainable_parameter_tensors": 2,
                "optimizer_unique_parameter_tensors": 2,
                "missing_trainable_names": [],
                "foreign_parameter_tensors": 0,
                "duplicate_parameter_occurrences": 0,
            },
        )
        with self.assertRaises(FrozenInstanceError):
            audit.model_parameter_tensors = 9

    def test_missing_trainable_names_are_sorted(self) -> None:
        kept, first, last = FakeParameter(), FakeParameter(), FakeParameter()
        audit = audit_optimizer_parameters(
            FakeModel([("z_head", last), ("body", kept), ("a_head", first)]),
            FakeOptimizer([kept]),
        )

        self.assertEqual(audit, OptimizerAudit(3, 3, 1, ("a_head", "z_head"), 0, 0))

    def test_replaced_head_with_identical_shape_and_value_is_foreign(self) -> None:
        body, old_head, new_head = FakeParameter(), FakeParameter(), FakeParameter()
        self.assertEqual(old_head.shape, new_head.shape)
        self.assertEqual(old_head.value, new_head.value)
        audit = audit_optimizer_parameters(
            FakeModel([("body", body), ("head", new_head)]),
            FakeOptimizer([body, old_head]),
        )

        self.assertEqual(audit, OptimizerAudit(2, 2, 2, ("head",), 1, 0))

    def test_frozen_parameters_are_excluded_from_missing(self) -> None:
        frozen, trainable = FakeParameter(requires_grad=False), FakeParameter()
        audit = audit_optimizer_parameters(
            FakeModel([("frozen", frozen), ("trainable", trainable)]),
            FakeOptimizer([trainable]),
        )

        self.assertEqual(audit, OptimizerAudit(2, 1, 1, (), 0, 0))
        included = audit_optimizer_parameters(
            FakeModel([("frozen", frozen)]), FakeOptimizer([frozen])
        )
        self.assertEqual(included, OptimizerAudit(1, 0, 1, (), 0, 0))

    def test_duplicates_and_foreign_tensors_across_multiple_groups(self) -> None:
        weight, bias, foreign = FakeParameter(), FakeParameter(), FakeParameter()
        audit = audit_optimizer_parameters(
            FakeModel([("weight", weight), ("bias", bias)]),
            FakeOptimizer([weight, weight], [], [bias, weight, foreign, foreign]),
        )

        self.assertEqual(audit, OptimizerAudit(2, 2, 3, (), 1, 3))

    def test_aliases_count_as_one_parameter_tensor(self) -> None:
        shared = FakeParameter()
        audit = audit_optimizer_parameters(
            FakeModel([("left", shared), ("right", shared)]), FakeOptimizer([shared])
        )

        self.assertEqual(audit, OptimizerAudit(1, 1, 1, (), 0, 0))

    def test_empty_model_and_optimizer(self) -> None:
        for optimizer in (FakeOptimizer(), FakeOptimizer([])):
            with self.subTest(groups=optimizer.param_groups):
                audit = audit_optimizer_parameters(FakeModel([]), optimizer)
                self.assertEqual(audit, OptimizerAudit(0, 0, 0, (), 0, 0))

    def test_empty_optimizer_reports_only_trainable_parameters(self) -> None:
        audit = audit_optimizer_parameters(
            FakeModel([("frozen", FakeParameter(requires_grad=False)), ("head", FakeParameter())]),
            FakeOptimizer(),
        )

        self.assertEqual(audit, OptimizerAudit(2, 1, 0, ("head",), 0, 0))

    def test_empty_model_reports_foreign_optimizer_parameters(self) -> None:
        foreign = FakeParameter()
        audit = audit_optimizer_parameters(FakeModel([]), FakeOptimizer([foreign]))

        self.assertEqual(audit, OptimizerAudit(0, 0, 1, (), 1, 0))


if __name__ == "__main__":
    unittest.main()
