"""Training acceptance must distinguish structural repair from performance recovery."""

import json
import unittest
from copy import deepcopy

from runsleuth.optimizer_source_training_policy import assess_source_training

INITIAL_HASH = "a" * 64


def training_reports():
    audit = {
        "model_parameter_tensors": 4,
        "trainable_parameter_tensors": 4,
        "optimizer_unique_parameter_tensors": 4,
        "missing_trainable_names": [],
        "foreign_parameter_tensors": 0,
        "duplicate_parameter_occurrences": 0,
    }
    metrics = {
        "id_validation_accuracy": 0.98,
        "id_validation_loss": 0.07,
        "ood_validation_accuracy": 0.90,
        "ood_validation_loss": 0.35,
    }
    clean = {
        "status": "completed",
        "error": None,
        "completed_epochs": 3,
        "initial_state_sha256": INITIAL_HASH,
        "initialization_metrics": {key: 0.5 for key in metrics},
        "final_metrics": {"epoch": 3, **metrics},
        "optimizer_audit": audit,
        "final_optimizer_audit": deepcopy(audit),
        "parameter_group_epochs": [
            {
                "epoch": epoch,
                "optimizer_steps": 2,
                "head": {"mean_gradient_l2_norm": 0.2, "mean_parameter_update_l2_norm": 0.003},
                "backbone": {"mean_gradient_l2_norm": 1.0, "mean_parameter_update_l2_norm": 0.04},
            }
            for epoch in (1, 2, 3)
        ],
    }
    stale = deepcopy(clean)
    for key in ("optimizer_audit", "final_optimizer_audit"):
        stale[key].update(
            missing_trainable_names=["fc.bias", "fc.weight"], foreign_parameter_tensors=2
        )
    for row in stale["parameter_group_epochs"]:
        row["head"]["mean_parameter_update_l2_norm"] = 0.0
    patched = deepcopy(clean)
    patched.update(
        source_patch_executed=True,
        optimizer_state_entries_before=0,
        model_state_sha256_before_constructor=INITIAL_HASH,
        model_state_sha256_after_constructor=INITIAL_HASH,
    )
    return clean, stale, patched


class SourceTrainingPolicyTests(unittest.TestCase):
    def setUp(self):
        self.clean, self.stale, self.patched = training_reports()

    def assess(self):
        return assess_source_training(
            self.clean, self.stale, self.patched, expected_hash=INITIAL_HASH
        )

    def test_structural_repair_and_both_domain_comparators_are_required(self):
        result = self.assess()
        self.assertEqual(result["decision"], "accepted")
        self.assertTrue(result["structure_verified"])
        self.assertTrue(result["performance_nonregression"])
        self.assertEqual(len(result["performance_checks"]), 8)
        self.assertEqual(result["policy"], {"max_accuracy_drop": 0.01, "max_loss_ratio": 1.1})

    def test_original_seed42_regression_is_rejected_despite_structural_success(self):
        for report in (self.clean, self.patched):
            report["final_metrics"].update(
                ood_validation_accuracy=0.9016, ood_validation_loss=0.37869434755183756
            )
        self.stale["final_metrics"].update(
            ood_validation_accuracy=0.9156, ood_validation_loss=0.3200979508943856
        )
        result = self.assess()
        self.assertEqual(result["decision"], "rejected")
        self.assertTrue(result["structure_verified"])
        failed = [check for check in result["performance_checks"] if not check["passed"]]
        self.assertEqual(
            [(check["comparator"], check["domain"], check["metric"]) for check in failed],
            [("stale_head", "ood", "accuracy_drop"), ("stale_head", "ood", "loss_ratio")],
        )

    def test_every_accuracy_and_loss_comparator_can_reject(self):
        for comparator in ("clean", "stale_head"):
            for domain in ("id", "ood"):
                for metric, value in (("accuracy", 1.0), ("loss", 0.001)):
                    with self.subTest(comparator=comparator, domain=domain, metric=metric):
                        self.clean, self.stale, self.patched = training_reports()
                        report = self.clean if comparator == "clean" else self.stale
                        report["final_metrics"][f"{domain}_validation_{metric}"] = value
                        result = self.assess()
                        self.assertTrue(result["structure_verified"])
                        self.assertEqual(result["decision"], "rejected")

    def test_final_epoch_only_no_best_ood_epoch_selection(self):
        self.patched["final_metrics"]["ood_validation_accuracy"] = 0.50
        self.patched["best_ood_validation_accuracy"] = 0.99
        self.assertEqual(self.assess()["decision"], "rejected")

    def test_audits_gradients_updates_state_and_budget_all_gate_acceptance(self):
        edits = [
            ("source_patch_executed", False),
            ("optimizer_state_entries_before", 2),
            ("model_state_sha256_after_constructor", "b" * 64),
            ("completed_epochs", 2),
            ("status", "failed"),
            ("error", {"message": "failed"}),
        ]
        for field, value in edits:
            with self.subTest(field=field):
                self.clean, self.stale, self.patched = training_reports()
                self.patched[field] = value
                result = self.assess()
                self.assertFalse(result["structure_verified"])
                self.assertEqual(result["decision"], "rejected")
        for field in ("mean_gradient_l2_norm", "mean_parameter_update_l2_norm"):
            self.clean, self.stale, self.patched = training_reports()
            self.patched["parameter_group_epochs"][0]["head"][field] = 0.0
            self.assertEqual(self.assess()["decision"], "rejected")
        self.clean, self.stale, self.patched = training_reports()
        self.patched["final_optimizer_audit"]["foreign_parameter_tensors"] = 2
        self.assertEqual(self.assess()["decision"], "rejected")

    def test_steps_initialization_and_control_evidence_must_match(self):
        self.patched["parameter_group_epochs"][1]["optimizer_steps"] = 1
        self.assertEqual(self.assess()["decision"], "rejected")
        self.clean, self.stale, self.patched = training_reports()
        self.patched["initialization_metrics"]["id_validation_accuracy"] = 0.6
        self.assertEqual(self.assess()["decision"], "rejected")
        self.clean, self.stale, self.patched = training_reports()
        self.stale["parameter_group_epochs"][0]["head"]["mean_parameter_update_l2_norm"] = 0.001
        self.assertEqual(self.assess()["decision"], "rejected")

    def test_invalid_numbers_fail_closed_and_serialize_as_strict_json(self):
        for value in (True, -1, "0.9", float("nan"), float("inf"), 10**500, None):
            with self.subTest(value=repr(value)[:20]):
                self.patched["final_metrics"]["ood_validation_accuracy"] = value
                result = self.assess()
                self.assertEqual(result["decision"], "rejected")
                json.dumps(result, allow_nan=False)

    def test_zero_reference_loss_only_allows_zero_repaired_loss(self):
        for report in (self.clean, self.stale, self.patched):
            report["final_metrics"]["ood_validation_loss"] = 0.0
        result = self.assess()
        self.assertEqual(result["decision"], "accepted")
        loss_checks = [
            check
            for check in result["performance_checks"]
            if check["domain"] == "ood" and check["metric"] == "loss_ratio"
        ]
        self.assertTrue(all(check["observed_value"] is None for check in loss_checks))
        self.patched["final_metrics"]["ood_validation_loss"] = 0.00001
        self.assertEqual(self.assess()["decision"], "rejected")

    def test_incomplete_receipt_and_boolean_epoch_never_pass(self):
        self.assertEqual(
            assess_source_training({}, {}, {}, expected_hash=INITIAL_HASH)["decision"], "rejected"
        )
        self.assertEqual(
            assess_source_training(
                self.clean, self.stale, self.patched, expected_hash=INITIAL_HASH, epochs=True
            )["decision"],
            "rejected",
        )


if __name__ == "__main__":
    unittest.main()
