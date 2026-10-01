"""Offline checks for the audit baseline and independent performance finding."""

import json
import unittest
from copy import deepcopy

from test_optimizer_agent_diagnosis import make_run

from runsleuth.optimizer_baseline import RULE_VERSION, diagnose_optimizer_rules


class OptimizerBaselineTests(unittest.TestCase):
    def setUp(self):
        self.reference = make_run("artifacts/reference", mismatch=False)
        self.candidate = make_run("artifacts/candidate", mismatch=False)

    def test_identical_metrics_and_healthy_binding_require_no_change(self):
        result = diagnose_optimizer_rules(self.reference, self.candidate)
        self.assertEqual(result["rule_version"], RULE_VERSION)
        self.assertEqual(result["implementation_status"], "no_optimizer_binding_defect")
        self.assertIsNone(result["root_cause"])
        self.assertEqual(result["performance_status"], "no_observed_degradation")
        self.assertEqual(result["recommended_action"], "no_change")
        self.assertEqual(
            [item["signed_improvement"] for item in result["evidence"]["final_metric_comparisons"]],
            [0.0] * 4,
        )
        json.dumps(result, allow_nan=False)

    def test_silent_binding_defect_is_detected_even_when_all_metrics_improve(self):
        result = diagnose_optimizer_rules(self.reference, make_run(mismatch=True))
        self.assertEqual(result["implementation_status"], "optimizer_binding_defect")
        self.assertEqual(result["root_cause"], "optimizer_parameter_mismatch")
        self.assertEqual(result["performance_status"], "no_observed_degradation")
        self.assertEqual(result["recommended_action"], "review_optimizer_binding")
        self.assertGreater(result["evidence"]["candidate_head_gradient_l2_norm"], 0)
        self.assertEqual(result["evidence"]["candidate_head_parameter_update_l2_norm"], 0.0)

    def test_each_audit_mismatch_is_detected_without_requiring_zero_head_updates(self):
        for field in (
            "missing_trainable_parameter_tensors",
            "foreign_parameter_tensors",
            "duplicate_parameter_occurrences",
        ):
            with self.subTest(field=field):
                candidate = deepcopy(self.candidate)
                candidate["optimizer_audit"][field] = 1
                if field == "missing_trainable_parameter_tensors":
                    candidate["optimizer_audit"]["missing_trainable_names"] = ["backbone.weight"]
                result = diagnose_optimizer_rules(self.reference, candidate)
                self.assertEqual(result["implementation_status"], "optimizer_binding_defect")
                self.assertGreater(result["evidence"]["candidate_head_parameter_update_l2_norm"], 0)

    def test_zero_head_updates_alone_do_not_establish_binding_mismatch(self):
        self.candidate["epochs"][-1]["head"]["mean_parameter_update_l2_norm"] = 0.0
        result = diagnose_optimizer_rules(self.reference, self.candidate)
        self.assertEqual(result["implementation_status"], "no_optimizer_binding_defect")
        self.assertIsNone(result["root_cause"])

    def test_loss_and_accuracy_use_opposite_improvement_directions(self):
        self.candidate["epochs"][-1].update(
            id_validation_accuracy=0.99,
            id_validation_loss=0.02,
            ood_validation_accuracy=0.95,
            ood_validation_loss=0.1,
        )
        result = diagnose_optimizer_rules(self.reference, self.candidate)
        self.assertEqual(result["performance_status"], "no_observed_degradation")
        for item in result["evidence"]["final_metric_comparisons"]:
            self.assertGreater(item["signed_improvement"], 0)
            self.assertEqual(
                item["better_when"], "higher" if item["metric"].endswith("accuracy") else "lower"
            )

    def test_mixed_and_degraded_performance_require_more_evidence_without_binding_defect(self):
        for status in ("mixed", "degraded"):
            with self.subTest(status=status):
                candidate = deepcopy(self.candidate)
                candidate["epochs"][-1].update(
                    id_validation_accuracy=0.9,
                    id_validation_loss=0.2,
                    ood_validation_accuracy=0.8,
                    ood_validation_loss=0.1 if status == "mixed" else 0.5,
                )
                result = diagnose_optimizer_rules(self.reference, candidate)
                self.assertEqual(result["performance_status"], status)
                self.assertEqual(result["recommended_action"], "collect_more_evidence")

    def test_incomparable_seed_initial_state_or_data_precludes_performance_claim(self):
        for changed in ("seed", "initial_state_sha256", "data_identity"):
            with self.subTest(changed=changed):
                candidate = deepcopy(self.candidate)
                if changed == "seed":
                    candidate["config"]["seed"] = 7
                elif changed == "data_identity":
                    candidate["data_identity"]["metadata_sha256"] = "0" * 64
                else:
                    candidate[changed] = "0" * 64
                result = diagnose_optimizer_rules(self.reference, candidate)
                self.assertEqual(result["performance_status"], "inconclusive")
                self.assertEqual(result["recommended_action"], "collect_more_evidence")
                self.assertFalse(result["evidence"]["comparable"])
                self.assertEqual(result["evidence"]["final_metric_comparisons"], [])

    def test_paths_variant_and_expected_labels_do_not_change_results_or_mutate_inputs(self):
        candidate = make_run(mismatch=True)
        original = deepcopy(candidate)
        expected = diagnose_optimizer_rules(self.reference, candidate)
        self.assertEqual(candidate, original)
        candidate["run_directory"] = "artifacts/healthy-perfect-control"
        candidate["variant"] = "clean"
        candidate["expected"] = {"implementation_status": "no_optimizer_binding_defect"}
        candidate["config"]["run_name"] = "clean"
        self.reference["run_directory"] = "artifacts/stale_head"
        self.assertEqual(diagnose_optimizer_rules(self.reference, candidate), expected)

    def test_boolean_audit_counts_are_invalid_in_either_input(self):
        for side in ("reference", "candidate"):
            with self.subTest(side=side):
                reference, candidate = deepcopy(self.reference), deepcopy(self.candidate)
                run = reference if side == "reference" else candidate
                run["optimizer_audit"]["foreign_parameter_tensors"] = False
                with self.assertRaisesRegex(ValueError, "Invalid audit count"):
                    diagnose_optimizer_rules(reference, candidate)

    def test_nonfinite_metric_or_parameter_telemetry_is_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            for field in ("id_validation_loss", "mean_gradient_l2_norm"):
                with self.subTest(value=value, field=field):
                    candidate = deepcopy(self.candidate)
                    row = candidate["epochs"][-1]
                    if field == "mean_gradient_l2_norm":
                        row["head"][field] = value
                    else:
                        row[field] = value
                    with self.assertRaisesRegex(ValueError, "Invalid nonnegative metric"):
                        diagnose_optimizer_rules(self.reference, candidate)

    def test_non_object_or_inconsistent_audit_evidence_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "JSON object"):
            diagnose_optimizer_rules(self.reference, [])
        self.candidate["optimizer_audit"]["missing_trainable_parameter_tensors"] = 1
        with self.assertRaisesRegex(ValueError, "names and count"):
            diagnose_optimizer_rules(self.reference, self.candidate)


if __name__ == "__main__":
    unittest.main()
