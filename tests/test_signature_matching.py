"""Tests for the declarative failure signature library and its matcher (no torch needed)."""

import contextlib
import copy
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from runsleuth import signature_matching as matching

HEAD_ELEMENTS, BACKBONE_ELEMENTS = 100, 10_000


def group(*, trainable=2, gradient=2, update=0.01, first=None, elements=100):
    stats = {
        "parameter_tensors": 2,
        "min_trainable_tensors": trainable,
        "max_trainable_tensors": trainable,
        "min_tensors_with_gradient": gradient,
        "max_tensors_with_gradient": gradient,
        "mean_gradient_l2_norm": 1.0 if gradient else 0.0,
        "mean_parameter_update_l2_norm": update,
    }
    if first is not None:
        stats.update(first_step_parameter_update_l2_norm=first, parameter_elements=elements)
    return stats


def run(
    *,
    lr=1e-4,
    head=None,
    missing=(),
    steps=10,
    forwards=10,
    eval_mode=0,
    lr_changes=0,
    lr_rebounds=0,
):
    """Two epochs of rows; the first step of AdamW moves each element by about lr.

    eval_mode forwards per epoch run in eval mode; the learning rate changes
    lr_changes times within each epoch and rises again after falling lr_rebounds times.
    """
    head = head or group(first=lr * math.sqrt(HEAD_ELEMENTS), elements=HEAD_ELEMENTS)
    backbone = group(first=lr * math.sqrt(BACKBONE_ELEMENTS), elements=BACKBONE_ELEMENTS)
    stepless = steps == 0
    rows = [
        {
            "epoch": epoch,
            "optimizer_steps": steps,
            "training_forwards": forwards,
            "eval_mode_training_forwards": eval_mode,
            "learning_rate": None
            if stepless
            else {
                "first": lr,
                "last": lr,
                "min": lr,
                "max": lr,
                "changes": lr_changes,
                "rebounds": lr_rebounds,
            },
            "head": None if stepless else head,
            "backbone": None if stepless else backbone,
        }
        for epoch in (1, 2)
    ]
    return {
        "status": "completed",
        "parameter_group_epochs": rows,
        "optimizer_audit": {
            "missing_trainable_names": list(missing),
            "foreign_parameter_tensors": len(missing),
            "duplicate_parameter_occurrences": 0,
        },
        "final_metrics": {
            "mean_gradient_norm": 1.0,
            "mean_parameter_update_norm": 0.0 if stepless else 0.1,
        },
    }


SCENARIOS = {
    "clean": run(),
    "stale_head": run(head=group(update=0.0), missing=("fc.bias", "fc.weight")),
    "frozen_head": run(head=group(trainable=0, gradient=0, update=0.0)),
    "high_learning_rate": run(lr=1e-2),
    "missing_optimizer_step": run(steps=0),
    "train_in_eval_mode": run(eval_mode=10),
    "scheduler_stepped_per_batch": run(lr_changes=9, lr_rebounds=3),
}


def diagnose(variant_run, reference=None, optimizer="AdamW"):
    evidence = matching.extract_evidence(variant_run, reference, optimizer)
    return matching.diagnose(evidence, matching.load_library())


def verdicts(result):
    return {item["id"]: item["verdict"] for item in result["verdicts"]}


class SignatureMatchingTests(unittest.TestCase):
    def test_library_loads_and_rejects_malformed_signatures(self):
        library = matching.load_library()
        self.assertEqual(
            [signature["id"] for signature in library["signatures"]],
            [
                "stale_optimizer_binding",
                "frozen_head",
                "high_learning_rate",
                "missing_optimizer_step",
                "train_eval_normalization_mismatch",
                "classifier_size_mismatch",
                "frozen_backbone_module",
                "train_in_eval_mode",
                "scheduler_stepped_per_batch",
                "train_evaluation_group_overlap",
                "zero_learning_rate_epoch",
            ],
        )
        broken = (
            ("op", "~="),
            ("value", "$unknown_threshold"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "library.json"
            for field, value in broken:
                with self.subTest(field=field):
                    changed = copy.deepcopy(library)
                    changed["signatures"][0]["requires"][0][field] = value
                    path.write_text(json.dumps(changed), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        matching.load_library(path)
            duplicated = copy.deepcopy(library)
            duplicated["signatures"].append(copy.deepcopy(library["signatures"][0]))
            path.write_text(json.dumps(duplicated), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unique"):
                matching.load_library(path)

    def test_each_fault_matches_only_its_signature(self):
        expected = {
            "clean": ("no_known_fault", "no_known_fault"),
            "stale_head": ("stale_optimizer_binding", "stale_optimizer_binding"),
            "frozen_head": ("pending_reference", "frozen_head"),
            "high_learning_rate": ("high_learning_rate", "high_learning_rate"),
            "missing_optimizer_step": ("missing_optimizer_step", "missing_optimizer_step"),
            "train_in_eval_mode": ("pending_reference", "train_in_eval_mode"),
            "scheduler_stepped_per_batch": ("pending_reference", "scheduler_stepped_per_batch"),
        }
        for name, (without, with_reference) in expected.items():
            with self.subTest(variant=name):
                self.assertEqual(diagnose(SCENARIOS[name])["diagnosis"], without)
                result = diagnose(SCENARIOS[name], SCENARIOS["clean"])
                self.assertEqual(result["diagnosis"], with_reference)
        self.assertEqual(diagnose(SCENARIOS["frozen_head"])["pending_reference"], ["frozen_head"])
        self.assertEqual(
            verdicts(diagnose(SCENARIOS["frozen_head"]))["stale_optimizer_binding"],
            "contradicted",
        )
        self.assertEqual(verdicts(diagnose(SCENARIOS["stale_head"]))["frozen_head"], "contradicted")
        self.assertEqual(
            verdicts(diagnose(SCENARIOS["missing_optimizer_step"]))["high_learning_rate"],
            "contradicted",
        )

    def test_every_condition_reports_its_observed_value(self):
        result = diagnose(SCENARIOS["high_learning_rate"], SCENARIOS["clean"])
        high = next(item for item in result["verdicts"] if item["id"] == "high_learning_rate")
        observed = {item["evidence"]: item["observed"] for item in high["requires"]}
        self.assertEqual(observed["optimizer_family"], "adam")
        self.assertAlmostEqual(observed["backbone_inferred_first_step_lr"], 1e-2)
        self.assertAlmostEqual(observed["backbone_inferred_lr_ratio_to_reference"], 100)
        self.assertEqual(high["requires"][1]["expected"], 0.001)
        json.dumps(result, allow_nan=False)

    def test_intended_eval_mode_or_per_step_schedule_in_the_reference_is_no_fault(self):
        # A deliberate BatchNorm freeze or a per-step schedule shows in the reference too.
        for name in ("train_in_eval_mode", "scheduler_stepped_per_batch"):
            with self.subTest(fault=name):
                result = diagnose(SCENARIOS[name], SCENARIOS[name])
                self.assertEqual(verdicts(result)[name], "not_supported")
                self.assertEqual(result["diagnosis"], "no_known_fault")

    def test_a_monotone_per_step_schedule_is_no_fault_with_or_without_a_reference(self):
        # Linear decay or warmup stepped per batch, as the Hugging Face Trainer does.
        per_step = run(lr_changes=9)
        for reference in (None, SCENARIOS["clean"]):
            with self.subTest(reference=reference is not None):
                result = diagnose(per_step, reference)
                self.assertEqual(verdicts(result)["scheduler_stepped_per_batch"], "not_supported")
                self.assertEqual(result["diagnosis"], "no_known_fault")

    def test_runs_recorded_before_the_loop_evidence_leave_it_unavailable(self):
        legacy = copy.deepcopy(SCENARIOS["clean"])
        for row in legacy["parameter_group_epochs"]:
            del row["eval_mode_training_forwards"], row["learning_rate"]
        evidence = matching.extract_evidence(legacy, None, "AdamW")
        self.assertIsNone(evidence["max_eval_mode_training_fraction"])
        self.assertIsNone(evidence["max_learning_rate_changes_per_epoch"])
        result = matching.diagnose(evidence, matching.load_library())
        self.assertEqual(verdicts(result)["train_in_eval_mode"], "insufficient_evidence")
        self.assertEqual(result["diagnosis"], "no_known_fault")

    def test_an_epoch_at_zero_learning_rate_is_a_fault_even_after_a_correct_first_epoch(self):
        wasted = copy.deepcopy(SCENARIOS["clean"])
        zero = {"first": 0.0, "last": 0.0, "min": 0.0, "max": 0.0, "changes": 0, "rebounds": 0}
        wasted["parameter_group_epochs"][1]["learning_rate"] = zero
        for reference in (None, SCENARIOS["clean"]):
            result = diagnose(wasted, reference)
            self.assertEqual(result["diagnosis"], "zero_learning_rate_epoch")
            self.assertEqual(result["evidence"]["min_epoch_peak_learning_rate"], 0)
        # A rate that touches zero within an epoch, as a cycling schedule does, is not this.
        cycling = copy.deepcopy(SCENARIOS["clean"])
        cycling["parameter_group_epochs"][1]["learning_rate"]["min"] = 0.0
        self.assertEqual(diagnose(cycling)["diagnosis"], "no_known_fault")
        # A loop without optimizer steps records no rate and stays its own fault.
        stepless = diagnose(SCENARIOS["missing_optimizer_step"])
        self.assertEqual(stepless["diagnosis"], "missing_optimizer_step")
        self.assertEqual(verdicts(stepless)["zero_learning_rate_epoch"], "not_supported")

    def test_training_on_a_held_out_group_is_a_fault_without_any_reference(self):
        def with_holdouts(shared, held_out_by="hospital"):
            leaky = copy.deepcopy(SCENARIOS["clean"])
            leaky["evaluation_holdouts"] = {
                # In-distribution validation shares its hospitals with training by design.
                "id": {"held_out_by": None, "training_samples_in_evaluation_groups": 900},
                "ood": {
                    "held_out_by": held_out_by,
                    "training_samples_in_evaluation_groups": shared,
                },
            }
            return leaky

        result = diagnose(with_holdouts(120))
        self.assertEqual(result["diagnosis"], "train_evaluation_group_overlap")
        self.assertEqual(result["evidence"]["max_training_samples_in_heldout_groups"], 120)
        self.assertEqual(diagnose(with_holdouts(0))["diagnosis"], "no_known_fault")
        undeclared = diagnose(with_holdouts(120, held_out_by=None))
        self.assertIsNone(undeclared["evidence"]["max_training_samples_in_heldout_groups"])
        self.assertEqual(
            verdicts(diagnose(SCENARIOS["clean"]))["train_evaluation_group_overlap"],
            "insufficient_evidence",
        )

    def test_intentionally_frozen_reference_does_not_support_frozen_head(self):
        result = diagnose(SCENARIOS["frozen_head"], SCENARIOS["frozen_head"])
        self.assertEqual(verdicts(result)["frozen_head"], "not_supported")
        self.assertEqual(result["diagnosis"], "no_known_fault")

    def test_reference_with_the_same_learning_rate_does_not_support_high_learning_rate(self):
        same = diagnose(SCENARIOS["high_learning_rate"], SCENARIOS["high_learning_rate"])
        self.assertEqual(verdicts(same)["high_learning_rate"], "not_supported")
        alone = diagnose(SCENARIOS["high_learning_rate"])
        ratio = next(
            item
            for item in next(
                verdict for verdict in alone["verdicts"] if verdict["id"] == "high_learning_rate"
            )["requires"]
            if item["evidence"] == "backbone_inferred_lr_ratio_to_reference"
        )
        self.assertEqual(ratio["status"], "skipped_without_reference")

    def test_learning_rate_is_inferred_only_for_adam_family_optimizers(self):
        for optimizer, verdict in (("SGD", "not_supported"), (None, "insufficient_evidence")):
            with self.subTest(optimizer=optimizer):
                result = diagnose(SCENARIOS["high_learning_rate"], optimizer=optimizer)
                self.assertEqual(verdicts(result)["high_learning_rate"], verdict)
                self.assertEqual(result["diagnosis"], "no_known_fault")

    def test_strict_rows_without_forward_counts_imply_one_step_per_forward(self):
        legacy = copy.deepcopy(SCENARIOS["clean"])
        for row in legacy["parameter_group_epochs"]:
            del row["training_forwards"]
        evidence = matching.extract_evidence(legacy, None, "AdamW")
        self.assertEqual(evidence["max_optimizer_steps_per_forward"], 1)

    def test_evaluate_reports_scores_labeled_variants(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for seed in (7, 2026):
                folder = Path(directory) / f"seed-{seed}"
                folder.mkdir()
                (folder / "environment.json").write_text(
                    json.dumps({"optimizer": "AdamW"}), encoding="utf-8"
                )
                report = {
                    "status": "completed",
                    "seed": seed,
                    "variants": {**SCENARIOS, "missing_optimizer_step_repaired": run()},
                }
                path = folder / "report.json"
                path.write_text(json.dumps(report), encoding="utf-8")
                paths.append(path)
            result = matching.evaluate_reports(paths, matching.load_library())
            reference_free = result["summary"]["reference_free"]
            with_reference = result["summary"]["with_reference"]
            self.assertEqual((with_reference["exact_matches"], with_reference["cases"]), (16, 16))
            self.assertEqual(reference_free["exact_matches"], 10)
            # Frozen head, eval mode and the scheduler wait for a reference on both seeds.
            self.assertEqual(reference_free["pending_reference_with_true_fault"], 6)
            self.assertEqual(
                (with_reference["healthy_false_positives"], with_reference["healthy_cases"]),
                (0, 4),
            )
            report = json.loads(paths[0].read_text(encoding="utf-8"))
            report["status"] = "failed"
            paths[0].write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not completed"):
                matching.evaluate_reports(paths, matching.load_library())

    def test_hugging_face_faults_match_only_their_signatures(self):
        def statistics(std):
            return {"channel_mean": [0.0] * 3, "channel_std": [std] * 3, "values_per_channel": 10}

        def vit_run(**changes):
            report = copy.deepcopy(SCENARIOS["clean"])
            report.update(
                head_prefix="classifier.",
                head_output_units=2,
                label_classes={"train": [0, 1], "id_eval": [0, 1]},
                input_statistics={"train": statistics(0.8), "id_eval": statistics(0.8)},
            )
            report.update(changes)
            return report

        clean = vit_run()
        frozen_backbone = vit_run()
        for row in frozen_backbone["parameter_group_epochs"]:
            row["backbone"] = group(trainable=1, gradient=1, first=1e-4 * 100, elements=10_000)
        cases = {
            "clean": (clean, "no_known_fault", "no_known_fault"),
            "normalization": (
                vit_run(input_statistics={"train": statistics(0.35), "id_eval": statistics(0.8)}),
                "train_eval_normalization_mismatch",
                "train_eval_normalization_mismatch",
            ),
            "head": (
                vit_run(head_output_units=1000),
                "classifier_size_mismatch",
                "classifier_size_mismatch",
            ),
            "frozen_backbone": (frozen_backbone, "pending_reference", "frozen_backbone_module"),
        }
        for name, (report, alone, with_reference) in cases.items():
            with self.subTest(case=name):
                self.assertEqual(diagnose(report)["diagnosis"], alone)
                self.assertEqual(diagnose(report, clean)["diagnosis"], with_reference)
        ratio = matching.extract_evidence(cases["normalization"][0])[
            "input_std_ratio_train_vs_eval"
        ]
        self.assertAlmostEqual(ratio, 0.8 / 0.35)
        for report in SCENARIOS.values():
            evidence = matching.extract_evidence(report, None, "AdamW")
            self.assertIsNone(evidence["input_std_ratio_train_vs_eval"])
            self.assertIsNone(evidence["head_outputs_beyond_label_classes"])

    def test_cli_diagnoses_one_run(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_report.json"
            path.write_text(json.dumps(SCENARIOS["missing_optimizer_step"]), encoding="utf-8")
            output = io.StringIO()
            arguments = ["matching", "diagnose", "--run", str(path), "--optimizer", "AdamW"]
            with patch.object(sys, "argv", arguments), contextlib.redirect_stdout(output):
                matching.main()
        self.assertEqual(json.loads(output.getvalue())["diagnosis"], "missing_optimizer_step")


if __name__ == "__main__":
    unittest.main()
