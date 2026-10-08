"""CPU tests for the DeiT fault runner with a tiny randomly initialized ViT and fake data."""

import copy
import unittest
from unittest.mock import patch

import runsleuth.camelyon_vit as runner
from runsleuth.camelyon_optimizer_probe import read_json

try:
    import torch
    import transformers  # noqa: F401
except ModuleNotFoundError as error:
    if error.name not in ("torch", "transformers"):
        raise
    torch = None

if torch is not None:
    from camelyon_harness import CamelyonHarness
else:
    CamelyonHarness = unittest.TestCase


def tiny_vit(plan):
    from transformers import ViTConfig, ViTForImageClassification

    config = ViTConfig(
        image_size=32,
        patch_size=8,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        num_labels=1000 if plan["num_labels"] is None else plan["num_labels"],
    )
    return ViTForImageClassification(config)


class RecordingCallback:
    def __init__(self):
        self.batches = []

    def observe_batch(self, split, inputs, labels=None):
        self.batches.append((split, inputs, labels))


@unittest.skipIf(torch is None, "Install torch and transformers for ViT runner tests")
class ViTFaultTests(CamelyonHarness):
    learning_rate = 1e-4

    def run_faults(self):
        with self.patched_builders():
            path = runner.run_vit_faults(
                self.reference,
                self.output,
                epochs=2,
                requested_device="cpu",
                model_factory=tiny_vit,
                image_size=32,
            )
        return read_json(path)

    def test_seven_variants_reproduce_faults_and_verify_repairs(self):
        report = self.run_faults()
        self.assertEqual(report["status"], "completed")
        self.assertTrue(report["mechanism_reproduced"], report["checks"])
        self.assertEqual(list(report["variants"]), list(runner.VARIANTS))
        variants = report["variants"]
        self.assertEqual(variants["imagenet_head_kept"]["head_output_units"], 1000)
        self.assertEqual(variants["clean"]["head_output_units"], 2)
        self.assertEqual(
            variants["frozen_patch_embedding"]["trainability"]["backbone"]["frozen_names"],
            [
                "vit.embeddings.patch_embeddings.projection.bias",
                "vit.embeddings.patch_embeddings.projection.weight",
            ],
        )
        shifts = report["observations"]["input_shift_train_vs_id_eval"]
        self.assertGreaterEqual(shifts["train_eval_normalization_mismatch"]["std_ratio"], 1.5)
        self.assertLess(shifts["clean"]["std_ratio"], 1.25)
        for fault, verification in report["repair_verification"].items():
            with self.subTest(fault=fault):
                self.assertTrue(verification["structure_verified"], verification)
                self.assertEqual(
                    verification["decision"],
                    "accepted" if verification["performance_nonregression"] else "rejected",
                )
        for row in variants["clean"]["parameter_group_epochs"]:
            self.assertEqual(row["metrics"].keys() >= set(runner.METRIC_NAMES.values()), True)
        self.assertTrue(report["determinism"]["deterministic_algorithms"])
        self.assertFalse(torch.are_deterministic_algorithms_enabled())
        self.assertEqual(
            report["observations"]["clean_configured_repairs_bitwise_equal_to_clean"],
            {"imagenet_head_kept_repaired": True, "frozen_patch_embedding_repaired": True},
        )

    def test_collate_then_preprocess_normalizes_per_split_resizes_and_records(self):
        plan = runner.variant_plan()["train_eval_normalization_mismatch"]
        callback = RecordingCallback()
        preprocess = runner.SplitPreprocessor(plan, callback, image_size=32)
        image = torch.zeros(3, 96, 96)  # ImageNet-normalized zeros are the ImageNet mean
        batch = runner.split_collate([{"image": image, "label": 1, "split": "train"}])
        self.assertEqual(
            (batch["split"], tuple(batch["pixel_values"].shape)), ("train", (1, 3, 96, 96))
        )
        train = preprocess(batch)
        evaluation = preprocess(
            runner.split_collate([{"image": image, "label": 0, "split": "id_eval"}])
        )
        self.assertNotIn("split", train)
        self.assertEqual(tuple(train["pixel_values"].shape), (1, 3, 32, 32))
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.assertTrue(
            torch.allclose(
                train["pixel_values"][0], ((mean - 0.5) / 0.5).expand(3, 32, 32), atol=1e-6
            )
        )
        self.assertTrue(
            torch.allclose(evaluation["pixel_values"], torch.zeros(1, 3, 32, 32), atol=1e-6)
        )
        self.assertEqual([split for split, _, _ in callback.batches], ["train", "id_eval"])
        with self.assertRaisesRegex(ValueError, "mixes splits"):
            runner.split_collate(
                [
                    {"image": image, "label": 0, "split": "train"},
                    {"image": image, "label": 0, "split": "id_eval"},
                ]
            )

    def test_assessment_rejects_a_mismatch_that_left_no_trace_in_inputs(self):
        report = self.run_faults()
        variants = copy.deepcopy(report["variants"])
        statistics = variants["train_eval_normalization_mismatch"]["input_statistics"]
        statistics["train"] = copy.deepcopy(statistics["id_eval"])
        result = runner.assess_vit_faults(variants, report["variant_plan"], 2)
        self.assertFalse(result["mechanism_reproduced"])
        verification = result["repair_verification"]["train_eval_normalization_mismatch"]
        self.assertEqual(verification["decision"], "rejected")

    def test_invalid_epoch_budget_does_no_work(self):
        for epochs in (0, 4, True):
            with self.subTest(epochs=epochs), patch.object(runner, "load_reference") as load:
                with self.assertRaises(ValueError):
                    runner.run_vit_faults(self.reference, self.output, epochs=epochs)
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
