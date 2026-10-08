"""RunSleuthCallback with a real Hugging Face Trainer and a tiny randomly initialized ViT (CPU)."""

import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from transformers import Trainer, TrainingArguments, ViTConfig, ViTForImageClassification
except ModuleNotFoundError as error:
    if error.name not in ("torch", "transformers"):
        raise
    torch = None

if torch is not None:
    from runsleuth import signature_matching as matching
    from runsleuth.hf_callback import RunSleuthCallback

    class PatchDataset(torch.utils.data.Dataset):
        def __init__(self, count, seed):
            generator = torch.Generator().manual_seed(seed)
            self.images = torch.randn(count, 3, 32, 32, generator=generator)
            self.labels = (self.images[:, 0].mean(dim=(1, 2)) > 0).long()

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            return {"pixel_values": self.images[index], "labels": self.labels[index]}


@unittest.skipIf(torch is None, "Install torch and transformers for Trainer tests")
class RunSleuthCallbackTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def train(self, name, *, freeze_head=False, accumulation=1):
        torch.manual_seed(0)
        config = ViTConfig(
            image_size=32,
            patch_size=8,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            num_labels=2,
        )
        model = ViTForImageClassification(config)
        if freeze_head:
            model.classifier.requires_grad_(False)
        arguments = TrainingArguments(
            output_dir=str(self.root / f"{name}-trainer"),
            num_train_epochs=2,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=accumulation,
            learning_rate=1e-4,
            eval_strategy="epoch",
            save_strategy="no",
            logging_strategy="no",
            report_to=[],
            use_cpu=True,
            seed=0,
            disable_tqdm=True,
        )
        callback = RunSleuthCallback(
            self.root / name,
            head="classifier",
            rename={"eval_id_loss": "id_validation_loss", "eval_ood_loss": "ood_validation_loss"},
        )
        trainer = Trainer(
            model=model,
            args=arguments,
            train_dataset=PatchDataset(32, seed=1),
            eval_dataset={"id": PatchDataset(16, seed=2), "ood": PatchDataset(16, seed=3)},
            callbacks=[callback],
        )
        trainer.train()
        return json.loads((self.root / name / "run_report.json").read_text(encoding="utf-8"))

    def diagnose(self, report, reference=None):
        evidence = matching.extract_evidence(report, reference)
        return matching.diagnose(evidence, matching.load_library())

    def test_trainer_run_produces_a_diagnosable_report(self):
        report = self.train("healthy")
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["optimizer"], "AdamW")
        self.assertEqual(report["head_prefix"], "classifier.")
        self.assertEqual(report["hf_trainer"]["learning_rate"], 1e-4)
        rows = report["parameter_group_epochs"]
        self.assertEqual(
            [(row["optimizer_steps"], row["training_forwards"]) for row in rows], [(4, 4), (4, 4)]
        )
        for row in rows:
            self.assertGreater(row["head"]["mean_parameter_update_l2_norm"], 0)
            self.assertGreater(row["backbone"]["mean_parameter_update_l2_norm"], 0)
            self.assertEqual(
                {"id_validation_loss", "ood_validation_loss"} - row["metrics"].keys(), set()
            )
            self.assertNotIn("epoch", row["metrics"])
        self.assertEqual(report["final_metrics"]["epoch"], 2)
        self.assertEqual(self.diagnose(report)["diagnosis"], "no_known_fault")

    def test_frozen_head_under_the_trainer_is_detected(self):
        healthy = self.train("healthy")
        frozen = self.train("frozen", freeze_head=True)
        self.assertEqual(
            frozen["trainability"]["head"]["frozen_names"], ["classifier.bias", "classifier.weight"]
        )
        self.assertEqual(self.diagnose(frozen)["diagnosis"], "pending_reference")
        self.assertEqual(self.diagnose(frozen, healthy)["diagnosis"], "frozen_head")

    def test_gradient_accumulation_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "gradient accumulation"):
            self.train("accumulating", accumulation=2)


if __name__ == "__main__":
    unittest.main()
