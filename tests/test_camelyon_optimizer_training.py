"""Offline paired-training tests using real CPU optimization and tiny models."""

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from runsleuth import camelyon_optimizer_training as experiment
from runsleuth.camelyon_config import CamelyonConfig

TORCH_AVAILABLE = bool(
    importlib.util.find_spec("torch") and importlib.util.find_spec("torchvision")
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ArgumentTests(unittest.TestCase):
    def test_invalid_epoch_budget_does_no_work(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "uncreated"
            with patch.object(experiment, "load_reference") as load_reference:
                for epochs in (0, -1, 4, True, 1.5):
                    with self.subTest(epochs=epochs), self.assertRaises(ValueError):
                        experiment.run_optimizer_training(
                            Path(directory) / "missing-reference",
                            output,
                            epochs=epochs,
                        )
                load_reference.assert_not_called()
            self.assertFalse(output.exists())


@unittest.skipUnless(TORCH_AVAILABLE, "Install torch and torchvision for CPU integration tests")
class OptimizerTrainingTests(unittest.TestCase):
    def setUp(self):
        import torch

        from runsleuth.camelyon_data import CamelyonData

        self.torch, self.data_type = torch, CamelyonData
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.reference = self.directory / "reference"
        self.reference.mkdir()
        self.output = self.directory / "paired"
        self.config = CamelyonConfig(
            device="cpu",
            epochs=3,
            batch_size=2,
            learning_rate=0.03,
            weight_decay=0,
            output_dir=str(self.output),
            seed=17,
        )
        self.config.save(self.reference / "config.json")
        self.manifest = {
            "dataset": "camelyon17",
            "dataset_version": "1.0",
            "subset_seed": 2026,
            "source_provenance": {
                "repo": "synthetic",
                "revision": self.config.hf_revision,
                "metadata_sha256": "0" * 64,
            },
            "splits": {
                name: {"selected_samples": 6, "selected_sample_ids": list(range(start, start + 6))}
                for name, start in (("train", 0), ("id_val", 6), ("val", 12))
            },
        }
        experiment.write_json(self.reference / "data_manifest.json", self.manifest)
        self.models = []

        class TinyClassifier(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Linear(3, 4)
                self.fc = torch.nn.Linear(4, 2)
                self.training_order = []

            def forward(self, inputs):
                if self.training:
                    self.training_order.append(inputs[:, 0, 0, 0].tolist())
                features = torch.tanh(self.backbone(inputs.mean(dim=(2, 3))))
                return self.fc(features)

        self.model_type = TinyClassifier
        torch.manual_seed(29)
        self.initial = {
            name: value.clone() for name, value in TinyClassifier().state_dict().items()
        }
        checkpoint = self.reference / "initial_state_dict.pt"
        torch.save(self.initial, checkpoint)
        experiment.write_json(
            self.reference / "run_report.json",
            {
                "task": "camelyon17_resnet18_development_baseline",
                "status": "completed",
                "completed_epochs": self.config.epochs,
                "initial_checkpoint_sha256": file_hash(checkpoint),
                "data_manifest_sha256": file_hash(self.reference / "data_manifest.json"),
            },
        )

    def build_model(self, *, pretrained):
        self.assertFalse(pretrained, "The saved reference initialization needs no download")
        model = self.model_type()
        self.models.append(model)
        return model

    def build_data(self, config, *, device, download):
        self.assertFalse(download, "The paired experiment must use cached data")
        self.assertEqual(device.type, "cpu")
        values = self.torch.arange(6).float() / 5
        images = values.reshape(6, 1, 1, 1).expand(6, 3, 96, 96).clone()
        labels = self.torch.tensor([0, 0, 0, 1, 1, 1])

        def loader(*, train=False, reverse=False):
            return self.torch.utils.data.DataLoader(
                self.torch.utils.data.TensorDataset(images, 1 - labels if reverse else labels),
                batch_size=config.batch_size,
                shuffle=train,
                generator=self.torch.Generator().manual_seed(config.seed),
            )

        return self.data_type(
            train_loader=loader(train=True),
            id_val_loader=loader(),
            ood_val_loader=loader(reverse=True),
            manifest=copy.deepcopy(self.manifest),
        )

    def run_pair(self):
        with (
            patch("runsleuth.camelyon.build_camelyon_model", side_effect=self.build_model) as model,
            patch(
                "runsleuth.camelyon_data.build_camelyon_data", side_effect=self.build_data
            ) as data,
            patch.object(experiment, "snapshot_sources", return_value={}),
        ):
            report_path = experiment.run_optimizer_training(
                self.reference,
                self.output,
                epochs=2,
                requested_device="cpu",
            )
        self.assertEqual(model.call_count, 2)
        self.assertEqual(data.call_count, 2)
        return report_path

    def test_pair_starts_identically_and_isolates_head_updates_through_both_epochs(self):
        from runsleuth.optimizer_probe import state_dict_sha256

        report_path = self.run_pair()
        report = read_json(report_path)
        self.assertEqual(report["status"], "completed")
        self.assertTrue(report["mechanism_reproduced"])
        self.assertTrue(all(report["checks"].values()))
        self.assertIn("performance_comparison", report)
        self.assertEqual(set(report["variants"]), {"clean", "stale_head"})
        for key, filename in (
            ("reference_config_sha256", "config.json"),
            ("reference_manifest_sha256", "data_manifest.json"),
            ("reference_initial_checkpoint_sha256", "initial_state_dict.pt"),
        ):
            self.assertEqual(report[key], file_hash(self.reference / filename))
        self.assertEqual(self.models[0].training_order, self.models[1].training_order)
        self.assertEqual(len(self.models[0].training_order), 6)
        initial_hash = state_dict_sha256(self.initial)
        baselines = []
        for name in ("clean", "stale_head"):
            with self.subTest(variant=name):
                directory = report_path.parent / name
                for filename in (
                    "config.json",
                    "data_manifest.json",
                    "baseline_metrics.json",
                    "metrics.jsonl",
                    "domain_metrics.jsonl",
                    "parameter_group_metrics.jsonl",
                    "model_state_dict.pt",
                    "run_report.json",
                ):
                    self.assertTrue((directory / filename).is_file(), filename)
                variant = report["variants"][name]
                self.assertEqual(variant["initial_state_sha256"], initial_hash)
                self.assertEqual(variant["completed_epochs"], 2)
                self.assertEqual(read_json(directory / "data_manifest.json"), self.manifest)
                baselines.append(read_json(directory / "baseline_metrics.json"))
                metrics = read_jsonl(directory / "metrics.jsonl")
                domains = read_jsonl(directory / "domain_metrics.jsonl")
                groups = read_jsonl(directory / "parameter_group_metrics.jsonl")
                self.assertEqual([row["epoch"] for row in metrics], [1, 2])
                self.assertEqual([row["epoch"] for row in domains], [1, 2])
                self.assertEqual(groups, variant["parameter_group_epochs"])
                self.assertEqual(len(groups), 2)
                for row in groups:
                    self.assertEqual(row["optimizer_steps"], 3)
                    self.assertGreater(row["head"]["mean_gradient_l2_norm"], 0)
                    self.assertGreater(row["backbone"]["mean_parameter_update_l2_norm"], 0)
                    if name == "clean":
                        self.assertGreater(row["head"]["mean_parameter_update_l2_norm"], 0)
                    else:
                        self.assertEqual(row["head"]["mean_parameter_update_l2_norm"], 0)
                state = self.torch.load(
                    directory / "model_state_dict.pt", map_location="cpu", weights_only=True
                )
                head_changed = any(
                    not self.torch.equal(state[key], self.initial[key])
                    for key in self.initial
                    if key.startswith("fc.")
                )
                backbone_changed = any(
                    not self.torch.equal(state[key], self.initial[key])
                    for key in self.initial
                    if key.startswith("backbone.")
                )
                self.assertEqual(head_changed, name == "clean")
                self.assertTrue(backbone_changed)
                saved_report = read_json(directory / "run_report.json")
                self.assertEqual(
                    saved_report["checkpoint_sha256"], file_hash(directory / "model_state_dict.pt")
                )
                self.assertEqual(saved_report["status"], "completed")
        self.assertEqual(baselines[0], baselines[1])
        clean = report["variants"]["clean"]["optimizer_audit"]
        stale = report["variants"]["stale_head"]["optimizer_audit"]
        self.assertEqual(clean["missing_trainable_names"], [])
        self.assertEqual(clean["foreign_parameter_tensors"], 0)
        self.assertEqual(set(stale["missing_trainable_names"]), {"fc.weight", "fc.bias"})
        self.assertEqual(stale["foreign_parameter_tensors"], 2)

    def test_changed_sample_identity_records_failure_before_any_training(self):
        self.manifest["splits"]["train"]["selected_sample_ids"][0] = 999
        with (
            patch("runsleuth.camelyon_data.build_camelyon_data", side_effect=self.build_data),
            patch("runsleuth.camelyon.build_camelyon_model", side_effect=self.build_model),
            patch("runsleuth.train.train_one_epoch") as train,
            patch.object(experiment, "snapshot_sources", return_value={}),
        ):
            with self.assertRaisesRegex(ValueError, "sample IDs differ"):
                experiment.run_optimizer_training(
                    self.reference,
                    self.output,
                    epochs=2,
                    requested_device="cpu",
                )
        train.assert_not_called()
        paths = list(self.output.glob("*/optimizer_training_report.json"))
        self.assertEqual(len(paths), 1)
        report = read_json(paths[0])
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error"]["type"], "ValueError")
        self.assertEqual(report["variants"]["clean"]["status"], "failed")
        self.assertEqual(report["variants"]["clean"]["completed_epochs"], 0)
        self.assertFalse(list(self.output.rglob("metrics.jsonl")))


if __name__ == "__main__":
    unittest.main()
