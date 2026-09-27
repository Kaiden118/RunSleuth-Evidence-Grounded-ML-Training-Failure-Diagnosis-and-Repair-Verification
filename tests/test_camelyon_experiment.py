"""Download-free model and experiment integration tests on tiny CPU tensors."""

import hashlib
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from runsleuth.camelyon_config import CamelyonConfig
from runsleuth.camelyon_data import CamelyonData

TORCH_AVAILABLE = bool(
    importlib.util.find_spec("torch") and importlib.util.find_spec("torchvision")
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@unittest.skipUnless(TORCH_AVAILABLE, "Install torch and torchvision for CPU integration tests")
class CamelyonExperimentTests(unittest.TestCase):
    def setUp(self):
        import torch

        from runsleuth import camelyon

        self.torch, self.camelyon = torch, camelyon
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config = CamelyonConfig(
            run_name="synthetic",
            output_dir=self.directory.name,
            device="cpu",
            epochs=2,
            batch_size=4,
            learning_rate=0.05,
            weight_decay=0,
        )
        targets = torch.tensor([0, 1] * 2)
        inputs = targets.float().view(4, 1, 1, 1).expand(4, 3, 96, 96).clone()

        def loader(labels):
            return torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(inputs, labels),
                batch_size=4,
            )

        self.data = CamelyonData(
            train_loader=loader(targets),
            id_val_loader=loader(targets),
            ood_val_loader=loader(1 - targets),
            manifest={
                "dataset": "synthetic-camelyon17",
                "splits": {name: {"selected_samples": 4} for name in ("train", "id_val", "val")},
                "majority_baseline": {
                    "label_from_train": 0,
                    "accuracy_by_split": {name: 0.5 for name in ("train", "id_val", "val")},
                },
            },
        )
        torch.manual_seed(7)
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 2, kernel_size=1),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
        )

    def run_synthetic(self, **options):
        with (
            patch.object(self.camelyon, "build_camelyon_data", return_value=self.data),
            patch.object(self.camelyon, "build_camelyon_model", return_value=self.model),
        ):
            return self.camelyon.run_camelyon_experiment(self.config, **options)

    def assert_common_artifacts(self, run_directory):
        for name in ("config.json", "data_manifest.json", "environment.json", "run_report.json"):
            self.assertTrue((run_directory / name).is_file(), name)
        self.assertEqual(read_json(run_directory / "data_manifest.json"), self.data.manifest)
        self.assertEqual(CamelyonConfig.load(run_directory / "config.json"), self.config)
        self.assertIsInstance(read_json(run_directory / "environment.json"), dict)

    def test_resnet18_has_two_outputs_and_every_parameter_is_trainable(self):
        with patch(
            "torch.hub.download_url_to_file", side_effect=AssertionError("Unexpected download")
        ):
            model = self.camelyon.build_camelyon_model(pretrained=False).eval()
        self.assertEqual([len(getattr(model, f"layer{i}")) for i in range(1, 5)], [2] * 4)
        self.assertEqual(model.fc.out_features, 2)
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        with self.torch.inference_mode():
            logits = model(self.torch.zeros(1, 3, 96, 96))
        self.assertEqual(tuple(logits.shape), (1, 2))
        self.assertTrue(self.torch.isfinite(logits).all().item())

    def test_prepare_only_records_data_without_building_or_training_model(self):
        with (
            patch.object(
                self.camelyon, "build_camelyon_data", return_value=self.data
            ) as build_data,
            patch.object(self.camelyon, "build_camelyon_model") as build_model,
            patch.object(self.camelyon, "train_one_epoch") as train,
            patch.object(self.camelyon, "evaluate") as evaluate,
        ):
            first = self.camelyon.run_camelyon_experiment(self.config, prepare_only=True)
            (first / "sentinel.txt").write_text("keep", encoding="utf-8")
            second = self.camelyon.run_camelyon_experiment(self.config, prepare_only=True)
        self.assertNotEqual(first, second)
        self.assertEqual((first / "sentinel.txt").read_text(encoding="utf-8"), "keep")
        self.assertEqual(build_data.call_count, 2)
        self.assertFalse(build_data.call_args.kwargs["download"])
        build_model.assert_not_called()
        train.assert_not_called()
        evaluate.assert_not_called()
        for directory in (first, second):
            self.assert_common_artifacts(directory)
            self.assertEqual(read_json(directory / "run_report.json")["status"], "prepared")
            self.assertFalse((directory / "model_state_dict.pt").exists())

    def test_cpu_training_records_real_updates_both_domains_and_final_checkpoint(self):
        run_directory = self.run_synthetic()
        self.assert_common_artifacts(run_directory)
        metrics = read_jsonl(run_directory / "metrics.jsonl")
        domains = read_jsonl(run_directory / "domain_metrics.jsonl")
        self.assertEqual([record["epoch"] for record in metrics], [1, 2])
        self.assertEqual([record["epoch"] for record in domains], [1, 2])
        for metric, domain in zip(metrics, domains, strict=True):
            self.assertGreater(metric["mean_gradient_norm"], 0)
            self.assertGreater(metric["mean_parameter_update_norm"], 0)
            self.assertTrue(all(math.isfinite(value) for value in metric.values()))
            self.assertEqual(metric["validation_loss"], domain["id_validation_loss"])
            self.assertEqual(metric["validation_accuracy"], domain["id_validation_accuracy"])
            self.assertTrue(math.isfinite(domain["epoch_seconds"]))
            self.assertGreaterEqual(domain["epoch_seconds"], 0)
            self.assertIsNone(domain["peak_cuda_memory_mb"])

        final_path = run_directory / "model_state_dict.pt"
        initial = self.torch.load(
            run_directory / "initial_state_dict.pt", map_location="cpu", weights_only=True
        )
        final = self.torch.load(final_path, map_location="cpu", weights_only=True)
        self.assertEqual(set(initial), set(final))
        self.assertTrue(any(not self.torch.equal(initial[name], final[name]) for name in initial))
        for name, parameter in self.model.state_dict().items():
            self.assertTrue(self.torch.equal(final[name], parameter.cpu()))

        baseline = read_json(run_directory / "baseline_metrics.json")
        self.assertEqual(baseline["epoch"], 0)
        for state, recorded in ((initial, baseline), (final, domains[-1])):
            self.model.load_state_dict(state)
            for name, loader in (
                ("id", self.data.id_val_loader),
                ("ood", self.data.ood_val_loader),
            ):
                actual = self.camelyon.evaluate(
                    self.model,
                    loader,
                    self.torch.nn.CrossEntropyLoss(),
                    self.torch.device("cpu"),
                )
                self.assertAlmostEqual(recorded[f"{name}_validation_loss"], actual.loss)
                self.assertAlmostEqual(recorded[f"{name}_validation_accuracy"], actual.accuracy)

        report = read_json(run_directory / "run_report.json")
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["completed_epochs"], self.config.epochs)
        self.assertEqual(Path(report["model_checkpoint"]).name, final_path.name)
        self.assertEqual(
            report["checkpoint_sha256"], hashlib.sha256(final_path.read_bytes()).hexdigest()
        )

    def test_training_failure_is_reported_and_reraised_with_completed_progress(self):
        original_train = self.camelyon.train_one_epoch
        calls = 0

        def fail_on_second_epoch(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("synthetic failure")
            return original_train(*args, **kwargs)

        with patch.object(self.camelyon, "train_one_epoch", side_effect=fail_on_second_epoch):
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                self.run_synthetic()
        directories = list(Path(self.directory.name).iterdir())
        self.assertEqual(len(directories), 1)
        self.assert_common_artifacts(directories[0])
        report = read_json(directories[0] / "run_report.json")
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["completed_epochs"], 1)
        self.assertEqual(report["error"], {"type": "RuntimeError", "message": "synthetic failure"})
        self.assertEqual(len(read_jsonl(directories[0] / "metrics.jsonl")), 1)


if __name__ == "__main__":
    unittest.main()
