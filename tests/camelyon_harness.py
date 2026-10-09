"""Shared CPU harness for Camelyon17 runner tests.

Provides a completed synthetic reference run, a tiny classifier and synthetic data,
so whole runners execute offline in seconds through the real training code.
"""

import copy
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torch

from runsleuth.camelyon_config import CamelyonConfig
from runsleuth.camelyon_data import CamelyonData
from runsleuth.camelyon_optimizer_probe import file_sha256
from runsleuth.camelyon_optimizer_training import write_json


class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(3, 4)
        self.fc = torch.nn.Linear(4, 2)
        self.training_order = []

    def forward(self, inputs):
        if self.training:
            self.training_order.append(inputs[:, 0, 0, 0].tolist())
        return self.fc(torch.tanh(self.backbone(inputs.mean(dim=(2, 3)))))


class TinyBatchNormClassifier(TinyClassifier):
    """TinyClassifier with BatchNorm, whose running statistics change only in train mode."""

    def __init__(self):
        super().__init__()
        self.norm = torch.nn.BatchNorm1d(4)

    def forward(self, inputs):
        if self.training:
            self.training_order.append(inputs[:, 0, 0, 0].tolist())
        return self.fc(torch.tanh(self.norm(self.backbone(inputs.mean(dim=(2, 3))))))


class CamelyonHarness(unittest.TestCase):
    """A reference run on disk plus patchable model and data builders."""

    # Nonzero weight decay: an untouched frozen head also shows AdamW skipped its decay.
    weight_decay = 0.01
    learning_rate = 0.03
    model_class = TinyClassifier
    training_samples = 6

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.reference = root / "reference"
        self.reference.mkdir()
        self.output = root / "output"
        self.config = CamelyonConfig(
            device="cpu",
            epochs=3,
            batch_size=2,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
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
        write_json(self.reference / "data_manifest.json", self.manifest)
        self.models = []
        self.nan_training_images = False
        torch.manual_seed(29)
        self.initial = {
            name: value.clone() for name, value in self.model_class().state_dict().items()
        }
        checkpoint = self.reference / "initial_state_dict.pt"
        torch.save(self.initial, checkpoint)
        write_json(
            self.reference / "run_report.json",
            {
                "task": "camelyon17_resnet18_development_baseline",
                "status": "completed",
                "completed_epochs": self.config.epochs,
                "initial_checkpoint_sha256": file_sha256(checkpoint),
                "data_manifest_sha256": file_sha256(self.reference / "data_manifest.json"),
            },
        )

    def build_model(self, *, pretrained):
        self.assertFalse(pretrained, "The saved reference initialization needs no download")
        model = self.model_class()
        self.models.append(model)
        return model

    def build_data(self, config, *, device, download):
        self.assertFalse(download, "Runners must use cached data")
        count = self.training_samples
        values = torch.arange(count).float() / (count - 1)
        images = values.reshape(count, 1, 1, 1).expand(count, 3, 96, 96).clone()
        labels = (torch.arange(count) >= count // 2).long()
        training_images = images.clone()
        if self.nan_training_images:
            training_images[-1] = float("nan")

        def loader(source, *, train=False, reverse=False):
            return torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(source, 1 - labels if reverse else labels),
                batch_size=config.batch_size,
                shuffle=train,
                generator=torch.Generator().manual_seed(config.seed),
            )

        return CamelyonData(
            train_loader=loader(training_images, train=True),
            id_val_loader=loader(images),
            ood_val_loader=loader(images, reverse=True),
            manifest=copy.deepcopy(self.manifest),
        )

    @contextmanager
    def patched_builders(self):
        with (
            patch("runsleuth.camelyon.build_camelyon_model", side_effect=self.build_model),
            patch("runsleuth.camelyon_data.build_camelyon_data", side_effect=self.build_data),
            patch("runsleuth.camelyon_optimizer_training.snapshot_sources", return_value={}),
            patch("runsleuth.camelyon_variant_runner.snapshot_sources", return_value={}),
            patch("builtins.print"),
        ):
            yield

    def reference_state(self):
        from runsleuth.optimizer_probe import state_dict_sha256

        config = CamelyonConfig.load(self.reference / "config.json")
        state = torch.load(
            self.reference / "initial_state_dict.pt", map_location="cpu", weights_only=True
        )
        return config, state, state_dict_sha256(state)
