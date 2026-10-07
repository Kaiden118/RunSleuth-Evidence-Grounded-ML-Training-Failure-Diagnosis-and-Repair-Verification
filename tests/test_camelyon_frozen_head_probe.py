"""Offline end-to-end tests of the frozen-head probe runner with a tiny model and fake data."""

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
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

import runsleuth.camelyon_frozen_head_probe as runner
from runsleuth.camelyon_config import CamelyonConfig

if torch is not None:
    from runsleuth.frozen_head_probe import VARIANTS

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


def manifest_fixture():
    return {
        "dataset": "camelyon17",
        "dataset_version": "1.0",
        "subset_seed": 2026,
        "source_provenance": {"repo": "fixture", "revision": "pinned", "metadata_sha256": "abc"},
        "splits": {
            "train": {"selected_sample_ids": [11, 12, 13, 14]},
            "id_val": {"selected_sample_ids": [21, 22]},
            "val": {"selected_sample_ids": [31, 32]},
        },
    }


@unittest.skipIf(torch is None, "PyTorch is not installed; runner tests require torch")
class FrozenHeadProbeRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.reference = root / "reference"
        self.reference.mkdir()
        self.output = root / "probes"
        self.config = replace(CamelyonConfig(), device="cpu")
        self.config.save(self.reference / "config.json")
        (self.reference / "data_manifest.json").write_text(
            json.dumps(manifest_fixture()), encoding="utf-8"
        )
        self.checkpoint = self.reference / "initial_state_dict.pt"
        torch.save(TinyModel().state_dict(), self.checkpoint)
        inputs = torch.tensor(
            [[0.2, 0.7, -0.4], [-0.5, 0.1, 0.8], [1.0, -0.2, 0.3], [-0.4, -0.6, 0.2]]
        )
        loader = DataLoader(TensorDataset(inputs, torch.tensor([0, 1, 1, 0])), batch_size=2)
        self.data = SimpleNamespace(manifest=manifest_fixture(), train_loader=loader)

    def run_probe(self, data_error=None):
        reference = (self.config, manifest_fixture(), self.checkpoint)
        with (
            patch.object(runner, "load_reference", return_value=reference),
            patch(
                "runsleuth.camelyon_data.build_camelyon_data",
                return_value=self.data,
                side_effect=data_error,
            ),
            patch(
                "runsleuth.camelyon.build_camelyon_model",
                side_effect=lambda pretrained: TinyModel(),
            ),
            patch("builtins.print"),
        ):
            return runner.run_frozen_head_probe(self.reference, self.output)

    def test_report_records_checks_candidates_and_predictions(self):
        report = json.loads(self.run_probe().read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "completed")
        self.assertTrue(report["mechanism_reproduced"])
        self.assertEqual(report["device"], "cpu")
        self.assertEqual(
            report["budget"], {"optimizer_steps": 5, "gpu_training_epochs": 0, "model_calls": 0}
        )
        self.assertEqual(report["batch"]["sample_ids"], [11, 12])
        self.assertEqual(list(report["variants"]), list(VARIANTS))
        self.assertEqual(
            {name: item["decision"] for name, item in report["candidates"].items()},
            {"frozen_head_repaired": "accepted", "frozen_head_optimizer_rebuilt": "rejected"},
        )
        self.assertTrue(all(report["predictions"].values()), report["predictions"])
        self.assertEqual(set(report["source_sha256"]), set(runner.SOURCE_FILES))

    def test_failure_writes_failed_report_and_reraises(self):
        with self.assertRaisesRegex(LookupError, "no cached data"):
            self.run_probe(data_error=LookupError("no cached data"))
        (path,) = self.output.glob("probe-*/frozen_head_probe_report.json")
        report = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error"], {"type": "LookupError", "message": "no cached data"})

    def test_cli_exits_nonzero_unless_completed(self):
        for status, exits in (("completed", False), ("inconclusive", True), ("failed", True)):
            with self.subTest(status=status):
                with (
                    patch.object(sys, "argv", ["probe", "--reference-run", "reference"]),
                    patch.object(
                        runner, "run_frozen_head_probe", return_value=Path("report.json")
                    ) as run,
                    patch.object(runner, "read_json", return_value={"status": status}),
                ):
                    if exits:
                        with self.assertRaises(SystemExit):
                            runner.main()
                    else:
                        runner.main()
                run.assert_called_once_with(Path("reference"), Path("artifacts/frozen_head_probes"))


if __name__ == "__main__":
    unittest.main()
