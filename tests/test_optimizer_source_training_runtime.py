"""Source training guards and tiny real-CPU training; no API or downloaded data."""

import ast
import hashlib
import importlib.util
import json
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from runsleuth.camelyon_config import CamelyonConfig
from runsleuth.optimizer_source_patch import propose_optimizer_order_patch
from runsleuth.optimizer_source_training_runtime import train_source_patch

SOURCE_PATH = Path(__file__).resolve().parents[1] / "src/runsleuth/optimizer_probe.py"
TORCH_AVAILABLE = bool(
    importlib.util.find_spec("torch") and importlib.util.find_spec("torchvision")
)


def source_job(root):
    original = SOURCE_PATH.read_text(encoding="utf-8")
    checkpoint = root / "initial.pt"
    checkpoint.write_bytes(b"test-checkpoint")
    return {
        "id": "seed-7-stale_head",
        "seed": 7,
        "epochs": 3,
        "config": CamelyonConfig(seed=7, device="cpu", batch_size=2),
        "original_source": original,
        "proposed_source": propose_optimizer_order_patch(original)["proposed_source"],
        "checkpoint": checkpoint,
        "baseline": {
            "files_sha256": {
                "initial_state_dict.pt": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            }
        },
    }


def read_report(path):
    return json.loads((path / "run_report.json").read_text(encoding="utf-8"))


class SourceTrainingRuntimeGuardsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.job = source_job(self.root)
        self.output = self.root / "run"

    def test_altered_proposal_rejected_before_import_or_output(self):
        self.job["proposed_source"] += "\n"
        with self.assertRaisesRegex(ValueError, "exact validated statement reorder"):
            train_source_patch(self.job, self.output)
        self.assertFalse(self.output.exists())

    def test_fixed_epoch_budget_not_adjustable(self):
        for epochs in (0, 1, 2, 4, True, 3.0):
            with self.subTest(epochs=epochs):
                self.job["epochs"] = epochs
                with self.assertRaisesRegex(ValueError, "exactly three"):
                    train_source_patch(self.job, self.output)
                self.assertFalse(self.output.exists())
        self.job["epochs"] = 3
        self.job["config"] = replace(self.job["config"], epochs=2)
        with self.assertRaisesRegex(ValueError, "exactly three"):
            train_source_patch(self.job, self.output)

    def test_bad_device_or_seed_refused_before_output(self):
        with self.assertRaisesRegex(ValueError, "Device"):
            train_source_patch(self.job, self.output, "cuda:12")
        self.job["seed"] = 8
        with self.assertRaisesRegex(ValueError, "Training seed"):
            train_source_patch(self.job, self.output)
        self.assertFalse(self.output.exists())

    def test_checkpoint_bytes_rechecked_and_failed_report_retained(self):
        self.job["checkpoint"].write_bytes(b"tampered-checkpoint")
        with self.assertRaisesRegex(ValueError, "Initial checkpoint before training load"):
            train_source_patch(self.job, self.output)
        report = read_report(self.output)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["completed_epochs"], 0)
        self.assertEqual(report["optimizer_steps_recorded"], 0)
        self.assertTrue(report["step_accounting_complete"])
        self.assertFalse(report["source_patch_executed"])
        self.assertFalse(report["partial_epoch_possible"])

    def test_existing_output_never_overwritten(self):
        self.output.mkdir()
        marker = self.output / "original.txt"
        marker.write_text("preserve", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            train_source_patch(self.job, self.output)
        self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")

    def test_runtime_parses_under_python_311(self):
        source = SOURCE_PATH.with_name("optimizer_source_training_runtime.py")
        ast.parse(source.read_text(encoding="utf-8"), feature_version=(3, 11))


@unittest.skipUnless(TORCH_AVAILABLE, "Torch and torchvision required for real CPU optimization")
class SourceTrainingRuntimeCPUTests(unittest.TestCase):
    def setUp(self):
        import torch
        from test_optimizer_evidence import OptimizerEvidenceTests
        from torch.utils.data import DataLoader, TensorDataset

        from runsleuth.optimizer_probe import state_dict_sha256

        self.torch = torch
        self.digest = state_dict_sha256
        fixture = OptimizerEvidenceTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root = fixture.root
        self.job = source_job(self.root)
        self.job["config"] = replace(
            fixture.config, seed=7, epochs=3, device="cpu", learning_rate=0.01
        )
        self.job["manifest"] = deepcopy(fixture.manifest)

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Linear(2, 3)
                self.fc = torch.nn.Linear(3, 2)

            def forward(self, inputs):
                return self.fc(torch.tanh(self.backbone(inputs)))

        self.model_type = TinyModel
        torch.manual_seed(17)
        self.initial = deepcopy(TinyModel().state_dict())
        torch.save(self.initial, self.job["checkpoint"])
        self.job["initial_state_sha256"] = self.digest(self.initial)
        self.job["baseline"]["files_sha256"]["initial_state_dict.pt"] = hashlib.sha256(
            self.job["checkpoint"].read_bytes()
        ).hexdigest()
        self.output = self.root / "source_training"

        def data_factory(config, *, device, download):
            self.assertFalse(download)
            dataset = TensorDataset(
                torch.tensor([[1.0, 2.0], [-1.0, 1.0], [2.0, -1.0]]), torch.tensor([0, 1, 0])
            )
            validation = TensorDataset(torch.tensor([[0.5, -0.5]]), torch.tensor([1]))
            return SimpleNamespace(
                train_loader=DataLoader(
                    dataset,
                    batch_size=2,
                    shuffle=True,
                    generator=torch.Generator().manual_seed(config.seed),
                ),
                id_val_loader=DataLoader(validation, batch_size=2),
                ood_val_loader=DataLoader(validation, batch_size=2),
                manifest=deepcopy(self.job["manifest"]),
            )

        self.data_factory = data_factory
        model_patch = patch(
            "runsleuth.camelyon.build_camelyon_model", side_effect=lambda **kw: TinyModel()
        )
        data_patch = patch("runsleuth.camelyon_data.build_camelyon_data", side_effect=data_factory)
        self.build_model = model_patch.start()
        self.build_data = data_patch.start()
        self.addCleanup(model_patch.stop)
        self.addCleanup(data_patch.stop)
        print_patch = patch("builtins.print")
        print_patch.start()
        self.addCleanup(print_patch.stop)

    def test_actual_patched_constructor_matches_clean_training(self):
        from runsleuth.camelyon_optimizer_training import _run_variant

        result = train_source_patch(self.job, self.output, "cpu")
        clean = _run_variant(
            "clean",
            self.job["config"],
            self.job["manifest"],
            self.initial,
            self.job["initial_state_sha256"],
            self.root / "clean",
            self.torch.device("cpu"),
        )
        clean_state = self.torch.load(
            clean["model_checkpoint"], weights_only=True, map_location="cpu"
        )
        self.assertEqual(result["final_state_sha256"], self.digest(clean_state))
        self.assertEqual(result["initialization_metrics"], clean["initialization_metrics"])
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["source_patch_executed"])
        self.assertEqual(result["source_execution"]["constructor_variant"], "stale_head")
        self.assertEqual(result["completed_epochs"], 3)
        self.assertEqual(result["optimizer_steps_recorded"], 6)
        self.assertTrue(result["step_accounting_complete"])
        self.assertFalse(result["partial_epoch_possible"])
        self.assertFalse(result["test_evaluated"])
        self.assertEqual(result["optimizer_state_entries_before"], 0)
        self.assertEqual(
            result["model_state_sha256_before_constructor"],
            result["model_state_sha256_after_constructor"],
        )
        for row in result["parameter_group_epochs"]:
            self.assertEqual(row["optimizer_steps"], 2)
            for group in ("head", "backbone"):
                self.assertGreater(row[group]["mean_parameter_update_l2_norm"], 0)
        for name in (
            "config.json",
            "data_manifest.json",
            "baseline_metrics.json",
            "metrics.jsonl",
            "domain_metrics.jsonl",
            "parameter_group_metrics.jsonl",
            "model_state_dict.pt",
            "run_report.json",
        ):
            self.assertTrue((self.output / name).is_file(), name)

    def test_loaded_model_state_mismatch_stops_before_data(self):
        self.job["initial_state_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Loaded initial model state"):
            train_source_patch(self.job, self.output)
        self.build_data.assert_not_called()
        self.assertFalse(read_report(self.output)["training_started"])

    def test_changed_data_identity_stops_before_model(self):
        real_factory = self.data_factory

        def changed_factory(*args, **kwargs):
            data = real_factory(*args, **kwargs)
            data.manifest["source_provenance"]["metadata_sha256"] = "f" * 64
            return data

        self.build_data.side_effect = changed_factory
        with self.assertRaisesRegex(ValueError, "Training selected data"):
            train_source_patch(self.job, self.output)
        self.build_model.assert_not_called()
        self.assertFalse(read_report(self.output)["training_started"])

    def test_interruption_after_update_preserves_unknown_partial_epoch_accounting(self):
        from runsleuth.train import train_one_epoch

        def fail_after_training(*args, **kwargs):
            train_one_epoch(*args, **kwargs)
            raise KeyboardInterrupt("interrupted after updates")

        with patch("runsleuth.train.train_one_epoch", side_effect=fail_after_training):
            with self.assertRaises(KeyboardInterrupt):
                train_source_patch(self.job, self.output)
        report = read_report(self.output)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["completed_epochs"], 0)
        self.assertEqual(report["optimizer_steps_recorded"], 0)
        self.assertTrue(report["partial_epoch_possible"])
        self.assertFalse(report["step_accounting_complete"])
        self.assertFalse((self.output / "model_state_dict.pt").exists())

    def test_environment_mismatch_precedes_data_and_model_work(self):
        self.job["expected_environment"] = {"torch": "wrong-version", "device": "cpu"}
        with self.assertRaisesRegex(ValueError, "PyTorch training version"):
            train_source_patch(self.job, self.output)
        self.build_data.assert_not_called()
        self.build_model.assert_not_called()

    def test_failure_during_validation_counts_finished_optimizer_steps(self):
        from runsleuth.camelyon import _evaluate_domains

        evaluations = 0

        def fail_after_initialization(*args, **kwargs):
            nonlocal evaluations
            evaluations += 1
            if evaluations > 1:
                raise RuntimeError("validation failure")
            return _evaluate_domains(*args, **kwargs)

        with patch("runsleuth.camelyon._evaluate_domains", side_effect=fail_after_initialization):
            with self.assertRaisesRegex(RuntimeError, "validation failure"):
                train_source_patch(self.job, self.output)
        report = read_report(self.output)
        self.assertEqual(report["completed_epochs"], 0)
        self.assertEqual(report["optimizer_steps_recorded"], 2)
        self.assertTrue(report["step_accounting_complete"])
        self.assertFalse(report["partial_epoch_possible"])


if __name__ == "__main__":
    unittest.main()
