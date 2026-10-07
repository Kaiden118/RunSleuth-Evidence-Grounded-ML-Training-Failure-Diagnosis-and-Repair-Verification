"""CPU integration tests for the stage-2 frozen-head training runner (tiny model, fake data)."""

import copy
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import runsleuth.camelyon_frozen_head_training as runner
import runsleuth.camelyon_optimizer_training as experiment
from runsleuth.camelyon_config import CamelyonConfig
from runsleuth.camelyon_optimizer_probe import file_sha256, read_json
from runsleuth.camelyon_optimizer_training import write_json

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None


@unittest.skipIf(torch is None, "Install torch and torchvision for CPU integration tests")
class FrozenHeadTrainingTests(unittest.TestCase):
    def setUp(self):
        from runsleuth.camelyon_data import CamelyonData

        self.data_type = CamelyonData
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.reference = root / "reference"
        self.reference.mkdir()
        self.output = root / "frozen"
        # Nonzero weight decay: an untouched frozen head also shows AdamW skipped its decay.
        self.config = CamelyonConfig(
            device="cpu",
            epochs=3,
            batch_size=2,
            learning_rate=0.03,
            weight_decay=0.01,
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

        self.model_type = TinyClassifier
        torch.manual_seed(29)
        initial = {name: value.clone() for name, value in TinyClassifier().state_dict().items()}
        checkpoint = self.reference / "initial_state_dict.pt"
        torch.save(initial, checkpoint)
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
        model = self.model_type()
        self.models.append(model)
        return model

    def build_data(self, config, *, device, download):
        self.assertFalse(download, "Stage 2 must use cached data")
        values = torch.arange(6).float() / 5
        images = values.reshape(6, 1, 1, 1).expand(6, 3, 96, 96).clone()
        labels = torch.tensor([0, 0, 0, 1, 1, 1])

        def loader(*, train=False, reverse=False):
            return torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(images, 1 - labels if reverse else labels),
                batch_size=config.batch_size,
                shuffle=train,
                generator=torch.Generator().manual_seed(config.seed),
            )

        return self.data_type(
            train_loader=loader(train=True),
            id_val_loader=loader(),
            ood_val_loader=loader(reverse=True),
            manifest=copy.deepcopy(self.manifest),
        )

    def run_training(self):
        with (
            patch("runsleuth.camelyon.build_camelyon_model", side_effect=self.build_model),
            patch("runsleuth.camelyon_data.build_camelyon_data", side_effect=self.build_data),
            patch.object(runner, "snapshot_sources", return_value={}),
            patch("builtins.print"),
        ):
            return read_json(
                runner.run_frozen_head_training(
                    self.reference, self.output, epochs=2, requested_device="cpu"
                )
            )

    def test_four_variants_reproduce_mechanisms_and_record_predictions(self):
        report = self.run_training()
        self.assertEqual(report["status"], "completed")
        self.assertTrue(report["mechanism_reproduced"])
        self.assertTrue(all(report["checks"].values()), report["checks"])
        self.assertEqual(list(report["variants"]), list(runner.VARIANTS))
        self.assertEqual(report["budget"], {"training_epoch_upper_bound": 8, "model_calls": 0})
        self.assertEqual(len({str(model.training_order) for model in self.models}), 1)
        verification = report["repair_verification"]
        self.assertTrue(verification["structure_verified"], verification["structural_checks"])
        self.assertEqual(verification["comparators"], ["clean", "frozen_head"])
        self.assertEqual(
            verification["decision"],
            "accepted" if verification["performance_nonregression"] else "rejected",
        )
        self.assertEqual(
            report["checkpoint_comparisons"]["head_unchanged_from_initial"],
            {
                "clean": False,
                "stale_head": True,
                "frozen_head": True,
                "frozen_head_repaired": False,
            },
        )
        predictions = report["predictions"]
        self.assertTrue(predictions["P1_frozen_head_bitwise_unchanged"])
        self.assertEqual(predictions["P2_frozen_vs_stale"]["outcome"], "bitwise_equal")
        self.assertEqual(predictions["P3_repaired_vs_clean"]["outcome"], "bitwise_equal")
        self.assertEqual(len(list(self.output.glob("run-*/repair_verification_policy.json"))), 1)
        snapshot = next(self.output.glob("run-*/source_snapshot"))
        self.assertEqual(
            sorted(path.name for path in snapshot.iterdir()), sorted(runner.EXTRA_SOURCES)
        )

    def test_assessment_rejects_broken_mechanisms_and_repairs(self):
        report = self.run_training()
        variants, comparisons = report["variants"], report["checkpoint_comparisons"]
        expected = report["reference_initial_state_sha256"]
        cases = (
            (
                ("frozen_head", "parameter_group_epochs", 0, "head", "max_tensors_with_gradient"),
                2,
                "frozen_head_has_no_gradients_or_updates_every_epoch",
            ),
            (
                ("frozen_head_repaired", "repair", "timing"),
                "after_first_optimizer_step",
                "repair_before_first_step_with_empty_state",
            ),
            (
                ("frozen_head_repaired", "repair", "model_state_sha256_after"),
                "changed",
                "repair_preserves_model_state",
            ),
            (
                (
                    "frozen_head_repaired",
                    "parameter_group_epochs",
                    1,
                    "head",
                    "min_trainable_tensors",
                ),
                0,
                "repaired_all_tensors_trainable_with_gradients_and_updates_every_epoch",
            ),
        )
        for path, value, failed_check in cases:
            with self.subTest(failed_check=failed_check):
                changed = copy.deepcopy(variants)
                target = changed
                for component in path[:-1]:
                    target = target[component]
                target[path[-1]] = value
                result = runner.assess_frozen_head_training(changed, expected, 2, comparisons)
                verification = result["repair_verification"]
                self.assertIs(verification["structural_checks"][failed_check], False)
                self.assertEqual(verification["decision"], "rejected")

        changed = copy.deepcopy(variants)
        frozen_loss = variants["frozen_head"]["final_metrics"]["ood_validation_loss"]
        changed["frozen_head_repaired"]["final_metrics"]["ood_validation_loss"] = (
            10 * frozen_loss + 1
        )
        verification = runner.assess_frozen_head_training(changed, expected, 2, comparisons)[
            "repair_verification"
        ]
        self.assertTrue(verification["structure_verified"])
        self.assertFalse(verification["performance_nonregression"])
        self.assertEqual(verification["decision"], "rejected")

    def test_variant_loop_matches_stale_experiment_runner(self):
        from runsleuth.optimizer_probe import state_dict_sha256

        config = CamelyonConfig.load(self.reference / "config.json")
        state = torch.load(
            self.reference / "initial_state_dict.pt", map_location="cpu", weights_only=True
        )
        expected = state_dict_sha256(state)
        timing = {"epoch_seconds", "peak_cuda_memory_mb"}
        loops = (("original", experiment._run_variant), ("mirror", runner._run_frozen_variant))
        for variant in ("clean", "stale_head"):
            with self.subTest(variant=variant):
                reports = []
                for name, run_variant in loops:
                    root = self.output / name
                    root.mkdir(parents=True, exist_ok=True)
                    variant_config = replace(
                        config, epochs=2, run_name=variant, output_dir=str(root), device="cpu"
                    )
                    with (
                        patch(
                            "runsleuth.camelyon.build_camelyon_model", side_effect=self.build_model
                        ),
                        patch(
                            "runsleuth.camelyon_data.build_camelyon_data",
                            side_effect=self.build_data,
                        ),
                        patch("builtins.print"),
                    ):
                        reports.append(
                            run_variant(
                                variant,
                                variant_config,
                                self.manifest,
                                state,
                                expected,
                                root / variant,
                                torch.device("cpu"),
                            )
                        )
                original, mirror = reports
                for key in (
                    "initial_state_sha256",
                    "optimizer_audit",
                    "final_optimizer_audit",
                    "initialization_metrics",
                    "parameter_group_epochs",
                ):
                    self.assertEqual(mirror[key], original[key], key)
                self.assertEqual(
                    {k: v for k, v in mirror["final_metrics"].items() if k not in timing},
                    {k: v for k, v in original["final_metrics"].items() if k not in timing},
                )
                left, right = (
                    torch.load(report["model_checkpoint"], map_location="cpu", weights_only=True)
                    for report in (original, mirror)
                )
                self.assertTrue(all(torch.equal(left[key], right[key]) for key in left))

    def test_performance_gate_matches_stale_experiment_policy(self):
        with (
            patch("runsleuth.camelyon.build_camelyon_model", side_effect=self.build_model),
            patch("runsleuth.camelyon_data.build_camelyon_data", side_effect=self.build_data),
            patch.object(experiment, "snapshot_sources", return_value={}),
            patch("builtins.print"),
        ):
            report = read_json(
                experiment.run_optimizer_training(
                    self.reference,
                    self.output / "stale",
                    epochs=2,
                    requested_device="cpu",
                    verify_rebind=True,
                )
            )
        variants, expected = report["variants"], report["reference_initial_state_sha256"]
        for changes in (
            {},
            {"ood_validation_loss": 10.0},
            {"id_validation_accuracy": 0.0},
            {"ood_validation_accuracy": float("nan")},
        ):
            with self.subTest(changes=changes):
                changed = copy.deepcopy(variants)
                changed["stale_head_repaired"]["final_metrics"].update(changes)
                self.assertEqual(
                    runner._performance_checks(
                        changed, "stale_head_repaired", ("clean", "stale_head")
                    ),
                    experiment.assess_rebind_training(changed, expected, 2)["performance_checks"],
                )

    def test_cli_exits_nonzero_unless_completed_and_accepted(self):
        cases = (
            ({"status": "completed", "repair_verification": {"decision": "accepted"}}, False),
            ({"status": "completed", "repair_verification": {"decision": "rejected"}}, True),
            ({"status": "inconclusive", "repair_verification": {"decision": "accepted"}}, True),
        )
        for report, exits in cases:
            with self.subTest(report=report):
                with (
                    patch.object(sys, "argv", ["training", "--reference-run", "reference"]),
                    patch.object(
                        runner, "run_frozen_head_training", return_value=Path("report.json")
                    ) as run,
                    patch.object(runner, "read_json", return_value=report),
                ):
                    if exits:
                        with self.assertRaises(SystemExit):
                            runner.main()
                    else:
                        runner.main()
                run.assert_called_once_with(
                    Path("reference"), Path("artifacts/frozen_head_training"), 3, None
                )

    def test_invalid_epoch_budget_does_no_work(self):
        for epochs in (0, 4, True, 2.0):
            with self.subTest(epochs=epochs), patch.object(runner, "load_reference") as load:
                with self.assertRaises(ValueError):
                    runner.run_frozen_head_training(self.reference, self.output, epochs=epochs)
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
