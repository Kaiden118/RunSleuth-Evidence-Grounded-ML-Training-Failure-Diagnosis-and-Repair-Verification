"""Offline paired-training tests using real CPU optimization and tiny models."""

import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from runsleuth import camelyon_optimizer_training as experiment
from runsleuth.camelyon_config import CamelyonConfig
from runsleuth.parameter_group_monitor import OptimizerAudit

TORCH_AVAILABLE = bool(
    importlib.util.find_spec("torch") and importlib.util.find_spec("torchvision")
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rebind_training_fixture():
    audit = OptimizerAudit(
        model_parameter_tensors=4,
        trainable_parameter_tensors=4,
        optimizer_unique_parameter_tensors=4,
        missing_trainable_names=(),
        foreign_parameter_tensors=0,
        duplicate_parameter_occurrences=0,
    ).to_dict()
    metrics = {
        "id_validation_accuracy": 0.95,
        "id_validation_loss": 0.2,
        "ood_validation_accuracy": 0.9,
        "ood_validation_loss": 0.4,
    }
    clean = {
        "status": "completed",
        "initial_state_sha256": "initial",
        "completed_epochs": 2,
        "initialization_metrics": {key: 0.5 for key in metrics},
        "optimizer_audit": audit,
        "final_optimizer_audit": copy.deepcopy(audit),
        "parameter_group_epochs": [
            {
                "epoch": epoch,
                "optimizer_steps": 3,
                "head": {"mean_gradient_l2_norm": 1.0, "mean_parameter_update_l2_norm": 0.01},
                "backbone": {"mean_gradient_l2_norm": 2.0, "mean_parameter_update_l2_norm": 0.02},
            }
            for epoch in (1, 2)
        ],
        "final_metrics": {"epoch": 2, **metrics},
    }
    stale = copy.deepcopy(clean)
    for key in ("optimizer_audit", "final_optimizer_audit"):
        stale[key].update(
            missing_trainable_names=["fc.bias", "fc.weight"], foreign_parameter_tensors=2
        )
    for row in stale["parameter_group_epochs"]:
        row["head"]["mean_parameter_update_l2_norm"] = 0.0
    repaired = copy.deepcopy(clean)
    repaired["repair"] = {
        "action": "rebuild_optimizer",
        "timing": "before_first_optimizer_step",
        "optimizer_state_entries_before": 0,
        "model_state_sha256_before": "initial",
        "model_state_sha256_after": "initial",
        "optimizer_audit_before": copy.deepcopy(stale["optimizer_audit"]),
    }
    return {"clean": clean, "stale_head": stale, "stale_head_repaired": repaired}


class RebindDecisionTests(unittest.TestCase):
    def test_acceptance_requires_structure_and_both_domain_comparators(self):
        variants = rebind_training_fixture()
        original = copy.deepcopy(variants)
        result = experiment.assess_rebind_training(variants, "initial", 2)
        self.assertEqual(result["decision"], "accepted")
        self.assertTrue(result["structure_verified"])
        self.assertTrue(all(result["structural_checks"].values()))
        self.assertTrue(result["performance_nonregression"])
        self.assertEqual(result["policy"], {"max_accuracy_drop": 0.01, "max_loss_ratio": 1.10})
        self.assertEqual(
            {
                (row["comparator"], row["domain"], row["metric"])
                for row in result["performance_checks"]
            },
            {
                (comparator, domain, metric)
                for comparator in ("clean", "stale_head")
                for domain in ("id", "ood")
                for metric in ("accuracy_drop", "loss_ratio")
            },
        )
        self.assertEqual(len(result["performance_checks"]), 8)
        self.assertTrue(all(row["passed"] for row in result["performance_checks"]))
        self.assertEqual(variants, original, "Verification must not change recorded evidence")
        self.assertEqual(
            experiment.assess_rebind_training(json.loads(json.dumps(variants)), "initial", 2),
            result,
            "In-memory audits and saved JSON must produce the same verification decision",
        )

    def test_metric_regression_rejects_despite_structural_repair(self):
        for domain in ("id", "ood"):
            for metric, value in (("accuracy", 0.5), ("loss", 1.0)):
                with self.subTest(domain=domain, metric=metric):
                    variants = rebind_training_fixture()
                    metrics = variants["stale_head_repaired"]["final_metrics"]
                    metrics[f"{domain}_validation_{metric}"] = value
                    result = experiment.assess_rebind_training(variants, "initial", 2)
                    self.assertTrue(result["structure_verified"])
                    self.assertFalse(result["performance_nonregression"])
                    self.assertEqual(result["decision"], "rejected")

    def test_accuracy_and_loss_thresholds_are_inclusive_without_accepting_regression(self):
        cases = ((0.89, 0.44, True), (0.889, 0.44, False), (0.89, 0.441, False))
        for accuracy, loss, accepted in cases:
            with self.subTest(accuracy=accuracy, loss=loss):
                variants = rebind_training_fixture()
                metrics = variants["stale_head_repaired"]["final_metrics"]
                metrics.update(ood_validation_accuracy=accuracy, ood_validation_loss=loss)
                result = experiment.assess_rebind_training(variants, "initial", 2)
                self.assertEqual(result["performance_nonregression"], accepted)
                self.assertEqual(result["decision"], "accepted" if accepted else "rejected")

    def test_stale_comparator_can_reject_even_when_clean_comparison_passes(self):
        variants = rebind_training_fixture()
        variants["stale_head"]["final_metrics"]["ood_validation_accuracy"] = 0.94
        result = experiment.assess_rebind_training(variants, "initial", 2)
        self.assertTrue(result["structure_verified"])
        self.assertTrue(
            all(
                row["passed"]
                for row in result["performance_checks"]
                if row["comparator"] == "clean"
            )
        )
        self.assertFalse(result["performance_nonregression"])
        self.assertEqual(result["decision"], "rejected")

    def test_structural_failures_reject_even_with_equal_final_performance(self):
        cases = (
            (("repair", "model_state_sha256_after"), "changed"),
            (("repair", "optimizer_state_entries_before"), 1),
            (("repair", "optimizer_audit_before", "missing_trainable_names"), []),
            (("optimizer_audit", "foreign_parameter_tensors"), 2),
            (("final_optimizer_audit", "missing_trainable_names"), ["fc.weight"]),
            (("parameter_group_epochs", 0, "head", "mean_parameter_update_l2_norm"), 0.0),
            (("parameter_group_epochs", 1, "head", "mean_gradient_l2_norm"), float("inf")),
            (("parameter_group_epochs", 1, "optimizer_steps"), 2),
            (("completed_epochs",), 1),
            (("final_metrics", "epoch"), 1),
            (("status",), "failed"),
        )
        for path, value in cases:
            with self.subTest(path=path):
                variants = rebind_training_fixture()
                target = variants["stale_head_repaired"]
                for component in path[:-1]:
                    target = target[component]
                target[path[-1]] = value
                result = experiment.assess_rebind_training(variants, "initial", 2)
                self.assertFalse(result["structure_verified"])
                self.assertTrue(result["performance_nonregression"])
                self.assertEqual(result["decision"], "rejected")

    def test_invalid_performance_values_fail_closed(self):
        for variant in ("clean", "stale_head", "stale_head_repaired"):
            for metric, value in (
                ("id_validation_accuracy", float("nan")),
                ("ood_validation_loss", float("inf")),
                ("id_validation_accuracy", 1.1),
                ("ood_validation_loss", -0.1),
            ):
                with self.subTest(variant=variant, metric=metric, value=value):
                    variants = rebind_training_fixture()
                    variants[variant]["final_metrics"][metric] = value
                    result = experiment.assess_rebind_training(variants, "initial", 2)
                    self.assertFalse(result["performance_nonregression"])
                    self.assertEqual(result["decision"], "rejected")
                    json.dumps(result, allow_nan=False)

    def test_zero_reference_loss_does_not_divide_by_zero(self):
        for repaired_loss in (0.0, 0.01):
            with self.subTest(repaired_loss=repaired_loss):
                variants = rebind_training_fixture()
                variants["clean"]["final_metrics"]["id_validation_loss"] = 0.0
                repaired = variants["stale_head_repaired"]["final_metrics"]
                repaired["id_validation_loss"] = repaired_loss
                result = experiment.assess_rebind_training(variants, "initial", 2)
                row = next(
                    row
                    for row in result["performance_checks"]
                    if (row["comparator"], row["domain"], row["metric"])
                    == ("clean", "id", "loss_ratio")
                )
                self.assertIsNone(row["observed_value"])
                self.assertEqual(row["passed"], repaired_loss == 0.0)
                expected = "accepted" if repaired_loss == 0.0 else "rejected"
                self.assertEqual(result["decision"], expected)
                json.dumps(result, allow_nan=False)


class ArgumentTests(unittest.TestCase):
    def test_cli_forwards_rebind_and_returns_failure_for_rejected_repair(self):
        for enabled, decision in ((False, None), (True, "accepted"), (True, "rejected")):
            with self.subTest(enabled=enabled, decision=decision):
                args = ["optimizer_training", "--reference-run", "reference", "--epochs", "2"]
                if enabled:
                    args.append("--verify-rebind")
                report = {"status": "completed", "repair_verification": {"decision": decision}}
                with (
                    patch.object(sys, "argv", args),
                    patch.object(
                        experiment, "run_optimizer_training", return_value=Path("report")
                    ) as run,
                    patch.object(experiment, "read_json", return_value=report),
                ):
                    if decision == "rejected":
                        with self.assertRaises(SystemExit) as error:
                            experiment.main()
                        self.assertEqual(error.exception.code, 1)
                    else:
                        experiment.main()
                self.assertEqual(run.call_args.args[2], 2)
                self.assertEqual(run.call_args.kwargs, {"verify_rebind": enabled})

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

    def run_pair(self, *, verify_rebind=False):
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
                verify_rebind=verify_rebind,
            )
        self.assertEqual(model.call_count, 3 if verify_rebind else 2)
        self.assertEqual(data.call_count, 3 if verify_rebind else 2)
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

    def test_rebuild_restores_updates_without_changing_training_control_or_downloads(self):
        from runsleuth.optimizer_probe import state_dict_sha256

        original_run_variant = experiment._run_variant
        observed_policies = []

        def guarded_run_variant(*args, **kwargs):
            directory = args[5]
            policy = directory.parent / "repair_verification_policy.json"
            self.assertTrue(policy.is_file(), "Freeze policy before any variant starts")
            observed_policies.append(policy.read_bytes())
            return original_run_variant(*args, **kwargs)

        with patch.object(experiment, "_run_variant", side_effect=guarded_run_variant):
            report_path = self.run_pair(verify_rebind=True)
        report = read_json(report_path)
        self.assertEqual(len(observed_policies), 3)
        self.assertTrue(all(value == observed_policies[0] for value in observed_policies))
        self.assertEqual(set(report["variants"]), {"clean", "stale_head", "stale_head_repaired"})
        clean = report["variants"]["clean"]
        stale = report["variants"]["stale_head"]
        repaired = report["variants"]["stale_head_repaired"]
        repair = repaired["repair"]
        initial_hash = state_dict_sha256(self.initial)
        self.assertEqual(repair["model_state_sha256_before"], initial_hash)
        self.assertEqual(repair["model_state_sha256_after"], initial_hash)
        self.assertEqual(repair["optimizer_state_entries_before"], 0)
        self.assertEqual(repair["optimizer_audit_before"], stale["optimizer_audit"])
        self.assertEqual(repaired["optimizer_audit"], clean["optimizer_audit"])
        self.assertEqual(repaired["final_optimizer_audit"], clean["final_optimizer_audit"])
        self.assertEqual(self.models[0].training_order, self.models[2].training_order)
        self.assertEqual(self.models[1].training_order, self.models[2].training_order)
        self.assertEqual(repaired["parameter_group_epochs"], clean["parameter_group_epochs"])
        for row in repaired["parameter_group_epochs"]:
            self.assertGreater(row["head"]["mean_parameter_update_l2_norm"], 0)
        for key, value in self.models[0].state_dict().items():
            self.assertTrue(self.torch.equal(value, self.models[2].state_dict()[key]), key)
        verification = report["repair_verification"]
        self.assertTrue(verification["structure_verified"])
        self.assertEqual(
            verification["decision"],
            "accepted" if verification["performance_nonregression"] else "rejected",
        )
        self.assertFalse(report["test_evaluated"])

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
