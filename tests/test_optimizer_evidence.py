"""Optimizer evidence validation uses only small JSON fixtures, never a GPU or API."""

import hashlib
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from runsleuth.camelyon_config import CamelyonConfig
from runsleuth.optimizer_evidence import FILES, inspect_optimizer_run


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class OptimizerEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.directory = self.root / "stale_head"
        self.directory.mkdir()
        self.config = CamelyonConfig(
            epochs=2,
            batch_size=2,
            max_train_samples=3,
            max_id_val_samples=1,
            max_ood_val_samples=1,
            run_name="stale_head",
            data_dir="C:/private-data",
        )
        write_json(self.directory / "config.json", asdict(self.config))
        splits = {}
        for split, ids in (("train", [10, 11, 12]), ("id_val", [20]), ("val", [30])):
            splits[split] = {
                "available_samples": len(ids),
                "selected_samples": len(ids),
                "selected_sample_ids": ids,
                "selected_sample_ids_sha256": hashlib.sha256(
                    json.dumps(ids, separators=(",", ":")).encode()
                ).hexdigest(),
            }
        self.manifest = {
            "dataset": "camelyon17",
            "dataset_version": "1.0",
            "subset_seed": 2026,
            "source_provenance": {
                "repo": "wltjr1007/Camelyon17-WILDS",
                "revision": self.config.hf_revision,
                "metadata_sha256": "b" * 64,
            },
            "transform": {
                "resolution": [96, 96],
                "color_mode": "RGB",
                "to_tensor": True,
                "normalize_mean": [0.485, 0.456, 0.406],
                "normalize_std": [0.229, 0.224, 0.225],
                "resize": False,
                "augmentation": False,
            },
            "splits": splits,
        }
        write_json(self.directory / "data_manifest.json", self.manifest)
        self.metrics, self.domains, self.groups = [], [], []
        for epoch in (1, 2):
            self.metrics.append(
                {
                    "epoch": epoch,
                    "train_loss": 0.12,
                    "train_accuracy": 0.95,
                    "validation_loss": 0.07,
                    "validation_accuracy": 0.98,
                    "mean_gradient_norm": 1.6,
                    "mean_parameter_update_norm": 0.04,
                }
            )
            self.domains.append(
                {
                    "epoch": epoch,
                    "id_validation_loss": 0.07,
                    "id_validation_accuracy": 0.98,
                    "ood_validation_loss": 0.32,
                    "ood_validation_accuracy": 0.91,
                    "epoch_seconds": 70.0,
                    "peak_cuda_memory_mb": None,
                }
            )
            self.groups.append(
                {
                    "epoch": epoch,
                    "optimizer_steps": 2,
                    "head": {"mean_gradient_l2_norm": 0.25, "mean_parameter_update_l2_norm": 0.0},
                    "backbone": {
                        "mean_gradient_l2_norm": 1.5,
                        "mean_parameter_update_l2_norm": 0.04,
                    },
                }
            )
        self.report = {
            "variant": "stale_head",
            "status": "completed",
            "error": None,
            "task": "camelyon17_controlled_optimizer_training",
            "completed_epochs": 2,
            "initial_state_sha256": "a" * 64,
            "data_manifest_sha256": digest(self.directory / "data_manifest.json"),
            "parameter_group_epochs": self.groups,
            "final_metrics": {**self.metrics[-1], **self.domains[-1]},
            "optimizer_audit": {
                "model_parameter_tensors": 62,
                "trainable_parameter_tensors": 62,
                "optimizer_unique_parameter_tensors": 62,
                "missing_trainable_names": ["fc.bias", "fc.weight"],
                "foreign_parameter_tensors": 2,
                "duplicate_parameter_occurrences": 0,
            },
        }
        self.save_telemetry()

    def save_telemetry(self):
        write_rows(self.directory / "metrics.jsonl", self.metrics)
        write_rows(self.directory / "domain_metrics.jsonl", self.domains)
        write_rows(self.directory / "parameter_group_metrics.jsonl", self.groups)
        write_json(self.directory / "run_report.json", self.report)

    def require_file(self, path):
        resolved = path.resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError("Outside project root")
        if not resolved.is_file():
            raise FileNotFoundError(path)
        return resolved

    def inspect(self):
        return inspect_optimizer_run(self.directory, require_file=self.require_file)

    def test_valid_compact_evidence_preserves_signals_without_injected_label(self):
        result = self.inspect()
        self.assertEqual(result["optimizer_audit"]["missing_trainable_parameter_tensors"], 2)
        self.assertEqual(result["epochs"][-1]["head"]["mean_parameter_update_l2_norm"], 0.0)
        self.assertEqual(result["epochs"][-1]["ood_validation_accuracy"], 0.91)
        self.assertFalse(result["provenance"]["live_optimizer_inspected"])
        self.assertEqual(set(result["provenance"]["files_sha256"]), set(FILES))
        self.assertNotIn("variant", result)
        for key in ("run_name", "data_dir", "output_dir"):
            self.assertNotIn(key, result["config"])
        self.assertNotIn("selected_sample_ids", result["data_identity"])
        self.assertEqual(result["initial_state_sha256"], "a" * 64)

    def test_every_read_passes_through_path_boundary_callback(self):
        visited = []

        def checked(path):
            visited.append(path.name)
            return self.require_file(path)

        inspect_optimizer_run(self.directory, require_file=checked)
        self.assertEqual(visited, list(FILES))

    def test_path_boundary_rejection_propagates_before_read(self):
        def reject(_path):
            raise ValueError("Outside project root")

        with self.assertRaisesRegex(ValueError, "Outside project root"):
            inspect_optimizer_run(self.directory, require_file=reject)

    def test_rejects_unfinished_or_wrong_task_report(self):
        for field, value in (("status", "failed"), ("task", "unrelated"), ("completed_epochs", 1)):
            with self.subTest(field=field):
                previous = self.report[field]
                self.report[field] = value
                self.save_telemetry()
                with self.assertRaises(ValueError):
                    self.inspect()
                self.report[field] = previous

    def test_rejects_report_telemetry_disagreement(self):
        self.report["final_metrics"]["ood_validation_accuracy"] = 0.99
        self.save_telemetry()
        with self.assertRaisesRegex(ValueError, "Final metrics"):
            self.inspect()

    def test_rejects_duplicate_group_report_disagreement(self):
        self.save_telemetry()
        self.groups[0]["head"]["mean_parameter_update_l2_norm"] = 0.001
        write_rows(self.directory / "parameter_group_metrics.jsonl", self.groups)
        with self.assertRaisesRegex(ValueError, "Parameter-group epochs"):
            self.inspect()

    def test_rejects_noncontiguous_or_missing_epochs(self):
        for rows in (self.metrics[:1], self.metrics[::-1], self.metrics + [self.metrics[-1]]):
            with self.subTest(rows=len(rows)):
                write_rows(self.directory / "metrics.jsonl", rows)
                with self.assertRaises(ValueError):
                    self.inspect()

    def test_rejects_invalid_numeric_telemetry(self):
        for value in (True, -0.1, float("nan"), float("inf"), "0", 10**500):
            with self.subTest(value=value):
                self.groups[0]["head"]["mean_gradient_l2_norm"] = value
                self.save_telemetry()
                with self.assertRaises(ValueError):
                    self.inspect()

    def test_rejects_accuracy_outside_range_and_id_disagreement(self):
        for value in (1.01, 0.5):
            with self.subTest(value=value):
                self.domains[0]["id_validation_accuracy"] = value
                self.save_telemetry()
                with self.assertRaises(ValueError):
                    self.inspect()

    def test_rejects_wrong_or_boolean_optimizer_step_count(self):
        for value in (True, 0, 3):
            with self.subTest(value=value):
                self.groups[0]["optimizer_steps"] = value
                self.save_telemetry()
                with self.assertRaises(ValueError):
                    self.inspect()

    def test_rejects_invalid_audit_counts_and_duplicate_names(self):
        audit = self.report["optimizer_audit"]
        for field, value in (
            ("foreign_parameter_tensors", -1),
            ("model_parameter_tensors", True),
            ("trainable_parameter_tensors", 100),
            ("optimizer_unique_parameter_tensors", 1),
            ("missing_trainable_names", ["fc.weight", "fc.weight"]),
        ):
            with self.subTest(field=field):
                old = audit[field]
                audit[field] = value
                self.save_telemetry()
                with self.assertRaises(ValueError):
                    self.inspect()
                audit[field] = old

    def test_rejects_manifest_file_hash_mismatch(self):
        self.manifest["subset_seed"] = 9
        write_json(self.directory / "data_manifest.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "manifest.*digest"):
            self.inspect()

    def test_rejects_selected_ids_digest_mismatch_even_with_new_manifest_hash(self):
        self.manifest["splits"]["train"]["selected_sample_ids"] = [10, 11, 13]
        write_json(self.directory / "data_manifest.json", self.manifest)
        self.report["data_manifest_sha256"] = digest(self.directory / "data_manifest.json")
        self.save_telemetry()
        with self.assertRaisesRegex(ValueError, "sample IDs.*digest"):
            self.inspect()

    def test_rejects_changed_preprocessing_even_with_new_manifest_hash(self):
        self.manifest["transform"]["normalize_mean"] = [0.5, 0.5, 0.5]
        write_json(self.directory / "data_manifest.json", self.manifest)
        self.report["data_manifest_sha256"] = digest(self.directory / "data_manifest.json")
        self.save_telemetry()
        with self.assertRaisesRegex(ValueError, "Input transform"):
            self.inspect()

    def test_rejects_duplicate_json_fields(self):
        path = self.directory / "config.json"
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace('"epochs": 2,', '"epochs": 2, "epochs": 3,'), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate JSON"):
            self.inspect()

    def test_rejects_oversized_artifact(self):
        with patch("runsleuth.optimizer_evidence.MAX_FILE_BYTES", 32):
            with self.assertRaisesRegex(ValueError, "size limit"):
                self.inspect()

    def test_missing_files_are_not_silently_ignored(self):
        (self.directory / "parameter_group_metrics.jsonl").unlink()
        with self.assertRaises(FileNotFoundError):
            self.inspect()


if __name__ == "__main__":
    unittest.main()
