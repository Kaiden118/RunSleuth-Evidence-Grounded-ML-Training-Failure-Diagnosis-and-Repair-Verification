"""Offline tests for reference identity and optimizer-probe decisions."""

import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from runsleuth.camelyon_config import CamelyonConfig
from runsleuth.camelyon_optimizer_probe import (
    assess_probe,
    check_matching_data,
    file_sha256,
    load_reference,
)


def write_json(path, value):
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def manifest_fixture():
    return {
        "dataset": "camelyon17",
        "dataset_version": "1.0",
        "subset_seed": 2026,
        "source_provenance": {"repo": "fixture", "revision": "pinned", "metadata_sha256": "abc"},
        "splits": {
            "train": {"selected_sample_ids": [1, 2]},
            "id_val": {"selected_sample_ids": [3, 4]},
            "val": {"selected_sample_ids": [5, 6]},
        },
    }


def variants_fixture():
    clean = {
        "initial_state_sha256": "initial",
        "pre_update_loss": 0.7,
        "optimizer_audit": {
            "missing_trainable_names": [],
            "foreign_parameter_tensors": 0,
            "duplicate_parameter_occurrences": 0,
        },
        "head": {"gradient_l2_norm": 1.0, "parameter_update_l2_norm": 0.01},
        "backbone": {"gradient_l2_norm": 2.0, "parameter_update_l2_norm": 0.02},
    }
    stale = copy.deepcopy(clean)
    stale["optimizer_audit"].update(
        {
            "missing_trainable_names": ["fc.bias", "fc.weight"],
            "foreign_parameter_tensors": 2,
        }
    )
    stale["head"]["parameter_update_l2_norm"] = 0.0
    return {"clean": clean, "stale_head": stale}


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.config = replace(CamelyonConfig(), device="cpu")
        self.config.save(self.directory / "config.json")
        # Only artifact hashes are tested here; no checkpoint is deserialized.
        (self.directory / "initial_state_dict.pt").write_bytes(b"test initial state")
        write_json(self.directory / "data_manifest.json", manifest_fixture())
        self.report = {
            "status": "completed",
            "completed_epochs": self.config.epochs,
            "task": "camelyon17_resnet18_development_baseline",
            "initial_checkpoint_sha256": file_sha256(self.directory / "initial_state_dict.pt"),
            "data_manifest_sha256": file_sha256(self.directory / "data_manifest.json"),
        }
        write_json(self.directory / "run_report.json", self.report)

    def test_completed_reference_loads_saved_initial_checkpoint(self):
        config, manifest, checkpoint = load_reference(self.directory)
        self.assertEqual(config, self.config)
        self.assertEqual(manifest, manifest_fixture())
        self.assertEqual(checkpoint.name, "initial_state_dict.pt")

    def test_changed_checkpoint_or_manifest_is_rejected(self):
        for name in ("initial_state_dict.pt", "data_manifest.json"):
            with self.subTest(name=name):
                path = self.directory / name
                original = path.read_bytes()
                path.write_bytes(original + b"changed")
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    load_reference(self.directory)
                path.write_bytes(original)

    def test_incomplete_reference_is_rejected(self):
        for changes in ({"status": "failed"}, {"completed_epochs": 2}, {"task": "other"}):
            with self.subTest(changes=changes):
                write_json(self.directory / "run_report.json", {**self.report, **changes})
                with self.assertRaisesRegex(ValueError, "completed Camelyon17"):
                    load_reference(self.directory)

    def test_same_count_different_sample_identity_is_rejected(self):
        reference = manifest_fixture()
        check_matching_data(reference, copy.deepcopy(reference))
        for split in ("train", "id_val", "val"):
            changed = copy.deepcopy(reference)
            changed["splits"][split]["selected_sample_ids"][0] = 999
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, "sample IDs differ"):
                check_matching_data(reference, changed)

    def test_changed_revision_or_metadata_is_rejected(self):
        reference = manifest_fixture()
        for field in ("repo", "revision", "metadata_sha256"):
            changed = copy.deepcopy(reference)
            changed["source_provenance"][field] = "different"
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "provenance mismatch"),
            ):
                check_matching_data(reference, changed)


class ProbeDecisionTests(unittest.TestCase):
    def test_expected_mechanism_satisfies_every_check(self):
        self.assertTrue(all(assess_probe(variants_fixture(), "initial").values()))

    def test_different_initial_state_prevents_success(self):
        variants = variants_fixture()
        variants["stale_head"]["initial_state_sha256"] = "different"
        self.assertFalse(all(assess_probe(variants, "initial").values()))

    def test_zero_head_update_without_gradients_is_insufficient(self):
        variants = variants_fixture()
        variants["stale_head"]["head"]["gradient_l2_norm"] = 0.0
        self.assertFalse(
            assess_probe(variants, "initial")["stale_head_has_gradients_without_updates"]
        )

    def test_unrelated_missing_parameter_does_not_count_as_head_fault(self):
        variants = variants_fixture()
        variants["stale_head"]["optimizer_audit"]["missing_trainable_names"] = ["backbone.weight"]
        self.assertFalse(
            assess_probe(variants, "initial")["stale_optimizer_misses_exactly_current_head"]
        )

    def test_missing_backbone_update_prevents_success(self):
        variants = variants_fixture()
        variants["stale_head"]["backbone"]["parameter_update_l2_norm"] = 0.0
        self.assertFalse(assess_probe(variants, "initial")["both_backbones_update"])


if __name__ == "__main__":
    unittest.main()
