"""Configuration checks that run without PyTorch or pytest."""

import json
import tempfile
import unittest
from pathlib import Path

from runsleuth.camelyon_config import CAMELYON_REVISION, CamelyonConfig


class CamelyonConfigTests(unittest.TestCase):
    def test_roundtrip_preserves_explicit_settings_and_null_caps(self):
        config = CamelyonConfig(
            run_name="audit-run",
            seed=7,
            subset_seed=19,
            epochs=2,
            max_train_samples=None,
            max_id_val_samples=11,
            max_ood_val_samples=None,
            weight_decay=0,
            device="cpu",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "config.json"
            config.save(path)
            self.assertEqual(CamelyonConfig.load(path), config)
            serialized = json.loads(path.read_text(encoding="utf-8"))
        self.assertIsNone(serialized["max_train_samples"])
        self.assertIsNone(serialized["max_ood_val_samples"])
        self.assertEqual(serialized["hf_revision"], CAMELYON_REVISION)

    def test_invalid_types_ranges_and_nonfinite_numbers_are_rejected(self):
        invalid = {
            "epochs": (0, -1, True, 1.5, "3"),
            "batch_size": (0, False, 2.5),
            "seed": (-1, 2**32, True, 1.0),
            "subset_seed": (-1, 2**32, False, "42"),
            "num_workers": (-1, True, 0.5),
            "max_train_samples": (0, -1, True, 2.5),
            "max_id_val_samples": (0, False, "5"),
            "max_ood_val_samples": (-1, True, 3.5),
            "learning_rate": (0, -0.1, True, "0.01", float("nan"), float("inf")),
            "weight_decay": (-0.1, False, "0.01", float("nan"), float("-inf")),
            "run_name": ("", "../run", "nested/run", "a b", "a" * 81, None),
            "device": ("gpu", "cuda:0", None),
            "data_dir": ("", "  ", None),
            "output_dir": ("", "\t", 12),
        }
        for field, values in invalid.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        CamelyonConfig(**{field: value})

    def test_baseline_identity_cannot_be_changed_silently(self):
        for field, value in {
            "dataset": "fashionmnist",
            "dataset_version": "2.0",
            "model": "resnet50",
            "pretrained_weights": None,
            "hf_revision": "main",
        }.items():
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    CamelyonConfig(**{field: value})

    def test_load_rejects_nonobject_unknown_fields_and_nonfinite_json(self):
        invalid_documents = (
            ("[]", ValueError),
            ("null", ValueError),
            ('{"optimizer_step_enabled": false}', TypeError),
            ('{"learning_rate": NaN}', ValueError),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for document, error in invalid_documents:
                with self.subTest(document=document):
                    path.write_text(document, encoding="utf-8")
                    with self.assertRaises(error):
                        CamelyonConfig.load(path)
            path.write_text('{"device": "cpu"}', encoding="utf-8-sig")
            self.assertEqual(CamelyonConfig.load(path).device, "cpu")

    def test_saving_does_not_replace_an_existing_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            original = CamelyonConfig(run_name="original")
            original.save(path)
            with self.assertRaises(FileExistsError):
                CamelyonConfig(run_name="replacement").save(path)
            self.assertEqual(CamelyonConfig.load(path), original)


if __name__ == "__main__":
    unittest.main()
