"""No network or model downloads; loader tests use a synthetic mirror dataset."""

import hashlib
import importlib.util
import json
import pickle
import random
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from runsleuth.camelyon_data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    NativeRGBPatch,
    PairDataset,
    build_camelyon_data,
    select_stratified_indices,
)


class StratificationTests(unittest.TestCase):
    def setUp(self):
        self.indices = list(range(100, 200))
        self.labels = [0] * 70 + [1] * 30
        self.hospitals = [0] * 50 + [3] * 20 + [3] * 20 + [4] * 10

    def select(self, cap=23, seed=2026):
        return select_stratified_indices(
            self.indices,
            self.labels,
            self.hospitals,
            cap,
            seed=seed,
        )

    def test_exact_proportional_count_and_unique_global_ids(self):
        selected = self.select(cap=23)
        self.assertEqual(len(selected), 23)
        self.assertEqual(len(set(selected)), 23)
        counts = Counter(
            (self.labels[index - 100], self.hospitals[index - 100]) for index in selected
        )
        self.assertEqual(counts, {(0, 0): 11, (0, 3): 5, (1, 3): 5, (1, 4): 2})

    def test_reproducible_independent_rng_and_seed_changes_selection(self):
        state = random.getstate()
        self.assertEqual(self.select(), self.select())
        self.assertEqual(state, random.getstate())
        self.assertNotEqual(self.select(seed=1), self.select(seed=2))

    def test_membership_independent_of_input_order(self):
        self.assertEqual(
            self.select(),
            select_stratified_indices(
                self.indices[::-1],
                self.labels[::-1],
                self.hospitals[::-1],
                23,
                seed=2026,
            ),
        )

    def test_full_and_oversize_caps(self):
        self.assertEqual(self.select(None), self.indices)
        self.assertEqual(self.select(200), self.indices)
        self.assertEqual(select_stratified_indices([], [], [], None, seed=0), [])

    def test_every_positive_cap_has_expected_length(self):
        for cap in range(1, 101):
            self.assertEqual(len(self.select(cap)), cap)

    def test_rejects_invalid_inputs(self):
        for cap in (0, -1, True, 2.5):
            with self.assertRaises(ValueError):
                self.select(cap)
        for indices, labels, hospitals in (
            ([0], [], [0]),
            ([0, 0], [0, 1], [0, 0]),
            ([0], [2], [0]),
        ):
            with self.assertRaises(ValueError):
                select_stratified_indices(indices, labels, hospitals, 1, seed=0)

    def test_native_image_check_and_rgb_conversion(self):
        rgb = NativeRGBPatch()(Image.new("L", (96, 96)))
        self.assertEqual((rgb.mode, rgb.size), ("RGB", (96, 96)))
        with self.assertRaisesRegex(ValueError, "96x96"):
            NativeRGBPatch()(Image.new("RGB", (224, 224)))

    def test_pair_adapter_preserves_targets_and_is_pickleable(self):
        adapter = PairDataset([("image0", 0, "metadata"), ("image1", 1, "metadata")], [1, 0])
        restored = pickle.loads(pickle.dumps(adapter))
        self.assertEqual(len(restored), 2)
        self.assertEqual(restored[0], ("image1", 1))


class SyntheticSubset:
    def __init__(self, dataset, indices, transform):
        self.dataset, self.indices, self.transform = dataset, indices, transform

    def __getitem__(self, position):
        index = self.indices[position]
        image = Image.new("RGB", (96, 96), (255, 0, 128))
        return (
            self.transform(image),
            self.dataset.y_array[index],
            self.dataset.metadata_array[index],
        )


class SyntheticWILDS:
    version = "1.0"
    split_scheme = "official"
    metadata_fields = ["hospital", "slide", "y"]
    split_dict = {"train": 0, "id_val": 1, "test": 2, "val": 3}

    def __init__(self, directory):
        import torch

        self.data_dir = directory
        self.source_metadata_path = Path(directory, "provenance_metadata.json")
        self.source_metadata_path.write_bytes(b'{"source":"synthetic"}\n')
        self.provenance = {"source": "synthetic", "equivalence_not_verified": True}
        self.sample_ids = list(range(1000, 1064))
        self.y_array = torch.tensor([0, 1] * 32)
        self.metadata_array = torch.tensor(
            [
                [
                    ([0, 3, 4][(index // 2) % 3] if index < 48 else (1 if index < 60 else 2)),
                    index // 2,
                    index % 2,
                ]
                for index in range(64)
            ]
        )
        self.split_array = torch.tensor([0] * 36 + [1] * 12 + [3] * 12 + [2] * 4)
        self.requests = []

    def get_subset(self, split, transform):
        assert split != "test", "Test subset must never be requested"
        self.requests.append(split)
        indices = (self.split_array == self.split_dict[split]).nonzero().flatten().tolist()
        return SyntheticSubset(self, indices, transform)


@unittest.skipUnless(
    importlib.util.find_spec("torch") and importlib.util.find_spec("torchvision"),
    "Install torch and torchvision to run synthetic loader integration tests",
)
class LoaderIntegrationTests(unittest.TestCase):
    def setUp(self):
        import torch

        self.torch = torch
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.dataset = SyntheticWILDS(self.directory.name)
        self.config = SimpleNamespace(
            data_dir=self.directory.name,
            dataset_version="1.0",
            subset_seed=2026,
            max_train_samples=18,
            max_id_val_samples=6,
            max_ood_val_samples=6,
            batch_size=4,
            num_workers=0,
            seed=42,
        )
        self.load_mirror = Mock(return_value=self.dataset)
        self.mock_mirror = patch("runsleuth.camelyon_data.load_camelyon_mirror", self.load_mirror)
        self.mock_mirror.start()
        self.addCleanup(self.mock_mirror.stop)

    def build(self):
        return build_camelyon_data(self.config, device=self.torch.device("cpu"))

    def test_official_splits_pairs_native_shape_normalization_and_hashes(self):
        data = self.build()
        self.load_mirror.assert_called_once_with(self.config, download=False)
        self.assertEqual(self.dataset.requests, ["train", "id_val", "val"])
        images, targets = next(iter(data.train_loader))
        self.assertEqual(tuple(images.shape), (4, 3, 96, 96))
        self.assertEqual(targets.dtype, self.torch.int64)
        expected = self.torch.tensor(
            [
                (v / 255 - m) / s
                for v, m, s in zip((255, 0, 128), IMAGENET_MEAN, IMAGENET_STD, strict=True)
            ]
        )
        self.assertTrue(self.torch.allclose(images[0, :, 0, 0], expected))
        selected_sets = [
            set(record["selected_indices"]) for record in data.manifest["splits"].values()
        ]
        self.assertEqual(len(set.union(*selected_sets)), sum(map(len, selected_sets)))
        for split, record in data.manifest["splits"].items():
            self.assertTrue(
                all(
                    self.dataset.split_array[i] == self.dataset.split_dict[split]
                    for i in record["selected_indices"]
                )
            )
            expected_hash = hashlib.sha256(
                json.dumps(record["selected_indices"], separators=(",", ":")).encode()
            ).hexdigest()
            self.assertEqual(record["selected_indices_sha256"], expected_hash)
            self.assertEqual(
                record["selected_sample_ids"], [1000 + i for i in record["selected_indices"]]
            )
        self.assertEqual(
            data.manifest["source_metadata_sha256"],
            hashlib.sha256(b'{"source":"synthetic"}\n').hexdigest(),
        )
        self.assertEqual(data.manifest["split_scheme"], "official_reconstructed_from_mirror")
        self.assertTrue(data.manifest["source_provenance"]["equivalence_not_verified"])
        self.assertEqual(
            data.manifest["majority_baseline"]["accuracy_by_split"],
            {"train": 0.5, "id_val": 0.5, "val": 0.5},
        )
        json.dumps(data.manifest)  # All manifest values must be JSON serializable.

    def test_training_seed_changes_order_but_not_subset_membership(self):
        first, same = self.build(), self.build()
        self.assertEqual(list(first.train_loader.sampler), list(same.train_loader.sampler))
        self.config.seed = 123
        different = self.build()
        self.assertEqual(first.manifest, different.manifest)
        self.assertNotEqual(list(same.train_loader.sampler), list(different.train_loader.sampler))

    def test_windows_pickle_roundtrip(self):
        data = self.build()
        restored = pickle.loads(pickle.dumps(data.train_loader.dataset))
        self.assertEqual(tuple(restored[0][0].shape), (3, 96, 96))

    def test_rejects_ood_hospital_overlap(self):
        self.dataset.metadata_array[48:60, 0] = 0
        with self.assertRaisesRegex(ValueError, "hospitals overlap"):
            self.build()

    def test_rejects_test_indices_mislabeled_as_train(self):
        original = self.dataset.get_subset

        def corrupt_subset(split, transform):
            subset = original(split, transform)
            if split == "train":
                subset.indices.append(60)
            return subset

        self.dataset.get_subset = corrupt_subset
        with self.assertRaisesRegex(ValueError, "another split"):
            self.build()


if __name__ == "__main__":
    unittest.main()
