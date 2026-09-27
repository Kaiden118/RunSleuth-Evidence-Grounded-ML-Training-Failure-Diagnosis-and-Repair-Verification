"""Network-free checks of mirror provenance, category decoding, and test sealing."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from runsleuth import camelyon_mirror as mirror


class ClassLabel:
    def __init__(self, names):
        self.names = names

    def int2str(self, value):
        if value < 0 or value >= len(self.names):
            raise ValueError(value)
        return self.names[value]


class FakeDataset:
    def __init__(self, rows, centers):
        self.rows = rows
        self.features = {"center": ClassLabel(centers), "label": ClassLabel(["tumor", "non-tumor"])}
        self.column_names = ["image", "image_id", "center", "slide", "label"]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def select_columns(self, names):
        return SimpleNamespace(
            to_dict=lambda: {key: [row[key] for row in self.rows] for key in names}
        )


def fixture():
    def row(index, encoded_center, hospital, label):
        return {
            "image": f"image-{index}",
            "image_id": index,
            "center": encoded_center,
            "slide": hospital * 10,
            "label": label,
        }

    train = [
        row(i * 2 + label, i, hospital, label)
        for i, hospital in enumerate((0, 3, 4))
        for label in (0, 1)
    ]
    validation = [
        row(6 + i * 2 + label, i, hospital, label)
        for i, hospital in enumerate((1, 0, 3, 4))
        for label in (0, 1)
    ]
    return {
        "train": FakeDataset(train, ["train-center1", "train-center2", "train-center3"]),
        "validation": FakeDataset(
            validation, ["validation-center", "train-center1", "train-center2", "train-center3"]
        ),
    }


COUNTS = {"train": 6, "id_val": 6, "val": 2}


class MetadataTests(unittest.TestCase):
    def test_split_local_class_codes_are_decoded_by_name(self):
        rows = mirror._metadata_rows(fixture(), COUNTS)
        self.assertEqual(rows[2], (2, 3, 30, 1, 0))
        self.assertEqual(rows[6], (6, 1, 10, 1, 3))
        self.assertEqual(rows[8], (8, 0, 0, 1, 1))

    def test_integer_values_and_invalid_encodings(self):
        self.assertEqual(mirror.canonical_value(4, object(), field="center"), 4)
        for value in (-1, 5, True, "1", 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                mirror.canonical_value(value, object(), field="center")
        with self.assertRaisesRegex(ValueError, "Unrecognized"):
            mirror.canonical_value(0, ClassLabel(["ambiguous"]), field="center")
        with self.assertRaisesRegex(ValueError, "Invalid encoded"):
            mirror.canonical_value(-1, ClassLabel(["tumor"]), field="label")

    def test_duplicate_identity_is_rejected_across_splits(self):
        raw = fixture()
        raw["validation"].rows[0]["image_id"] = raw["train"].rows[0]["image_id"]
        with self.assertRaisesRegex(ValueError, "Duplicate source image ID"):
            mirror._metadata_rows(raw, COUNTS)

    def test_wrong_counts_missing_classes_and_test_hospital_fail(self):
        with self.assertRaisesRegex(ValueError, "expected"):
            mirror._metadata_rows(fixture(), {**COUNTS, "train": 7})
        raw = fixture()
        raw["validation"].rows[1]["label"] = 0
        with self.assertRaisesRegex(ValueError, "both binary labels"):
            mirror._metadata_rows(raw, COUNTS)
        raw = fixture()
        raw["validation"].features["center"].names[0] = "test-center"
        with self.assertRaisesRegex(ValueError, "Unexpected hospital 2"):
            mirror._metadata_rows(raw, COUNTS)

    def test_subset_offsets_transform_and_test_sealing(self):
        dataset = mirror.CamelyonMirror.__new__(mirror.CamelyonMirror)
        dataset.raw, dataset.train_size = fixture(), 6
        rows = mirror._metadata_rows(dataset.raw, COUNTS)
        dataset.split_array = [row[4] for row in rows]
        dataset.y_array = [row[3] for row in rows]
        dataset.metadata_array = [row[1:4] for row in rows]
        subset = dataset.get_subset("id_val", transform=str.upper)
        self.assertEqual(subset.indices, list(range(8, 14)))
        self.assertEqual(subset[0], ("IMAGE-8", 1, (0, 0, 1)))
        with self.assertRaisesRegex(ValueError, "test remains sealed"):
            dataset.get_subset("test")


class DownloadTests(unittest.TestCase):
    def test_pinned_local_only_shards_and_stable_provenance(self):
        raw = fixture()
        download = Mock(side_effect=lambda **kw: "/fake/" + kw["filename"])
        load = Mock(side_effect=lambda _format, **kw: raw[kw["split"]])
        modules = {
            "datasets": SimpleNamespace(load_dataset=load),
            "huggingface_hub": SimpleNamespace(hf_hub_download=download),
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict("sys.modules", modules),
            patch.object(mirror, "EXPECTED_COUNTS", COUNTS),
            patch.object(mirror, "CamelyonMirror", side_effect=lambda *args: args),
        ):
            config = SimpleNamespace(
                data_dir=directory, dataset_version="1.0", hf_revision=mirror.MIRROR_REVISION
            )
            result = mirror.load_camelyon_mirror(config)
            first_bytes = result[2].read_bytes()
            self.assertTrue(result[3]["equivalence_not_verified"])
            expected_hub_cache = (
                Path(directory).resolve()
                / "community_mirror"
                / mirror.MIRROR_REVISION
                / "cache"
                / "hub"
            )
            expected_arrow_cache = Path(directory).resolve() / "arrow"
            # Hub paths must remain unchanged: users already downloaded the shards.
            for call in download.call_args_list:
                self.assertEqual(Path(call.kwargs["cache_dir"]), expected_hub_cache)
            # Arrow conversion needs a short cache root for Windows lock filenames.
            for call in load.call_args_list:
                self.assertEqual(Path(call.kwargs["cache_dir"]), expected_arrow_cache)

            self.assertEqual(len(download.call_args_list), 17)
            self.assertEqual(
                [call.kwargs["filename"] for call in download.call_args_list],
                [f"data/train-{i:05d}-of-00014.parquet" for i in range(14)]
                + [f"data/validation-{i:05d}-of-00003.parquet" for i in range(3)],
            )
            for call in download.call_args_list:
                self.assertEqual(call.kwargs["revision"], mirror.MIRROR_REVISION)
                self.assertEqual(call.kwargs["repo_id"], mirror.MIRROR_REPO)
                self.assertTrue(call.kwargs["local_files_only"])
                self.assertNotIn("test", call.kwargs["filename"])
            self.assertEqual(
                [call.kwargs["split"] for call in load.call_args_list], ["train", "validation"]
            )
            mirror.load_camelyon_mirror(config, download=True)
            self.assertEqual(first_bytes, result[2].read_bytes())
            self.assertFalse(download.call_args.kwargs["local_files_only"])
            self.assertEqual(json.loads(first_bytes)["split_counts"], COUNTS)
            self.assertIn(mirror.MIRROR_REVISION, str(result[2]))
            self.assertEqual(Path(result[2]).name, "metadata_provenance.json")

    def test_unreviewed_revision_is_rejected_before_import_or_network(self):
        config = SimpleNamespace(dataset_version="1.0", hf_revision="main")
        with self.assertRaisesRegex(ValueError, "pinned mirror revision"):
            mirror.load_camelyon_mirror(config, download=True)


if __name__ == "__main__":
    unittest.main()
