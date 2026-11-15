"""CPU tests for training programs written from scratch: the task statement and the
arrays a program is given, on the harness's smoke-test data."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

if torch is not None:
    import numpy as np

    from runsleuth import pipeline_workspace, standalone_task


@unittest.skipIf(torch is None, "Install torch and torchvision for the standalone tests")
class StandaloneTaskTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_the_statement_fixes_the_call_the_files_and_the_six_requirements(self):
        text = standalone_task.statement("camelyon")
        self.assertIn(
            "python train.py --data DIR --out DIR --seed N --epochs N "
            "[--stop-after K] [--resume] [--device cuda]",
            text,
        )
        for name in ("epoch_{n}.pt", "delivered.pt", "log.jsonl", *standalone_task.LOG_FIELDS):
            self.assertIn(f"`{name}`", text)
        for name in standalone_task.ARRAYS:
            self.assertIn(f"`{name}.npy`", text)
        self.assertIn("(count, 3, 96, 96) holding RGB patches", text)
        for number in range(1, 7):
            self.assertIn(f"\n{number}. ", text)
        self.assertIn("the same run made twice writes the same weights", text)
        self.assertIn("are already in torchvision's cache", text)
        self.assertEqual(standalone_task.STATEMENT_VERSION, 2)
        self.assertIn("trains\n   exactly as the uninterrupted run would have", text)
        self.assertIn("the earliest such epoch on a tie", text)
        for claim in standalone_task.CLAIMS:
            self.assertIn(f"`CLAIM: {claim}`", text)
        self.assertNotIn("{", text.replace("epoch_{n}.pt", ""))

    def test_only_the_dataset_with_flips_is_asked_to_flip(self):
        camelyon = standalone_task.statement("camelyon")
        ham = standalone_task.statement("ham10000")
        self.assertNotIn("flip", camelyon)
        self.assertIn("flip every image left-right and up-down", ham)
        self.assertIn("(count, 3, 128, 128) holding RGB images", ham)
        self.assertIn("(count, 3, 16, 16)", standalone_task.statement("ham10000", 16))
        with self.assertRaisesRegex(ValueError, "Unknown dataset"):
            standalone_task.statement("mnist")

    def test_the_export_keeps_the_training_hospitals_and_gives_the_bytes_back(self):
        data = pipeline_workspace.synthetic_pipeline_data(7)
        record = standalone_task.export(data, self.root)
        arrays = {name: np.load(self.root / f"{name}.npy") for name in standalone_task.ARRAYS}
        kept = [p for p in range(len(data.train)) if data.train.hospitals[p] != 1]
        self.assertEqual(len(kept), 24)
        self.assertEqual(arrays["train_images"].shape, (24, 3, 16, 16))
        self.assertEqual(arrays["train_images"].dtype, np.uint8)
        self.assertEqual(arrays["train_labels"].dtype, np.int64)
        self.assertEqual(arrays["train_labels"].tolist(), [data.train.labels[p] for p in kept])
        first = (data.train.image(kept[0]) * 255).round().to(torch.uint8).numpy()
        self.assertTrue((arrays["train_images"][0] == first).all())
        self.assertEqual(arrays["validation_images"].shape, (12, 3, 16, 16))
        self.assertEqual(arrays["validation_labels"].tolist(), list(data.validation.labels))
        self.assertEqual(
            record["train"], json.loads((self.root / "data.json").read_text())["train"]
        )
        self.assertEqual(record["train"]["samples"], 24)
        self.assertEqual(record["train"]["of"], 24)
        self.assertIsNone(record["train"]["subset_seed"])
        self.assertEqual(record["image_size"], 16)
        self.assertEqual(sorted(record["sha256"]), sorted(standalone_task.ARRAYS))

    def test_a_capped_export_is_a_seeded_subset_in_order(self):
        data = pipeline_workspace.synthetic_pipeline_data(7)
        whole = standalone_task.export(data, self.root / "whole")
        capped = standalone_task.export(data, self.root / "capped", max_train=10, subset_seed=3)
        again = standalone_task.export(data, self.root / "again", max_train=10, subset_seed=3)
        other = standalone_task.export(data, self.root / "other", max_train=10, subset_seed=4)
        self.assertEqual(capped["train"]["samples"], 10)
        self.assertEqual(capped["train"]["subset_seed"], 3)
        self.assertEqual(capped["sha256"], again["sha256"])
        self.assertNotEqual(capped["sha256"]["train_images"], other["sha256"]["train_images"])
        self.assertEqual(
            capped["sha256"]["validation_images"], whole["sha256"]["validation_images"]
        )
        images = np.load(self.root / "whole" / "train_images.npy").reshape(24, -1).tolist()
        positions = [
            images.index(image)
            for image in np.load(self.root / "capped" / "train_images.npy").reshape(10, -1).tolist()
        ]
        self.assertEqual(positions, sorted(positions))

    def test_a_tool_using_model_gets_the_statement_without_the_answer_format(self):
        whole = standalone_task.statement("camelyon")
        body = standalone_task.statement("camelyon", answer=False)
        self.assertTrue(whole.startswith(body))
        self.assertIn("## Your answer", whole)
        self.assertNotIn("## Your answer", body)
        self.assertNotIn("CLAIM", body)
        self.assertTrue(body.rstrip().endswith("the earliest such epoch on a tie."))

    def test_a_small_copy_is_a_seeded_subset_of_both_splits(self):
        data = pipeline_workspace.synthetic_pipeline_data(7)
        standalone_task.export(data, self.root / "whole")
        options = {"max_train": 10, "max_validation": 5}
        record = standalone_task.shrink(self.root / "whole", self.root / "small", **options)
        again = standalone_task.shrink(self.root / "whole", self.root / "again", **options)
        self.assertEqual((record["train"]["samples"], record["train"]["of"]), (10, 24))
        self.assertEqual((record["validation"]["samples"], record["validation"]["of"]), (5, 12))
        self.assertEqual(record["sha256"], again["sha256"])
        images = np.load(self.root / "small" / "validation_images.npy")
        labels = np.load(self.root / "small" / "validation_labels.npy")
        self.assertEqual((images.shape, images.dtype), ((5, 3, 16, 16), np.uint8))
        self.assertEqual(record["validation"]["positive"], int(labels.sum()))
        whole = np.load(self.root / "whole" / "validation_images.npy").reshape(12, -1).tolist()
        positions = [whole.index(image) for image in images.reshape(5, -1).tolist()]
        self.assertEqual(positions, sorted(positions))

    def test_ham_labels_come_from_the_malignant_diagnoses(self):
        data = pipeline_workspace.synthetic_ham_pipeline_data(7)
        record = standalone_task.export(data, self.root)
        labels = np.load(self.root / "train_labels.npy").tolist()
        expected = [int(dx in ("mel", "bcc", "akiec")) for dx in data.train.diagnoses]
        self.assertEqual(labels, expected)
        self.assertEqual(record["train"]["positive"], sum(expected))
        images = np.load(self.root / "train_images.npy")
        self.assertTrue((images[3] == data.train.image(3).numpy()).all())

    def test_the_commands_print_the_statement_and_write_the_data(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            standalone_task.main(["statement", "--dataset", "ham10000"])
        self.assertEqual(output.getvalue(), standalone_task.statement("ham10000"))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            standalone_task.main(
                ["export", "--dataset", "ham10000", "--synthetic", "--out", str(self.root)]
            )
        self.assertIn("24 of 24 training and 12 validation images, 16x16", output.getvalue())
        self.assertTrue((self.root / "data.json").is_file())


if __name__ == "__main__":
    unittest.main()
