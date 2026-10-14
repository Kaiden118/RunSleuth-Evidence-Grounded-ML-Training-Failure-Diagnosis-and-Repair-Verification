"""CPU tests for HAM10000 preparation and splits, on a small made-up copy of its files."""

import csv
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path

from runsleuth import ham10000_data
from runsleuth.ham10000_data import HamConfig

try:
    import numpy
    import torch
    from PIL import Image
except ModuleNotFoundError as error:
    if error.name not in ("numpy", "torch", "PIL"):
        raise
    torch = None

if torch is not None:
    from test_agent_workspace import tiny_source

    from runsleuth import agent_tasks, agent_workspace
    from runsleuth.agent_tasks import HEALTHY
    from runsleuth.signature_matching import diagnose, extract_evidence, load_library

SOURCES = {"near": 0, "far": 1, "sealed": 2, "other": 3}
# (lesion, source, diagnoses of its images)
LESIONS = [
    ("L1", "near", ["nv", "nv", "nv"]),
    ("L2", "near", ["mel", "mel"]),
    ("L3", "near", ["bkl"]),
    ("L4", "other", ["bcc", "bcc"]),
    ("L5", "other", ["nv"]),
    ("L6", "other", ["akiec", "akiec", "akiec"]),
    ("L7", "near", ["nv", "nv"]),
    ("L8", "other", ["df"]),
    ("F1", "far", ["mel"]),
    ("F2", "far", ["nv", "nv"]),
    ("F3", "far", ["bcc"]),
    ("F4", "far", ["nv"]),
    ("S1", "sealed", ["mel", "nv"]),
]


def jpeg(seed: int) -> bytes:
    """A 30x20 image: noise in the central 20x20 square, black on its sides.

    Every image's noise has the same distribution, so training and validation inputs
    look alike, as they do in the real data.
    """
    noise = numpy.random.RandomState(seed).randint(60, 200, size=(20, 20, 3), dtype=numpy.uint8)
    image = Image.new("RGB", (30, 20), (0, 0, 0))
    image.paste(Image.fromarray(noise), (5, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def made_up_ham10000(root: Path) -> tuple[HamConfig, dict[str, str]]:
    """HAM10000's three published files, small, with their MD5 sums."""
    download = root / "download"
    download.mkdir(parents=True)
    rows, number = [], 0
    for lesion, source, diagnoses in LESIONS:
        for dx in diagnoses:
            number += 1
            rows.append([lesion, f"ISIC_{number:07d}", dx, "histo", "50.0", "male", "back", source])
    with (download / "HAM10000_metadata.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["lesion_id", "image_id", "dx", "dx_type", "age", "sex", "localization", "dataset"]
        )
        writer.writerows(rows)
    for part, prefix in ((1, ""), (2, "HAM10000_images_part_2/")):
        with zipfile.ZipFile(download / f"HAM10000_images_part_{part}.zip", "w") as archive:
            for index, row in enumerate(rows):
                if index % 2 == part - 1:
                    archive.writestr(f"{prefix}{row[1]}.jpg", jpeg(index))
    published = {
        name: hashlib.md5((download / name).read_bytes()).hexdigest()
        for name in ham10000_data.PUBLISHED
    }
    config = HamConfig(
        data_dir=str(root),
        sources=SOURCES,
        ood_source="far",
        sealed_source="sealed",
        image_size=8,
        subset_seed=3,
        id_val_fraction=0.5,
        max_ood_val_samples=3,
        batch_size=4,
    )
    return config, published


def admit_training(number: int) -> bool:
    return number not in (SOURCES["far"], SOURCES["sealed"])


def by_image(image_id: str, _lesion_id: str) -> str:
    return image_id


@unittest.skipIf(torch is None, "Install torch, numpy and Pillow for the HAM10000 tests")
class Ham10000DataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config, self.published = made_up_ham10000(self.root)
        ham10000_data.prepare(self.config, published=self.published)
        self.data = ham10000_data.load(self.config)
        self.rows = self.data.rows

    def test_preparing_checks_the_files_and_stores_every_image_shrunk(self):
        self.assertEqual(self.data.images.shape, (22, 8, 8, 3))
        self.assertEqual(self.data.images.dtype, numpy.uint8)
        # The central square is kept: the black sides do not survive the crop.
        self.assertGreater(float(self.data.images[:, :, 0, :].mean()), 30.0)
        self.assertGreater(float(self.data.images[:, :, -1, :].mean()), 30.0)
        malignant = [row for row in self.rows if row["label"] == 1]
        self.assertEqual({row["dx"] for row in malignant}, {"mel", "bcc", "akiec"})
        self.assertEqual(len(malignant), 10)
        self.assertEqual(
            [row["image_id"] for row in self.rows], sorted(row["image_id"] for row in self.rows)
        )
        manifest = self.data.manifest
        self.assertEqual(
            (manifest["doi"], manifest["license"]), (ham10000_data.DOI, "CC BY-NC 4.0")
        )
        self.assertEqual(manifest["published_md5"], self.published)
        self.assertEqual(manifest["images"], 22)
        wrong = {**self.published, "HAM10000_metadata.csv": "0" * 32}
        with self.assertRaisesRegex(ValueError, "MD5 differs .* HAM10000_metadata.csv"):
            ham10000_data.prepare(self.config, published=wrong)
        unknown = replace(self.config, sources={"near": 0, "far": 1, "sealed": 2, "elsewhere": 3})
        with self.assertRaisesRegex(ValueError, "sources the config does not"):
            ham10000_data.prepare(unknown, published=self.published)
        with self.assertRaisesRegex(FileNotFoundError, "run prepare first"):
            ham10000_data.load(replace(self.config, image_size=16))

    def test_validation_is_fixed_and_takes_one_image_of_each_held_out_lesion(self):
        evaluation = ham10000_data.evaluation_split(self.config, self.rows)
        self.assertEqual(evaluation, ham10000_data.evaluation_split(self.config, self.rows))
        lesions = [self.rows[index]["lesion_id"] for index in evaluation["id"]]
        self.assertEqual(len(lesions), len(set(lesions)), "one image per lesion")
        self.assertTrue(lesions)
        self.assertEqual(
            {self.rows[index]["source"] for index in evaluation["id"]} - {"near", "other"}, set()
        )
        self.assertEqual(len(evaluation["ood"]), 3)
        self.assertEqual({self.rows[index]["source"] for index in evaluation["ood"]}, {"far"})
        # Another seed holds out other lesions.
        other = ham10000_data.evaluation_split(replace(self.config, subset_seed=4), self.rows)
        self.assertNotEqual(other["id"], evaluation["id"])

    def test_a_script_keyed_by_image_trains_on_the_other_images_of_validation_lesions(self):
        healthy = ham10000_data.training_split(self.config, self.rows, admit_training)
        leaky = ham10000_data.training_split(self.config, self.rows, admit_training, by_image)
        evaluation = ham10000_data.evaluation_split(self.config, self.rows)
        held_out = {self.rows[index]["lesion_id"] for index in evaluation["id"]}
        siblings = [
            index
            for index, row in enumerate(self.rows)
            if row["lesion_id"] in held_out and index not in evaluation["id"]
        ]
        self.assertTrue(siblings, "the made-up data must hold a lesion with several images")
        self.assertEqual(set(leaky) - set(healthy), set(siblings))
        self.assertFalse(set(healthy) & set(siblings))
        self.assertFalse(set(leaky) & set(evaluation["id"]), "the validation images never train")
        self.assertEqual(
            ham10000_data.holdouts(self.config, self.rows, healthy)["id"][
                "training_samples_in_evaluation_groups"
            ],
            0,
        )
        self.assertEqual(
            ham10000_data.holdouts(self.config, self.rows, leaky)["id"][
                "training_samples_in_evaluation_groups"
            ],
            len(siblings),
        )

    def test_admitting_the_validation_source_leaks_and_the_sealed_source_never_trains(self):
        everything = ham10000_data.training_split(self.config, self.rows, lambda _number: True)
        sources = {self.rows[index]["source"] for index in everything}
        self.assertEqual(sources, {"near", "far", "other"})
        evaluation = ham10000_data.evaluation_split(self.config, self.rows)
        self.assertFalse(set(everything) & set(evaluation["ood"]))
        ood = ham10000_data.holdouts(self.config, self.rows, everything)["ood"]
        far = [index for index in everything if self.rows[index]["source"] == "far"]
        self.assertEqual(ood["training_samples_in_evaluation_groups"], len(far))
        self.assertGreater(len(far), 0)
        healthy = ham10000_data.training_split(self.config, self.rows, admit_training)
        self.assertEqual(
            ham10000_data.holdouts(self.config, self.rows, healthy)["ood"][
                "training_samples_in_evaluation_groups"
            ],
            0,
        )
        summary = ham10000_data.summary(self.config, self.rows)
        self.assertEqual(summary["splits"]["training (keyed by lesion)"], len(healthy))
        self.assertEqual(summary["sources"]["far"], {"number": 1, "benign": 3, "malignant": 2})

    def test_the_harness_serves_the_images_and_reports_the_held_out_groups(self):
        build = agent_workspace.ham10000_task_data(self.config)
        data = build(admit_training, torch.device("cpu"), 7)
        images, labels = next(iter(data.train_loader))
        self.assertEqual(tuple(images.shape[1:]), (3, 8, 8))
        self.assertEqual(images.dtype, torch.float32)
        self.assertTrue(0.0 <= float(images.min()) and float(images.max()) <= 1.0)
        self.assertEqual(set(labels.tolist()) - {0, 1}, set())
        self.assertEqual(set(data.evaluation_loaders), {"id", "ood"})
        record = data.record
        self.assertEqual((record["source"], record["training_sources"]), ("ham10000", [0, 3]))
        self.assertEqual(record["provenance"]["license"], "CC BY-NC 4.0")
        self.assertEqual(data.holdouts["id"]["held_out_by"], "lesion")
        again = build(admit_training, torch.device("cpu"), 7).record
        self.assertEqual(again["training_indices_sha256"], record["training_indices_sha256"])
        leaky = build(admit_training, torch.device("cpu"), 7, by_image)
        self.assertGreater(leaky.holdouts["id"]["training_samples_in_evaluation_groups"], 0)
        self.assertGreater(leaky.record["training_samples"], record["training_samples"])
        with self.assertRaisesRegex(ValueError, "admits no image"):
            build(lambda _number: False, torch.device("cpu"), 7)

    def test_a_workspace_script_trains_on_it_and_a_leak_is_diagnosed(self):
        config_path = self.root / "config.json"
        values = {name: getattr(self.config, name) for name in HamConfig.__dataclass_fields__}
        config_path.write_text(json.dumps({**values, "image_size": 8}), encoding="utf-8")
        reports = {}
        for task in (HEALTHY, "validation_hospital_in_training"):
            workspace = self.root / "workspaces" / task
            workspace.mkdir(parents=True)
            (workspace / "train.py").write_text(tiny_source(task), encoding="utf-8")
            path = agent_workspace.run_workspace(
                workspace,
                self.root / "runs" / task,
                agent_workspace.data_builder("ham10000", config_path),
                epochs=2,
                seed=7,
                device=torch.device("cpu"),
            )
            reports[task] = json.loads(path.read_text(encoding="utf-8"))
        healthy, leaky = reports[HEALTHY], reports["validation_hospital_in_training"]
        self.assertEqual(healthy["status"], "completed")
        self.assertEqual(healthy["workspace"]["data"]["source"], "ham10000")
        self.assertIn("ood_validation_accuracy", healthy["final_metrics"])
        library = load_library()
        self.assertEqual(
            diagnose(extract_evidence(healthy, healthy), library)["diagnosis"], "no_known_fault"
        )
        # Admitting source 1 trains on the out-of-distribution source, and the run says so.
        self.assertEqual(
            diagnose(extract_evidence(leaky), library)["diagnosis"],
            "train_evaluation_group_overlap",
        )
        self.assertEqual(agent_tasks.task_source(HEALTHY).count("VALIDATION_HOSPITAL = 1"), 1)


if __name__ == "__main__":
    unittest.main()
