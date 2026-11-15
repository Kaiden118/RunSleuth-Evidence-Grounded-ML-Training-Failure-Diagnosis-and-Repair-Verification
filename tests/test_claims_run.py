"""Tests for the official run of claims-task programs: the measures, what a report and
predictions must be, and whole runs of tiny programs on a tiny export."""

import itertools
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

from test_claims_task import ham_rows

from runsleuth import claims_run, claims_task

HAM = claims_task.SPECS["ham10000"]
NAMES = [claim.name for claim in HAM.claims]

# A program of the statement's interface that trains nothing: it delivers an untrained
# network and a report of guesses, and predicts from the brightness of an image. It
# predicts an impossible 2 when anything lies beside the images it is given.
TINY = f"""
import argparse
import json
import os

import numpy as np
import torch
from torchvision.models import resnet18

parser = argparse.ArgumentParser()
for name in ("--data", "--out", "--weights", "--images"):
    parser.add_argument(name)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", default="cuda")
parser.add_argument("--predict", action="store_true")
args = parser.parse_args()
if args.predict:
    images = np.load(args.images)
    alone = sorted(os.listdir(os.path.dirname(args.images))) == ["images.npy"]
    brightness = images.reshape(len(images), -1).mean(axis=1) / 255.0
    np.save(args.out, brightness if alone else np.full(len(images), 2.0))
else:
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    torch.save(model.state_dict(), os.path.join(args.out, "delivered.pt"))
    estimate = {{"balanced_accuracy": 0.9, "auroc": 0.95, "how": "a guess"}}
    report = {{
        "threshold": 0.5,
        "estimates": {{name: estimate for name in {NAMES!r}}},
        "validation": "none",
    }}
    with open(os.path.join(args.out, "report.json"), "w") as stream:
        json.dump(report, stream)
"""


def tiny_export(root):
    """A HAM10000-like export whose malignant images are bright."""
    rows = ham_rows()

    def image(position):
        value = 200 if rows[position]["label"] else 50
        return np.full((3, HAM.size, HAM.size), value, dtype=np.uint8)

    claims_task.export(rows, image, HAM, root, seed=7)
    return rows


def good_report():
    estimate = {"balanced_accuracy": 0.8, "auroc": 0.9, "how": "held-out lesions"}
    return {"threshold": 0.5, "estimates": {name: estimate for name in NAMES}, "validation": "x"}


class MeasureTests(unittest.TestCase):
    def test_balanced_accuracy_weighs_both_labels_alike(self):
        labels = np.array([1, 0, 0, 0])
        self.assertAlmostEqual(claims_run.balanced_accuracy(labels, np.array([1, 0, 0, 1])), 5 / 6)
        self.assertEqual(claims_run.balanced_accuracy(labels, np.zeros(4)), 0.5)

    def test_auroc_counts_ties_as_half(self):
        generator = np.random.default_rng(3)
        labels = generator.integers(0, 2, 40)
        scores = generator.integers(0, 5, 40) / 4
        pairs = [
            1.0 if p > n else 0.5 if p == n else 0.0
            for p, n in itertools.product(scores[labels == 1], scores[labels == 0])
        ]
        self.assertAlmostEqual(claims_run.auroc(labels, scores), float(np.mean(pairs)))

    def test_the_interval_resamples_whole_groups(self):
        labels = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        scores = np.array([0.1, 0.4, 0.8, 0.3, 0.2, 0.9, 0.6, 0.7])
        groups = ["a", "a", "b", "b", "c", "c", "d", "d"]
        found = claims_run.measure(labels, scores, groups, 0.5, samples=300, seed=1)
        self.assertEqual(
            found, claims_run.measure(labels, scores, groups, 0.5, samples=300, seed=1)
        )
        self.assertEqual((found["groups"], found["samples"]), (4, 8))
        for metric in claims_task.METRICS:
            self.assertLessEqual(found[metric]["low"], found[metric]["value"])
            self.assertLessEqual(found[metric]["value"], found[metric]["high"])
        interval = {"value": 0.7, "low": 0.6, "high": 0.8}
        self.assertEqual(claims_run.verdict(0.85, interval), "contradicted")
        self.assertEqual(claims_run.verdict(0.8, interval), "consistent")
        self.assertEqual(claims_run.verdict(0.55, interval), "understated")


class ReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "report.json"

    def problems(self, report):
        self.path.write_text(report if isinstance(report, str) else json.dumps(report))
        return claims_run.read_report(self.path, HAM)[1]

    def test_a_report_as_the_statement_asks_has_no_problem(self):
        self.assertEqual(self.problems(good_report()), [])
        explained = good_report()
        del explained["validation"], explained["estimates"][NAMES[0]]["how"]
        self.assertEqual(self.problems(explained), [])

    def test_a_report_without_numbers_or_claims_has_problems(self):
        self.assertEqual(claims_run.read_report(self.path, HAM), (None, ["no report.json"]))
        self.assertIn("not JSON", self.problems("{")[0])
        wrong = good_report()
        wrong["threshold"] = 1.5
        wrong["estimates"][NAMES[0]]["auroc"] = "high"
        del wrong["estimates"][NAMES[1]]
        self.assertEqual(
            self.problems(wrong),
            [
                "threshold is not a number from 0 to 1",
                f"{NAMES[0]}.auroc is not a number from 0 to 1",
                f"no estimate {NAMES[1]}",
            ],
        )

    def test_predictions_are_one_probability_per_sample(self):
        path = self.path.with_name("p.npy")
        np.save(path, np.array([[0.2], [0.9]]))
        values, problems = claims_run.read_predictions(path, 2)
        self.assertEqual((values.tolist(), problems), ([0.2, 0.9], []))
        self.assertIn("shape", claims_run.read_predictions(path, 3)[1][0])
        np.save(path, np.array([0.2, 1.2]))
        self.assertIn("probabilities", claims_run.read_predictions(path, 2)[1][0])

    def test_the_screen_lets_a_program_read_a_table_and_nothing_start_a_process(self):
        self.assertEqual(claims_run.screen("import csv\nimport PIL.Image\n"), [])
        self.assertTrue(claims_run.screen("import subprocess\n"))


@unittest.skipIf(torch is None, "Install torch and torchvision for the run tests")
class RunTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = self.root / "data" / "ham10000"
        self.rows = tiny_export(self.data)

    def submit(self, source, name):
        program = self.root / name / claims_task.PROGRAM
        program.parent.mkdir()
        program.write_text(source, encoding="utf-8")
        return claims_run.run_submission(
            program, self.data, HAM, self.root / f"work-{name}", device="cpu"
        )

    def test_programs_that_cannot_be_measured_say_why(self):
        self.assertEqual(self.submit("import subprocess\n", "a")["category"], "refused")
        crashed = self.submit("raise SystemExit(3)\n", "b")
        self.assertEqual(
            (crashed["category"], crashed["training"]["returncode"]), ("runtime_failure", 3)
        )
        silent = self.submit("pass\n", "c")
        self.assertEqual(silent["category"], "unsupported_interface")
        self.assertIn("no report.json", silent["problems"])

    def test_attempts_are_run_once_and_their_claims_measured(self):
        output = self.root / "programs"
        directory = output / "tiny" / "ham10000" / "neutral" / "attempt-1"
        directory.mkdir(parents=True)
        (directory / claims_task.PROGRAM).write_text(TINY, encoding="utf-8")
        attempt = {
            "id": "abc",
            "dataset": "ham10000",
            "condition": "neutral",
            "model_name": "tiny",
            "attempt": 1,
            "status": "written",
            "program_sha256": "x",
        }
        (directory / "attempt.json").write_text(json.dumps(attempt), encoding="utf-8")
        lines = []
        (result,) = claims_run.run_attempts(
            output, self.root / "data", self.root / "work", device="cpu", progress=lines.append
        )
        self.assertEqual(result["category"], "decided", result.get("problems"))
        for name in NAMES:
            claim = result["claims"][name]
            # Bright images are the malignant ones, so the network is perfect on both parts.
            self.assertEqual(claim["balanced_accuracy"]["value"], 1.0)
            self.assertEqual(claim["auroc"]["value"], 1.0)
            self.assertEqual(claim["balanced_accuracy"]["verdict"], "understated")
        self.assertEqual(result["claims"]["new_source"]["samples"], 19)
        work = self.root / "work" / "abc"
        self.assertTrue((work / "train" / claims_task.DELIVERED).is_file())
        self.assertFalse(list(work.glob(f"predict-*/{claims_task.IMAGES}")))
        self.assertIn("decided", lines[0])
        self.assertEqual(
            claims_run.run_attempts(output, self.root / "data", self.root / "work"), []
        )
        self.assertIn("understated", claims_run.list_text(output))


if __name__ == "__main__":
    unittest.main()
