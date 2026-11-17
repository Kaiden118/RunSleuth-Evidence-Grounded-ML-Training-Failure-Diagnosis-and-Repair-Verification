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

HAM = claims_task.spec("ham10000-vienna_dias")
NAMES = [claim.name for claim in HAM.claims]


# A program of the statement's interface that trains nothing: it delivers an untrained
# network and a report of guesses, and predicts from the brightness of an image. It
# predicts an impossible 2 when anything lies beside the images it is given. Its partition
# record trains on every row, and measures and chooses on the first lesion's images.
def tiny_program(abstain=()):
    return f"""
import argparse
import csv
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
    with open(os.path.join(args.data, "metadata.csv"), encoding="utf-8") as stream:
        table = list(csv.DictReader(stream))
    first = [int(line["row"]) for line in table if line["lesion_id"] == table[0]["lesion_id"]]
    estimate = {{
        "balanced_accuracy": 0.9, "balanced_accuracy_95": [0.85, 0.95],
        "auroc": 0.95, "auroc_95": [0.9, 0.99], "how": "a guess",
    }}
    abstain = {list(abstain)!r}
    estimates = {{
        name: {{"not_estimable": "no such data"}} if name in abstain else estimate
        for name in {NAMES!r}
    }}
    report = {{"threshold": 0.5, "estimates": estimates, "validation": "none"}}
    with open(os.path.join(args.out, "report.json"), "w") as stream:
        json.dump(report, stream)
    with open(os.path.join(args.out, "report.md"), "w") as stream:
        for section in {list(claims_task.SECTIONS)!r}:
            stream.write("## " + section + "\\n\\nText.\\n\\n")
    delivered = {{
        "name": "delivered",
        "train": [int(line["row"]) for line in table],
        "select": first,
        "estimates": {{name: first for name in {NAMES!r} if name not in abstain}},
    }}
    with open(os.path.join(args.out, "partition.json"), "w") as stream:
        json.dump({{"fits": [delivered]}}, stream)
"""


TINY = tiny_program()


def tiny_export(root):
    """A HAM10000-like export whose malignant images are bright."""
    rows = ham_rows()

    def image(position):
        value = 200 if rows[position]["label"] else 50
        return np.full((3, HAM.size, HAM.size), value, dtype=np.uint8)

    claims_task.export(rows, image, HAM, root, seed=7)
    return rows


def good_report():
    estimate = {
        "balanced_accuracy": 0.8,
        "balanced_accuracy_95": [0.7, 0.85],
        "auroc": 0.9,
        "auroc_95": [0.8, 0.95],
        "how": "held-out lesions",
    }
    return {
        "threshold": 0.5,
        "estimates": {name: dict(estimate) for name in NAMES},
        "validation": "x",
    }


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
        self.assertEqual(claims_run.interval_verdict([0.81, 0.9], interval), "contradicted")
        self.assertEqual(claims_run.interval_verdict([0.75, 0.95], interval), "consistent")
        self.assertEqual(claims_run.interval_verdict([0.4, 0.59], interval), "understated")


class ReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "report.json"

    def problems(self, report):
        self.path.write_text(report if isinstance(report, str) else json.dumps(report))
        return claims_run.read_report(self.path, HAM)[1]

    def test_a_report_as_the_statement_asks_has_no_problem_and_no_gap(self):
        self.assertEqual(self.problems(good_report()), [])
        self.assertEqual(claims_run.report_gaps(good_report(), HAM), [])

    def test_missing_intervals_and_explanations_are_gaps_not_problems(self):
        report = good_report()
        del report["validation"], report["estimates"][NAMES[0]]["how"]
        del report["estimates"][NAMES[0]]["auroc_95"]
        report["estimates"][NAMES[1]]["balanced_accuracy_95"] = [0.9, 0.95]
        report["estimates"][NAMES[1]]["auroc_95"] = [0.95, 0.9]
        self.assertEqual(self.problems(report), [])
        self.assertEqual(
            claims_run.report_gaps(report, HAM),
            [
                "no validation",
                f"{NAMES[0]} has no how",
                f"{NAMES[0]}.auroc_95 is not an interval",
                f"{NAMES[1]}.balanced_accuracy lies outside its interval",
                f"{NAMES[1]}.auroc_95 is not an interval",
            ],
        )

    def test_a_use_may_be_called_not_estimable_with_a_reason(self):
        report = good_report()
        report["estimates"][NAMES[1]] = {"not_estimable": "one source only"}
        self.assertEqual(self.problems(report), [])
        self.assertEqual(claims_run.report_gaps(report, HAM), [])
        report["estimates"][NAMES[1]] = {"not_estimable": " ", "auroc": 0.5}
        self.assertEqual(self.problems(report), [])
        self.assertEqual(
            claims_run.estimate_problems(report, HAM),
            {
                NAMES[0]: [],
                NAMES[1]: [
                    f"{NAMES[1]}.not_estimable gives no reason",
                    f"{NAMES[1]} is both estimated and not estimable",
                ],
            },
        )

    def test_the_markdown_report_needs_every_section_as_a_heading(self):
        path = self.path.with_name("report.md")
        self.assertEqual(claims_run.markdown_gaps(path), ["no report.md"])
        sections = [f"##  {section.upper()}  " for section in claims_task.SECTIONS[1:]]
        path.write_text("# Report\nIntended use: none.\n" + "\n".join(sections))
        self.assertEqual(claims_run.markdown_gaps(path), ["report.md has no section Intended use"])
        path.write_text("### 1. Intended use of the network\n" + "\n".join(sections))
        self.assertEqual(claims_run.markdown_gaps(path), [])

    def test_a_report_without_numbers_or_claims_has_problems(self):
        self.assertEqual(claims_run.read_report(self.path, HAM), (None, ["no report.json"]))
        self.assertIn("not JSON", self.problems("{")[0])
        wrong = good_report()
        wrong["threshold"] = 1.5
        wrong["estimates"][NAMES[0]]["auroc"] = "high"
        del wrong["estimates"][NAMES[1]]
        self.assertEqual(self.problems(wrong), ["threshold is not a number from 0 to 1"])
        self.assertEqual(
            claims_run.estimate_problems(wrong, HAM),
            {
                NAMES[0]: [f"{NAMES[0]}.auroc is not a number from 0 to 1"],
                NAMES[1]: [f"no estimate {NAMES[1]}"],
            },
        )
        # A use that cannot be read has no reporting gap of its own.
        self.assertEqual(claims_run.report_gaps(wrong, HAM), [])
        wrong["estimates"] = []
        self.assertEqual(
            self.problems(wrong),
            ["threshold is not a number from 0 to 1", "estimates is not an object"],
        )

    def test_an_estimate_is_judged_with_its_optimism_and_its_own_interval(self):
        measured = {
            "balanced_accuracy": {"value": 0.70, "low": 0.62, "high": 0.76},
            "auroc": {"value": 0.80, "low": 0.75, "high": 0.85},
        }
        estimate = {
            "balanced_accuracy": 0.80,
            "balanced_accuracy_95": [0.74, 0.86],
            "auroc": 0.86,
            "auroc_95": [0.90, 0.95],
        }
        found = claims_run.judged(estimate, measured)
        accuracy, area = found["balanced_accuracy"], found["auroc"]
        self.assertEqual((accuracy["verdict"], accuracy["material"]), ("contradicted", True))
        self.assertAlmostEqual(accuracy["optimism"], 0.10)
        # Its interval holds the estimate, misses the measured value and reaches the
        # measured interval.
        self.assertEqual(
            (
                accuracy["interval_scored"],
                accuracy["interval_covers"],
                accuracy["interval_verdict"],
            ),
            (True, False, "consistent"),
        )
        # An interval that does not hold its own estimate is not scored.
        self.assertEqual((area["verdict"], area["material"]), ("contradicted", True))
        self.assertEqual(
            (area["interval_scored"], area["interval_covers"], area["interval_verdict"]),
            (False, None, None),
        )
        # Above the interval by less than MATERIAL: contradicted, not materially optimistic.
        near = {
            "balanced_accuracy": {"value": 0.75, "low": 0.70, "high": 0.78},
            "auroc": measured["auroc"],
        }
        slight = claims_run.judged({"balanced_accuracy": 0.79, "auroc": 0.8}, near)
        self.assertEqual(
            (slight["balanced_accuracy"]["verdict"], slight["balanced_accuracy"]["material"]),
            ("contradicted", False),
        )
        self.assertEqual(slight["auroc"]["verdict"], "consistent")
        self.assertIsNone(slight["auroc"]["interval"])

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
        self.data = self.root / "data" / HAM.name
        self.rows = tiny_export(self.data)

    def submit(self, source, name):
        program = self.root / name / claims_task.PROGRAM
        program.parent.mkdir()
        program.write_text(source, encoding="utf-8")
        return claims_run.run_submission(
            program, self.data, HAM, self.root / f"work-{name}", device="cpu"
        )

    def test_a_use_called_not_estimable_is_measured_and_kept_as_abstained(self):
        result = self.submit(tiny_program(abstain=["new_source"]), "abstains")
        self.assertEqual(result["category"], "decided", result.get("problems"))
        source = result["claims"]["new_source"]
        self.assertEqual(source["abstained"], "no such data")
        self.assertEqual(source["balanced_accuracy"]["value"], 1.0)
        self.assertNotIn("estimate", source["balanced_accuracy"])
        self.assertEqual(sorted(result["partition"]), ["new_lesions"])
        silent = self.submit(TINY.replace('"partition.json"', '"other.json"'), "silent")
        self.assertEqual(silent["reporting"]["partition.json"], ["no partition.json"])
        self.assertEqual({found["warrant"] for found in silent["partition"].values()}, {"unknown"})
        attempt = {"model_name": "m", "dataset": "ham10000", "condition": "neutral", "attempt": 1}
        line = claims_run.result_line(attempt, result)
        self.assertIn("new_source not estimable, BA 1.000", line)

    def test_the_program_reads_its_copy_of_the_given_data_and_nothing_of_the_export(self):
        result = self.submit(TINY, "clean")
        self.assertEqual(result["category"], "decided", result.get("problems"))
        work = self.root / "work-clean"
        given = self.root / claims_run.GIVEN_COPIES / HAM.name
        self.assertEqual(result["training"]["arguments"][1], str(given.resolve()))
        self.assertEqual(sorted(p.name for p in given.iterdir()), ["images.npy", "metadata.csv"])
        self.assertEqual(
            sorted(p.name for p in given.parent.iterdir()), [HAM.name, f"{HAM.name}.json"]
        )
        self.assertEqual((work / "program" / claims_task.PROGRAM).read_text(encoding="utf-8"), TINY)
        # The copy is made once.
        made = (given / claims_task.IMAGES).stat().st_mtime_ns
        self.submit(TINY, "again")
        self.assertEqual((given / claims_task.IMAGES).stat().st_mtime_ns, made)
        sealed = self.data / claims_task.SEALED / "new_site" / claims_task.METADATA
        for name, lines in (
            ("labels", f"open({str(sealed)!r}, encoding='utf-8').read()\n"),
            ("record", f"open({str(self.data / claims_task.RECORD)!r}, encoding='utf-8').read()\n"),
            (
                "beside",
                "import os, sys\n"
                "if '--data' in sys.argv:\n"
                "    os.listdir(os.path.dirname(sys.argv[sys.argv.index('--data') + 1]))\n",
            ),
        ):
            refused = self.submit(lines + TINY, name)
            self.assertEqual(refused["category"], "runtime_failure", name)
            output = (self.root / f"work-{name}" / claims_run.OUTPUT_FILE).read_text(
                encoding="utf-8"
            )
            self.assertIn("outside what this program may read", output)

    def test_a_prediction_sees_the_images_and_the_weights_alone(self):
        looks = TINY.replace(
            "    images = np.load(args.images)\n",
            "    images = np.load(args.images)\n"
            "    where = os.path.dirname(os.path.dirname(os.path.dirname(args.weights)))\n"
            "    open(os.path.join(where, 'train', 'report.json')).read()\n",
        )
        self.assertNotEqual(looks, TINY)
        result = self.submit(looks, "looks")
        self.assertEqual(result["category"], "unsupported_interface")
        self.assertIn("prediction of new_groups: crashed", result["problems"])
        self.assertIn(
            "outside what this program may read", result["predictions"]["new_groups"]["output"]
        )
        self.assertEqual(
            result["claims"]["new_lesions"]["problems"][0], "prediction of new_groups: crashed"
        )
        for part in ("new_groups", "new_site"):
            place = self.root / "work-looks" / f"predict-{part}"
            self.assertEqual([p.name for p in place.iterdir()], [])

    def test_a_use_that_cannot_be_judged_leaves_the_other_judged(self):
        fails = TINY.replace(
            "    images = np.load(args.images)\n",
            "    images = np.load(args.images)\n    if len(images) == 19:\n        raise SystemExit(5)\n",
        )
        result = self.submit(fails, "fails")
        self.assertEqual(result["category"], "partial")
        self.assertEqual(
            result["problems"], ["prediction of new_site: crashed", "no predictions written"]
        )
        self.assertEqual(
            result["claims"]["new_lesions"]["balanced_accuracy"]["verdict"], "understated"
        )
        self.assertNotIn("balanced_accuracy", result["claims"]["new_source"])
        attempt = {
            "model_name": "m",
            "world": HAM.name,
            "dataset": "ham10000",
            "condition": "neutral",
            "attempt": 1,
        }
        self.assertIn("new_source not measured", claims_run.result_line(attempt, result))
        unreadable = TINY.replace(
            '    report = {"threshold": 0.5,',
            '    estimates["new_source"] = {"auroc": "high"}\n    report = {"threshold": 0.5,',
        )
        self.assertNotEqual(unreadable, TINY)
        result = self.submit(unreadable, "unreadable")
        self.assertEqual(result["category"], "partial")
        source = result["claims"]["new_source"]
        self.assertEqual(
            source["problems"],
            [
                "new_source.balanced_accuracy is not a number from 0 to 1",
                "new_source.auroc is not a number from 0 to 1",
            ],
        )
        # The use is measured all the same, without a verdict.
        self.assertEqual(source["balanced_accuracy"]["value"], 1.0)
        self.assertNotIn("verdict", source["balanced_accuracy"])
        self.assertEqual(sorted(result["partition"]), ["new_lesions"])
        self.assertIn("new_source not judged", claims_run.result_line(attempt, result))

    def test_the_interval_resamples_the_sealing_units_of_the_export(self):
        units = json.loads((self.data / claims_task.UNITS).read_text(encoding="utf-8"))
        self.assertEqual(
            self.submit(TINY, "lesions")["claims"]["new_source"]["groups"],
            len(set(units["new_site"])),
        )
        units["new_site"] = [f"unit-{n % 2}" for n in range(len(units["new_site"]))]
        (self.data / claims_task.UNITS).write_text(json.dumps(units), encoding="utf-8")
        self.assertEqual(self.submit(TINY, "merged")["claims"]["new_source"]["groups"], 2)

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
            "world": HAM.name,
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
            self.assertEqual(claim["balanced_accuracy"]["interval"], [0.85, 0.95])
            self.assertEqual(claim["balanced_accuracy"]["interval_verdict"], "understated")
        self.assertEqual(result["claims"]["new_source"]["samples"], 19)
        self.assertEqual(
            result["reporting"], {"report.json": [], "report.md": [], "partition.json": []}
        )
        # The record measures on images of a lesion it trains on, and chooses on them.
        for name in NAMES:
            found = result["partition"][name]
            self.assertEqual(found["fits"], ["delivered"])
            self.assertTrue(found["group_leakage"])
            self.assertTrue(found["selection_on_estimate_data"])
            self.assertFalse(found["estimate_not_of_the_delivered_network"])
        self.assertTrue(result["partition"]["new_source"]["new_site_from_known_sites"])
        self.assertEqual(
            {found["warrant"] for found in result["partition"].values()}, {"not_supported"}
        )
        facts = json.loads(
            (self.root / "work" / "abc" / claims_run.PROVENANCE_FILE).read_text(encoding="utf-8")
        )
        self.assertEqual(sorted(facts["uses"]), NAMES)
        self.assertNotIn("value", json.dumps(facts))
        work = self.root / "work" / "abc"
        self.assertTrue((work / "train" / claims_task.DELIVERED).is_file())
        for part in ("new_groups", "new_site"):
            self.assertEqual(
                [p.name for p in (work / f"predict-{part}").iterdir()], [claims_run.PREDICTIONS]
            )
        self.assertIn("decided", lines[0])
        # What the run prints holds no sealed number.
        self.assertIn("new_source estimated, measured", lines[0])
        self.assertNotIn("1.000", lines[0])
        self.assertNotIn("understated", claims_run.list_text(output))
        self.assertEqual(
            claims_run.run_attempts(output, self.root / "data", self.root / "work"), []
        )
        self.assertIn("understated", claims_run.list_text(output, results=True))


if __name__ == "__main__":
    unittest.main()
