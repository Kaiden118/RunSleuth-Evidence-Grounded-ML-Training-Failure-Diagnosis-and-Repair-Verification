"""Tests for the replication and regime-matched probes of a claims-task program."""

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

from test_claims_run import HAM, NAMES, TINY, tiny_export

from runsleuth import claims_probe, claims_task


def table(directory):
    with (Path(directory) / claims_task.METADATA).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


class ForecastTests(unittest.TestCase):
    def test_the_forecast_is_the_mean_with_a_prediction_interval_over_folds(self):
        found = claims_probe.forecast([0.6, 0.7, 0.8])
        half = 4.303 * 0.1 * math.sqrt(1 + 1 / 3)
        self.assertAlmostEqual(found["value"], 0.7)
        self.assertAlmostEqual(found["sd"], 0.1)
        self.assertAlmostEqual(found["low"], 0.7 - half)
        self.assertEqual((found["high"], found["folds"]), (1.0, 3))
        self.assertEqual(
            claims_probe.forecast([0.65]), {"folds": 1, "value": 0.65, "low": None, "high": None}
        )
        self.assertIsNone(claims_probe.forecast([])["value"])

    def test_a_report_is_read_use_by_use(self):
        report = {
            "threshold": 0.5,
            "estimates": {
                NAMES[0]: {"balanced_accuracy": 0.8, "auroc": 0.9, "auroc_95": [0.85, 0.95]},
                NAMES[1]: {"not_estimable": "one source"},
            },
        }
        found = claims_probe.estimates_of(report, HAM)
        self.assertEqual(
            found[NAMES[0]],
            {
                "balanced_accuracy": {"estimate": 0.8, "interval": None},
                "auroc": {"estimate": 0.9, "interval": [0.85, 0.95]},
            },
        )
        self.assertEqual(found[NAMES[1]], {"not_estimable": "one source"})
        report["estimates"][NAMES[1]] = {"auroc": 2}
        self.assertIn("problems", claims_probe.estimates_of(report, HAM)[NAMES[1]])
        self.assertIn("problems", claims_probe.estimates_of(None, HAM)[NAMES[0]])


class FoldTests(unittest.TestCase):
    def test_every_development_site_is_sealed_in_one_fold_by_the_worlds_rule(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = tiny_export(root / "export")
            folds = claims_probe.build_folds(root / "export", HAM, root / "folds")
            given = table(root / "export" / claims_task.GIVEN)
            sites = sorted({line["source"] for line in given})
            self.assertEqual([site for site, _ in folds], sites)
            self.assertNotIn("vienna_dias", sites)
            for site, directory in folds:
                record = json.loads((directory / claims_task.RECORD).read_text(encoding="utf-8"))
                self.assertEqual(record["world"], f"{HAM.name}.fold-{site}")
                held_out = table(directory / claims_task.SEALED / "new_site")
                self.assertEqual({line["source"] for line in held_out}, {site})
                self.assertNotIn(site, {line["source"] for line in table(directory / "given")})
                units = json.loads((directory / claims_task.UNITS).read_text(encoding="utf-8"))
                parts = [set(units[part]) for part in claims_task.PARTS]
                self.assertFalse(parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])
                # Every fold row is a row of the world's given part.
                ids = {line["image_id"] for line in given}
                for part in ("given", "sealed/new_groups", "sealed/new_site"):
                    self.assertTrue({line["image_id"] for line in table(directory / part)} <= ids)
            self.assertLess(len(rows), 1000)
            made = (folds[0][1] / claims_task.RECORD).stat().st_mtime_ns
            claims_probe.build_folds(root / "export", HAM, root / "folds")
            self.assertEqual((folds[0][1] / claims_task.RECORD).stat().st_mtime_ns, made)


@unittest.skipIf(torch is None, "Install torch and torchvision for the probe runs")
class ProbeTests(unittest.TestCase):
    def test_a_probe_replicates_the_program_and_measures_it_on_every_fold(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tiny_export(root / "data" / HAM.name)
            output = root / "programs"
            directory = output / "tiny" / HAM.name / "neutral" / "attempt-1"
            directory.mkdir(parents=True)
            (directory / claims_task.PROGRAM).write_text(TINY, encoding="utf-8")
            attempt = {
                "id": "abc",
                "world": HAM.name,
                "dataset": "ham10000",
                "condition": "neutral",
                "model_name": "tiny",
                "attempt": 1,
            }
            (directory / "attempt.json").write_text(json.dumps(attempt), encoding="utf-8")
            lines = []
            (result,) = claims_probe.run_attempts(
                output,
                root / "data",
                root / "folds",
                root / "probes",
                device="cpu",
                progress=lines.append,
                seeds=(2,),
            )
            self.assertEqual(result["category"], "probed")
            (replicate,) = result["replication"]
            self.assertEqual(replicate["training"]["status"], "completed")
            self.assertEqual(replicate["estimates"][NAMES[1]]["balanced_accuracy"]["estimate"], 0.9)
            self.assertEqual(len(result["regime"]), 3)
            for fold in result["regime"]:
                self.assertEqual(fold["problems"], [])
                # Bright images are the malignant ones: every fold is measured perfect.
                self.assertEqual(fold["measured"]["new_site"]["balanced_accuracy"]["value"], 1.0)
                self.assertEqual(fold["estimates"][NAMES[1]]["auroc"]["estimate"], 0.95)
            source = result["forecast"]["new_source"]["balanced_accuracy"]
            self.assertEqual((source["value"], source["folds"]), (1.0, 3))
            self.assertAlmostEqual(source["own_estimate_excess"], 0.9 - 1.0)
            kept = json.loads((directory / claims_probe.PROBE_FILE).read_text(encoding="utf-8"))
            self.assertEqual(kept["forecast"], result["forecast"])
            self.assertTrue((root / "probes" / "abc" / claims_probe.PROBE_FILE).is_file())
            # The fold run trained on the fold's given part only.
            arguments = result["regime"][0]["training"]["arguments"]
            self.assertTrue(arguments[1].endswith(str(Path("fold-rosendahl") / "given")))
            self.assertEqual(
                claims_probe.run_attempts(output, root / "data", root / "folds", root / "probes"),
                [],
            )
            refused = root / "bad.py"
            refused.write_text("import subprocess\n", encoding="utf-8")
            found = claims_probe.probe(
                refused, root / "data" / HAM.name, HAM, root / "folds", root / "probes" / "bad"
            )
            self.assertEqual(found["category"], "refused")


if __name__ == "__main__":
    unittest.main()
