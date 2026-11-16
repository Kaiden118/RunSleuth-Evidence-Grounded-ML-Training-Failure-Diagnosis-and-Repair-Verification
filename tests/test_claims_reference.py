"""Tests for the reference program: it passes the screen, runs as an attempt, and its
estimates are supported by what it records."""

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

from test_claims_run import HAM, tiny_export

from runsleuth import claims_reference, claims_run, claims_task


class ReferenceTests(unittest.TestCase):
    def test_the_reference_passes_the_screen_and_is_written_once(self):
        source = claims_reference.REFERENCE.read_text(encoding="utf-8")
        self.assertEqual(claims_run.screen(source), [])
        with tempfile.TemporaryDirectory() as temporary:
            directory = claims_reference.write(Path(temporary), HAM.name)
            record = json.loads((directory / "attempt.json").read_text(encoding="utf-8"))
            self.assertEqual(
                (record["world"], record["model_name"], record["reference"]),
                (HAM.name, "reference", True),
            )
            self.assertEqual((directory / claims_task.PROGRAM).read_text(encoding="utf-8"), source)
            (directory / claims_task.PROGRAM).write_text("changed", encoding="utf-8")
            claims_reference.write(Path(temporary), HAM.name)
            self.assertEqual((directory / claims_task.PROGRAM).read_text(), "changed")

    @unittest.skipIf(torch is None, "Install torch and torchvision for the reference run")
    def test_the_reference_estimates_every_use_with_support_on_its_record(self):
        source = claims_reference.REFERENCE.read_text(encoding="utf-8")
        # One epoch, a short bootstrap, and no ImageNet weights, which a test host may lack
        # and the guard does not let a program fetch.
        quick = (
            source.replace("EPOCHS = 6\n", "EPOCHS = 1\n")
            .replace("BOOTSTRAP = 1000\n", "BOOTSTRAP = 50\n")
            .replace("resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)", "resnet18(weights=None)")
        )
        self.assertEqual(quick.count("weights=None"), 1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data" / HAM.name
            tiny_export(data)
            program = root / claims_task.PROGRAM
            program.write_text(quick, encoding="utf-8")
            result = claims_run.run_submission(program, data, HAM, root / "work", device="cpu")
            self.assertEqual(result["category"], "decided", result.get("problems"))
            self.assertEqual(
                result["reporting"], {"report.json": [], "report.md": [], "partition.json": []}
            )
            for use, found in result["partition"].items():
                self.assertEqual(found["warrant"], "supported", (use, found))
            source_fits = result["partition"]["new_source"]["fits"]
            self.assertEqual(len(source_fits), 3)
            for claim in result["claims"].values():
                self.assertNotIn("abstained", claim)
                self.assertTrue(claim["balanced_accuracy"]["interval_scored"])


if __name__ == "__main__":
    unittest.main()
