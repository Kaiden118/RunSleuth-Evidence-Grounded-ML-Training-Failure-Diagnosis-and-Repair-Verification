"""The committed example runs keep diagnosing as the README and the Docker check say.

The examples are a held-out frozen-head run (seed 85302029) and its clean reference.
"""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from runsleuth import diagnose_run
from runsleuth.camelyon_variant_runner import verify_repair

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def report(name: str) -> Path:
    return EXAMPLES / name / "run_report.json"


class ExampleTests(unittest.TestCase):
    def diagnose(self, name, *arguments):
        with tempfile.TemporaryDirectory() as directory:
            argv = ["diagnose", "--run", str(report(name)), *arguments]
            argv += ["--no-llm", "--output-dir", directory]
            with (
                patch.object(sys, "argv", argv),
                contextlib.redirect_stdout(io.StringIO()) as printed,
            ):
                diagnose_run.main()
            (saved,) = Path(directory).glob("diagnosis-*.json")
            return json.loads(saved.read_text(encoding="utf-8")), printed.getvalue()

    def test_the_frozen_head_is_found_with_the_reference_and_deferred_without(self):
        result, text = self.diagnose("frozen_head", "--reference", str(report("clean")))
        self.assertEqual(result["final_diagnosis"], "frozen_head")
        self.assertIn("Diagnosis: frozen_head", text)
        self.assertEqual(self.diagnose("frozen_head")[0]["final_diagnosis"], "pending_reference")

    def test_the_clean_example_is_healthy(self):
        for arguments in ((), ("--reference", str(report("clean")))):
            with self.subTest(reference=bool(arguments)):
                self.assertEqual(
                    self.diagnose("clean", *arguments)[0]["final_diagnosis"], "no_known_fault"
                )

    def test_the_fault_is_silent_in_validation_metrics(self):
        runs = {
            name: json.loads(report(name).read_text("utf-8")) for name in ("clean", "frozen_head")
        }
        gate = verify_repair(runs, "frozen_head", reference="clean", faulty=None)
        self.assertTrue(gate["performance_nonregression"])


if __name__ == "__main__":
    unittest.main()
