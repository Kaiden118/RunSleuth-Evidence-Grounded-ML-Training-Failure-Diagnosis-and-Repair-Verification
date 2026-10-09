"""Offline tests for the runsleuth command: diagnose, verify and the usage text."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from runsleuth import cli

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def path(name: str) -> str:
    return str(EXAMPLES / name / "run_report.json")


def load(name: str) -> dict:
    return json.loads(Path(path(name)).read_text(encoding="utf-8"))


def only_id_split(run: dict) -> dict:
    """A typical project logs one validation split."""
    metrics = {key: value for key, value in run["final_metrics"].items() if "ood" not in key}
    return {**run, "final_metrics": metrics}


class VerifyTests(unittest.TestCase):
    def verify(self, candidate, faulty=None, reference=None):
        reference = load("clean") if reference is None else reference
        return cli.verify_runs(candidate, reference, faulty, optimizer_name="AdamW")

    def test_the_repaired_example_is_accepted_and_the_faulty_run_is_informational(self):
        result = self.verify(load("frozen_head_repaired"), load("frozen_head"))
        self.assertEqual(
            (result["decision"], result["re_diagnosis"]), ("accepted", "no_known_fault")
        )
        self.assertEqual(result["validation_splits"], ["id", "ood"])
        self.assertEqual(
            (result["comparators"], result["informational_comparators"]),
            (["reference"], ["faulty"]),
        )
        # Gate v1 would have rejected this correct repair: it trails the faulty run's loss.
        self.assertFalse(all(check["passed"] for check in result["informational_checks"]))
        self.assertTrue(result["faulty_passes_gate_vs_reference"])
        self.assertIn("the fault was silent in metrics", cli.verification_text(result))

    def test_an_unrepaired_run_is_rejected_by_re_diagnosis_although_its_metrics_pass(self):
        result = self.verify(load("frozen_head"))
        self.assertEqual(result["decision"], "rejected")
        self.assertTrue(result["performance_nonregression"])
        self.assertEqual(
            result["reasons"], ["re-diagnosis against the reference found frozen_head"]
        )

    def test_a_regression_against_the_reference_is_rejected(self):
        candidate = load("frozen_head_repaired")
        candidate["final_metrics"]["ood_validation_accuracy"] -= 0.05
        result = self.verify(candidate)
        self.assertEqual(result["decision"], "rejected")
        self.assertEqual(len(result["reasons"]), 1)
        self.assertTrue(result["reasons"][0].startswith("ood accuracy_drop is 0.05"))

    def test_only_validation_splits_both_runs_logged_are_compared(self):
        result = self.verify(only_id_split(load("frozen_head_repaired")), load("frozen_head"))
        self.assertEqual((result["decision"], result["validation_splits"]), ("accepted", ["id"]))
        self.assertEqual(len(result["performance_checks"]), 2)
        self.assertEqual(len(result["informational_checks"]), 2)

        bare = {**load("frozen_head_repaired"), "final_metrics": {"validation_accuracy": 0.9}}
        result = self.verify(bare)
        self.assertEqual(result["decision"], "rejected")
        self.assertIn("no validation split", result["reasons"][-1])
        self.assertEqual(result["performance_checks"], [])


class CommandTests(unittest.TestCase):
    def run_command(self, *argv):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            try:
                cli.main(list(argv))
                code = 0
            except SystemExit as exit:
                code = exit.code
        return code, output.getvalue()

    def test_verify_writes_a_report_and_exits_1_when_it_rejects(self):
        with tempfile.TemporaryDirectory() as directory:
            options = ["--reference", path("clean"), "--output-dir", directory]
            code, text = self.run_command(
                "verify", "--candidate", path("frozen_head_repaired"), *options
            )
            self.assertEqual(code, 0)
            self.assertIn("Decision: accepted", text)
            (saved,) = Path(directory).glob("verification-*.json")
            report = json.loads(saved.read_text(encoding="utf-8"))
            self.assertEqual(report["inputs"]["candidate"], path("frozen_head_repaired"))
            code, text = self.run_command("verify", "--candidate", path("frozen_head"), *options)
        self.assertEqual(code, 1)
        self.assertIn("Reason: re-diagnosis against the reference found frozen_head", text)

    def test_diagnose_runs_the_diagnosis(self):
        with tempfile.TemporaryDirectory() as directory:
            code, text = self.run_command(
                "diagnose",
                "--run",
                path("frozen_head"),
                "--reference",
                path("clean"),
                "--no-llm",
                "--output-dir",
                directory,
            )
        self.assertEqual(code, 0)
        self.assertIn("Diagnosis: frozen_head", text)

    def test_usage_lists_commands_and_each_command_has_its_own_help(self):
        code, text = self.run_command("--help")
        self.assertEqual(code, 0)
        for command in cli.COMMANDS:
            self.assertIn(command, text)
        self.assertEqual(self.run_command()[0], 2)
        self.assertEqual(self.run_command("repair")[0], 2)
        code, text = self.run_command("verify", "--help")
        self.assertEqual(code, 0)
        self.assertIn("usage: runsleuth verify", text)


if __name__ == "__main__":
    unittest.main()
