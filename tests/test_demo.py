"""CPU tests of the end-to-end demo: train, diagnose, repair from the diagnosis, verify."""

import importlib.util
import json
import random
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import runsleuth.demo as demo

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

if torch is not None:
    from camelyon_harness import CamelyonHarness
else:
    CamelyonHarness = unittest.TestCase


@unittest.skipIf(torch is None, "Install torch and torchvision for CPU integration tests")
class DemoTests(CamelyonHarness):
    # A healthy fine-tuning rate, below the library's 1e-3 ceiling.
    learning_rate = 1e-4

    def demo(self, fault, reference=None, **options):
        with self.patched_builders():
            path = demo.run_demo(
                self.reference,
                reference,
                self.output,
                fault=fault,
                epochs=2,
                requested_device="cpu",
                **options,
            )
        return json.loads(path.read_text(encoding="utf-8")), path

    def healthy_reference(self) -> Path:
        _, path = self.demo("clean")
        return path.parent / "faulty" / "run_report.json"

    def test_clean_run_needs_no_repair(self):
        report, _ = self.demo("clean")
        self.assertEqual(report["diagnosis"]["final"], "no_known_fault")
        self.assertTrue(report["diagnosis_correct"])
        self.assertIsNone(report["repair"])
        self.assertEqual(report["verification"]["decision"], "no_repair_applied")

    def test_each_fault_is_diagnosed_repaired_and_reverified(self):
        reference = self.healthy_reference()
        for fault in ("stale_head", "frozen_head", "high_learning_rate", "missing_optimizer_step"):
            with self.subTest(fault=fault):
                report, path = self.demo(fault, reference)
                expected = demo.EXPECTED_DIAGNOSIS[fault]
                self.assertEqual(report["diagnosis"]["final"], expected)
                self.assertTrue(report["diagnosis_correct"])
                self.assertEqual(report["repair"]["action"], demo.REPAIRS[expected])
                verification = report["verification"]
                self.assertEqual(verification["re_diagnosis"], "no_known_fault")
                self.assertTrue(verification["structure_verified"])
                self.assertEqual(
                    (verification["comparators"], verification["informational_comparators"]),
                    (["reference"], ["faulty"]),
                )
                self.assertEqual(
                    verification["decision"],
                    "accepted" if verification["performance_nonregression"] else "rejected",
                )
                self.assertTrue((path.parent / "repaired" / "run_report.json").is_file())
                if fault == "high_learning_rate":
                    self.assertEqual(
                        (
                            report["repair"]["learning_rate"],
                            report["repair"]["learning_rate_source"],
                        ),
                        (1e-4, "reference_config"),
                    )

    def test_wrong_diagnosis_applies_a_wrong_repair_that_is_rejected(self):
        reference = self.healthy_reference()
        real = demo.diagnose_run
        calls = []

        def misdiagnose(*args, **kwargs):
            result = real(*args, **kwargs)
            calls.append(result["final_diagnosis"])
            if len(calls) == 1:
                result = {**result, "final_diagnosis": "high_learning_rate"}
            return result

        with patch.object(demo, "diagnose_run", side_effect=misdiagnose):
            report, _ = self.demo("frozen_head", reference)
        self.assertEqual(calls, ["frozen_head", "frozen_head"])
        self.assertFalse(report["diagnosis_correct"])
        self.assertEqual(report["repair"]["action"], "restore_learning_rate")
        self.assertEqual(report["verification"]["re_diagnosis"], "frozen_head")
        self.assertEqual(report["verification"]["decision"], "rejected")

    def test_random_blind_choice_is_recorded_and_hidden_until_the_reveal(self):
        expected = random.Random(11).choice(demo.FAULTS)
        with self.patched_builders(), patch("builtins.print") as printed:
            path = demo.run_demo(
                self.reference,
                None,
                self.output,
                fault="random",
                blind=True,
                epochs=2,
                requested_device="cpu",
                choice_seed=11,
            )
        report = json.loads(path.read_text(encoding="utf-8"))
        lines = [str(call.args[0]) for call in printed.call_args_list if call.args]
        self.assertEqual((report["injected_fault"], report["fault_choice_seed"]), (expected, 11))
        self.assertIn("injected_fault=[hidden]", lines)
        self.assertTrue(lines[-2].startswith(f"reveal: injected={expected} "))

    def test_unknown_fault_is_rejected_before_any_work(self):
        with patch.object(demo, "load_reference") as load:
            with self.assertRaisesRegex(ValueError, "Unknown fault"):
                demo.run_demo(self.reference, None, self.output, fault="typo")
        load.assert_not_called()


@unittest.skipIf(importlib.util.find_spec("openai") is None, "Install the llm extra")
class DemoCommandLineTests(unittest.TestCase):
    def test_the_chosen_provider_reviews_the_demo(self):
        arguments = ["demo", "--baseline", "baseline", "--provider", "ollama"]
        with (
            patch.object(sys, "argv", arguments),
            patch(
                "runsleuth.llm_client.create_llm_client", return_value=("client", "qwen3:8b")
            ) as create,
            patch.object(demo, "run_demo") as run,
        ):
            demo.main()
        create.assert_called_once_with("ollama")
        options = run.call_args.kwargs
        self.assertEqual((options["client"], options["model_name"]), ("client", "qwen3:8b"))


if __name__ == "__main__":
    unittest.main()
