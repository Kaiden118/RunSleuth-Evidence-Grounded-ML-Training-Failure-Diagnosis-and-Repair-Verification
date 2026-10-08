"""Offline tests for re-scoring recorded gate v1 repair verifications under gate v2."""

import contextlib
import copy
import io
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_camelyon_variant_runner import final_metrics

from runsleuth import rescore_repairs as module
from runsleuth.camelyon_variant_runner import performance_checks


def v1_verification(variants, candidate, comparators, structure=True):
    """A verification as gate v1 recorded it: every comparator gates."""
    checks = performance_checks(variants, candidate, comparators)
    passed = all(check["passed"] for check in checks)
    return {
        "decision": "accepted" if structure and passed else "rejected",
        "structure_verified": structure,
        "performance_checks": checks,
        "performance_nonregression": passed,
    }


def fault_report():
    """Per-fault shape (config faults, DeiT): a silent fault and a diverged one."""
    variants = {
        "clean": final_metrics(0.90, 0.30),
        "imagenet_head_kept": final_metrics(0.93, 0.25),
        "imagenet_head_kept_repaired": final_metrics(0.90, 0.30),
        "high_learning_rate": {"status": "diverged"},
        "high_learning_rate_repaired": final_metrics(0.90, 0.30),
    }
    verification = {}
    for fault, comparators in (
        ("imagenet_head_kept", ("clean", "imagenet_head_kept")),
        ("high_learning_rate", ("clean",)),
    ):
        candidate = f"{fault}_repaired"
        verification[fault] = {
            "candidate": candidate,
            **v1_verification(variants, candidate, comparators),
        }
    return {
        "status": "completed",
        "task": "fault_training",
        "seed": 7,
        "variants": variants,
        "repair_verification": verification,
    }


def stale_report(reference_run=""):
    """Single-repair shape of the stale-binding fixture: no candidate, no seed."""
    variants = {
        "clean": final_metrics(0.90, 0.30),
        "stale_head": final_metrics(0.89, 0.31),
        "stale_head_repaired": final_metrics(0.90, 0.30),
    }
    return {
        "status": "completed",
        "task": "rebind_verification",
        "reference_run": reference_run,
        "variants": variants,
        "repair_verification": v1_verification(
            variants, "stale_head_repaired", ("clean", "stale_head")
        ),
    }


class RescoreTests(unittest.TestCase):
    def test_silent_fault_rejected_by_v1_is_accepted_by_v2(self):
        silent, diverged = module.rescore_report(fault_report())
        self.assertEqual(
            (silent["fault"], silent["v1_decision"], silent["v2_decision"]),
            ("imagenet_head_kept", "rejected", "accepted"),
        )
        self.assertTrue(
            all(name.startswith("imagenet_head_kept:") for name in silent["v1_failed_checks"])
        )
        self.assertEqual(silent["v2_failed_checks"], [])
        self.assertTrue(silent["faulty_passes_gate_vs_reference"])
        # The diverged faulty run was skipped by v1 and stays out of v2.
        self.assertEqual(
            (diverged["v1_decision"], diverged["v2_decision"]), ("accepted", "accepted")
        )
        self.assertIsNone(diverged["faulty_passes_gate_vs_reference"])

    def test_single_repair_reports_take_the_faulty_run_from_their_comparators(self):
        (case,) = module.rescore_report(stale_report())
        self.assertEqual((case["fault"], case["candidate"]), ("stale_head", "stale_head_repaired"))
        self.assertEqual((case["v1_decision"], case["v2_decision"]), ("accepted", "accepted"))

    def test_a_structural_failure_stays_rejected(self):
        report = fault_report()
        report["repair_verification"]["imagenet_head_kept"].update(
            structure_verified=False, decision="rejected"
        )
        silent, _ = module.rescore_report(report)
        self.assertEqual(silent["v2_decision"], "rejected")

    def test_no_v1_acceptance_becomes_a_v2_rejection(self):
        generator = random.Random(0)
        for _ in range(200):
            variants = {
                name: final_metrics(generator.uniform(0.8, 1.0), generator.uniform(0.1, 0.5))
                for name in ("clean", "stale_head", "stale_head_repaired")
            }
            report = stale_report()
            report["variants"] = variants
            report["repair_verification"] = v1_verification(
                variants, "stale_head_repaired", ("clean", "stale_head")
            )
            (case,) = module.rescore_report(report)
            if case["v1_decision"] == "accepted":
                self.assertEqual(case["v2_decision"], "accepted", variants)

    def test_reports_that_cannot_be_rescored_are_refused(self):
        tampered = fault_report()
        tampered["repair_verification"]["imagenet_head_kept"]["performance_checks"][0]["passed"] = (
            False
        )
        already_v2 = fault_report()
        already_v2["repair_verification"]["imagenet_head_kept"]["gate"] = {"version": 2}
        incomplete = {**fault_report(), "status": "inconclusive"}
        for report, message in (
            (tampered, "disagree with the recorded metrics"),
            (already_v2, "already verified under gate v2"),
            (incomplete, "not completed"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                module.rescore_report(copy.deepcopy(report))

    def test_cli_writes_hashed_sources_and_prints_a_table(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline"
            baseline.mkdir()
            (baseline / "config.json").write_text('{"seed": 2026}', encoding="utf-8")
            paths = (root / "faults.json", root / "stale.json")
            paths[0].write_text(json.dumps(fault_report()), encoding="utf-8")
            paths[1].write_text(json.dumps(stale_report(str(baseline))), encoding="utf-8")
            output_path = root / "results" / "rescore.json"
            arguments = ["rescore", *map(str, paths), "--output", str(output_path)]
            printed = io.StringIO()
            with patch.object(sys, "argv", arguments), contextlib.redirect_stdout(printed):
                module.main()
            result = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual([case["seed"] for case in result["cases"]], [7, 7, 2026])
        self.assertEqual(
            (result["summary"]["v1_accepted"], result["summary"]["v2_accepted"]), (2, 3)
        )
        self.assertEqual(
            result["summary"]["changed"],
            ["fault_training seed 7 imagenet_head_kept: rejected -> accepted"],
        )
        self.assertEqual(len(result["reports"][0]["sha256"]), 64)
        self.assertIn("post hoc", result["scope"])
        self.assertIn("accepted: v1 2/3, v2 3/3", printed.getvalue())


if __name__ == "__main__":
    unittest.main()
