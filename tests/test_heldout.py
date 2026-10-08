"""Offline tests for held-out seeds: the draw, the resumable batch and its records."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_rescore_repairs import stale_report
from test_signature_matching import SCENARIOS

from runsleuth import heldout as module
from runsleuth.rescore_repairs import rescore_report

# Variant -> scenario; repaired variants train like clean.
VARIANTS = {
    "frozen_head": {
        "clean": "clean",
        "stale_head": "stale_head",
        "frozen_head": "frozen_head",
        "frozen_head_repaired": "clean",
    },
    "config_faults": {
        "clean": "clean",
        "high_learning_rate": "high_learning_rate",
        "high_learning_rate_repaired": "clean",
        "missing_optimizer_step": "missing_optimizer_step",
        "missing_optimizer_step_repaired": "clean",
    },
    "vit": {"clean": "clean", "imagenet_head_kept_repaired": "clean"},
}


def verification(faulty, decision="accepted", silent=False):
    return {
        "decision": decision,
        "structure_verified": True,
        "informational_comparators": [faulty],
        "faulty_passes_gate_vs_reference": silent,
    }


VERIFICATIONS = {
    "frozen_head": verification("frozen_head", silent=True),
    "config_faults": {
        "high_learning_rate": verification("high_learning_rate"),
        "missing_optimizer_step": verification("missing_optimizer_step"),
    },
    "vit": {"imagenet_head_kept": verification("imagenet_head_kept", decision="rejected")},
}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


class FakeGit:
    def __init__(self, dirty="", changed=""):
        self.dirty, self.changed = dirty, changed

    def __call__(self, command, *arguments):
        return {"status": self.dirty, "rev-parse": "abc123", "diff": self.changed}[command]


class FakeTraining:
    """The sweep and the three experiment runners, writing minimal completed reports."""

    def __init__(self, failing=()):
        self.failing = set(failing)
        self.calls = []

    def sweep(self, reference_run, seeds):
        self.calls.append(("sweep", tuple(seeds)))
        cases = [
            {
                "seed": seed,
                "status": "completed",
                "baseline_run": f"artifacts/sweep/seed-{seed}/baseline",
                "training_report": {
                    "path": write(
                        f"artifacts/sweep/seed-{seed}/stale.json", stale_report()
                    ).as_posix()
                },
            }
            for seed in seeds
        ]
        return write("artifacts/sweep/sweep_summary.json", {"status": "completed", "cases": cases})

    def runners(self):
        def runner(experiment):
            def run(reference_run, output_dir, epochs, device):
                self.calls.append((experiment, reference_run.as_posix()))
                if experiment in self.failing:
                    raise RuntimeError("CUDA out of memory")
                write(output_dir / "run-1" / "environment.json", {"optimizer": "AdamW"})
                variants = {name: SCENARIOS[kind] for name, kind in VARIANTS[experiment].items()}
                report = {
                    "status": "completed",
                    "variants": variants,
                    "repair_verification": VERIFICATIONS[experiment],
                }
                return write(output_dir / "run-1" / "report.json", report)

            return run

        return {experiment: runner(experiment) for experiment in module.EXPERIMENTS}


class HeldoutTests(unittest.TestCase):
    seeds_path = Path("evaluations/heldout_seeds.json")

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = contextlib.chdir(temporary.name)
        directory.__enter__()
        self.addCleanup(directory.__exit__, None, None, None)

    def draw(self, git=None):
        draws = iter([7, 123, 123, 456])  # a development seed and a repeat are skipped
        return module.draw_seeds(
            2, self.seeds_path, randbelow=lambda _: next(draws), git=git or FakeGit()
        )

    def run_batch(self, training, git=None):
        with contextlib.redirect_stdout(io.StringIO()):
            path = module.run_heldout(
                self.seeds_path,
                Path("baseline"),
                sweep=training.sweep,
                runners=training.runners(),
                git=git or FakeGit(),
            )
        return json.loads(path.read_text(encoding="utf-8"))

    def test_draw_records_new_seeds_with_the_commit_and_frozen_components(self):
        record = self.draw()
        self.assertEqual(record["seeds"], [123, 456])
        self.assertEqual(record["code_commit"], "abc123")
        self.assertEqual(record["frozen"], module.frozen_components())
        self.assertEqual(record["frozen"]["llm_prompt_version"], 2)
        self.assertEqual(json.loads(self.seeds_path.read_text("utf-8")), record)
        with self.assertRaises(FileExistsError):
            self.draw()

    def test_draw_refuses_uncommitted_source_and_unsupported_counts(self):
        with self.assertRaisesRegex(RuntimeError, "Commit the source"):
            self.draw(FakeGit(dirty=" M src/runsleuth/diagnose_run.py"))
        self.assertFalse(self.seeds_path.exists())
        for count in (0, 4, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                module.draw_seeds(count, self.seeds_path, git=FakeGit())

    def test_batch_trains_every_seed_writes_records_and_resumes(self):
        self.draw()
        training = FakeTraining()
        state = self.run_batch(training)
        self.assertEqual(state["status"], "completed")
        self.assertFalse(state["source_changed_since_draw"])
        self.assertEqual(len(training.calls), 1 + 2 * len(module.EXPERIMENTS))
        self.assertIn(("vit", "artifacts/sweep/seed-456/baseline"), training.calls)

        repairs = json.loads(Path(state["results"]["repairs"]).read_text("utf-8"))
        faults = repairs["faults"]
        self.assertEqual(
            faults["frozen_head"], {"cases": 2, "accepted": 2, "faulty_passes_gate_vs_reference": 2}
        )
        self.assertEqual(faults["imagenet_head_kept"]["accepted"], 0)
        silent = sum(
            case["faulty_passes_gate_vs_reference"] for case in rescore_report(stale_report())
        )
        self.assertEqual(faults["stale_head"]["faulty_passes_gate_vs_reference"], 2 * silent)
        self.assertIn("drawn after the signatures", repairs["scope"])

        diagnosis = json.loads(Path(state["results"]["signature_matching"]).read_text("utf-8"))
        self.assertEqual(diagnosis["summary"]["with_reference"]["exact_matches"], 22)
        self.assertEqual(
            diagnosis["summary"]["reference_free"]["pending_reference_with_true_fault"], 2
        )
        text = module.summary_text(state)
        self.assertIn("diagnosis with_reference: 22/22 exact", text)
        self.assertIn("frozen_head", text)

        self.run_batch(training)
        self.assertEqual(len(training.calls), 1 + 2 * len(module.EXPERIMENTS))

    def test_a_failed_experiment_is_recorded_and_retried_alone(self):
        self.draw()
        state = self.run_batch(FakeTraining(failing={"vit"}))
        self.assertEqual(state["status"], "completed_with_errors")
        self.assertEqual(state["seeds"]["123"]["vit"]["error"]["message"], "CUDA out of memory")
        self.assertEqual(state["seeds"]["123"]["config_faults"]["status"], "completed")
        retry = FakeTraining()
        state = self.run_batch(retry, FakeGit(changed="src/runsleuth/camelyon_vit.py"))
        self.assertEqual([call[0] for call in retry.calls], ["vit", "vit"])
        self.assertEqual(state["status"], "completed")
        self.assertTrue(state["source_changed_since_draw"])

    def test_a_changed_frozen_component_blocks_the_batch(self):
        self.draw()
        training = FakeTraining()
        changed = {**module.frozen_components(), "llm_prompt_version": 3}
        with patch.object(module, "frozen_components", return_value=changed):
            with self.assertRaisesRegex(RuntimeError, "frozen component changed"):
                self.run_batch(training)
        self.assertEqual(training.calls, [])

    def test_cli_draw_prints_the_seeds_file(self):
        record = {"seeds": [123, 456], "code_commit": "abc123"}
        arguments = ["heldout", "draw", "--count", "2"]
        with (
            patch.object(sys, "argv", arguments),
            patch.object(module, "draw_seeds", return_value=record) as draw,
            contextlib.redirect_stdout(io.StringIO()) as printed,
        ):
            module.main()
        draw.assert_called_once_with(2, module.DEFAULT_SEEDS_FILE)
        self.assertIn("seeds=[123, 456] commit=abc123", printed.getvalue())


if __name__ == "__main__":
    unittest.main()
