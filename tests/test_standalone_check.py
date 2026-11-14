"""CPU tests for the checks of a program from outside: the screen, the verdicts on
made-up runs, and whole checks of a tiny program and of faulty edits of it."""

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

if torch is not None:
    from torch import nn

    from runsleuth import pipeline_workspace, standalone_check, standalone_task

# A program for the statement's interface with a network small enough for the tests.
TINY = """\
import argparse
import json
import os

import numpy as np
import torch
from torch import nn


def build():
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(4, 2),
    )


def prepare(images):
    images = torch.from_numpy(images).float() / 255
    mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    return (images - mean) / std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--out")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    load = lambda name: np.load(os.path.join(args.data, name + ".npy"))
    images, labels = prepare(load("train_images")), torch.from_numpy(load("train_labels"))
    held, targets = prepare(load("validation_images")), torch.from_numpy(load("validation_labels"))
    torch.manual_seed(args.seed)
    model = build()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_function = nn.CrossEntropyLoss()
    start, best = 1, -1.0
    state_path = os.path.join(args.out, "state.pt")
    if args.resume:
        state = torch.load(state_path)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start, best = state["epoch"] + 1, state["best"]
    for epoch in range(start, (args.stop_after or args.epochs) + 1):
        generator = torch.Generator().manual_seed(1000 * args.seed + epoch)
        order = torch.randperm(len(labels), generator=generator)
        rate = optimizer.param_groups[0]["lr"]
        model.train()
        for first in range(0, len(order), 8):
            batch = order[first : first + 8]
            optimizer.zero_grad()
            loss_function(model(images[batch]), labels[batch]).backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            logits = model(held)
        accuracy = float((logits.argmax(1) == targets).float().mean())
        torch.save(model.state_dict(), os.path.join(args.out, f"epoch_{epoch}.pt"))
        with open(os.path.join(args.out, "log.jsonl"), "a") as log:
            entry = {"epoch": epoch, "learning_rate": rate, "validation_accuracy": accuracy}
            entry["validation_loss"] = float(loss_function(logits, targets))
            log.write(json.dumps(entry) + "\\n")
        if accuracy > best:
            best = accuracy
            torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best": best,
            },
            state_path,
        )
    if not args.stop_after:
        torch.save(torch.load(os.path.join(args.out, "best.pt")), os.path.join(args.out, "delivered.pt"))


main()
"""
EDITS = {
    "optimizer_not_restored": ('        optimizer.load_state_dict(state["optimizer"])\n', ""),
    # The learning rates are restored, Adam's moments are not.
    "moments_not_restored": (
        '        optimizer.load_state_dict(state["optimizer"])\n',
        '        optimizer.load_state_dict({**state["optimizer"], "state": {}})\n',
    ),
    "scheduler_not_restored": ('        scheduler.load_state_dict(state["scheduler"])\n', ""),
    "shuffled_by_the_clock": (
        "torch.Generator().manual_seed(1000 * args.seed + epoch)",
        'torch.Generator().manual_seed(int.from_bytes(os.urandom(4), "little"))',
    ),
    "resume_fails": ("        state = torch.load(state_path)\n", "        state = {}['state']\n"),
    "nothing_delivered": ("    if not args.stop_after:\n", "    if False:\n"),
    "last_epoch_delivered": ('"best.pt")), os.path.join', '"epoch_3.pt")), os.path.join'),
}


def tiny_model():
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, 2),
    )


def made_up_run(correct, delivered, reported=None, epochs=None, rates=None, last=0.0):
    """A run by what the checker measured: epoch n has the weights "w<n>" unless named,
    was trained at the rate 0.1 unless given, and the last weights are the number last."""
    names = epochs or [f"w{n}" for n in range(1, len(correct) + 1)]
    accuracies = reported or [count / 100 for count in correct]
    rates = rates or [0.1] * len(correct)
    return {
        "last_state": {"w": torch.tensor([last])},
        "epochs": [
            {"sha256": name, "correct": count} for name, count in zip(names, correct, strict=True)
        ],
        "delivered": {
            "sha256": delivered,
            "correct": dict(zip(names, correct, strict=True)).get(delivered),
        },
        "log": [
            {"epoch": n, "validation_accuracy": accuracy, "learning_rate": rate}
            for n, (accuracy, rate) in enumerate(zip(accuracies, rates, strict=True), start=1)
        ],
        "log_order": None,
    }


@unittest.skipIf(torch is None, "Install torch and torchvision for the standalone tests")
class ScreenTests(unittest.TestCase):
    def test_a_training_program_may_run(self):
        self.assertEqual(standalone_check.screen(TINY), [])
        allowed = "import shutil\nfrom urllib.parse import urlparse\nimport torchvision.models\n"
        self.assertEqual(standalone_check.screen(allowed), [])

    def test_processes_the_network_and_tree_removal_are_refused(self):
        cases = {
            "import subprocess\n": "import of 'subprocess' is not allowed",
            "import urllib.request\n": "import of 'urllib.request' is not allowed",
            "from os import system\n": "import of 'system' is not allowed",
            "import os\nos.system('dir')\n": "attribute 'system' is not allowed",
            "import shutil\nshutil.rmtree('.')\n": "attribute 'rmtree' is not allowed",
            "exec('x = 1')\n": "'exec' is not allowed",
            "from . import helper\n": "relative imports are not allowed",
            "def broken(:\n": "syntax error",
        }
        for source, reason in cases.items():
            with self.subTest(source=source):
                (problem,) = standalone_check.screen(source)
                self.assertIn(reason, problem)
        self.assertIn("longer than", standalone_check.screen("x = 1\n" * 10_000)[0])


@unittest.skipIf(torch is None, "Install torch and torchvision for the standalone tests")
class VerdictTests(unittest.TestCase):
    def test_delivery_holds_for_the_earliest_best_epoch(self):
        verdict = standalone_check.delivery_verdict(made_up_run([70, 90, 90, 80], "w2"))
        self.assertEqual(verdict["status"], "holds")
        self.assertEqual((verdict["best_epoch"], verdict["delivered_epochs"]), (2, [2]))
        self.assertFalse(verdict["best_is_last"])
        self.assertNotIn("history", verdict)

    def test_a_worse_epoch_a_later_tie_and_other_weights_are_told_apart(self):
        delivery = standalone_check.delivery_verdict
        self.assertEqual(delivery(made_up_run([70, 90, 85, 80], "w4"))["status"], "violated")
        self.assertEqual(delivery(made_up_run([70, 90, 90, 80], "w3"))["status"], "violated_tie")
        self.assertEqual(delivery(made_up_run([70, 90, 85, 80], "other"))["status"], "no_epoch")

    def test_equal_weights_of_two_epochs_count_as_the_earliest(self):
        run = made_up_run([70, 90, 90], "same", epochs=["w1", "same", "same"])
        verdict = standalone_check.delivery_verdict(run)
        self.assertEqual((verdict["status"], verdict["delivered_epochs"]), ("holds", [2, 3]))

    def test_the_programs_own_best_epoch_leaves_delivery_undecided(self):
        # The checker counts epoch 2 best, the program's log epoch 3, which it delivered.
        run = made_up_run([70, 90, 89], "w3", reported=[0.70, 0.899, 0.90])
        self.assertEqual(standalone_check.delivery_verdict(run)["status"], "undecided")
        # Delivering an epoch that neither ranking puts first is a violation all the same.
        run = made_up_run([70, 90, 89], "w1", reported=[0.70, 0.899, 0.90])
        self.assertEqual(standalone_check.delivery_verdict(run)["status"], "violated")
        # A log without one line per epoch is no ranking: the checker's own decides.
        run = {**made_up_run([70, 90, 89], "w3"), "log_order": "the log's epochs are [1, 1, 2]"}
        verdict = standalone_check.delivery_verdict(run)
        self.assertEqual((verdict["status"], verdict["reported_best_epoch"]), ("violated", None))

    def test_the_history_of_a_resumed_run(self):
        # Best before the stop, the best of the later epochs delivered: the stricter event.
        forgot = standalone_check.delivery_verdict(made_up_run([70, 92, 80, 88, 85], "w4"), 2)
        self.assertEqual(forgot["status"], "violated")
        self.assertEqual(
            forgot["history"],
            {
                "best_before_stop": True,
                "strictly": True,
                "later_best_epoch": 4,
                "later_best_delivered": True,
            },
        )
        kept = standalone_check.delivery_verdict(made_up_run([70, 92, 80, 88, 85], "w2"), 2)
        self.assertEqual(kept["status"], "holds")
        self.assertFalse(kept["history"]["later_best_delivered"])
        late = standalone_check.delivery_verdict(made_up_run([70, 80, 85, 93, 90], "w4"), 2)
        self.assertEqual((late["status"], late["history"]["best_before_stop"]), ("holds", False))
        tie = standalone_check.delivery_verdict(made_up_run([70, 90, 85, 90, 80], "w4"), 2)
        self.assertEqual(tie["status"], "violated_tie")
        self.assertEqual(
            (tie["history"]["best_before_stop"], tie["history"]["strictly"]), (True, False)
        )

    def test_the_same_run_twice_must_write_the_same_weights(self):
        straight = made_up_run([70, 90, 80], "w2")
        same = standalone_check.repeat_verdict(straight, made_up_run([70, 90, 80], "w2"))
        self.assertEqual(
            same,
            {
                "status": "holds",
                "first_difference": None,
                "largest_difference": 0.0,
                "learning_rates_agree": True,
            },
        )
        other = made_up_run([70, 90, 81], "w2", epochs=["w1", "x2", "x3"], last=0.25)
        verdict = standalone_check.repeat_verdict(straight, other)
        self.assertEqual((verdict["status"], verdict["first_difference"]), ("violated", 2))
        self.assertEqual(verdict["largest_difference"], 0.25)

    def test_restart_is_decided_by_the_log_before_the_weights(self):
        restart = standalone_check.restart_verdict
        straight = made_up_run([70, 90, 80], "w2", rates=[0.1, 0.075, 0.025])
        repeats = standalone_check.repeat_verdict(straight, straight)
        wanders = {**repeats, "status": "violated"}
        same = made_up_run([70, 90, 80], "w2", rates=[0.1, 0.075, 0.025 + 1e-17])
        self.assertEqual(restart(straight, same, repeats)["status"], "holds")
        self.assertIsNone(restart(straight, same, repeats)["witness"])
        # Other weights at the end: a violation for a program that repeats itself only.
        ends = ["w1", "w2", "y3"]
        moved = made_up_run([70, 90, 79], "w2", epochs=ends, rates=[0.1, 0.075, 0.025], last=0.5)
        verdict = restart(straight, moved, repeats)
        self.assertEqual((verdict["status"], verdict["witness"]), ("violated", "weights"))
        self.assertEqual((verdict["first_difference"], verdict["largest_difference"]), (3, 0.5))
        self.assertEqual(restart(straight, moved, wanders)["status"], "undecided")
        # The schedule one epoch behind shows in the log, whatever the weights do.
        behind = made_up_run([70, 90, 79], "w2", epochs=ends, rates=[0.1, 0.075, 0.075], last=0.5)
        for repeat in (repeats, wanders):
            verdict = restart(straight, behind, repeat)
            self.assertEqual(
                (verdict["status"], verdict["witness"]), ("violated", "learning_rates")
            )
        # Learning rates that already differ between two uninterrupted runs prove nothing.
        unsteady = {**wanders, "learning_rates_agree": False}
        self.assertEqual(restart(straight, behind, unsteady)["status"], "undecided")
        # An epoch logged twice by the resumed run alone.
        twice = {**same, "log_order": "the log's epochs are [1, 2, 2, 3], not 1 to 3"}
        verdict = restart(straight, twice, wanders)
        self.assertEqual((verdict["status"], verdict["witness"]), ("violated", "log_epochs"))


@unittest.skipIf(torch is None, "Install torch and torchvision for the standalone tests")
class CheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        standalone_task.export(pipeline_workspace.synthetic_pipeline_data(7), cls.root / "data")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def check(self, name, **options):
        source = TINY
        if name in EDITS:
            old, new = EDITS[name]
            self.assertEqual(source.count(old), 1)
            source = source.replace(old, new)
        program = self.root / f"{name}.py"
        program.write_text(source, encoding="utf-8")
        return standalone_check.check_program(
            program,
            self.root / "data",
            self.root / "work" / name,
            seed=3,
            epochs=3,
            device="cpu",
            model_factory=tiny_model,
            **options,
        )

    def names(self):
        return {"model_name": "m", "dataset": "camelyon", "attempt": 1, "claim": "meets"}

    def test_a_program_that_resumes_exactly_holds_on_its_runs(self):
        record = self.check("healthy")
        self.assertEqual(record["category"], "decided")
        self.assertEqual(record["screen"], [])
        self.assertEqual(record["stops"], [1, 2])
        self.assertEqual(
            list(record["trajectories"]), ["straight", "repeat", "resumed-1", "resumed-2"]
        )
        self.assertEqual(record["repeat"]["status"], "holds")
        self.assertEqual(record["restart"]["status"], "holds")
        self.assertEqual(sorted(record["restart"]["stops"]), ["1", "2"])
        for verdict in record["restart"]["stops"].values():
            self.assertEqual((verdict["status"], verdict["witness"]), ("holds", None))
            self.assertEqual(verdict["largest_difference"], 0.0)
            self.assertTrue(verdict["learning_rates_agree"])
        straight = record["delivery"]["straight"]
        self.assertEqual(straight["status"], "holds")
        for stop, verdict in record["delivery"]["resumed"].items():
            self.assertEqual(verdict["status"], "holds")
            self.assertEqual(verdict["correct"], straight["correct"])
            self.assertEqual(
                verdict["history"]["best_before_stop"], straight["best_epoch"] <= int(stop)
            )
        resumed = record["trajectories"]["resumed-1"]
        self.assertEqual(resumed["calls"][0]["arguments"][-2:], ["--stop-after", "1"])
        self.assertEqual(resumed["calls"][1]["arguments"][-1], "--resume")
        self.assertIn("epoch_1.pt", resumed["files_after_stop"])
        self.assertNotIn("delivered.pt", resumed["files_after_stop"])
        self.assertNotIn("epoch_2.pt", resumed["files_after_stop"])
        self.assertIn("epoch_2.pt", record["trajectories"]["resumed-2"]["files_after_stop"])
        # The log's accuracies are the checker's own counts over the validation samples.
        run = record["trajectories"]["straight"]
        logged = [round(entry["validation_accuracy"] * 12) for entry in run["log"]]
        self.assertEqual(logged, [epoch["correct"] for epoch in run["epochs"]])
        self.assertEqual(record["validation_samples"], 12)
        self.assertFalse((self.root / "work" / "healthy").exists())
        self.assertEqual(standalone_check.findings(record), [])
        json.dumps(record)

    def test_an_optimizer_left_behind_shows_in_the_log(self):
        # The new optimizer starts again at the initial learning rate, which the log shows.
        record = self.check("optimizer_not_restored", stops=[1])
        verdict = record["restart"]["stops"]["1"]
        self.assertEqual((verdict["status"], verdict["witness"]), ("violated", "learning_rates"))
        self.assertEqual(verdict["first_difference"], 2)

    def test_moments_left_behind_show_in_the_weights_alone(self):
        record = self.check("moments_not_restored", stops=[1])
        self.assertEqual(record["category"], "decided")
        self.assertEqual(record["repeat"]["status"], "holds")
        verdict = record["restart"]["stops"]["1"]
        self.assertEqual((verdict["status"], verdict["witness"]), ("violated", "weights"))
        self.assertEqual(verdict["first_difference"], 2)
        self.assertGreater(verdict["largest_difference"], 0)
        self.assertTrue(verdict["learning_rates_agree"])
        self.assertEqual(record["restart"]["status"], "violated")
        self.assertEqual(standalone_check.findings(record), ["restart (weights)"])

    def test_a_schedule_left_behind_shows_in_the_log(self):
        record = self.check("scheduler_not_restored", stops=[1])
        verdict = record["restart"]["stops"]["1"]
        self.assertEqual((verdict["status"], verdict["witness"]), ("violated", "learning_rates"))
        self.assertFalse(verdict["learning_rates_agree"])
        self.assertEqual(standalone_check.findings(record), ["restart (learning_rates)"])

    def test_a_program_that_does_not_repeat_itself_violates_repeat_and_leaves_restart_undecided(
        self,
    ):
        record = self.check("shuffled_by_the_clock", stops=[1])
        self.assertEqual(record["category"], "decided")
        self.assertEqual(record["repeat"]["status"], "violated")
        self.assertEqual(record["repeat"]["first_difference"], 1)
        self.assertTrue(record["repeat"]["learning_rates_agree"])
        self.assertEqual(record["restart"]["status"], "undecided")
        self.assertIn(record["delivery"]["straight"]["status"], ("holds", "undecided"))
        self.assertEqual(standalone_check.findings(record)[0], "repeat")

    def test_a_resume_that_fails_is_told_from_a_run_that_fails(self):
        record = self.check("resume_fails")
        self.assertEqual(record["category"], "resume_failure")
        self.assertEqual(list(record["trajectories"]), ["straight", "repeat", "resumed-1"])
        resumed = record["trajectories"]["resumed-1"]
        self.assertEqual([call["status"] for call in resumed["calls"]], ["completed", "crashed"])
        self.assertIn("KeyError", resumed["calls"][1]["output"])
        self.assertEqual(record["restart"], {"status": None, "stops": {}})
        self.assertEqual(record["repeat"]["status"], "holds")
        self.assertEqual(record["delivery"]["straight"]["status"], "holds")
        self.assertEqual(standalone_check.findings(record), ["resume_failure"])
        text = standalone_check.line({**record, **self.names()})
        self.assertIn("resume_failure (resumed-1: completed crashed); repeat holds", text)
        self.assertIn("found: resume_failure; claims meets", text)

    def test_missing_files_are_an_unsupported_interface_and_end_the_check(self):
        record = self.check("nothing_delivered")
        self.assertEqual(record["category"], "unsupported_interface")
        self.assertEqual(list(record["trajectories"]), ["straight"])
        self.assertEqual(
            record["trajectories"]["straight"]["problems"], ["delivered.pt is missing"]
        )
        self.assertEqual(record["delivery"], {"straight": None, "resumed": {}})
        self.assertEqual(standalone_check.findings(record), ["unsupported_interface"])

    def test_the_last_epoch_delivered_is_judged_by_the_checkers_counts(self):
        record = self.check("last_epoch_delivered", stops=[])
        self.assertEqual(list(record["trajectories"]), ["straight", "repeat"])
        verdict = record["delivery"]["straight"]
        self.assertEqual(verdict["delivered_epochs"][-1], 3)
        expected = "holds" if verdict["best_epoch"] in verdict["delivered_epochs"] else None
        if expected is None:
            best, last = (verdict["correct"][n - 1] for n in (verdict["best_epoch"], 3))
            expected = "violated" if last < best else "violated_tie"
        self.assertEqual(verdict["status"], expected)
        self.assertEqual(
            standalone_check.findings(record), [] if expected == "holds" else ["delivery"]
        )

    def test_a_refused_program_is_not_run(self):
        program = self.root / "refused.py"
        program.write_text("import subprocess\n", encoding="utf-8")
        record = standalone_check.check_program(
            program, self.root / "data", self.root / "work" / "refused"
        )
        self.assertEqual(record["category"], "refused")
        self.assertEqual(record["trajectories"], {})
        self.assertFalse((self.root / "work" / "refused").exists())
        self.assertIn("refused (line 1", standalone_check.line({**record, **self.names()}))

    def test_an_existing_work_directory_is_never_touched(self):
        work = self.root / "work" / "taken"
        work.mkdir(parents=True)
        (work / "mine.txt").write_text("keep", encoding="utf-8")
        program = self.root / "healthy.py"
        program.write_text(TINY, encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "remove it or choose another"):
            standalone_check.check_program(
                program, self.root / "data", work, model_factory=tiny_model
            )
        self.assertEqual((work / "mine.txt").read_text(encoding="utf-8"), "keep")

    def test_the_report_counts_categories_verdicts_and_claims(self):
        self.assertEqual(standalone_check.report_text([]), "no checks")
        records = [
            {**self.check("healthy", stops=[1], keep_weights=True), **self.names()},
            {**self.check("scheduler_not_restored", stops=[1]), **self.names(), "attempt": 2},
            {**self.check("nothing_delivered"), **self.names(), "attempt": 3, "claim": None},
        ]
        self.assertTrue((self.root / "work" / "healthy" / "resumed-1" / "delivered.pt").is_file())
        text = standalone_check.report_text(records)
        self.assertIn(
            "m camelyon 1 seed 3: decided; repeat holds; restart holds [1:holds]; "
            "delivery holds, resumed [1:holds]; correct [",
            text,
        )
        self.assertIn("found: nothing; claims meets", text)
        self.assertIn("restart violated [1:violated]", text)
        self.assertIn("found: restart (learning_rates); claims meets", text)
        self.assertIn(
            "m camelyon 3 seed 3: unsupported_interface "
            "(straight: completed; delivered.pt is missing); claims None",
            text,
        )
        self.assertIn("3 checks: 1 unsupported_interface, 2 decided", text)
        self.assertIn("repeat: {'holds': 2}", text)
        self.assertIn("restart: {'holds': 1, 'violated': 1}", text)
        self.assertIn("delivery, resumed runs: {'holds': 2}", text)
        self.assertIn(
            "claims to meet every requirement: 2 of 3, of which something was found in 1", text
        )


if __name__ == "__main__":
    unittest.main()
