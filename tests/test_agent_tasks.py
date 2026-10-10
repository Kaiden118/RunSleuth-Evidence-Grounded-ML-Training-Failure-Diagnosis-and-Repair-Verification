"""Tests for the agent repair tasks: source edits, workspaces and each fault's mechanism."""

import contextlib
import io
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from runsleuth import agent_tasks
from runsleuth.agent_tasks import FAULTS, HEALTHY, LOOKALIKES, TASKS, Fault
from runsleuth.signature_matching import load_library

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None


class TaskSourceTests(unittest.TestCase):
    def test_healthy_task_is_the_template(self):
        self.assertEqual(
            agent_tasks.task_source(HEALTHY),
            agent_tasks.TEMPLATE_PATH.read_text(encoding="utf-8"),
        )
        self.assertEqual(agent_tasks.source_diff(HEALTHY), "")

    def test_every_variant_changes_the_script_and_stays_valid_python(self):
        sources = {task: agent_tasks.task_source(task) for task in TASKS}
        self.assertEqual(len(set(sources.values())), len(TASKS), "tasks must differ")
        for task in (*LOOKALIKES, *FAULTS):
            with self.subTest(task=task):
                compile(sources[task], agent_tasks.SCRIPT_NAME, "exec")
                diff = agent_tasks.source_diff(task)
                self.assertIn(f"+++ {task}/train.py", diff)
                changed = [
                    line
                    for line in diff.splitlines()
                    if line[:1] in "+-" and line[:3] not in ("+++", "---")
                ]
                self.assertLessEqual(len(changed), 6, "a variant is a small edit")

    def test_an_edit_must_match_exactly_once(self):
        for old in ("not in the script", "model"):
            with (
                self.subTest(old=old),
                patch.dict(FAULTS, {"broken": Fault("none", "none", ((old, ""),))}),
                self.assertRaisesRegex(ValueError, "exactly once"),
            ):
                agent_tasks.task_source("broken")

    def test_unknown_task_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown task"):
            agent_tasks.task_source("no_such_task")

    def test_faults_map_to_failure_signatures(self):
        library = {signature["id"] for signature in load_library()["signatures"]}
        self.assertEqual({fault.signature for fault in FAULTS.values()} - library, set())
        for task in TASKS:
            expected = FAULTS[task].signature if task in FAULTS else "no_known_fault"
            self.assertEqual(agent_tasks.expected_diagnosis(task), expected)
        self.assertEqual(
            {lookalike.resembles for lookalike in LOOKALIKES.values()} - set(FAULTS), set()
        )

    def test_workspace_holds_only_the_script_and_a_neutral_statement(self):
        with tempfile.TemporaryDirectory() as temporary:
            statements = set()
            for task in TASKS:
                workspace = agent_tasks.materialize(task, Path(temporary) / task)
                self.assertEqual(
                    sorted(path.name for path in workspace.iterdir()), ["TASK.md", "train.py"]
                )
                self.assertEqual(
                    (workspace / "train.py").read_text(encoding="utf-8"),
                    agent_tasks.task_source(task),
                )
                statements.add((workspace / "TASK.md").read_text(encoding="utf-8"))
            self.assertEqual(len(statements), 1, "every task gets the same statement")
            statement = statements.pop().lower()
            signatures = (fault.signature for fault in FAULTS.values())
            for hint in (*FAULTS, *LOOKALIKES, *signatures, "fault", "bug"):
                self.assertNotIn(hint.lower(), statement)
            with self.assertRaises(FileExistsError):
                agent_tasks.materialize(HEALTHY, Path(temporary) / HEALTHY)

    def test_command_line_lists_and_shows_tasks(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            agent_tasks.main(["list"])
            agent_tasks.main(["show", "frozen_head"])
        text = output.getvalue()
        for task in TASKS:
            self.assertIn(f"{task}:", text)
        self.assertIn("+    model.fc.requires_grad_(False)", text)


if torch is not None:

    class TinyResNet(torch.nn.Module):
        """Stands in for torchvision's resnet18: BatchNorm and a 1000-way fc head."""

        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 4, kernel_size=3)
            self.bn1 = torch.nn.BatchNorm2d(4)
            self.fc = torch.nn.Linear(4, 1000)

        def forward(self, inputs):
            return self.fc(torch.relu(self.bn1(self.conv1(inputs))).mean(dim=(2, 3)))


def load_task(task: str) -> types.ModuleType:
    """Execute a task's script as a module, with a tiny model instead of ResNet18."""
    module = types.ModuleType(f"task_{task}")
    with patch("torchvision.models.resnet18", lambda weights=None: TinyResNet()):
        exec(compile(agent_tasks.task_source(task), agent_tasks.SCRIPT_NAME, "exec"), vars(module))
    return module


def profile(task: str) -> dict:
    """What one epoch of a task's script does, as the mechanisms the faults change."""
    torch.manual_seed(5)
    batches = [(torch.rand(4, 3, 16, 16), torch.tensor([0, 1, 0, 1])) for _ in range(5)]
    module = load_task(task)
    model, optimizer, scheduler = module.build(torch.device("cpu"), 2, len(batches))
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    learning_rate = optimizer.param_groups[0]["lr"]
    model.eval()  # The harness evaluates before the first epoch and after every epoch.
    modes, rates = [], []
    hooks = [
        model.register_forward_pre_hook(lambda module_, _inputs: modes.append(module_.training)),
        optimizer.register_step_pre_hook(
            lambda optimizer_, *_: rates.append(optimizer_.param_groups[0]["lr"])
        ),
    ]
    module.train_one_epoch(model, batches, optimizer, scheduler, torch.nn.CrossEntropyLoss())
    for hook in hooks:
        hook.remove()
    images = torch.rand(2, 3, 16, 16)
    return {
        "head_outputs": model.fc.out_features,
        "head_trainable": all(p.requires_grad for p in model.fc.parameters()),
        "head_in_optimizer": all(id(p) in optimized for p in model.fc.parameters()),
        "learning_rate": learning_rate,
        "trains_in_train_mode": all(modes),
        "scheduler_steps_per_epoch": scheduler.last_epoch,
        "rate_rises_within_epoch": any(b > a for a, b in zip(rates, rates[1:], strict=False)),
        "trains_in_second_epoch": optimizer.param_groups[0]["lr"] > 0,
        "same_preprocessing": torch.equal(
            module.prepare_training_batch(images), module.prepare_evaluation_batch(images)
        ),
        "training_hospitals": tuple(h for h in range(5) if module.use_for_training(h)),
    }


@unittest.skipIf(torch is None, "Install torch and torchvision for the mechanism tests")
class TaskMechanismTests(unittest.TestCase):
    healthy = {
        "head_outputs": 2,
        "head_trainable": True,
        "head_in_optimizer": True,
        "learning_rate": 1e-4,
        "trains_in_train_mode": True,
        "scheduler_steps_per_epoch": 1,
        "rate_rises_within_epoch": False,
        "trains_in_second_epoch": True,
        "same_preprocessing": True,
        "training_hospitals": (0, 3, 4),
    }
    # Each fault changes exactly one mechanism and leaves the rest healthy.
    changes = {
        "stale_head": {"head_in_optimizer": False},
        "frozen_head": {"head_trainable": False},
        "high_learning_rate": {"learning_rate": 1e-2},
        "train_eval_normalization_mismatch": {"same_preprocessing": False},
        "imagenet_head_kept": {"head_outputs": 1000},
        "train_in_eval_mode": {"trains_in_train_mode": False},
        "scheduler_stepped_per_batch": {
            "scheduler_steps_per_epoch": 5,
            "rate_rises_within_epoch": True,
        },
        "validation_hospital_in_training": {"training_hospitals": (0, 1, 3, 4)},
        "schedule_ends_an_epoch_early": {"trains_in_second_epoch": False},
    }
    # A look-alike keeps every mechanism healthy, or changes one in a way that is correct.
    lookalikes = {
        "head_replaced_after_moving": {},
        "cosine_stepped_per_batch_over_the_run": {"scheduler_steps_per_epoch": 5},
    }

    def test_healthy_script_trains_as_the_statement_describes(self):
        self.assertEqual(profile(HEALTHY), self.healthy)

    def test_each_fault_changes_one_mechanism_and_look_alikes_stay_correct(self):
        self.assertEqual(set(self.changes), set(FAULTS))
        self.assertEqual(set(self.lookalikes), set(LOOKALIKES))
        for task, change in {**self.changes, **self.lookalikes}.items():
            with self.subTest(task=task):
                self.assertEqual(profile(task), {**self.healthy, **change})


if __name__ == "__main__":
    unittest.main()
