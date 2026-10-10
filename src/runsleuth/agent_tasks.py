"""Repair tasks for agents: one training script with a fault written into its source.

Each fault is an exact edit of the healthy script in agent_task_template.py, so the
healthy script is its ground-truth repair. materialize() writes a workspace holding
train.py and TASK.md, the only files an agent is given. The task statement never names
a fault, and the healthy control gets the same statement.

    python -m runsleuth.agent_tasks list
    python -m runsleuth.agent_tasks show train_in_eval_mode
    python -m runsleuth.agent_tasks write stale_head workspaces/stale_head
"""

import argparse
import difflib
from dataclasses import dataclass
from pathlib import Path

TEMPLATE_PATH = Path(__file__).with_name("agent_task_template.py")
SCRIPT_NAME = "train.py"
STATEMENT_NAME = "TASK.md"
HEALTHY = "healthy"
STATEMENT = """\
# Task

`train.py` fine-tunes an ImageNet-pretrained ResNet18 to tell tumor from normal tissue
in 96x96 histopathology patches (Camelyon17). It is meant to fine-tune the whole network
with a new two-class head, AdamW at a fine-tuning learning rate, and a cosine
learning-rate schedule that spans the epochs. Training and evaluation normalize their
inputs in the same way.

A harness imports `train.py`, supplies the patches and calls its functions. After every
epoch the harness itself evaluates the model on two validation sets:

- in-distribution: held-out patches from the hospitals used for training;
- out-of-distribution: patches from hospital 1, which training must never see.

Hospital 2 is the sealed test set.

Your job:

1. Decide whether `train.py` trains the model correctly.
2. If it does not, fix `train.py`, changing only what is needed.
3. Say whether training is now correct and what you changed.

The script may already be correct.
"""


@dataclass(frozen=True, slots=True)
class Fault:
    """A fault as source edits; signature is its id in the failure signature library."""

    signature: str
    summary: str
    edits: tuple[tuple[str, str], ...]


FAULTS = {
    "stale_head": Fault(
        "stale_optimizer_binding",
        "The optimizer is built before the head is replaced, so the new head is never updated.",
        (
            (
                "    model.fc = nn.Linear(model.fc.in_features, 2)\n"
                "    model.to(device)\n"
                "    optimizer = torch.optim.AdamW("
                "model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)\n",
                "    model.to(device)\n"
                "    optimizer = torch.optim.AdamW("
                "model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)\n"
                "    model.fc = nn.Linear(model.fc.in_features, 2).to(device)\n",
            ),
        ),
    ),
    "frozen_head": Fault(
        "frozen_head",
        "The head has requires_grad=False although the whole network should train.",
        (
            (
                "    model.fc = nn.Linear(model.fc.in_features, 2)\n    model.to(device)\n",
                "    model.fc = nn.Linear(model.fc.in_features, 2)\n"
                "    model.fc.requires_grad_(False)\n"
                "    model.to(device)\n",
            ),
        ),
    ),
    "high_learning_rate": Fault(
        "high_learning_rate",
        "The learning rate is 100 times the fine-tuning rate.",
        (("LEARNING_RATE = 1e-4\n", "LEARNING_RATE = 1e-2\n"),),
    ),
    "train_eval_normalization_mismatch": Fault(
        "train_eval_normalization_mismatch",
        "Evaluation normalizes inputs with 0.5 while training uses the ImageNet statistics.",
        (
            (
                '    """Turn RGB patches in [0, 1] into model inputs for evaluation."""\n'
                "    return normalize(images)\n",
                '    """Turn RGB patches in [0, 1] into model inputs for evaluation."""\n'
                "    return (images - 0.5) / 0.5\n",
            ),
        ),
    ),
    "imagenet_head_kept": Fault(
        "classifier_size_mismatch",
        "The 1000-class ImageNet head is kept for the two-class task.",
        (("    model.fc = nn.Linear(model.fc.in_features, 2)\n", ""),),
    ),
    "train_in_eval_mode": Fault(
        "train_in_eval_mode",
        "model.train() is never called, so training runs in eval mode after an evaluation.",
        (("    model.train()\n    for images, labels", "    for images, labels"),),
    ),
    "scheduler_stepped_per_batch": Fault(
        "scheduler_stepped_per_batch",
        "The per-epoch cosine schedule is stepped after every batch.",
        (
            (
                "        optimizer.step()\n    scheduler.step()\n",
                "        optimizer.step()\n        scheduler.step()\n",
            ),
        ),
    ),
    "validation_hospital_in_training": Fault(
        "train_evaluation_group_overlap",
        "Only the test hospital is kept out of training, so the out-of-distribution "
        "validation hospital is trained on.",
        (
            (
                "    return hospital not in (VALIDATION_HOSPITAL, TEST_HOSPITAL)\n",
                "    return hospital != TEST_HOSPITAL\n",
            ),
        ),
    ),
}
TASKS = (HEALTHY, *FAULTS)


def healthy_source() -> str:
    return TEMPLATE_PATH.read_text(encoding="utf-8")


def task_source(task: str) -> str:
    """The script for a task: the healthy script, or it with one fault's edits applied."""
    source = healthy_source()
    if task == HEALTHY:
        return source
    if task not in FAULTS:
        raise ValueError(f"Unknown task {task!r}; choose from {', '.join(TASKS)}")
    for old, new in FAULTS[task].edits:
        if source.count(old) != 1:
            raise ValueError(f"{task}: every edit must match the script exactly once")
        source = source.replace(old, new)
    return source


def source_diff(task: str) -> str:
    """Unified diff from the healthy script to a task's script."""
    return "".join(
        difflib.unified_diff(
            healthy_source().splitlines(keepends=True),
            task_source(task).splitlines(keepends=True),
            f"{HEALTHY}/{SCRIPT_NAME}",
            f"{task}/{SCRIPT_NAME}",
        )
    )


def materialize(task: str, directory: Path) -> Path:
    """Write a fresh workspace for a task and return it; the directory must not exist."""
    source = task_source(task)
    directory = Path(directory)
    directory.mkdir(parents=True)
    (directory / SCRIPT_NAME).write_text(source, encoding="utf-8", newline="\n")
    (directory / STATEMENT_NAME).write_text(STATEMENT, encoding="utf-8", newline="\n")
    return directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agent repair tasks.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="Name every task with its ground truth.")
    show = commands.add_parser("show", help="Print a task's diff from the healthy script.")
    show.add_argument("task", choices=TASKS)
    write = commands.add_parser("write", help="Write a task's workspace to a new directory.")
    write.add_argument("task", choices=TASKS)
    write.add_argument("directory", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "list":
        print(f"{HEALTHY}: the healthy script, used as a control")
        for name, fault in FAULTS.items():
            print(f"{name}: {fault.summary} [{fault.signature}]")
    elif args.command == "show":
        print(source_diff(args.task) or "No difference: this is the healthy script.", end="")
    else:
        print(materialize(args.task, args.directory))


if __name__ == "__main__":
    main()
