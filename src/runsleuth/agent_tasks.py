"""Repair tasks for agents: one training script with a fault written into its source.

Each fault is an exact edit of the healthy script in agent_task_template.py, so the
healthy script is its ground-truth repair. Most faults act from the first optimizer
step; schedule_ends_an_epoch_early leaves the first epoch untouched. FAULTS each have
a signature in the failure signature library. UNLISTED_FAULTS have none: the library
was not written for them, so they show what a verifier does with a fault it does not
know. A look-alike is also an edit, but one that keeps training correct while
resembling a fault or being written differently, to measure false alarms and what
correct scripts cost a verifier. materialize() writes a workspace holding train.py and TASK.md,
the only files an agent is given. The task statement is the public specification of
the script: it states everything a script is later tested for (agent_contract), never
names a fault, and is the same for every task.

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
in 96x96 histopathology patches (Camelyon17).

A harness imports `train.py`, supplies the patches and calls five functions, which
must keep their names and arguments: `use_for_training`, `prepare_training_batch`,
`prepare_evaluation_batch`, `build` and `train_one_epoch`. Before the first epoch and
after every epoch the harness itself evaluates the model, in evaluation mode, on two
validation sets:

- in-distribution: held-out patches from the hospitals used for training;
- out-of-distribution: patches from hospital 1.

Hospital 2 is the sealed test set.

## What the script must do

1. Data: train on the patches of every hospital (0 to 4) except hospitals 1 and 2.
2. Inputs: patches arrive as RGB values in [0, 1]. For training and for evaluation
   alike, subtract the ImageNet mean (0.485, 0.456, 0.406) and divide by the ImageNet
   standard deviation (0.229, 0.224, 0.225) per channel, and change nothing else.
3. Model: the pretrained ResNet18 with its final layer replaced by a new layer of two
   outputs.
4. Training: every parameter of the network is trained, and the network is in
   training mode whenever it is trained.
5. Optimizer: AdamW with learning rate 1e-4 and weight decay 1e-2 for all parameters,
   taking one step per batch.
6. Schedule: the learning rate follows a cosine from 1e-4 at the start of the run to 0
   at its end, without warmup or restarts. It may be updated once per epoch or once
   per batch.

## Your job

1. Decide whether `train.py` does all of the above.
2. If it does not, fix `train.py`, changing only what is needed.
3. Say whether training is now correct and what you changed.

The script may already be correct.
"""


@dataclass(frozen=True, slots=True)
class Fault:
    """A fault as source edits.

    signature is its id in the failure signature library, or None for a fault the
    library has no signature for.
    """

    signature: str | None
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
    "schedule_ends_an_epoch_early": Fault(
        "zero_learning_rate_epoch",
        "The cosine schedule is one epoch too short, so the last epoch trains at a learning "
        "rate of zero. The first epoch is the healthy script's, step for step; the fault needs "
        "at least two epochs.",
        (
            (
                "CosineAnnealingLR(optimizer, T_max=epochs)",
                "CosineAnnealingLR(optimizer, T_max=epochs - 1)",
            ),
        ),
    ),
}


_HEAD = "    model.fc = nn.Linear(model.fc.in_features, 2)\n"
_COSINE = "torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)"
_LAMBDA = "torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: "
UNLISTED_FAULTS = {
    "schedule_period_in_steps_stepped_per_epoch": Fault(
        None,
        "The schedule's period counts batches but it is stepped per epoch, so the learning "
        "rate hardly falls. An agent wrote this as a repair in the first pilot.",
        (("T_max=epochs)", "T_max=epochs * steps_per_epoch)"),),
    ),
    "learning_rate_halved": Fault(
        None,
        "The learning rate is half of what the statement asks for.",
        (("LEARNING_RATE = 1e-4\n", "LEARNING_RATE = 5e-5\n"),),
    ),
    "same_wrong_normalization_twice": Fault(
        None,
        "Training and evaluation normalize alike, but not with the ImageNet statistics.",
        (
            ("MEAN = (0.485, 0.456, 0.406)", "MEAN = (0.5, 0.5, 0.5)"),
            ("STD = (0.229, 0.224, 0.225)", "STD = (0.5, 0.5, 0.5)"),
        ),
    ),
    "head_bias_left_frozen": Fault(
        None,
        "The new head's weight trains and its bias does not.",
        ((_HEAD, _HEAD + "    model.fc.bias.requires_grad_(False)\n"),),
    ),
    "head_frozen_from_the_second_epoch": Fault(
        None,
        "The head trains in the first epoch only; the first epoch is the healthy script's.",
        (
            (
                "    model.train()\n    for images, labels in batches:\n",
                "    model.train()\n"
                "    if scheduler.last_epoch >= 1:\n"
                "        model.fc.requires_grad_(False)\n"
                "    for images, labels in batches:\n",
            ),
        ),
    ),
    "weight_decay_dropped": Fault(
        None,
        "AdamW is built without weight decay.",
        (("weight_decay=WEIGHT_DECAY", "weight_decay=0.0"),),
    ),
    "training_hospital_left_out": Fault(
        None,
        "Hospital 0 is kept out of training along with the two held-out hospitals.",
        (("hospital not in (VALIDATION_HOSPITAL, TEST_HOSPITAL)", "hospital in (3, 4)"),),
    ),
}
ALL_FAULTS = {**FAULTS, **UNLISTED_FAULTS}


@dataclass(frozen=True, slots=True)
class Lookalike:
    """A correct script that resembles a fault or is written differently from the
    healthy script; resembles names the fault it could be taken for."""

    resembles: str
    summary: str
    edits: tuple[tuple[str, str], ...]


LOOKALIKES = {
    "head_replaced_after_moving": Lookalike(
        "stale_head",
        "The head is replaced after the model moves to the device, yet before the optimizer "
        "is built: the same training as the healthy script.",
        (
            (
                "    model.fc = nn.Linear(model.fc.in_features, 2)\n    model.to(device)\n",
                "    model.to(device)\n"
                "    model.fc = nn.Linear(model.fc.in_features, 2).to(device)\n",
            ),
        ),
    ),
    "cosine_stepped_per_batch_over_the_run": Lookalike(
        "scheduler_stepped_per_batch",
        "The schedule is stepped after every batch and its period is the whole run: a correct "
        "per-step cosine, though not the healthy script's trajectory.",
        (
            ("T_max=epochs)", "T_max=epochs * steps_per_epoch)"),
            (
                "        optimizer.step()\n    scheduler.step()\n",
                "        optimizer.step()\n        scheduler.step()\n",
            ),
        ),
    ),
    "two_parameter_groups": Lookalike(
        "stale_head",
        "The head and the rest of the network are separate optimizer groups with the same "
        "settings.",
        (
            (
                "torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)",
                "torch.optim.AdamW(\n"
                "        [\n"
                "            {'params': [p for n, p in model.named_parameters() "
                "if not n.startswith('fc.')]},\n"
                "            {'params': list(model.fc.parameters())},\n"
                "        ],\n"
                "        lr=LEARNING_RATE,\n"
                "        weight_decay=WEIGHT_DECAY,\n"
                "    )",
            ),
        ),
    ),
    "cosine_written_as_a_formula": Lookalike(
        "schedule_ends_an_epoch_early",
        "The per-epoch cosine is a LambdaLR with the formula written out.",
        (
            ("import torch\n", "import math\n\nimport torch\n"),
            (_COSINE, _LAMBDA + "0.5 * (1 + math.cos(math.pi * epoch / epochs)))"),
        ),
    ),
    "normalization_written_out": Lookalike(
        "train_eval_normalization_mismatch",
        "Evaluation normalizes with its own tensors instead of the shared function.",
        (
            (
                '    """Turn RGB patches in [0, 1] into model inputs for evaluation."""\n'
                "    return normalize(images)\n",
                '    """Turn RGB patches in [0, 1] into model inputs for evaluation."""\n'
                "    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device)\n"
                "    std = torch.tensor([0.229, 0.224, 0.225], device=images.device)\n"
                "    return (images - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)\n",
            ),
        ),
    ),
}
TASKS = (HEALTHY, *LOOKALIKES, *FAULTS, *UNLISTED_FAULTS)
NO_FAULT = "no_known_fault"


def is_faulty(task: str) -> bool:
    """Whether a task's script has a fault, with or without a signature of its own."""
    return task in ALL_FAULTS


def kind(task: str) -> str:
    """healthy, lookalike, listed_fault or unlisted_fault."""
    if task in FAULTS:
        return "listed_fault"
    if task in UNLISTED_FAULTS:
        return "unlisted_fault"
    return "lookalike" if task in LOOKALIKES else HEALTHY


def expected_diagnosis(task: str) -> str:
    """The signature a task's run should be diagnosed with.

    None is expected for a correct script, and for a fault the library has no
    signature for.
    """
    return (ALL_FAULTS[task].signature if task in ALL_FAULTS else None) or NO_FAULT


def healthy_source() -> str:
    return TEMPLATE_PATH.read_text(encoding="utf-8")


def task_source(task: str) -> str:
    """The script for a task: the healthy script, or it with the task's edits applied."""
    source = healthy_source()
    if task == HEALTHY:
        return source
    variants = {**LOOKALIKES, **ALL_FAULTS}
    if task not in variants:
        raise ValueError(f"Unknown task {task!r}; choose from {', '.join(TASKS)}")
    for old, new in variants[task].edits:
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
        for name, lookalike in LOOKALIKES.items():
            print(f"{name}: {lookalike.summary} [correct; resembles {lookalike.resembles}]")
        for name, fault in ALL_FAULTS.items():
            print(f"{name}: {fault.summary} [{fault.signature or 'no signature'}]")
    elif args.command == "show":
        print(source_diff(args.task) or "No difference: this is the healthy script.", end="")
    else:
        print(materialize(args.task, args.directory))


if __name__ == "__main__":
    main()
