"""A training program written from scratch: its task statement and the data it is given.

The repair tasks hand a model a project written for RunSleuth's harness. Here a model
writes the whole program, one file `train.py` with a command line, from a statement
that fixes how the program is called and what it leaves behind: the network after
every epoch, a log line per epoch and the delivered weights. Two of the statement's
requirements can then be decided from outside the program, with no reference
implementation: that a stopped and resumed run trains as the uninterrupted one, and
that the delivered weights are the best epoch's.

    python -m runsleuth.standalone_task statement --dataset camelyon
    python -m runsleuth.standalone_task export --dataset camelyon \\
        --config configs/camelyon17_probe.json --out artifacts/standalone/data/camelyon

The statement keeps the wording of the repair tasks' requirements (pipeline_tasks) where
the two tasks ask the same thing. What the repair tasks tested and this one does not is
given to the program already done: the training hospitals are filtered and the labels
are made, so the data are plain arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from runsleuth import ham10000_data, pipeline_workspace

PROGRAM = "train.py"
EPOCH_WEIGHTS = "epoch_{epoch}.pt"
DELIVERED_WEIGHTS = "delivered.pt"
LOG = "log.jsonl"
LOG_FIELDS = ("epoch", "learning_rate", "validation_loss", "validation_accuracy")
ARRAYS = ("train_images", "train_labels", "validation_images", "validation_labels")
RECORD = "data.json"
# The hospitals a Camelyon17 project trains on; hospital 1 is for evaluation only.
TRAINING_HOSPITALS = (0, 3, 4)
CLAIMS = ("meets every requirement", "does not meet every requirement")


@dataclass(frozen=True, slots=True)
class Dataset:
    """One dataset a program is written for, and the smoke-test data that stand in for it."""

    name: str
    source: str
    synthetic: str
    size: int
    task: str
    samples: str
    positive: str
    flips: bool


DATASETS = {
    dataset.name: dataset
    for dataset in (
        Dataset(
            "camelyon",
            "camelyon",
            "synthetic",
            96,
            "tumor from normal tissue in histopathology patches (Camelyon17)",
            "patches",
            "tumor",
            False,
        ),
        Dataset(
            "ham10000",
            "ham10000",
            "synthetic_ham",
            128,
            "malignant from benign skin lesions in dermatoscopic images (HAM10000)",
            "images",
            "a malignant lesion",
            True,
        ),
    )
}

_FLIPS = (
    " In training, and only there, flip every image left-right and up-down, each with "
    "probability one half, drawn anew for every image in every epoch from nothing but the "
    "run's seed and the epoch."
)

_STATEMENT = """\
# Task

Write one Python file, `{program}`, that fine-tunes an ImageNet-pretrained ResNet18 to tell
{task}.

Python 3.11 with torch, torchvision and numpy is installed; there is no network and no
other package.

## How the program is called

    python {program} --data DIR --out DIR --seed N --epochs N [--stop-after K] [--resume] [--device cuda]

- `--data`: a directory with `train_images.npy` and `validation_images.npy`, uint8 arrays
  of shape (count, 3, {size}, {size}) holding RGB {samples}, and `train_labels.npy` and
  `validation_labels.npy`, int64 arrays with 1 for {positive} and 0 otherwise.
- `--out`: an existing directory, the program's only storage. On a call with `--resume`
  it holds what the earlier call of the same run wrote.
- `--seed` and `--epochs`: the run's seed, and how many epochs the run trains in all.
- `--stop-after K`: stop once epoch K is complete, that is trained, validated, logged
  and stored, and exit with status 0 without delivering.
- `--resume`: continue the run stored in `--out` from where it stopped.
- `--device`: where to train, `cuda` unless given.

A run is either one call without `--stop-after`, or a call with `--stop-after K` followed
by a call with `--resume`; both calls have the same `--data`, `--out`, `--seed` and
`--epochs`. Samples are read in the main process (`num_workers=0`).

## What the program writes into `--out`

- After every epoch, numbered from 1: `{epoch_weights}`, the network's `state_dict()`
  after that epoch, written with `torch.save`.
- After every epoch: one more line in `{log}`, a JSON object with `epoch`,
  `learning_rate` (the one the epoch was trained with), `validation_loss` and
  `validation_accuracy`.
- At the end of the run, and not when it stops early: `{delivered}`, the `state_dict()`
  it delivers.

It may write other files there for its own use.

## What the program must do

1. Training data: every epoch trains on each training sample exactly once, with its own
   label. Batches hold 32 {samples} (the last of an epoch may hold fewer), in an order
   that is shuffled and depends on nothing but the run's seed and the epoch.
2. Inputs: divide the {samples} by 255, then subtract the ImageNet mean (0.485, 0.456,
   0.406) and divide by the ImageNet standard deviation (0.229, 0.224, 0.225) per
   channel, in training and validation alike.{flips} Change nothing else. The network
   is given the {samples} so prepared.
3. Model: `torchvision.models.resnet18` with the weights `ResNet18_Weights.IMAGENET1K_V1`
   and its final layer replaced by a new `nn.Linear` of two outputs, so that the
   `state_dict()` has torchvision's keys. Every parameter is trained, and the network
   is in training mode whenever it is trained and in evaluation mode whenever it is
   validated.
4. Optimizer: AdamW with learning rate 1e-4 and weight decay 1e-2 for all parameters,
   one step per batch, minimizing the cross-entropy loss. The learning rate follows a
   cosine from 1e-4 at the start of the run to 0 at its end, updated once per epoch.
5. Repeats and restarts: a run depends on nothing but its arguments and the data, so
   the same run made twice writes the same weights. After every epoch the program
   stores what it needs, so that a run stopped with `--stop-after` and resumed trains
   exactly as the uninterrupted run would have: the same epochs, batches, weights,
   optimizer state and learning rates.
6. Validation and delivery: after every epoch the program measures the mean
   cross-entropy loss and the accuracy of the network on all validation {samples}, in
   batches of 32 in the order given, and logs them. At the end it delivers the weights
   of the epoch with the highest validation accuracy among all the epochs of the run,
   before and after any restart, the earliest such epoch on a tie.

## Your answer

Reply with the whole of `{program}` in one ```python block and nothing else of code.
After the block write one line, either `CLAIM: {meets}` or `CLAIM: {fails}`, and, in
the second case, one line saying which requirement and why.
"""


def dataset(name: str) -> Dataset:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; choose from {', '.join(DATASETS)}")
    return DATASETS[name]


def statement(name: str, size: int | None = None) -> str:
    """The task statement for a dataset; size is the images' side when not the dataset's."""
    chosen = dataset(name)
    return _STATEMENT.format(
        program=PROGRAM,
        task=chosen.task,
        size=size or chosen.size,
        samples=chosen.samples,
        positive=chosen.positive,
        flips=_FLIPS if chosen.flips else "",
        epoch_weights=EPOCH_WEIGHTS.format(epoch="{n}"),
        log=LOG,
        delivered=DELIVERED_WEIGHTS,
        meets=CLAIMS[0],
        fails=CLAIMS[1],
    )


def _labels(samples: pipeline_workspace.Samples) -> list[int]:
    """The labels a program is given: Camelyon17's own, HAM10000's malignant diagnoses."""
    if hasattr(samples, "labels"):
        return [int(label) for label in samples.labels]
    return [int(diagnosis in ham10000_data.MALIGNANT) for diagnosis in samples.diagnoses]


def _array(samples: pipeline_workspace.Samples, positions: list[int]) -> np.ndarray:
    """The images at the positions as uint8, (count, 3, height, width)."""
    images = []
    for position in positions:
        image = samples.image(position)
        if image.dtype != torch.uint8:
            # Camelyon17 patches come as bytes over 255; this gives the bytes back.
            image = (image * 255).round().clamp(0, 255).to(torch.uint8)
        images.append(image)
    return torch.stack(images).numpy()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(
    data: pipeline_workspace.PipelineData,
    directory: Path,
    *,
    max_train: int | None = None,
    subset_seed: int = 2026,
) -> dict[str, Any]:
    """Write the arrays of a program's `--data` directory and their record.

    Training holds the samples a project should train on: for data with hospitals those
    of the training hospitals, in the order given. max_train keeps that many of them,
    drawn with subset_seed and kept in order. Validation is written whole.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    train = data.train
    positions = list(range(len(train)))
    if hasattr(train, "hospitals"):
        positions = [p for p in positions if train.hospitals[p] in TRAINING_HOSPITALS]
    candidates = len(positions)
    if max_train is not None and max_train < candidates:
        generator = torch.Generator().manual_seed(subset_seed)
        kept = torch.randperm(candidates, generator=generator)[:max_train].tolist()
        positions = [positions[index] for index in sorted(kept)]
    train_labels, validation_labels = _labels(train), _labels(data.validation)
    arrays = {
        "train_images": _array(train, positions),
        "train_labels": np.asarray([train_labels[p] for p in positions], dtype=np.int64),
        "validation_images": _array(data.validation, list(range(len(data.validation)))),
        "validation_labels": np.asarray(validation_labels, dtype=np.int64),
    }
    for name in ARRAYS:
        np.save(directory / f"{name}.npy", arrays[name])
    record = {
        "schema_version": 1,
        "task": "standalone_data",
        "source": data.record,
        "image_size": int(arrays["train_images"].shape[-1]),
        "train": {
            "samples": len(positions),
            "of": candidates,
            "max_train": max_train,
            "subset_seed": subset_seed if len(positions) < candidates else None,
            "positive": int(arrays["train_labels"].sum()),
        },
        "validation": {
            "samples": len(validation_labels),
            "positive": int(arrays["validation_labels"].sum()),
        },
        "sha256": {name: _sha256(directory / f"{name}.npy") for name in ARRAYS},
    }
    (directory / RECORD).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Training programs written from scratch.")
    commands = parser.add_subparsers(dest="command", required=True)
    stated = commands.add_parser("statement", help="Print a dataset's task statement.")
    exported = commands.add_parser("export", help="Write a dataset's --data directory.")
    for command in (stated, exported):
        command.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    stated.add_argument("--size", type=int, help="Image size; the dataset's own if unset.")
    exported.add_argument("--out", required=True, type=Path)
    exported.add_argument("--config", type=Path, help="The dataset's config file.")
    exported.add_argument("--synthetic", action="store_true", help="Smoke-test data.")
    exported.add_argument("--max-train", type=int, help="Keep this many training samples.")
    exported.add_argument("--subset-seed", type=int, default=2026)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    chosen = dataset(args.dataset)
    if args.command == "statement":
        print(statement(chosen.name, args.size), end="")
        return 0
    source = chosen.synthetic if args.synthetic else chosen.source
    data = pipeline_workspace.data_builder(source, args.config)(args.subset_seed)
    record = export(data, args.out, max_train=args.max_train, subset_seed=args.subset_seed)
    print(
        f"{args.out}: {record['train']['samples']} of {record['train']['of']} training and "
        f"{record['validation']['samples']} validation {chosen.samples}, "
        f"{record['image_size']}x{record['image_size']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
