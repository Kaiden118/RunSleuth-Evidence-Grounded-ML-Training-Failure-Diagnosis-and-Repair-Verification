"""Contract tests for training projects: does a project do what its statement says?

TASK.md (pipeline_tasks) is the public specification an agent works from. This module
turns its requirements into checks that the agent never sees and that run only after
the agent has finished:

    training_samples         1. every epoch trains on each sample it should (Camelyon17:
                                the patches of hospitals 0, 3 and 4; HAM10000: every
                                image) exactly once, and on no other, inside run.epoch
    training_labels          1. each of them is trained on with its own label
    inputs                   2. the network is given every sample prepared as stated, in
                                training and validation: divided by 255 for HAM10000,
                                normalized once by the ImageNet mean and standard
                                deviation, and otherwise unchanged
    flips                    2. HAM10000 only: training images are flipped left-right
                                and up-down anew every epoch, validation images never
    model                    3. the network is the stated model as built: the
                                pretrained weights and a new final layer of two
                                outputs, perhaps inside a wrapper that adds nothing
    all_parameters_trained   3. every epoch changes every parameter of the network
    training_mode            3. the network and every module in it are in training
                                mode whenever it is trained
    optimizer                4. AdamW with learning rate 1e-4 and weight decay 1e-2,
                                holding every parameter once, one step per batch,
                                minimizing the mean cross-entropy of the batch
    learning_rate_schedule   4. every epoch trains with one learning rate, the first
                                one's times a cosine from 1 at the start of the run to 0
                                at its end
    restart                  5. a run stopped as it starts an epoch and restarted
                                trains exactly as the uninterrupted run
    validation               6. after every epoch run.log reports the mean
                                cross-entropy and the accuracy of the network on all
                                the validation samples
    delivery                 6. the delivered weights are those of the epoch whose
                                reported validation accuracy is the highest in the whole
                                run, the earliest on a tie

Version 1 had the first two; version 2 (2026-10-22) added restart; version 3
(2026-10-22) added validation and delivery; version 4 (2026-10-23) added the HAM10000
family and flips, and judges delivery by the accuracy the project reports, which
validation checks, so that a wrong measurement fails validation and a wrong choice
fails delivery. Version 5 adds the clauses of requirements 2 to 4, so that every
requirement of the statement has its checks.

The checks call the project's main() as the harness does, with a run of their own: on
the CPU, with samples whose pixels say which sample they are, in a pattern that
per-channel normalization keeps and, for HAM10000, that can be read in any flip. A
sample counts as trained on when the gradient of a backward pass reaches the network's
output for it, so a forward without a backward (validation, or measuring training
accuracy) does not count, whether or not gradients were enabled. Its label is the
target of the cross-entropy computed on that output. For the training data the project
trains two epochs on samples in two layouts, grouped as the dataset lists them and
mixed, with more than a batch. For the restart it trains three epochs uninterrupted,
and again stopped after the first and after the second epoch and restarted, as the
harness restarts it: main is called again in freshly loaded modules with the
checkpoints written so far. The restarted runs must train the same epochs, on the same
batches flipped the same way, with the same learning rates at every step, ending every
epoch with the same weights; the optimizer's state shows in the weights of the epochs
after it.

Validation and delivery are judged by what the project did, not by how: the checks
compute each epoch's validation accuracy and loss themselves, from the network's
logits on the validation samples in that run and their labels, and require run.log to
report them; and the delivered weights must be those the network had when the epoch
with the highest reported accuracy was validated. Every run above is judged so. Two
more runs make the best epoch fall on either side of a restart, with labels chosen
from the predictions of the uninterrupted three-epoch run and fixed for the whole run:
labelled as the first epoch predicted and stopped after it, the best epoch is the
first, which the restarted run must keep; labelled as the first epoch that predicted
otherwise and stopped before it, the best epoch comes after the restart, which the
restarted run must take. If the network never changes its predictions, the second
cannot be made and the result says so.

Requirements 2 to 4 are judged on the network, the module whose output a backward
pass reached. Every input it is given is compared with the statement's preparation of
the sample the input shows, in the flip it shows it in. Its weights before training are
compared with those of the healthy project's model as built, but for the values of the
new final layer, which are random; and its parameters at the end of every epoch with
those at the end of the epoch before. The gradient a backward pass brings to its output
must be that of the mean cross-entropy of that output and the loss's targets, which
tells a weighted, smoothed or summed loss from the stated one. The learning rates are
judged on the uninterrupted runs, of two epochs and of three: over two epochs a linear
decay gives the same rates as the cosine.

Passing every check is this benchmark's working definition of a project that meets the
requirements the checks cover. It is not a proof that the project is correct.

The checks share no code with the verifier or with the harness's recording: they use
pipeline_workspace only to read and load a project under its guards, and never read a
run report, evidence or signatures.

    python -m runsleuth.pipeline_contract run --project projects/labels_not_filtered
    python -m runsleuth.pipeline_contract check --family ham10000_pipeline
"""

import argparse
import hashlib
import io
import json
import math
import operator
import subprocess
import sys
import types
from collections import Counter
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim.optimizer import register_optimizer_step_post_hook

from runsleuth import pipeline_tasks
from runsleuth.agent_workspace import clean_environment
from runsleuth.pipeline_tasks import HEALTHY
from runsleuth.pipeline_workspace import check_project, load_project, project_sources

CONTRACT_VERSION = 5
TRAINING_HOSPITALS = (0, 3, 4)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
# How close an input must be to the stated preparation of its sample, relative to 1 or
# to the value if larger; and a gradient at the network's output to the mean
# cross-entropy's, relative, beyond an absolute 1e-6.
INPUT_TOLERANCE = 1e-4
GRADIENT_TOLERANCE = 1e-3
# Camelyon17 candidates per hospital: unequal, and more than a batch of 32 in all;
# validation patches too, so that a project that measures one batch only is seen.
CANDIDATES = {0: 9, 1: 7, 3: 14, 4: 17}
VALIDATION = {0: 12, 3: 14, 4: 14}
# HAM10000: the images of each training lesion, several for some lesions so that a
# cache keyed by lesion is seen, and the lesions' diagnoses, with akiec often.
HAM_LESION_IMAGES = (2,) * 12 + (3,) * 2 + (1,) * 16
HAM_DIAGNOSES = ("nv", "akiec", "mel", "bkl", "bcc", "nv", "df", "akiec", "vasc", "nv")
HAM_MALIGNANT = ("mel", "bcc", "akiec")
HAM_VALIDATION = 40
# How close a reported validation loss must be to the checks' own, relative.
LOSS_TOLERANCE = 1e-4
EPOCHS = 2
# The restart: three epochs, stopped after the first or the second. After the first, a
# schedule computed step by step from the current rate and not restored shows only in
# the third epoch.
RESTART_EPOCHS = 3
STOPS = (1, 2)
PATCH_SIZE = 96
HAM_SIZE = 128
# A sample's number is written in bits of channel 0 and their complement in channel 1,
# after a dark and a bright reference pixel; validation samples are numbered from
# VALIDATION_CODES on.
BITS = 12
VALIDATION_CODES = 1 << 11
# The flips a HAM10000 image may reach the network with, as the dimensions they flip.
FLIPS = {"none": (), "left_right": (-1,), "up_down": (-2,), "both": (-2, -1)}


def _passed(detail: str = "") -> dict[str, Any]:
    return {"passed": True, "detail": detail}


def _failed(detail: str) -> dict[str, Any]:
    return {"passed": False, "detail": detail}


def coded_images(
    codes: list[int], brightness: list[float], size: int = PATCH_SIZE, uint8: bool = False
) -> torch.Tensor:
    """RGB images showing their codes, in [0, 1] or as uint8.

    The other pixels are noise of width 0.2 around the image's brightness, so that a
    network can learn to tell bright from dark images.
    """
    images = torch.empty(len(codes), 3, size * size)
    for row, (code, level) in enumerate(zip(codes, brightness, strict=True)):
        noise = torch.rand(images.shape[1:], generator=torch.Generator().manual_seed(code))
        images[row] = level - 0.1 + 0.2 * noise
    images[:, :, 0], images[:, :, 1] = 0.0, 1.0
    bits = torch.tensor([[(code >> bit) & 1 for bit in range(BITS)] for code in codes])
    images[:, 0, 2 : 2 + BITS] = bits.float()
    images[:, 1, 2 : 2 + BITS] = 1 - bits.float()
    images = images.view(len(codes), 3, size, size)
    return (images * 255).round().to(torch.uint8) if uint8 else images


def _readable(inputs: torch.Tensor) -> bool:
    return inputs.dim() == 4 and inputs.shape[1] == 3 and inputs[0, 0].numel() >= 2 + BITS


def _canonical(inputs: torch.Tensor) -> tuple[list[int], list[bool]]:
    values = inputs.detach().to("cpu", torch.float64).flatten(2)[:, :2, : 2 + BITS]
    dark, bright = values[:, :, :1], values[:, :, 1:2]
    bits = values[:, :, 2:] > (dark + bright) / 2
    shown = (bright > dark).all(dim=2).all(dim=1) & (bits[:, 0] != bits[:, 1]).all(dim=1)
    codes = (bits[:, 0].long() * 2 ** torch.arange(BITS)).sum(dim=1)
    return [int(code) for code in codes.tolist()], shown.tolist()


def read_codes(inputs: torch.Tensor) -> list[int | None]:
    """The code of every sample in a batch, after any per-channel increasing affine map.

    None for a sample whose pixels show no code: changed otherwise, or not a sample.
    """
    count = len(inputs) if inputs.dim() else 0
    if not _readable(inputs):
        return [None] * count
    codes, shown = _canonical(inputs)
    return [code if ok else None for code, ok in zip(codes, shown, strict=True)]


def read_flipped(inputs: torch.Tensor) -> list[tuple[int, str] | None]:
    """The code and flip of every sample in a batch, as read_codes but in any flip.

    None for a sample that shows no code in any flip, or one in more than one.
    """
    count = len(inputs) if inputs.dim() else 0
    if not _readable(inputs):
        return [None] * count
    readings = {
        name: _canonical(inputs.flip(dims) if dims else inputs) for name, dims in FLIPS.items()
    }
    found = []
    for row in range(count):
        shown = [(codes[row], name) for name, (codes, ok) in readings.items() if ok[row]]
        found.append(shown[0] if len(shown) == 1 else None)
    return found


def _normalized(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(MEAN, dtype=images.dtype).view(1, 3, 1, 1)
    std = torch.tensor(STD, dtype=images.dtype).view(1, 3, 1, 1)
    return (images - mean) / std


def _flipped(image: torch.Tensor, flip: str | None) -> torch.Tensor:
    dims = FLIPS.get(flip or "none", ())
    return image.flip(dims) if dims else image


def input_kinds(
    inputs: torch.Tensor,
    codes: list[int | None],
    flips: list[str | None],
    image_of: Callable[[int], torch.Tensor | None],
) -> list[str | None]:
    """How each input of a batch was prepared from the sample it shows.

    stated is the statement's preparation of the sample's image (divided by 255 if it is
    uint8, then normalized) in the flip it was read in; unnormalized and twice are that
    preparation without the normalization or with it twice; unscaled is a uint8 image
    normalized without the division; other is anything else. None where no code was read.
    """
    kinds: list[str | None] = [None if code is None else "other" for code in codes]
    rows = [
        row for row, code in enumerate(codes) if code is not None and image_of(code) is not None
    ]
    if not rows:
        return kinds
    images = torch.stack([_flipped(image_of(codes[row]), flips[row]) for row in rows])
    uint8 = images.dtype == torch.uint8
    images = images.to(torch.float64)
    received = inputs.detach()[rows].to("cpu", torch.float64)
    if received.shape != images.shape:
        return kinds
    scaled = images / 255 if uint8 else images
    candidates = {
        "stated": _normalized(scaled),
        "unnormalized": scaled,
        "twice": _normalized(_normalized(scaled)),
    }
    if uint8:
        candidates["unscaled"] = _normalized(images)
    for name, expected in candidates.items():
        close = (received - expected).abs() <= INPUT_TOLERANCE * expected.abs().clamp(min=1)
        for row, matched in zip(rows, close.flatten(1).all(dim=1).tolist(), strict=True):
            if matched and kinds[row] == "other":
                kinds[row] = name
    return kinds


class CodedSamples:
    """Samples as the harness gives them: samples[i], samples.image(i) and the columns.

    samples[i] is (image, the value of the target column); the true labels are private,
    as the harness keeps them.
    """

    def __init__(
        self,
        codes: list[int],
        labels: list[int],
        columns: dict[str, list[Any]],
        brightness: list[float],
        *,
        target: str,
        size: int,
        uint8: bool,
    ) -> None:
        self._images = coded_images(codes, brightness, size, uint8)
        self.codes, self._labels, self._target = tuple(codes), tuple(labels), target
        self._columns = {name: tuple(values) for name, values in columns.items()}
        for name, values in self._columns.items():
            setattr(self, name, values)

    def __len__(self) -> int:
        return len(self._labels)

    def _position(self, index) -> int:
        position = operator.index(index)
        position += len(self) if position < 0 else 0
        if not 0 <= position < len(self):
            raise IndexError(f"sample {index} is out of range for {len(self)} samples")
        return position

    def __getitem__(self, index) -> tuple[torch.Tensor, Any]:
        position = self._position(index)
        return self._images[position], self._columns[self._target][position]

    def image(self, index) -> torch.Tensor:
        return self._images[self._position(index)]


def _spread(count: int) -> list[float]:
    """Brightness spread evenly between the two training classes', in a fixed shuffle,
    so that some validation samples lie near the network's boundary and its predictions
    change as it trains."""
    order = torch.randperm(count, generator=torch.Generator().manual_seed(count))
    return [0.3 + 0.4 * rank / (count - 1) for rank in order.tolist()]


def candidates(layout: str, seed: int) -> CodedSamples:
    """Camelyon17 candidates by position, labels drawn at random, label 1 brighter."""
    hospitals = [hospital for hospital, count in CANDIDATES.items() for _ in range(count)]
    if layout == "mixed":
        order = torch.randperm(len(hospitals), generator=torch.Generator().manual_seed(seed))
        hospitals = [hospitals[position] for position in order.tolist()]
    generator = torch.Generator().manual_seed(1000 + seed)
    labels = torch.randint(0, 2, (len(hospitals),), generator=generator).tolist()
    return CodedSamples(
        list(range(len(hospitals))),
        labels,
        {"labels": labels, "hospitals": hospitals},
        [0.3 + 0.4 * label for label in labels],
        target="labels",
        size=PATCH_SIZE,
        uint8=False,
    )


def validation_patches(labels: dict[int, int] | None = None) -> CodedSamples:
    """Camelyon17 validation patches labelled by brightness, or by labels[code]."""
    hospitals = [hospital for hospital, count in VALIDATION.items() for _ in range(count)]
    codes = [VALIDATION_CODES + position for position in range(len(hospitals))]
    brightness = _spread(len(codes))
    given = [
        int(level > 0.5) if labels is None else labels[code]
        for code, level in zip(codes, brightness, strict=True)
    ]
    return CodedSamples(
        codes,
        given,
        {"labels": given, "hospitals": hospitals},
        brightness,
        target="labels",
        size=PATCH_SIZE,
        uint8=False,
    )


def _ham_label(diagnosis: str) -> int:
    return int(diagnosis in HAM_MALIGNANT)


def ham_images(layout: str, seed: int) -> CodedSamples:
    """HAM10000 training images by position: lesions of one to three images, the
    lesion's diagnosis on each, malignant images brighter."""
    rows = [
        (lesion, HAM_DIAGNOSES[lesion % len(HAM_DIAGNOSES)])
        for lesion, count in enumerate(HAM_LESION_IMAGES)
        for _ in range(count)
    ]
    if layout == "mixed":
        order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(seed))
        rows = [rows[position] for position in order.tolist()]
    labels = [_ham_label(diagnosis) for _, diagnosis in rows]
    codes = list(range(len(rows)))
    return CodedSamples(
        codes,
        labels,
        {
            "image_ids": [f"ISIC_{9000000 + code}" for code in codes],
            "lesion_ids": [f"HAM_{lesion:07d}" for lesion, _ in rows],
            "diagnoses": [diagnosis for _, diagnosis in rows],
            "sources": ["vidir_modern" if lesion % 2 else "vidir_molemax" for lesion, _ in rows],
        },
        [0.3 + 0.4 * label for label in labels],
        target="diagnoses",
        size=HAM_SIZE,
        uint8=True,
    )


def ham_validation(labels: dict[int, int] | None = None) -> CodedSamples:
    """HAM10000 validation images, one per lesion, labelled by brightness or by
    labels[code], with a diagnosis of that label: akiec for every third malignant one."""
    codes = [VALIDATION_CODES + position for position in range(HAM_VALIDATION)]
    brightness = _spread(len(codes))
    given = [
        int(level > 0.5) if labels is None else labels[code]
        for code, level in zip(codes, brightness, strict=True)
    ]
    diagnoses = [
        ("akiec", "mel", "bcc")[position % 3]
        if label
        else ("nv", "bkl", "df", "vasc")[position % 4]
        for position, label in enumerate(given)
    ]
    return CodedSamples(
        codes,
        given,
        {
            "image_ids": [f"ISIC_{8000000 + code}" for code in codes],
            "lesion_ids": [f"HAM_V{position:06d}" for position in range(len(codes))],
            "diagnoses": diagnoses,
            "sources": ["vidir_modern"] * len(codes),
        },
        brightness,
        target="diagnoses",
        size=HAM_SIZE,
        uint8=True,
    )


@dataclass(frozen=True, slots=True)
class Spec:
    """What a family's statement asks of the training data, and the samples to check it on.

    trained_on lists the positions of the candidates every epoch must train on; kept
    names them in a failure; group names a column to count the wrong samples by; flips
    says whether training flips its images (and the codes are read in any flip).
    """

    clauses: tuple[str, ...]
    layouts: tuple[tuple[str, int], ...]
    candidates: Callable[[str, int], CodedSamples]
    validation: Callable[[dict[int, int] | None], CodedSamples]
    trained_on: Callable[[CodedSamples], list[int]]
    noun: str
    kept: str
    group: str | None
    flips: bool


SPECS = {
    "camelyon_pipeline": Spec(
        (
            "training_samples",
            "training_labels",
            "inputs",
            "model",
            "all_parameters_trained",
            "training_mode",
            "optimizer",
            "learning_rate_schedule",
            "restart",
            "validation",
            "delivery",
        ),
        (("by_hospital", 3), ("mixed", 5)),
        candidates,
        validation_patches,
        lambda train: [
            position
            for position, hospital in enumerate(train.hospitals)
            if hospital in TRAINING_HOSPITALS
        ],
        "patch",
        "patches of hospitals 0, 3 and 4",
        "hospitals",
        False,
    ),
    "ham10000_pipeline": Spec(
        (
            "training_samples",
            "training_labels",
            "inputs",
            "flips",
            "model",
            "all_parameters_trained",
            "training_mode",
            "optimizer",
            "learning_rate_schedule",
            "restart",
            "validation",
            "delivery",
        ),
        (("by_lesion", 3), ("mixed", 5)),
        ham_images,
        ham_validation,
        lambda train: list(range(len(train))),
        "image",
        "images of run.data.train",
        None,
        True,
    ),
}


def spec(family: str) -> Spec:
    if family not in SPECS:
        raise ValueError(f"No contract for the family {family!r}; choose from {', '.join(SPECS)}")
    return SPECS[family]


class Stopped(BaseException):
    """Raised into a project when the checks stop its run; a project cannot name it."""


class MemoryCheckpoints:
    """Checkpoints kept in memory, read back as torch.load reads them."""

    def __init__(self) -> None:
        self._files: dict[str, bytes] = {}
        self._stopped = False

    def write(self, name: str, state: Any) -> None:
        if self._stopped:
            raise Stopped  # A stopped run writes nothing more.
        buffer = io.BytesIO()
        torch.save(state, buffer)
        self._files[name] = buffer.getvalue()

    def read(self, name: str) -> Any:
        return torch.load(io.BytesIO(self._files[name]), map_location="cpu", weights_only=True)

    def __contains__(self, name: object) -> bool:
        return name in self._files


class ContractRun:
    """The run a project gets in the checks: everything the statement says run gives.

    With stop_after, the run is stopped as it starts an epoch after that many epochs.
    events receives every ended epoch (epoch_ended), log (logged) and delivery
    (delivered).
    """

    def __init__(
        self,
        train: CodedSamples,
        validation: CodedSamples,
        *,
        seed: int,
        epochs: int,
        checkpoints: MemoryCheckpoints,
        events: types.SimpleNamespace,
        stop_after: int | None = None,
    ) -> None:
        self.device, self.seed, self.epochs = torch.device("cpu"), seed, epochs
        self.data = types.SimpleNamespace(train=train, validation=validation)
        self.checkpoints = checkpoints
        self._stop_after, self._events = stop_after, events
        # The epoch being trained, and the last one whose block was entered.
        self._epoch: int | None = None
        self._started: int | None = None
        self._trained = 0

    def track(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
        pass

    @contextmanager
    def epoch(self, number: int):
        if self.checkpoints._stopped:
            raise Stopped
        if self._stop_after is not None and self._trained >= self._stop_after:
            self.checkpoints._stopped = True
            raise Stopped
        if self._epoch is not None:
            raise RuntimeError("run.epoch cannot be nested")
        self._epoch = self._started = number
        try:
            yield
        finally:
            self._epoch = None
        self._trained += 1
        self._events.epoch_ended(number)

    def log(self, **metrics: float) -> None:
        if self.checkpoints._stopped:
            raise Stopped
        self._events.logged(metrics)

    def deliver(self, state_dict) -> None:
        if self.checkpoints._stopped:
            raise Stopped
        self._events.delivered(state_dict)


def fingerprint(state: dict[str, torch.Tensor]) -> str:
    """A hash of a state_dict's tensors, by name, dtype, shape and bytes."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}|{value.dtype}|{tuple(value.shape)}|".encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def loss_kind(
    output: torch.Tensor, targets: list[int] | None, gradient: torch.Tensor
) -> str | None:
    """The loss the gradient at the network's output shows: mean, summed or other.

    mean is the gradient of the mean cross-entropy of the output and the targets, summed
    that of their summed cross-entropy. None without targets to compare with.
    """
    if targets is None or output.dim() != 2 or len(targets) != len(output):
        return None
    classes = output.shape[1]
    if gradient.shape != output.shape or not all(0 <= target < classes for target in targets):
        return "other"
    logits = output.detach().to("cpu", torch.float64)
    one_hot = nn.functional.one_hot(torch.tensor(targets), classes).to(torch.float64)
    expected = logits.softmax(dim=1) - one_hot
    received = gradient.detach().to("cpu", torch.float64)
    for name, scale in (("mean", 1 / len(targets)), ("summed", 1.0)):
        if torch.allclose(received, expected * scale, rtol=GRADIENT_TOLERANCE, atol=1e-6):
            return name
    return "other"


def optimizer_settings(
    optimizer: torch.optim.Optimizer, network: nn.Module | None
) -> dict[str, Any]:
    """What the statement asks of an optimizer at a step: whether it is AdamW, the weight
    decay of every group, and the network's parameters it misses or holds more than once."""
    groups = optimizer.param_groups
    held = Counter(id(parameter) for group in groups for parameter in group["params"])
    counts = (
        []
        if network is None
        else [(name, held[id(parameter)]) for name, parameter in network.named_parameters()]
    )
    # AdamW is Adam with decoupled weight decay, which Adam can be asked for as well.
    adamw = isinstance(optimizer, torch.optim.AdamW) or (
        isinstance(optimizer, torch.optim.Adam)
        and all(group.get("decoupled_weight_decay", False) for group in groups)
    )
    return {
        "kind": type(optimizer).__name__,
        "adamw": adamw,
        "weight_decays": [group.get("weight_decay") for group in groups],
        "missing": [name for name, count in counts if count == 0],
        "repeated": [name for name, count in counts if count > 1],
    }


def observe(
    sources: dict[str, str],
    rules: Spec,
    layout: str,
    seed: int,
    *,
    epochs: int = EPOCHS,
    stop_after: int | None = None,
    validation_labels: dict[int, int] | None = None,
    capture: bool = False,
) -> dict[str, Any]:
    """Run a project on coded samples, perhaps stopped and restarted; what it did.

    trained has one row per backward pass that reached the network's output: the
    attempt and the epoch it happened in (None outside run.epoch), the codes, flips and
    preparation (input_kinds) of the batch the network was given, the first of its
    modules not in training mode, the targets of the cross-entropy on that output (None
    if there was none) and the loss the gradient at the output shows (mean, summed,
    other, or None without targets). steps has the attempt, epoch and learning rates of
    every optimizer step, and optimizers the settings at that step (optimizer_settings);
    ended has the attempt, number and network fingerprint of every epoch. validated maps
    (attempt, the attempt's last ended epoch) to the network's fingerprint, and logits,
    flip and preparation per validation code, in the first pass over them after that
    epoch; logged and delivered hold the same key with every run.log, and the attempt and
    fingerprint of every delivery. With stop_after, the first attempt is stopped as it
    starts an epoch after that many, and main is called again in freshly loaded modules
    with its checkpoints. validation_labels relabel the validation samples. With
    capture, built is the network's state before its first training step, and unchanged
    has the attempt, number and unchanged parameters of every epoch after it.
    """
    train, validation = rules.candidates(layout, seed), rules.validation(validation_labels)
    checkpoints = MemoryCheckpoints()
    seen: dict[str, Any] = {"layout": layout, "train": train, "trained": [], "steps": []}
    seen.update(ended=[], validated={}, logged=[], delivered=[], validation=validation)
    seen.update(epochs=epochs, optimizers=[], built=None, unchanged=[])
    state: dict[str, Any] = {"depth": 0, "batch": None, "network": None, "attempt": 0}
    state.update(last_ended=None, previous=None)
    run: ContractRun | None = None
    validation_codes = set(validation.codes)

    def read(inputs: torch.Tensor) -> tuple[list[int | None], list[str | None]]:
        if not rules.flips:
            codes = read_codes(inputs)
            return codes, ["none" if code is not None else None for code in codes]
        found = read_flipped(inputs)
        return [item and item[0] for item in found], [item and item[1] for item in found]

    def image_of(code: int) -> torch.Tensor | None:
        if 0 <= code < len(train):
            return train.image(code)
        position = code - VALIDATION_CODES
        return validation.image(position) if 0 <= position < len(validation) else None

    def before(module, args) -> None:
        state["depth"] += 1
        if state["depth"] == 1 and args and isinstance(args[0], torch.Tensor):
            if args[0].dim() == 4:
                # The outermost call on a batch of images: the network.
                codes, flips = read(args[0])
                state["batch"] = {
                    "codes": codes,
                    "flips": flips,
                    "inputs": input_kinds(args[0], codes, flips, image_of),
                    "evaluating": next(
                        (
                            name or "the network"
                            for name, part in module.named_modules()
                            if not part.training
                        ),
                        None,
                    ),
                    "module": module,
                    "output": None,
                    "targets": None,
                }
                if capture and seen["built"] is None:
                    # Before the forward, which updates batch norm statistics in training.
                    state["batch"]["weights"] = {
                        name: value.detach().clone() for name, value in module.state_dict().items()
                    }

    def after(module, _args, output) -> None:
        state["depth"] -= 1
        batch = state["batch"]
        if state["depth"] or batch is None or batch["output"] is not None:
            return
        if isinstance(output, torch.Tensor):
            batch["output"] = output
            validated(module, batch, output)
            if output.requires_grad:
                output.register_hook(lambda gradient, batch=batch: backward(batch, gradient))

    def validated(module, batch: dict[str, Any], output: torch.Tensor) -> None:
        """Keep the first logits and flip per validation sample after each ended epoch."""
        codes = batch["codes"]
        if state["last_ended"] is None or output.dim() != 2 or len(output) != len(codes):
            return
        if not validation_codes.intersection(codes):
            return
        key = (state["attempt"], state["last_ended"])
        if key not in seen["validated"]:
            seen["validated"][key] = {
                "fingerprint": fingerprint(module.state_dict()),
                "logits": {},
                "flips": {},
                "inputs": {},
            }
        kept = seen["validated"][key]
        rows = output.detach().to("cpu", torch.float64)
        for code, flip, kind, row in zip(codes, batch["flips"], batch["inputs"], rows, strict=True):
            if code in validation_codes and code not in kept["logits"]:
                kept["logits"][code], kept["flips"][code], kept["inputs"][code] = row, flip, kind

    def backward(batch: dict[str, Any], gradient: torch.Tensor) -> None:
        network = state["network"] = batch["module"]
        if capture and seen["built"] is None and "weights" in batch:
            seen["built"] = batch["weights"]
            state["previous"] = {
                name: seen["built"][name]
                for name, _ in network.named_parameters()
                if name in seen["built"]
            }
        seen["trained"].append(
            {
                "attempt": state["attempt"],
                "epoch": run._epoch,
                "codes": batch["codes"],
                "flips": batch["flips"],
                "inputs": batch["inputs"],
                "evaluating": batch["evaluating"],
                "targets": batch["targets"],
                "loss": loss_kind(batch["output"], batch["targets"], gradient),
            }
        )

    def stepped(optimizer, _args, _kwargs) -> None:
        rates = [float(group["lr"]) for group in optimizer.param_groups]
        seen["steps"].append((state["attempt"], run._epoch, rates))
        settings = optimizer_settings(optimizer, state["network"])
        seen["optimizers"].append({**settings, "started": run._started})

    def epoch_ended(number: int) -> None:
        network = state["network"]
        weights = network and fingerprint(network.state_dict())
        seen["ended"].append((state["attempt"], number, weights))
        state["last_ended"] = number
        previous = state["previous"]
        if capture and network is not None and previous is not None:
            current = {name: value.detach().clone() for name, value in network.named_parameters()}
            unchanged = [
                name
                for name, value in current.items()
                if name in previous and torch.equal(value, previous[name])
            ]
            seen["unchanged"].append((state["attempt"], number, unchanged))
            state["previous"] = current

    def logged(metrics: dict[str, Any]) -> None:
        seen["logged"].append(((state["attempt"], state["last_ended"]), dict(metrics)))

    def delivered(weights) -> None:
        try:
            weights = fingerprint(dict(weights))
        except (TypeError, ValueError, AttributeError):
            weights = None  # Not a state_dict: no epoch's weights.
        seen["delivered"].append((state["attempt"], weights))

    events = types.SimpleNamespace(epoch_ended=epoch_ended, logged=logged, delivered=delivered)

    cross_entropy = nn.functional.cross_entropy

    def recorded_cross_entropy(input, target, *args, **kwargs):
        batch = state["batch"]
        if batch is not None and input is batch["output"] and isinstance(target, torch.Tensor):
            batch["targets"] = [int(value) for value in target.detach().flatten().tolist()]
        return cross_entropy(input, target, *args, **kwargs)

    # nn.CrossEntropyLoss calls nn.functional.cross_entropy, so both are seen.
    nn.functional.cross_entropy = recorded_cross_entropy
    handles = [
        nn.modules.module.register_module_forward_pre_hook(before),
        nn.modules.module.register_module_forward_hook(after),
        register_optimizer_step_post_hook(stepped),
    ]
    try:
        for stop in (stop_after, None) if stop_after is not None else (None,):
            state.update(attempt=state["attempt"] + 1, depth=0, batch=None, network=None)
            state["last_ended"] = None
            run = ContractRun(
                train,
                validation,
                seed=seed,
                epochs=epochs,
                checkpoints=checkpoints,
                events=events,
                stop_after=stop,
            )
            torch.manual_seed(seed)
            try:
                vars(load_project(sources)["train"])["main"](run)
            except Stopped:
                pass
            # Whatever the project did with the stop, the attempt ended there.
            stopped, checkpoints._stopped = checkpoints._stopped, False
            if not stopped:
                break
    finally:
        nn.functional.cross_entropy = cross_entropy
        for handle in handles:
            handle.remove()
    return seen


def _plural(noun: str) -> str:
    return noun + ("es" if noun.endswith("ch") else "s")


def _count(number: int, noun: str) -> str:
    return f"{number} {noun if number == 1 else _plural(noun)}"


def _training_samples(seen: dict[str, Any], rules: Spec) -> dict[str, Any]:
    train, trained, layout, noun = seen["train"], seen["trained"], seen["layout"], rules.noun
    if not trained:
        return _failed("the network was never trained: no backward pass reached its output")
    if any(row["epoch"] is None for row in trained):
        return _failed("the network is trained outside run.epoch")
    validation = set(seen["validation"].codes)
    codes = [code for row in trained for code in row["codes"]]
    unread = sum(code is None or not (code < len(train) or code in validation) for code in codes)
    if unread:
        return _failed(
            f"{_count(unread, 'trained input')} could not be read back as {_plural(noun)}: the "
            f"network must be given the {_plural(noun)} prepared as stated and otherwise unchanged"
        )
    expected = rules.trained_on(train)
    wanted = set(expected)
    for epoch in sorted({row["epoch"] for row in trained}):
        counts = Counter(code for row in trained if row["epoch"] == epoch for code in row["codes"])
        problems = []
        validated = sum(count for code, count in counts.items() if code in validation)
        if validated:
            problems.append(f"trains on {_count(validated, 'validation ' + noun)}")
        others = [code for code, count in counts.items() for _ in range(count) if code < len(train)]
        others = [code for code in others if code not in wanted]
        if others and rules.group:
            groups = Counter(getattr(train, rules.group)[code] for code in others)
            problems.extend(
                f"trains on {_count(count, noun)} from {rules.group[:-1]} {group}"
                for group, count in sorted(groups.items())
            )
        elif others:
            problems.append(f"trains on {_count(len(others), noun)} it should not")
        missing = sum(counts[position] == 0 for position in expected)
        if missing:
            problems.append(f"leaves out {missing} of the {len(expected)} {rules.kept}")
        repeated = sum(counts[position] > 1 for position in expected)
        if repeated:
            problems.append(f"trains on {_count(repeated, noun)} more than once")
        if problems:
            return _failed(f"epoch {epoch} ({layout} layout): " + "; ".join(problems))
    return _passed()


def _training_labels(seen: dict[str, Any], rules: Spec) -> dict[str, Any]:
    train, trained, noun = seen["train"], seen["trained"], rules.noun
    unlabelled = sum(
        len(row["codes"])
        for row in trained
        if row["targets"] is None or len(row["targets"]) != len(row["codes"])
    )
    if unlabelled:
        return _failed(
            f"{_count(unlabelled, 'trained ' + noun)} had no target in a cross-entropy loss on "
            "the network's output"
        )
    pairs = [
        (code, target)
        for row in trained
        for code, target in zip(row["codes"], row["targets"], strict=True)
        if code is not None and code < len(train)
    ]
    wrong = sum(target != train._labels[code] for code, target in pairs)
    if wrong:
        return _failed(
            f"{wrong} of the {len(pairs)} trained candidate {_plural(noun)} ({seen['layout']} layout) "
            "were trained with a label that is not theirs"
        )
    return _passed()


def _flips(seen: dict[str, Any]) -> dict[str, Any]:
    """Training images flipped both ways, anew every epoch; validation images never."""
    where = seen["context"]
    flipped = sum(
        flip != "none"
        for validated in seen["validated"].values()
        for flip in validated["flips"].values()
    )
    if flipped:
        return _failed(f"{where}: {_count(flipped, 'validation image')} reach the network flipped")
    order = [(attempt, number) for attempt, number, _ in seen["ended"]]
    by_epoch: dict[tuple[int, int], dict[int, str]] = {key: {} for key in order}
    for row in seen["trained"]:
        kept = by_epoch.setdefault((row["attempt"], row["epoch"]), {})
        for code, flip in zip(row["codes"], row["flips"], strict=True):
            if code is not None:
                kept.setdefault(code, flip)
    for (_, number), flips in by_epoch.items():
        for name, ways in (
            ("left-right", ("left_right", "both")),
            ("up-down", ("up_down", "both")),
        ):
            if flips and not any(flip in ways for flip in flips.values()):
                return _failed(f"{where}: epoch {number} flips no training image {name}")
    for earlier, later in zip(order, order[1:], strict=False):
        common = set(by_epoch[earlier]) & set(by_epoch[later])
        changed = sum(by_epoch[earlier][code] != by_epoch[later][code] for code in common)
        # Drawn anew, three flips in four differ from the epoch before.
        if common and changed < len(common) / 4:
            return _failed(
                f"{where}: epochs {earlier[1]} and {later[1]} flip {len(common) - changed} of "
                f"the {len(common)} images the same way: the flips are not drawn anew"
            )
    return _passed()


def _some(names: list[str]) -> str:
    shown = ", ".join(names[:3])
    return shown + (f" and {len(names) - 3} more" if len(names) > 3 else "")


def _in_epoch(epoch: int | None) -> str:
    return "outside run.epoch" if epoch is None else f"in epoch {epoch}"


INPUT_PROBLEMS = {
    "unnormalized": "not normalized",
    "twice": "normalized twice",
    "unscaled": "normalized without being divided by 255",
    "other": "changed otherwise than stated",
}


def _inputs(seen: dict[str, Any], rules: Spec) -> dict[str, Any]:
    """Every sample the network is trained and validated on, prepared as stated."""
    where, noun = seen["context"], rules.noun
    given = {
        "trained": [kind for row in seen["trained"] for kind in row["inputs"]],
        "validated": [
            kind
            for validated in seen["validated"].values()
            for kind in validated["inputs"].values()
        ],
    }
    for use, kinds in given.items():
        wrong = Counter("other" if kind is None else kind for kind in kinds if kind != "stated")
        if wrong:
            kind, count = wrong.most_common(1)[0]
            more = sum(wrong.values()) - count
            return _failed(
                f"{where}: the network is {use} on {_count(count, noun)} {INPUT_PROBLEMS[kind]}"
                + (f", and on {more} more prepared otherwise" if more else "")
            )
    return _passed()


_REFERENCES: dict[str, dict[str, Any]] = {}


def reference_model(sources: dict[str, str]) -> dict[str, Any]:
    """The stated model as the healthy project builds it: its state and final layer."""
    key = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    if key not in _REFERENCES:
        torch.manual_seed(0)
        model = vars(load_project(sources)["train"])["build_model"]()
        _REFERENCES[key] = {
            "state": {name: value.detach().clone() for name, value in model.state_dict().items()},
            "final_layer": list(model.named_children())[-1][0],
        }
    return _REFERENCES[key]


def _wrapped(built: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> str | None:
    """The prefix under which built holds exactly the entries of expected, if any: empty,
    or the name of the stated model inside a wrapper that adds nothing."""
    anchor = next(iter(expected))
    for name in built:
        prefix = name[: len(name) - len(anchor)]
        if name.endswith(anchor) and (not prefix or prefix.endswith(".")):
            if set(built) == {prefix + key for key in expected}:
                return prefix
    return None


def _model(seen: dict[str, Any], stated: dict[str, Any]) -> dict[str, Any]:
    """The network before training against the stated model, but the new layer's values."""
    where, built, expected = seen["context"], seen["built"], stated["state"]
    if built is None:
        return _failed(f"{where}: the network was not seen before its first training step")
    prefix = _wrapped(built, expected)
    if prefix is None:
        missing = sorted(set(expected) - set(built))
        extra = sorted(set(built) - set(expected))
        return _failed(
            f"{where}: the network's parameters and buffers are not the stated model's: "
            f"{len(missing)} of its {len(expected)} are missing ({_some(missing) or 'none'}) "
            f"and {len(extra)} others are present ({_some(extra) or 'none'})"
        )
    for name, value in expected.items():
        shape = tuple(built[prefix + name].shape)
        if shape != tuple(value.shape):
            return _failed(f"{where}: {prefix + name} has shape {shape}, not {tuple(value.shape)}")
    final = stated["final_layer"] + "."
    altered = [
        prefix + name
        for name, value in expected.items()
        if not name.startswith(final) and not torch.equal(built[prefix + name], value)
    ]
    if altered:
        return _failed(f"{where}: the pretrained values of {_some(altered)} were not kept")
    return _passed()


def _all_parameters_trained(seen: dict[str, Any]) -> dict[str, Any]:
    where = seen["context"]
    if not seen["unchanged"]:
        return _failed(f"{where}: no epoch was seen to train the network")
    for _, number, names in seen["unchanged"]:
        if names:
            return _failed(
                f"{where}: epoch {number} leaves {_count(len(names), 'parameter')} of the "
                f"network unchanged: {_some(names)}"
            )
    return _passed()


def _training_mode(seen: dict[str, Any]) -> dict[str, Any]:
    where = seen["context"]
    for row in seen["trained"]:
        if row["evaluating"]:
            name, whole = row["evaluating"], row["evaluating"] == "the network"
            what, trained = (
                (name, "it") if whole else (f"the network's module {name}", "the network")
            )
            return _failed(
                f"{where}: {what} is in evaluation mode while {trained} is trained "
                f"{_in_epoch(row['epoch'])}"
            )
    return _passed()


def _decay(value: Any) -> bool:
    try:
        return math.isclose(float(value), WEIGHT_DECAY, rel_tol=1e-6)
    except (TypeError, ValueError):
        return False


def _optimizer(seen: dict[str, Any]) -> dict[str, Any]:
    """AdamW as stated, holding every parameter once, one step per batch, on the mean
    cross-entropy."""
    where, steps, settings = seen["context"], seen["steps"], seen["optimizers"]
    if not steps:
        return _failed(f"{where}: the optimizer never steps")
    for setting in settings:
        if not setting["adamw"]:
            return _failed(f"{where}: the optimizer is {setting['kind']}, not AdamW")
        decays = [decay for decay in setting["weight_decays"] if not _decay(decay)]
        if decays:
            return _failed(f"{where}: the weight decay is {decays[0]}, not {WEIGHT_DECAY:g}")
        for names, how in ((setting["missing"], "does not hold"), (setting["repeated"], "holds")):
            if names:
                times = "" if how == "does not hold" else " more than once"
                return _failed(
                    f"{where}: the optimizer {how} {_count(len(names), 'parameter')} of the "
                    f"network{times}: {_some(names)}"
                )
    first = steps[0][2]
    if not all(math.isclose(rate, LEARNING_RATE, rel_tol=1e-6) for rate in first):
        rates = ", ".join(f"{rate:g}" for rate in first)
        return _failed(f"{where}: the first step's learning rate is {rates}, not {LEARNING_RATE:g}")
    batches = Counter((row["attempt"], row["epoch"]) for row in seen["trained"])
    taken = Counter((attempt, epoch) for attempt, epoch, _ in steps)
    for key in sorted(batches.keys() | taken.keys(), key=lambda key: (key[0], key[1] or 0)):
        if batches[key] != taken[key]:
            return _failed(
                f"{where}: {_in_epoch(key[1])} the optimizer takes "
                f"{_count(taken[key], 'step')} for {_count(batches[key], 'batch')}"
            )
    for row in seen["trained"]:
        when = _in_epoch(row["epoch"])
        if row["loss"] == "summed":
            return _failed(
                f"{where}: {when} the loss is the cross-entropy summed over the batch, not its mean"
            )
        if row["loss"] == "other":
            return _failed(
                f"{where}: {when} the gradient at the network's output is not that of the mean "
                "cross-entropy of the batch: the loss is weighted, smoothed or changed otherwise"
            )
    return _passed()


def _cosine(progress: float) -> float:
    return (1 + math.cos(math.pi * progress)) / 2


def _learning_rate_schedule(seen: dict[str, Any]) -> dict[str, Any]:
    """An uninterrupted run's rates, relative to the first step's, against a cosine over
    the run updated once per epoch."""
    where, epochs = seen["context"], seen["epochs"]
    rates = [
        (setting["started"], step_rates)
        for (_, _, step_rates), setting in zip(seen["steps"], seen["optimizers"], strict=True)
    ]
    if not rates or not all(rate > 0 for rate in rates[0][1]):
        return _failed(f"{where}: the first optimizer step has no positive learning rate")
    first = rates[0][1]
    factors: dict[int | None, list[list[float]]] = {}
    for epoch, step_rates in rates:
        if len(step_rates) != len(first):
            return _failed(f"{where}: the optimizer's parameter groups change during the run")
        factors.setdefault(epoch, []).append(
            [rate / start for rate, start in zip(step_rates, first, strict=True)]
        )
    if None in factors:
        return _failed(f"{where}: the optimizer steps before the first run.epoch")
    for epoch, found in sorted(factors.items()):
        if any(
            not math.isclose(factor, start, rel_tol=1e-9, abs_tol=1e-12)
            for step in found
            for factor, start in zip(step, found[0], strict=True)
        ):
            return _failed(
                f"{where}: the learning rate changes within epoch {epoch}; it must be updated "
                "once per epoch"
            )
    numbers = sorted(factors)
    expected = [_cosine((number - 1) / epochs) for number in numbers]
    if all(
        math.isclose(factor, wanted, abs_tol=1e-6)
        for number, wanted in zip(numbers, expected, strict=True)
        for factor in factors[number][0]
    ):
        return _passed()
    starts = ", ".join(f"{factors[number][0][0]:.4g}" for number in numbers)
    cosine = ", ".join(f"{wanted:.4g}" for wanted in expected)
    return _failed(
        f"{where}: the learning rates of epochs {_listed(numbers)} are {starts} times the "
        f"first; a cosine from the start of the run to 0 at its end gives {cosine}"
    )


def _listed(numbers: list[int]) -> str:
    return ", ".join(map(str, numbers)) or "none"


def _restart_difference(reference: dict[str, Any], restarted: dict[str, Any]) -> str | None:
    """The first way a restarted run trains differently from the uninterrupted one."""
    epochs = [number for _, number, _ in reference["ended"]]
    again = [number for _, number, _ in restarted["ended"]]
    if again != epochs:
        return f"trains epochs {_listed(again)} instead of {_listed(epochs)}"
    for number in epochs:
        batches = [
            [row["codes"] for row in seen["trained"] if row["epoch"] == number]
            for seen in (reference, restarted)
        ]
        if batches[0] != batches[1]:
            return f"trains epoch {number} on other batches"
        flips = [
            [row["flips"] for row in seen["trained"] if row["epoch"] == number]
            for seen in (reference, restarted)
        ]
        if flips[0] != flips[1]:
            return f"flips the images of epoch {number} otherwise"
        rates = [
            [step_rates for _, epoch, step_rates in seen["steps"] if epoch == number]
            for seen in (reference, restarted)
        ]
        if len(rates[0]) != len(rates[1]):
            return f"takes {len(rates[1])} optimizer steps in epoch {number} instead of {len(rates[0])}"
        for expected, found in zip(*rates, strict=True):
            if len(expected) != len(found) or not all(
                math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-15)
                for a, b in zip(expected, found, strict=True)
            ):
                return (
                    f"trains epoch {number} with learning rate {found[0]:.4g} "
                    f"instead of {expected[0]:.4g}"
                )
    for (_, number, expected), (_, _, found) in zip(
        reference["ended"], restarted["ended"], strict=True
    ):
        # The network is known once a backward pass reached it; a project that trains
        # outside run.epoch may end an epoch before that, which training_samples reports.
        if None not in (expected, found) and found != expected:
            return f"ends epoch {number} with other weights"
    return None


def _restart(reference: dict[str, Any], restarted: dict[str, Any]) -> dict[str, Any]:
    difference = _restart_difference(reference, restarted)
    if difference:
        return _failed(f"{restarted['context']}, the run {difference}")
    return _passed()


def measured_epochs(seen: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    """Accuracy, mean cross-entropy and weights of every epoch validated on every sample.

    Computed by the checks from the network's logits after the epoch and the labels the
    validation samples had in that run; keyed by (attempt, epoch).
    """
    validation = seen["validation"]
    targets = torch.tensor(validation._labels)
    measured = {}
    for key, validated in seen["validated"].items():
        logits = validated["logits"]
        if len(logits) < len(validation):
            continue
        rows = torch.stack([logits[code] for code in validation.codes])
        measured[key] = {
            "accuracy": float((rows.argmax(dim=1) == targets).double().mean()),
            "loss": float(nn.functional.cross_entropy(rows, targets)),
            "fingerprint": validated["fingerprint"],
        }
    return measured


def _reported(seen: dict[str, Any]) -> dict[tuple[int, int | None], dict[str, Any]]:
    reported: dict[tuple[int, int | None], dict[str, Any]] = {}
    for key, metrics in seen["logged"]:
        reported[key] = {**reported.get(key, {}), **metrics}
    return reported


def _validation(seen: dict[str, Any]) -> dict[str, Any]:
    measured, where, total = measured_epochs(seen), seen["context"], len(seen["validation"])
    reported = _reported(seen)
    for attempt, number, _ in seen["ended"]:
        key, after = (attempt, number), f"after epoch {number}"
        if key not in measured:
            validated = len(seen["validated"].get(key, {}).get("logits", {}))
            return _failed(
                f"{where}: {after} the network is measured on {validated} of the {total} "
                "validation samples"
            )
        metrics = reported.get(key, {})
        for name in ("validation_accuracy", "validation_loss"):
            if name not in metrics:
                return _failed(f"{where}: run.log reports no {name} {after}")
        accuracy, loss = measured[key]["accuracy"], measured[key]["loss"]
        if not math.isclose(float(metrics["validation_accuracy"]), accuracy, abs_tol=1e-9):
            return _failed(
                f"{where}: {after} run.log reports a validation accuracy of "
                f"{float(metrics['validation_accuracy']):.4g}; the network's is {accuracy:.4g}"
            )
        if not math.isclose(
            float(metrics["validation_loss"]), loss, rel_tol=LOSS_TOLERANCE, abs_tol=1e-6
        ):
            return _failed(
                f"{where}: {after} run.log reports a validation loss of "
                f"{float(metrics['validation_loss']):.4g}; the network's is {loss:.4g}"
            )
    return _passed()


def _reported_accuracy(metrics: dict[str, Any]) -> float | None:
    try:
        value = float(metrics["validation_accuracy"])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _delivery(seen: dict[str, Any]) -> dict[str, Any]:
    """The delivered weights against the epoch whose reported accuracy is the highest.

    Where an epoch's accuracy is not reported, the checks' own measurement stands in;
    whether the reports are right is validation's to say.
    """
    measured, reported, where = measured_epochs(seen), _reported(seen), seen["context"]
    delivered = [weights for _, weights in seen["delivered"]]
    if len(delivered) != 1:
        times = "no weights" if not delivered else f"weights {len(delivered)} times"
        return _failed(f"{where}: the run delivers {times}")
    epochs = [(attempt, number) for attempt, number, _ in seen["ended"]]
    if not epochs:
        return _failed(f"{where}: the run trained no epoch")
    accuracies = {}
    for key in epochs:
        accuracy = _reported_accuracy(reported.get(key, {}))
        if accuracy is None and key in measured:
            accuracy = measured[key]["accuracy"]
        if accuracy is None:
            return _failed(
                f"{where}: the delivery cannot be checked, since after epoch {key[1]} no "
                "validation accuracy is reported or measured on every validation sample"
            )
        accuracies[key] = accuracy
    weights = {(attempt, number): ended for attempt, number, ended in seen["ended"]}
    for key, validated in seen["validated"].items():
        weights[key] = validated["fingerprint"]
    # max keeps the first of equals, and epochs are in the order they were trained.
    best = max(epochs, key=accuracies.__getitem__)
    if delivered[0] == weights.get(best):
        return _passed()
    matching = [key for key in epochs if weights.get(key) == delivered[0]]
    what = f"the weights of epoch {matching[0][1]}" if matching else "weights no epoch ended with"
    return _failed(
        f"{where}: the run delivers {what} instead of those of epoch {best[1]}, whose "
        f"validation accuracy of {accuracies[best]:.3g} is the highest"
    )


def seen_complete(seen: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    """The validation passes of a run that cover every validation sample."""
    total = len(seen["validation"])
    return {key: value for key, value in seen["validated"].items() if len(value["logits"]) == total}


def steered_runs(
    sources: dict[str, str], rules: Spec, reference: dict[str, Any]
) -> tuple[list[dict[str, Any]], str]:
    """Runs whose best epoch falls before, and after, a restart; and what was not tested.

    The validation labels are the predictions of an epoch of the uninterrupted
    three-epoch run, which makes that epoch's accuracy 1 if the run repeats.
    """
    layout, seed = rules.layouts[0]
    predictions = {
        number: {code: int(row.argmax()) for code, row in validated["logits"].items()}
        for (_, number), validated in seen_complete(reference).items()
    }
    runs, untested = [], ""
    if 1 in predictions:
        seen = observe(
            sources,
            rules,
            layout,
            seed,
            epochs=RESTART_EPOCHS,
            stop_after=1,
            validation_labels=predictions[1],
        )
        seen["context"] = "labelled for epoch 1 to be the best, stopped after it and restarted"
        runs.append(seen)
    later = [
        number
        for number in range(2, RESTART_EPOCHS + 1)
        if number in predictions and predictions[number] != predictions.get(1)
    ]
    if later:
        seen = observe(
            sources,
            rules,
            layout,
            seed,
            epochs=RESTART_EPOCHS,
            stop_after=later[0] - 1,
            validation_labels=predictions[later[0]],
        )
        seen["context"] = (
            f"labelled for epoch {later[0]} to be the best, stopped before it and restarted"
        )
        runs.append(seen)
    elif predictions:
        untested = (
            "a best epoch after a restart was not tested: the network's validation "
            "predictions did not change from epoch to epoch"
        )
    return runs, untested


def contract_result(
    sources: dict[str, str],
    family: str = "camelyon_pipeline",
    healthy: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Check a project's modules against every clause of its family.

    healthy are the modules of the healthy project, whose model is the stated one: by
    default the family's own.
    """
    rules = spec(family)
    healthy = healthy or pipeline_tasks.healthy_sources(family)
    clauses: dict[str, dict[str, Any]] = {}

    def stated_model(seen: dict[str, Any]) -> dict[str, Any]:
        return _model(seen, reference_model(healthy))

    def check(name: str, test, *arguments: Any) -> None:
        if name in clauses and not clauses[name]["passed"]:
            return  # Keep the first failure of a clause.
        try:
            clauses[name] = test(*arguments)
        except Exception as error:
            clauses[name] = _failed(f"the check failed with {type(error).__name__}: {error}"[:300])

    def failed_with(error: Exception, where: str = "the project") -> dict[str, Any]:
        return _failed(f"{where} failed with {type(error).__name__}: {error}"[:300])

    layout, seed = rules.layouts[0]
    observations = []
    try:
        for name, number in rules.layouts:
            seen = observe(sources, rules, name, number, capture=True)
            seen["context"] = f"{name} layout, {EPOCHS} epochs"
            observations.append(seen)
            check("training_samples", _training_samples, seen, rules)
            check("training_labels", _training_labels, seen, rules)
            if rules.flips:
                check("flips", _flips, seen)
            check("model", stated_model, seen)
            seen["built"] = None  # A network's worth of tensors, needed no more.
            check("all_parameters_trained", _all_parameters_trained, seen)
            check("learning_rate_schedule", _learning_rate_schedule, seen)
        reference = observe(sources, rules, layout, seed, epochs=RESTART_EPOCHS)
        reference["context"] = f"{layout} layout, {RESTART_EPOCHS} epochs"
        observations.append(reference)
        # Over three epochs a linear decay no longer gives the cosine's rates.
        check("learning_rate_schedule", _learning_rate_schedule, reference)
    except Exception as error:
        failure = failed_with(error)
        for name in rules.clauses:
            if name not in clauses or clauses[name]["passed"]:
                clauses[name] = failure
        return _result(clauses, family)
    for stop in STOPS:
        where = f"stopped after epoch {stop} and restarted"
        try:
            seen = observe(sources, rules, layout, seed, epochs=RESTART_EPOCHS, stop_after=stop)
        except Exception as error:
            check("restart", failed_with, error, where + ", the project")
            continue
        seen["context"] = where
        observations.append(seen)
        check("restart", _restart, reference, seen)
    untested = ""
    try:
        steered, untested = steered_runs(sources, rules, reference)
        observations.extend(steered)
    except Exception as error:
        check("delivery", failed_with, error, "a run labelled to test the delivery")
    for seen in observations:
        check("inputs", _inputs, seen, rules)
        check("training_mode", _training_mode, seen)
        check("optimizer", _optimizer, seen)
        check("validation", _validation, seen)
        check("delivery", _delivery, seen)
    if untested and clauses["delivery"]["passed"]:
        clauses["delivery"] = _passed(untested)
    return _result(clauses, family)


def _result(clauses: dict[str, dict[str, Any]], family: str) -> dict[str, Any]:
    names = spec(family).clauses
    failed = [name for name in names if not clauses[name]["passed"]]
    return {
        "schema_version": 1,
        "task": "pipeline_contract",
        "contract_version": CONTRACT_VERSION,
        "family": family,
        "family_sha256": pipeline_tasks.family_sha256(family),
        "status": "failed" if failed else "passed",
        "failed": failed,
        "clauses": {name: clauses[name] for name in names},
    }


def _refused(problems: list[str], family: str = "camelyon_pipeline") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": "pipeline_contract",
        "contract_version": CONTRACT_VERSION,
        "family": family,
        "family_sha256": pipeline_tasks.family_sha256(family),
        "status": "refused",
        "failed": list(spec(family).clauses),
        "clauses": {},
        "problems": problems,
    }


def run_contract(
    project: Path, family: str = "camelyon_pipeline", healthy: Path | None = None
) -> dict[str, Any]:
    """Check a project folder in this process; a project the harness refuses fails.

    healthy is a folder with the healthy project, whose model is the stated one: by
    default the family's own.
    """
    sources = project_sources(project)
    problems = check_project(sources)
    if problems:
        return _refused(problems, family)
    return contract_result(sources, family, healthy and project_sources(healthy))


def run_contract_in_subprocess(
    project: Path,
    family: str = "camelyon_pipeline",
    *,
    healthy: Path | None = None,
    timeout: float = 900.0,
) -> dict[str, Any]:
    """Check a project in a child process without credentials; status may be crashed or timeout."""
    problems = check_project(project_sources(project))
    if problems:
        return _refused(problems, family)
    command = [sys.executable, "-m", "runsleuth.pipeline_contract", "run"]
    command += ["--project", str(Path(project).resolve()), "--family", family]
    if healthy is not None:
        command += ["--healthy", str(Path(healthy).resolve())]
    unfinished = {**_refused([], family), "status": "crashed"}
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=clean_environment(),
        )
    except subprocess.TimeoutExpired:
        return {**unfinished, "status": "timeout", "problems": [f"no result in {timeout:g} s"]}
    try:
        return json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        return {**unfinished, "problems": [(completed.stdout + completed.stderr)[-1000:]]}


@dataclass(frozen=True, slots=True)
class Case:
    """A project, as edits (module, old, new) of the healthy one, and the clauses it should fail."""

    fails: tuple[str, ...]
    summary: str
    edits: tuple[tuple[str, str, str], ...]


VERSION_CLAUSES = {
    "camelyon_pipeline": {
        # Its images come from the wrong positions while its labels come from the right ones.
        "positions_index_base": ("training_samples", "training_labels"),
        "labels_not_filtered": ("training_labels",),
        "scheduler_not_restored": ("restart",),
        "optimizer_not_restored": ("restart",),
        "last_epoch_delivered": ("delivery",),
        "best_accuracy_not_restored": ("delivery",),
        "pretrained_weights_delivered": ("delivery",),
    },
    "ham10000_pipeline": {
        "epoch_count_not_restored": ("restart",),
        "shuffle_generator_made_once": ("restart",),
        # A restarted run caches other images of a lesion, or other flips, than the
        # uninterrupted run did, so it does not train as that run either.
        "cache_keyed_by_lesion": ("training_samples", "restart"),
        "flipped_image_cached": ("flips", "restart"),
        "akiec_left_out_of_validation": ("validation",),
        "akiec_left_out_of_training": ("training_labels",),
    },
}
_IMPORT = "from torch.utils.data import DataLoader, Dataset\n"
_ITEM = "        return self.patches.image(self.indices[position]), self.labels[position]\n"
_SET = "    return HospitalSubset(patches, config.TRAINING_HOSPITALS)\n"
_TRAINING = (
    "        with run.epoch(epoch):\n"
    "            train_one_epoch(model, loader, optimizer, loss_function, run.device)\n"
)
# Lines of train.py that both families share.
_TRAINED_WITH = "            train_one_epoch(model, loader, optimizer, loss_function, run.device)\n"
_TRAINED_LOSS = "        loss = loss_function(model(data.normalize(images)), labels)\n"
_VALIDATED_LOGITS = "        logits = model(data.normalize(images))\n"
_HEAD = "    model.fc = nn.Linear(model.fc.in_features, 2)\n"
_LOSS = "    loss_function = nn.CrossEntropyLoss()\n"
# Projects beyond the versions, to see that the checks neither accept only the healthy
# project's wording nor reject only the faults they were written against.
CAMELYON_CASES = {
    "torch_subset": Case(
        (),
        "The training set is a torch Subset of the kept indices.",
        (
            ("data", _IMPORT, _IMPORT.replace("Dataset", "Dataset, Subset")),
            (
                "data",
                _SET,
                "    kept = [\n"
                "        index\n"
                "        for index, hospital in enumerate(patches.hospitals)\n"
                "        if hospital in config.TRAINING_HOSPITALS\n"
                "    ]\n"
                "    return Subset(patches, kept)\n",
            ),
        ),
    ),
    "label_read_with_the_patch": Case(
        (),
        "The wrapper returns the patch and label the patches give for a kept index.",
        (("data", _ITEM, "        return self.patches[self.indices[position]]\n"),),
    ),
    "validation_with_gradients": Case(
        (),
        "Validation runs without torch.no_grad, so its forwards build graphs nobody uses.",
        (("train", "@torch.no_grad()\ndef evaluate", "def evaluate"),),
    ),
    "training_accuracy_measured": Case(
        (),
        "After every epoch the project also measures its accuracy on the training patches.",
        (
            (
                "train",
                "        scheduler.step()\n",
                "        scheduler.step()\n"
                "        evaluate(model, data.training_loader(training, run.seed, epoch), "
                "loss_function, run.device)\n",
            ),
        ),
    ),
    "validation_hospital_kept": Case(
        ("training_samples",),
        "Hospital 1 is among the training hospitals.",
        (("config", "TRAINING_HOSPITALS = (0, 3, 4)", "TRAINING_HOSPITALS = (0, 1, 3, 4)"),),
    ),
    "training_hospital_left_out": Case(
        ("training_samples",),
        "Hospital 4 is not among the training hospitals.",
        (("config", "TRAINING_HOSPITALS = (0, 3, 4)", "TRAINING_HOSPITALS = (0, 3)"),),
    ),
    "last_batch_dropped": Case(
        ("training_samples",),
        "The loader drops the last, incomplete batch of every epoch.",
        (("data", "generator=generator)\n", "generator=generator, drop_last=True)\n"),),
    ),
    "every_patch_twice": Case(
        ("training_samples",),
        "The training set holds the kept patches twice.",
        (
            ("data", _IMPORT, _IMPORT.replace("DataLoader", "ConcatDataset, DataLoader")),
            (
                "data",
                _SET,
                "    kept = HospitalSubset(patches, config.TRAINING_HOSPITALS)\n"
                "    return ConcatDataset([kept, kept])\n",
            ),
        ),
    ),
    "positions_index_base_with_their_labels": Case(
        ("training_samples",),
        "The wrapper reads patch and label at its own positions: wrong patches, their labels.",
        (("data", _ITEM, "        return self.patches[position]\n"),),
    ),
    "labels_one_position_off": Case(
        ("training_labels",),
        "Every patch gets the label of the patch kept before it.",
        (("data", "self.labels[position]\n", "self.labels[position - 1]\n"),),
    ),
    "trained_outside_the_epoch": Case(
        ("training_samples",),
        "The epoch is trained after its run.epoch block instead of inside it.",
        (
            (
                "train",
                _TRAINING,
                "        with run.epoch(epoch):\n"
                "            pass\n"
                "        train_one_epoch(model, loader, optimizer, loss_function, run.device)\n",
            ),
        ),
    ),
    "epoch_counter_restarts": Case(
        ("restart",),
        "A restart restores the checkpoint but counts the epochs from 1 again.",
        (
            (
                "train",
                '        start, best_accuracy = state["epoch"] + 1, state["best_accuracy"]\n',
                '        start, best_accuracy = 1, state["best_accuracy"]\n',
            ),
        ),
    ),
    "weights_not_restored": Case(
        ("restart",),
        "A restart restores the optimizer and the schedule but not the weights.",
        (("train", '        model.load_state_dict(state["model"])\n', ""),),
    ),
    "checkpoint_written_before_the_schedule_steps": Case(
        ("restart",),
        "The schedule steps after the checkpoint is written, so a restart resumes one step "
        "behind; an uninterrupted run trains as the healthy one.",
        (
            (
                "train",
                "        scheduler.step()\n        metrics = evaluate",
                "        metrics = evaluate",
            ),
            (
                "train",
                "            },\n        )\n    run.deliver",
                "            },\n        )\n        scheduler.step()\n    run.deliver",
            ),
        ),
    ),
    "shuffle_counted_from_the_restart": Case(
        ("restart",),
        "The shuffle is seeded by the epochs trained since the start of the attempt, so a "
        "restarted run repeats the first epoch's order.",
        (
            (
                "train",
                "        loader = data.training_loader(training, run.seed, epoch)\n",
                "        loader = data.training_loader(training, run.seed, epoch - start + 1)\n",
            ),
        ),
    ),
    "best_before_the_restart_kept_for_good": Case(
        ("delivery",),
        "A repair of a lost best accuracy that, after a restart, keeps the best checkpoint "
        "written before it whatever comes later.",
        (
            (
                "train",
                '        start, best_accuracy = state["epoch"] + 1, state["best_accuracy"]\n',
                '        start, best_accuracy = state["epoch"] + 1, float("inf")\n',
            ),
        ),
    ),
    "accuracy_reported_as_a_percentage": Case(
        ("validation",),
        "run.log reports the validation accuracy times 100; the choice of the best epoch "
        "is unchanged.",
        (
            (
                "train",
                '"validation_accuracy": correct / count',
                '"validation_accuracy": 100 * correct / count',
            ),
        ),
    ),
    "validation_loss_summed": Case(
        ("validation",),
        "run.log reports the validation loss summed over the patches instead of averaged.",
        (("train", '"validation_loss": loss / count', '"validation_loss": loss'),),
    ),
    "validation_on_the_first_batch": Case(
        ("validation",),
        "Validation stops after its first batch, so it measures 32 of the patches, and the "
        "best epoch is chosen by what it reports.",
        (
            (
                "train",
                "        count += len(labels)\n    return",
                "        count += len(labels)\n        break\n    return",
            ),
        ),
    ),
    "training_patches_not_normalized": Case(
        ("inputs",),
        "Training gives the network the patches as they come; validation normalizes them.",
        (("train", _TRAINED_LOSS, _TRAINED_LOSS.replace("data.normalize(images)", "images")),),
    ),
    "validation_normalized_twice": Case(
        ("inputs",),
        "Validation normalizes the patches twice.",
        (
            (
                "train",
                _VALIDATED_LOGITS,
                "        logits = model(data.normalize(data.normalize(images)))\n",
            ),
        ),
    ),
    "normalized_inside_the_network": Case(
        ("inputs",),
        "The model normalizes the patches in a first layer of its own, so the network is "
        "given them as they come, which the statement rules out.",
        (
            (
                "train",
                "\n\ndef build_model",
                "\n\nclass Normalize(nn.Module):\n"
                "    def forward(self, images):\n"
                "        return data.normalize(images)\n"
                "\n\ndef build_model",
            ),
            (
                "train",
                _HEAD + "    return model\n",
                _HEAD + "    return nn.Sequential(Normalize(), model)\n",
            ),
            ("train", _TRAINED_LOSS, _TRAINED_LOSS.replace("data.normalize(images)", "images")),
            ("train", _VALIDATED_LOGITS, "        logits = model(images)\n"),
        ),
    ),
    "network_in_a_sequential": Case(
        (),
        "The model is the stated network inside an nn.Sequential, which adds nothing.",
        (("train", _HEAD + "    return model\n", _HEAD + "    return nn.Sequential(model)\n"),),
    ),
    "imagenet_head_kept": Case(
        ("model",),
        "The model keeps ResNet18's ImageNet head of 1000 outputs, which the cross-entropy "
        "still trains on labels 0 and 1.",
        (("train", _HEAD, ""),),
    ),
    "backbone_frozen": Case(
        ("all_parameters_trained",),
        "The pretrained layers are frozen before the new head is added, so only the head "
        "is trained.",
        (
            (
                "train",
                _HEAD,
                "    for parameter in model.parameters():\n"
                "        parameter.requires_grad_(False)\n" + _HEAD,
            ),
        ),
    ),
    "trained_in_eval_mode": Case(
        ("training_mode",),
        "Training puts the network in evaluation mode instead of training mode.",
        (("train", "    model.train()\n", "    model.eval()\n"),),
    ),
    "sgd_instead_of_adamw": Case(
        ("optimizer",),
        "The optimizer is SGD with momentum, at the stated learning rate and weight decay.",
        (
            ("train", "torch.optim.AdamW(", "torch.optim.SGD("),
            (
                "train",
                "weight_decay=config.WEIGHT_DECAY\n",
                "weight_decay=config.WEIGHT_DECAY, momentum=0.9\n",
            ),
        ),
    ),
    "label_smoothing": Case(
        ("optimizer",),
        "Training minimizes a cross-entropy with label smoothing 0.1; validation measures "
        "the plain one.",
        (
            ("train", _LOSS, _LOSS + "    smoothed = nn.CrossEntropyLoss(label_smoothing=0.1)\n"),
            ("train", _TRAINED_WITH, _TRAINED_WITH.replace("loss_function", "smoothed")),
        ),
    ),
    "head_left_out_of_the_optimizer": Case(
        ("all_parameters_trained", "optimizer"),
        "The optimizer holds every parameter but the new head's, which keep their first values.",
        (
            (
                "train",
                "        model.parameters(), lr=",
                "        [value for name, value in model.named_parameters() "
                'if not name.startswith("fc.")],\n'
                "        lr=",
            ),
        ),
    ),
    "linear_decay": Case(
        ("learning_rate_schedule",),
        "The learning rate decays linearly to 0 over the run: the cosine's rates over two "
        "epochs, but not over three.",
        (
            (
                "train",
                "torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=run.epochs)",
                "torch.optim.lr_scheduler.LinearLR(\n"
                "        optimizer, start_factor=1.0, end_factor=0.0, total_iters=run.epochs\n"
                "    )",
            ),
        ),
    ),
}
_HAM_RETURNED = "        return flip(self.cache[key], self.generator), self.labels[position]\n"
_HAM_PREPARED = "        images, labels = images.to(device).float() / 255, torch.tensor(labels, device=device)\n"
HAM_CASES = {
    "no_flips": Case(
        ("flips",),
        "Training images are never flipped.",
        (("data", _HAM_RETURNED, "        return self.cache[key], self.labels[position]\n"),),
    ),
    "validation_flipped_too": Case(
        ("flips",),
        "Validation flips every image left-right as well.",
        (
            (
                "train",
                _HAM_PREPARED,
                "        images = images.to(device).float().flip(-1) / 255\n"
                "        labels = torch.tensor(labels, device=device)\n",
            ),
        ),
    ),
    "akiec_benign_everywhere": Case(
        ("training_labels", "validation"),
        "Both lists of malignant diagnoses leave out akiec.",
        (
            ("config", 'MALIGNANT = ("mel", "bcc", "akiec")', 'MALIGNANT = ("mel", "bcc")'),
            (
                "train",
                'VALIDATED_MALIGNANT = ("mel", "bcc", "akiec")',
                'VALIDATED_MALIGNANT = ("mel", "bcc")',
            ),
        ),
    ),
    "last_batch_dropped": Case(
        ("training_samples",),
        "The loader drops the last, incomplete batch of every epoch.",
        (("data", "generator=generator)\n", "generator=generator, drop_last=True)\n"),),
    ),
    "last_epoch_delivered": Case(
        ("delivery",),
        "The run delivers the weights of its last epoch instead of the best checkpoint.",
        (
            (
                "train",
                "    run.deliver(run.checkpoints.read(config.BEST_CHECKPOINT))\n",
                "    run.deliver(model.state_dict())\n",
            ),
        ),
    ),
    "flips_drawn_once_per_attempt": Case(
        ("restart",),
        "The flips come from a generator made when main starts, so a restarted run flips "
        "the images as the first epoch did; the shuffle is the epoch's.",
        (
            (
                "data",
                "    dataset.generator = generator\n",
                "    if dataset.generator is None:\n"
                "        dataset.generator = torch.Generator().manual_seed(seed)\n",
            ),
        ),
    ),
    "validation_not_divided_by_255": Case(
        ("inputs",),
        "Validation normalizes the images without dividing them by 255 first.",
        (("train", _HAM_PREPARED, _HAM_PREPARED.replace(".float() / 255", ".float()")),),
    ),
    "normalized_with_torchvision": Case(
        (),
        "The images are normalized by torchvision's normalize.",
        (
            ("data", _IMPORT, _IMPORT + "from torchvision.transforms import functional\n"),
            (
                "data",
                "    mean = torch.tensor(config.MEAN, device=images.device).view(1, 3, 1, 1)\n"
                "    std = torch.tensor(config.STD, device=images.device).view(1, 3, 1, 1)\n"
                "    return (images - mean) / std\n",
                "    return functional.normalize(images, config.MEAN, config.STD)\n",
            ),
        ),
    ),
    "pretrained_weights_not_loaded": Case(
        ("model",),
        "The ResNet18 is built without its pretrained weights.",
        (
            (
                "train",
                "resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)",
                "resnet18(weights=None)",
            ),
        ),
    ),
    "head_frozen": Case(
        ("all_parameters_trained",),
        "The new head is frozen, so only the pretrained layers are trained.",
        (("train", _HEAD, _HEAD + "    model.fc.requires_grad_(False)\n"),),
    ),
    "batch_norm_in_eval_mode": Case(
        ("training_mode",),
        "Training keeps batch normalization in evaluation mode, with the pretrained statistics.",
        (
            (
                "train",
                "    model.train()\n",
                "    model.train()\n"
                "    for module in model.modules():\n"
                "        if isinstance(module, nn.BatchNorm2d):\n"
                "            module.eval()\n",
            ),
        ),
    ),
    "adam_instead_of_adamw": Case(
        ("optimizer",),
        "The optimizer is Adam, whose weight decay is an L2 term in the gradient instead "
        "of AdamW's decoupled decay.",
        (("train", "torch.optim.AdamW(", "torch.optim.Adam("),),
    ),
    "loss_weighted_by_class": Case(
        ("optimizer",),
        "Training weights the cross-entropy of malignant images three times; validation "
        "measures the plain one.",
        (
            (
                "train",
                _LOSS,
                _LOSS + "    weighted = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0], "
                "device=run.device))\n",
            ),
            ("train", _TRAINED_WITH, _TRAINED_WITH.replace("loss_function", "weighted")),
        ),
    ),
    "schedule_period_doubled": Case(
        ("learning_rate_schedule",),
        "The cosine runs over twice the run's epochs, so the learning rate ends far from 0.",
        (("train", "T_max=run.epochs", "T_max=2 * run.epochs"),),
    ),
}
CASES = {"camelyon_pipeline": CAMELYON_CASES, "ham10000_pipeline": HAM_CASES}


def expected_failures(name: str, family: str = "camelyon_pipeline") -> tuple[str, ...]:
    """The clauses a version or case should fail; none for a correct project."""
    if name in CASES[family]:
        return CASES[family][name].fails
    if name == HEALTHY or pipeline_tasks.version(name, family).correct:
        return ()
    return VERSION_CLAUSES[family][name]


def case_sources(name: str, family: str = "camelyon_pipeline") -> dict[str, str]:
    """The modules of a version or of a case."""
    if name in pipeline_tasks.family(family).names:
        return pipeline_tasks.version_sources(name, family)
    sources = pipeline_tasks.healthy_sources(family)
    for module, old, new in CASES[family][name].edits:
        if sources[module].count(old) != 1:
            raise ValueError(f"{name}: every edit must match its module exactly once")
        sources[module] = sources[module].replace(old, new)
    return sources


def check_cases(*, sources=None, family: str = "camelyon_pipeline") -> list[dict[str, Any]]:
    """Run the contract on every version and case and compare with what each should fail."""
    sources = sources or (lambda name: case_sources(name, family))
    versions = pipeline_tasks.family(family).names
    healthy = sources(HEALTHY)
    rows = []
    for name in (*versions, *CASES[family]):
        result = contract_result(sources(name), family, healthy)
        expected = list(expected_failures(name, family))
        rows.append(
            {
                "name": name,
                "kind": "version" if name in versions else "case",
                "expected": expected,
                "failed": result["failed"],
                "as_expected": sorted(result["failed"]) == sorted(expected),
                "details": [result["clauses"][clause]["detail"] for clause in result["failed"]],
                "notes": [
                    f"{clause}: {outcome['detail']}"
                    for clause, outcome in result["clauses"].items()
                    if outcome["passed"] and outcome["detail"]
                ],
            }
        )
    return rows


def cases_text(rows: list[dict[str, Any]]) -> str:
    """One line per project, why it failed, and what a passing clause could not test."""
    lines = []
    for row in rows:
        verdict = "as expected" if row["as_expected"] else f"EXPECTED {row['expected'] or 'none'}"
        lines.append(f"{row['name']:<40} {', '.join(row['failed']) or 'passes':<36} {verdict}")
        lines.extend(f"    {detail}" for detail in row["details"])
        lines.extend(f"    note, {note}" for note in row.get("notes", []))
    agreed = sum(row["as_expected"] for row in rows)
    correct = sum(not row["expected"] for row in rows)
    lines.append(
        f"{agreed}/{len(rows)} projects as expected "
        f"({correct} that should pass, {len(rows) - correct} that should fail)"
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Contract tests for training projects.")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Check one project; prints the result as JSON.")
    run.add_argument("--project", type=Path, required=True)
    run.add_argument(
        "--healthy", type=Path, help="The healthy project, by default the family's own."
    )
    check = commands.add_parser("check", help="Check every version and case against expectations.")
    for command in (run, check):
        command.add_argument("--family", choices=tuple(SPECS), default="camelyon_pipeline")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        print(json.dumps(run_contract(args.project, args.family, args.healthy)))
        return 0
    rows = check_cases(family=args.family)
    print(cases_text(rows))
    return 0 if all(row["as_expected"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
