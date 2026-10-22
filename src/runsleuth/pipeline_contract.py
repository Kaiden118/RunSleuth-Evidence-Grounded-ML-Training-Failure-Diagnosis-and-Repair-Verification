"""Contract tests for training projects: does a project do what its statement says?

TASK.md (pipeline_tasks) is the public specification an agent works from. This module
turns its requirements into checks that the agent never sees and that run only after
the agent has finished:

    training_samples   1. every epoch trains on each patch of hospitals 0, 3 and 4
                          exactly once, and on no other patch, inside run.epoch
    training_labels    1. each of those patches is trained on with its own label
    restart            5. a run stopped as it starts an epoch and restarted trains
                          exactly as the uninterrupted run
    validation         6. after every epoch run.log reports the mean cross-entropy and
                          the accuracy of the network on all the validation patches
    delivery           6. the delivered weights are those of the epoch with the highest
                          validation accuracy in the whole run, the earliest on a tie

The other requirements get their clauses with the tasks that need them. Version 1 had
the first two; version 2 (2026-10-22) added restart; version 3 (2026-10-22) adds
validation and delivery, and gives the patches content a network can learn.

The checks call the project's main() as the harness does, with a run of their own: on
the CPU, with candidates whose pixels say which patch they are, in a pattern that
per-channel normalization keeps. A patch counts as trained on when the gradient of a
backward pass reaches the network's output for it, so a forward without a backward
(validation, or measuring training accuracy) does not count, whether or not gradients
were enabled. Its label is the target of the cross-entropy computed on that output.
For the training data the project trains two epochs on candidates in two layouts,
hospital by hospital as Camelyon17 lists them and mixed, with unequal numbers of
patches per hospital and more than a batch. For the restart it trains three epochs
uninterrupted, and again stopped after the first and after the second epoch and
restarted, as the harness restarts it: main is called again in freshly loaded modules
with the checkpoints written so far. The restarted runs must train the same epochs, on
the same batches, with the same learning rates at every step, ending every epoch with
the same weights; the optimizer's state shows in the weights of the epochs after it.

Validation and delivery are judged by what the project did, not by how: whatever
keeps the best epoch, the checks compute each epoch's validation accuracy themselves,
from the network's predictions on the validation patches in that run and their
labels, and require that run.log reports it and that the delivered weights are those
the network had when the best epoch was validated. Every run above is judged so. Two
more runs make the best epoch fall on either side of a restart, with labels chosen
from the predictions of the uninterrupted three-epoch run and fixed for the whole run:
labelled as the first epoch predicted and stopped after it, the best epoch is the
first, which the restarted run must keep; labelled as the first epoch that predicted
otherwise and stopped before it, the best epoch comes after the restart, which the
restarted run must take. If the network never changes its predictions, the second
cannot be made and the result says so.

Passing every check is this benchmark's working definition of a project that meets the
requirements the checks cover. It is not a proof that the project is correct.

The checks share no code with the verifier or with the harness's recording: they use
pipeline_workspace only to read and load a project under its guards, and never read a
run report, evidence or signatures.

    python -m runsleuth.pipeline_contract run --project projects/labels_not_filtered
    python -m runsleuth.pipeline_contract check
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

CONTRACT_VERSION = 3
CLAUSES = ("training_samples", "training_labels", "restart", "validation", "delivery")
TRAINING_HOSPITALS = (0, 3, 4)
# Candidates per hospital: unequal, and more than a batch of 32 in all; validation
# patches too, so that a project that measures one batch only is seen.
CANDIDATES = {0: 9, 1: 7, 3: 14, 4: 17}
VALIDATION = {0: 12, 3: 14, 4: 14}
# How close a reported validation loss must be to the checks' own, relative.
LOSS_TOLERANCE = 1e-4
# Each layout runs with a seed of its own.
LAYOUTS = (("by_hospital", 3), ("mixed", 5))
EPOCHS = 2
# The restart: three epochs, stopped after the first or the second. After the first, a
# schedule computed step by step from the current rate and not restored shows only in
# the third epoch.
RESTART_EPOCHS = 3
STOPS = (1, 2)
PATCH_SIZE = 96
# A patch's number is written in bits of channel 0 and their complement in channel 1,
# after a dark and a bright reference pixel; validation patches are numbered from
# VALIDATION_CODES on.
BITS = 12
VALIDATION_CODES = 1 << 11


def _passed(detail: str = "") -> dict[str, Any]:
    return {"passed": True, "detail": detail}


def _failed(detail: str) -> dict[str, Any]:
    return {"passed": False, "detail": detail}


def coded_images(codes: list[int], brightness: list[float]) -> torch.Tensor:
    """RGB patches in [0, 1], each showing its code.

    The other pixels are noise of width 0.2 around the patch's brightness, so that a
    network can learn to tell bright from dark patches.
    """
    images = torch.empty(len(codes), 3, PATCH_SIZE * PATCH_SIZE)
    for row, (code, level) in enumerate(zip(codes, brightness, strict=True)):
        noise = torch.rand(images.shape[1:], generator=torch.Generator().manual_seed(code))
        images[row] = level - 0.1 + 0.2 * noise
    images[:, :, 0], images[:, :, 1] = 0.0, 1.0
    bits = torch.tensor([[(code >> bit) & 1 for bit in range(BITS)] for code in codes])
    images[:, 0, 2 : 2 + BITS] = bits.float()
    images[:, 1, 2 : 2 + BITS] = 1 - bits.float()
    return images.view(len(codes), 3, PATCH_SIZE, PATCH_SIZE)


def read_codes(inputs: torch.Tensor) -> list[int | None]:
    """The code of every patch in a batch, after any per-channel increasing affine map.

    None for a patch whose pixels show no code: changed otherwise, or not a patch.
    """
    count = len(inputs) if inputs.dim() else 0
    if inputs.dim() != 4 or inputs.shape[1] != 3 or inputs[0, 0].numel() < 2 + BITS:
        return [None] * count
    values = inputs.detach().to("cpu", torch.float64).flatten(2)[:, :2, : 2 + BITS]
    dark, bright = values[:, :, :1], values[:, :, 1:2]
    bits = values[:, :, 2:] > (dark + bright) / 2
    shown = (bright > dark).all(dim=2).all(dim=1) & (bits[:, 0] != bits[:, 1]).all(dim=1)
    codes = (bits[:, 0].long() * 2 ** torch.arange(BITS)).sum(dim=1)
    return [
        int(code) if ok else None for code, ok in zip(codes.tolist(), shown.tolist(), strict=True)
    ]


class CodedPatches:
    """Patches as the harness gives them: samples[i], samples.image(i), labels, hospitals."""

    def __init__(
        self,
        codes: list[int],
        labels: list[int],
        hospitals: list[int],
        brightness: list[float],
    ) -> None:
        self._images = coded_images(codes, brightness)
        self.codes, self.labels = tuple(codes), tuple(labels)
        self.hospitals = tuple(hospitals)

    def __len__(self) -> int:
        return len(self.labels)

    def _position(self, index) -> int:
        position = operator.index(index)
        position += len(self) if position < 0 else 0
        if not 0 <= position < len(self):
            raise IndexError(f"patch {index} is out of range for {len(self)} patches")
        return position

    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        position = self._position(index)
        return self._images[position], self.labels[position]

    def image(self, index) -> torch.Tensor:
        return self._images[self._position(index)]


def candidates(layout: str, seed: int) -> CodedPatches:
    """Training candidates numbered by position, labels drawn at random, label 1 brighter."""
    hospitals = [hospital for hospital, count in CANDIDATES.items() for _ in range(count)]
    if layout == "mixed":
        order = torch.randperm(len(hospitals), generator=torch.Generator().manual_seed(seed))
        hospitals = [hospitals[position] for position in order.tolist()]
    generator = torch.Generator().manual_seed(1000 + seed)
    labels = torch.randint(0, 2, (len(hospitals),), generator=generator).tolist()
    brightness = [0.3 + 0.4 * label for label in labels]
    return CodedPatches(list(range(len(hospitals))), labels, hospitals, brightness)


def validation_patches(labels: dict[int, int] | None = None) -> CodedPatches:
    """Validation patches labelled by brightness, or by labels[code].

    Their brightness is spread evenly between the two training classes', so that some
    lie near the network's boundary and its predictions change as it trains.
    """
    hospitals = [hospital for hospital, count in VALIDATION.items() for _ in range(count)]
    codes = [VALIDATION_CODES + position for position in range(len(hospitals))]
    order = torch.randperm(len(codes), generator=torch.Generator().manual_seed(len(codes)))
    brightness = [0.3 + 0.4 * rank / (len(codes) - 1) for rank in order.tolist()]
    given = [int(level > 0.5) for level in brightness] if labels is None else None
    return CodedPatches(codes, given or [labels[code] for code in codes], hospitals, brightness)


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
        train: CodedPatches,
        validation: CodedPatches,
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
        self._epoch: int | None = None
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
        self._epoch = number
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


def observe(
    sources: dict[str, str],
    layout: str,
    seed: int,
    *,
    epochs: int = EPOCHS,
    stop_after: int | None = None,
    validation_labels: dict[int, int] | None = None,
) -> dict[str, Any]:
    """Run a project on coded candidates, perhaps stopped and restarted; what it did.

    trained has one row per backward pass that reached the network's output: the
    attempt and the epoch it happened in (None outside run.epoch), the codes of the
    batch the network was given, and the targets of the cross-entropy on that output
    (None if there was none). steps has the attempt, epoch and learning rates of every
    optimizer step, and ended the attempt, number and network fingerprint of every
    epoch. validated maps (attempt, the attempt's last ended epoch) to the network's
    fingerprint and logits per validation code in the first pass over them after that
    epoch; logged and delivered hold the same key with every run.log, and the attempt
    and fingerprint of every delivery. With stop_after, the first attempt is stopped as
    it starts an epoch after that many, and main is called again in freshly loaded
    modules with its checkpoints. validation_labels relabel the validation patches.
    """
    train, validation = candidates(layout, seed), validation_patches(validation_labels)
    checkpoints = MemoryCheckpoints()
    seen: dict[str, Any] = {"layout": layout, "train": train, "trained": [], "steps": []}
    seen.update(ended=[], validated={}, logged=[], delivered=[], validation=validation)
    state: dict[str, Any] = {"depth": 0, "batch": None, "network": None, "attempt": 0}
    state["last_ended"] = None
    run: ContractRun | None = None
    validation_codes = set(validation.codes)

    def before(module, args) -> None:
        state["depth"] += 1
        if state["depth"] == 1 and args and isinstance(args[0], torch.Tensor):
            if args[0].dim() == 4:
                # The outermost call on a batch of images: the network.
                codes = read_codes(args[0])
                state["batch"] = {"codes": codes, "module": module, "output": None, "targets": None}

    def after(module, _args, output) -> None:
        state["depth"] -= 1
        batch = state["batch"]
        if state["depth"] or batch is None or batch["output"] is not None:
            return
        if isinstance(output, torch.Tensor):
            batch["output"] = output
            validated(module, batch["codes"], output)
            if output.requires_grad:
                output.register_hook(lambda _gradient, batch=batch: backward(batch))

    def validated(module, codes: list[int | None], output: torch.Tensor) -> None:
        """Keep the first logits per validation patch after each ended epoch."""
        if state["last_ended"] is None or output.dim() != 2 or len(output) != len(codes):
            return
        if not validation_codes.intersection(codes):
            return
        key = (state["attempt"], state["last_ended"])
        if key not in seen["validated"]:
            seen["validated"][key] = {"fingerprint": fingerprint(module.state_dict()), "logits": {}}
        logits = seen["validated"][key]["logits"]
        for code, row in zip(codes, output.detach().to("cpu", torch.float64), strict=True):
            if code in validation_codes and code not in logits:
                logits[code] = row

    def backward(batch: dict[str, Any]) -> None:
        state["network"] = batch["module"]
        seen["trained"].append(
            {
                "attempt": state["attempt"],
                "epoch": run._epoch,
                "codes": batch["codes"],
                "targets": batch["targets"],
            }
        )

    def stepped(optimizer, _args, _kwargs) -> None:
        rates = [float(group["lr"]) for group in optimizer.param_groups]
        seen["steps"].append((state["attempt"], run._epoch, rates))

    def epoch_ended(number: int) -> None:
        network = state["network"]
        weights = network and fingerprint(network.state_dict())
        seen["ended"].append((state["attempt"], number, weights))
        state["last_ended"] = number

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


def _count(number: int, noun: str) -> str:
    plural = noun + ("es" if noun.endswith("ch") else "s")
    return f"{number} {noun if number == 1 else plural}"


def _training_samples(seen: dict[str, Any]) -> dict[str, Any]:
    train, trained, layout = seen["train"], seen["trained"], seen["layout"]
    if not trained:
        return _failed("the network was never trained: no backward pass reached its output")
    if any(row["epoch"] is None for row in trained):
        return _failed("the network is trained outside run.epoch")
    validation = range(VALIDATION_CODES, VALIDATION_CODES + sum(VALIDATION.values()))
    codes = [code for row in trained for code in row["codes"]]
    unread = sum(code is None or not (code < len(train) or code in validation) for code in codes)
    if unread:
        return _failed(
            f"{_count(unread, 'trained input')} could not be read back as a patch: the network "
            "must be given the patches normalized as stated and otherwise unchanged"
        )
    expected = [
        position
        for position, hospital in enumerate(train.hospitals)
        if hospital in TRAINING_HOSPITALS
    ]
    for epoch in sorted({row["epoch"] for row in trained}):
        counts = Counter(code for row in trained if row["epoch"] == epoch for code in row["codes"])
        problems = []
        validated = sum(count for code, count in counts.items() if code in validation)
        if validated:
            problems.append(f"trains on {_count(validated, 'validation patch')}")
        others = Counter(
            train.hospitals[code]
            for code, count in counts.items()
            for _ in range(count)
            if code < len(train) and train.hospitals[code] not in TRAINING_HOSPITALS
        )
        problems.extend(
            f"trains on {_count(count, 'patch')} from hospital {hospital}"
            for hospital, count in sorted(others.items())
        )
        missing = sum(counts[position] == 0 for position in expected)
        if missing:
            problems.append(
                f"leaves out {missing} of the {len(expected)} patches of hospitals 0, 3 and 4"
            )
        repeated = sum(counts[position] > 1 for position in expected)
        if repeated:
            problems.append(f"trains on {_count(repeated, 'patch')} more than once")
        if problems:
            return _failed(f"epoch {epoch} ({layout} layout): " + "; ".join(problems))
    return _passed()


def _training_labels(seen: dict[str, Any]) -> dict[str, Any]:
    train, trained = seen["train"], seen["trained"]
    unlabelled = sum(
        len(row["codes"])
        for row in trained
        if row["targets"] is None or len(row["targets"]) != len(row["codes"])
    )
    if unlabelled:
        return _failed(
            f"{_count(unlabelled, 'trained patch')} had no target in a cross-entropy loss on "
            "the network's output"
        )
    pairs = [
        (code, target)
        for row in trained
        for code, target in zip(row["codes"], row["targets"], strict=True)
        if code is not None and code < len(train)
    ]
    wrong = sum(target != train.labels[code] for code, target in pairs)
    if wrong:
        return _failed(
            f"{wrong} of the {len(pairs)} trained candidate patches ({seen['layout']} layout) "
            "were trained with a label that is not theirs"
        )
    return _passed()


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
    """Accuracy, mean cross-entropy and weights of every epoch validated on every patch.

    Computed by the checks from the network's logits after the epoch and the labels the
    validation patches had in that run; keyed by (attempt, epoch).
    """
    validation = seen["validation"]
    targets = torch.tensor(validation.labels)
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


def _validation(seen: dict[str, Any]) -> dict[str, Any]:
    measured, where, total = measured_epochs(seen), seen["context"], len(seen["validation"])
    reported: dict[tuple[int, int | None], dict[str, Any]] = {}
    for key, metrics in seen["logged"]:
        reported[key] = {**reported.get(key, {}), **metrics}
    for attempt, number, _ in seen["ended"]:
        key, after = (attempt, number), f"after epoch {number}"
        if key not in measured:
            validated = len(seen["validated"].get(key, {}).get("logits", {}))
            return _failed(
                f"{where}: {after} the network is measured on {validated} of the {total} "
                "validation patches"
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


def _delivery(seen: dict[str, Any]) -> dict[str, Any]:
    measured, where = measured_epochs(seen), seen["context"]
    delivered = [weights for _, weights in seen["delivered"]]
    if len(delivered) != 1:
        times = "no weights" if not delivered else f"weights {len(delivered)} times"
        return _failed(f"{where}: the run delivers {times}")
    epochs = [(attempt, number) for attempt, number, _ in seen["ended"]]
    if not epochs:
        return _failed(f"{where}: the run trained no epoch")
    if any(key not in measured for key in epochs):
        return _failed(
            f"{where}: the delivery cannot be checked, since not every epoch is measured on "
            "all the validation patches"
        )
    # max keeps the first of equals, and epochs are in the order they were trained.
    best = max(epochs, key=lambda key: measured[key]["accuracy"])
    if delivered[0] == measured[best]["fingerprint"]:
        return _passed()
    matching = [key for key in epochs if measured[key]["fingerprint"] == delivered[0]]
    what = f"the weights of epoch {matching[0][1]}" if matching else "weights no epoch ended with"
    return _failed(
        f"{where}: the run delivers {what} instead of those of epoch {best[1]}, whose "
        f"validation accuracy of {measured[best]['accuracy']:.3g} is the highest"
    )


def steered_runs(
    sources: dict[str, str], reference: dict[str, Any]
) -> tuple[list[dict[str, Any]], str]:
    """Runs whose best epoch falls before, and after, a restart; and what was not tested.

    The validation labels are the predictions of an epoch of the uninterrupted
    three-epoch run, which makes that epoch's accuracy 1 if the run repeats.
    """
    layout, seed = LAYOUTS[0]
    predictions = {
        number: {code: int(row.argmax()) for code, row in validated["logits"].items()}
        for (_, number), validated in seen_complete(reference).items()
    }
    runs, untested = [], ""
    if 1 in predictions:
        seen = observe(
            sources,
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


def seen_complete(seen: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    """The validation passes of a run that cover every validation patch."""
    total = len(seen["validation"])
    return {key: value for key, value in seen["validated"].items() if len(value["logits"]) == total}


def contract_result(sources: dict[str, str], family: str = "camelyon_pipeline") -> dict[str, Any]:
    """Check a project's modules against every clause."""
    clauses: dict[str, dict[str, Any]] = {}

    def check(name: str, test, *arguments: Any) -> None:
        if name in clauses and not clauses[name]["passed"]:
            return  # Keep the first failure of a clause.
        try:
            clauses[name] = test(*arguments)
        except Exception as error:
            clauses[name] = _failed(f"the check failed with {type(error).__name__}: {error}"[:300])

    def failed_with(error: Exception, where: str = "the project") -> dict[str, Any]:
        return _failed(f"{where} failed with {type(error).__name__}: {error}"[:300])

    layout, seed = LAYOUTS[0]
    observations = []
    try:
        for name, number in LAYOUTS:
            seen = observe(sources, name, number)
            seen["context"] = f"{name} layout, {EPOCHS} epochs"
            observations.append(seen)
            check("training_samples", _training_samples, seen)
            check("training_labels", _training_labels, seen)
        reference = observe(sources, layout, seed, epochs=RESTART_EPOCHS)
        reference["context"] = f"{layout} layout, {RESTART_EPOCHS} epochs"
        observations.append(reference)
    except Exception as error:
        failure = failed_with(error)
        for name in CLAUSES:
            if name not in clauses or clauses[name]["passed"]:
                clauses[name] = failure
        return _result(clauses, family)
    for stop in STOPS:
        where = f"stopped after epoch {stop} and restarted"
        try:
            seen = observe(sources, layout, seed, epochs=RESTART_EPOCHS, stop_after=stop)
        except Exception as error:
            check("restart", failed_with, error, where + ", the project")
            continue
        seen["context"] = where
        observations.append(seen)
        check("restart", _restart, reference, seen)
    untested = ""
    try:
        steered, untested = steered_runs(sources, reference)
        observations.extend(steered)
    except Exception as error:
        check("delivery", failed_with, error, "a run labelled to test the delivery")
    for seen in observations:
        check("validation", _validation, seen)
        check("delivery", _delivery, seen)
    if untested and clauses["delivery"]["passed"]:
        clauses["delivery"] = _passed(untested)
    return _result(clauses, family)


def _result(clauses: dict[str, dict[str, Any]], family: str) -> dict[str, Any]:
    failed = [name for name in CLAUSES if not clauses[name]["passed"]]
    return {
        "schema_version": 1,
        "task": "pipeline_contract",
        "contract_version": CONTRACT_VERSION,
        "family": family,
        "family_sha256": pipeline_tasks.family_sha256(family),
        "status": "failed" if failed else "passed",
        "failed": failed,
        "clauses": {name: clauses[name] for name in CLAUSES},
    }


def _refused(problems: list[str], family: str = "camelyon_pipeline") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": "pipeline_contract",
        "contract_version": CONTRACT_VERSION,
        "family": family,
        "family_sha256": pipeline_tasks.family_sha256(family),
        "status": "refused",
        "failed": list(CLAUSES),
        "clauses": {},
        "problems": problems,
    }


def run_contract(project: Path, family: str = "camelyon_pipeline") -> dict[str, Any]:
    """Check a project folder in this process; a project the harness refuses fails."""
    sources = project_sources(project)
    problems = check_project(sources)
    if problems:
        return _refused(problems, family)
    return contract_result(sources, family)


def run_contract_in_subprocess(
    project: Path, family: str = "camelyon_pipeline", *, timeout: float = 900.0
) -> dict[str, Any]:
    """Check a project in a child process without credentials; status may be crashed or timeout."""
    problems = check_project(project_sources(project))
    if problems:
        return _refused(problems, family)
    command = [sys.executable, "-m", "runsleuth.pipeline_contract", "run"]
    command += ["--project", str(Path(project).resolve()), "--family", family]
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
    # Its images come from the wrong positions while its labels come from the right ones.
    "positions_index_base": ("training_samples", "training_labels"),
    "labels_not_filtered": ("training_labels",),
    "scheduler_not_restored": ("restart",),
    "optimizer_not_restored": ("restart",),
    "last_epoch_delivered": ("delivery",),
    "best_accuracy_not_restored": ("delivery",),
    "pretrained_weights_delivered": ("delivery",),
}
_IMPORT = "from torch.utils.data import DataLoader, Dataset\n"
_ITEM = "        return self.patches.image(self.indices[position]), self.labels[position]\n"
_SET = "    return HospitalSubset(patches, config.TRAINING_HOSPITALS)\n"
_TRAINING = (
    "        with run.epoch(epoch):\n"
    "            train_one_epoch(model, loader, optimizer, loss_function, run.device)\n"
)
# Projects beyond the versions, to see that the checks neither accept only the healthy
# project's wording nor reject only the faults they were written against.
CASES = {
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
        ("validation", "delivery"),
        "Validation stops after its first batch, so it measures 32 of the patches; the best "
        "epoch is then chosen on them.",
        (
            (
                "train",
                "        count += len(labels)\n    return",
                "        count += len(labels)\n        break\n    return",
            ),
        ),
    ),
}


def expected_failures(name: str, family: str = "camelyon_pipeline") -> tuple[str, ...]:
    """The clauses a version or case should fail; none for a correct project."""
    if name in CASES:
        return CASES[name].fails
    if name == HEALTHY or pipeline_tasks.version(name, family).correct:
        return ()
    return VERSION_CLAUSES[name]


def case_sources(name: str, family: str = "camelyon_pipeline") -> dict[str, str]:
    """The modules of a version or of a case."""
    if name in pipeline_tasks.family(family).names:
        return pipeline_tasks.version_sources(name, family)
    sources = pipeline_tasks.healthy_sources(family)
    for module, old, new in CASES[name].edits:
        if sources[module].count(old) != 1:
            raise ValueError(f"{name}: every edit must match its module exactly once")
        sources[module] = sources[module].replace(old, new)
    return sources


def check_cases(*, sources=None, family: str = "camelyon_pipeline") -> list[dict[str, Any]]:
    """Run the contract on every version and case and compare with what each should fail."""
    sources = sources or (lambda name: case_sources(name, family))
    versions = pipeline_tasks.family(family).names
    rows = []
    for name in (*versions, *CASES):
        result = contract_result(sources(name), family)
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
    check = commands.add_parser("check", help="Check every version and case against expectations.")
    for command in (run, check):
        command.add_argument(
            "--family", choices=tuple(pipeline_tasks.FAMILIES), default="camelyon_pipeline"
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        print(json.dumps(run_contract(args.project, args.family)))
        return 0
    rows = check_cases(family=args.family)
    print(cases_text(rows))
    return 0 if all(row["as_expected"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
