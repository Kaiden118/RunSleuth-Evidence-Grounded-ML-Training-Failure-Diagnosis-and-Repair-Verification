"""Contract tests for training projects: does a project do what its statement says?

TASK.md (pipeline_tasks) is the public specification an agent works from. This module
turns its requirements into checks that the agent never sees and that run only after
the agent has finished:

    training_samples   1. every epoch trains on each patch of hospitals 0, 3 and 4
                          exactly once, and on no other patch, inside run.epoch
    training_labels    1. each of those patches is trained on with its own label
    restart            5. a run stopped as it starts an epoch and restarted trains
                          exactly as the uninterrupted run

The other requirements get their clauses with the tasks that need them. Version 1 had
the first two; version 2 (2026-10-22) adds restart.

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

CONTRACT_VERSION = 2
CLAUSES = ("training_samples", "training_labels", "restart")
TRAINING_HOSPITALS = (0, 3, 4)
# Candidates per hospital: unequal, and more than a batch of 32 in all.
CANDIDATES = {0: 9, 1: 7, 3: 14, 4: 17}
VALIDATION = {0: 2, 3: 2, 4: 2}
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


def _passed() -> dict[str, Any]:
    return {"passed": True, "detail": ""}


def _failed(detail: str) -> dict[str, Any]:
    return {"passed": False, "detail": detail}


def coded_images(codes: list[int]) -> torch.Tensor:
    """RGB patches in [0, 1], each showing its code."""
    images = torch.full((len(codes), 3, PATCH_SIZE * PATCH_SIZE), 0.5)
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

    def __init__(self, codes: list[int], labels: list[int], hospitals: list[int]) -> None:
        self._images = coded_images(codes)
        self.labels, self.hospitals = tuple(labels), tuple(hospitals)

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
    """Training candidates numbered by position, with labels drawn at random."""
    hospitals = [hospital for hospital, count in CANDIDATES.items() for _ in range(count)]
    if layout == "mixed":
        order = torch.randperm(len(hospitals), generator=torch.Generator().manual_seed(seed))
        hospitals = [hospitals[position] for position in order.tolist()]
    generator = torch.Generator().manual_seed(1000 + seed)
    labels = torch.randint(0, 2, (len(hospitals),), generator=generator).tolist()
    return CodedPatches(list(range(len(hospitals))), labels, hospitals)


def validation_patches() -> CodedPatches:
    hospitals = [hospital for hospital, count in VALIDATION.items() for _ in range(count)]
    codes = [VALIDATION_CODES + position for position in range(len(hospitals))]
    return CodedPatches(codes, [position % 2 for position in range(len(codes))], hospitals)


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
    """

    def __init__(
        self,
        train: CodedPatches,
        validation: CodedPatches,
        *,
        seed: int,
        epochs: int,
        checkpoints: MemoryCheckpoints,
        stop_after: int | None = None,
        epoch_ended=lambda _number: None,
    ) -> None:
        self.device, self.seed, self.epochs = torch.device("cpu"), seed, epochs
        self.data = types.SimpleNamespace(train=train, validation=validation)
        self.checkpoints = checkpoints
        self._stop_after, self._epoch_ended = stop_after, epoch_ended
        self._epoch: int | None = None
        self._trained = 0
        self._delivered = None

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
        self._epoch_ended(number)

    def log(self, **metrics: float) -> None:
        pass

    def deliver(self, state_dict) -> None:
        self._delivered = state_dict


def fingerprint(module: nn.Module) -> str:
    """A hash of a module's parameters and buffers, by name, dtype, shape and bytes."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
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
) -> dict[str, Any]:
    """Run a project on coded candidates, perhaps stopped and restarted; what it did.

    trained has one row per backward pass that reached the network's output: the
    attempt and the epoch it happened in (None outside run.epoch), the codes of the
    batch the network was given, and the targets of the cross-entropy on that output
    (None if there was none). steps has the attempt, epoch and learning rates of every
    optimizer step, and ended the attempt, number and network fingerprint of every
    epoch. With stop_after, the first attempt is stopped as it starts an epoch after
    that many, and main is called again in freshly loaded modules with its checkpoints.
    """
    train, validation = candidates(layout, seed), validation_patches()
    checkpoints = MemoryCheckpoints()
    seen: dict[str, Any] = {"layout": layout, "train": train, "trained": [], "steps": []}
    seen["ended"] = []
    state: dict[str, Any] = {"depth": 0, "batch": None, "network": None, "attempt": 0}
    run: ContractRun | None = None

    def before(module, args) -> None:
        state["depth"] += 1
        if state["depth"] == 1 and args and isinstance(args[0], torch.Tensor):
            if args[0].dim() == 4:
                # The outermost call on a batch of images: the network.
                codes = read_codes(args[0])
                state["batch"] = {"codes": codes, "module": module, "output": None, "targets": None}

    def after(_module, _args, output) -> None:
        state["depth"] -= 1
        batch = state["batch"]
        if state["depth"] or batch is None or batch["output"] is not None:
            return
        if isinstance(output, torch.Tensor):
            batch["output"] = output
            if output.requires_grad:
                output.register_hook(lambda _gradient, batch=batch: backward(batch))

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
        seen["ended"].append((state["attempt"], number, network and fingerprint(network)))

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
            run = ContractRun(
                train,
                validation,
                seed=seed,
                epochs=epochs,
                checkpoints=checkpoints,
                stop_after=stop,
                epoch_ended=epoch_ended,
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


def _restart(sources: dict[str, str]) -> dict[str, Any]:
    layout, seed = LAYOUTS[0]
    reference = observe(sources, layout, seed, epochs=RESTART_EPOCHS)
    for stop in STOPS:
        try:
            restarted = observe(sources, layout, seed, epochs=RESTART_EPOCHS, stop_after=stop)
        except Exception as error:
            return _failed(
                f"stopped after epoch {stop} and restarted, the project failed with "
                f"{type(error).__name__}: {error}"[:300]
            )
        difference = _restart_difference(reference, restarted)
        if difference:
            return _failed(f"stopped after epoch {stop} and restarted, the run {difference}")
    return _passed()


def contract_result(sources: dict[str, str], family: str = "camelyon_pipeline") -> dict[str, Any]:
    """Check a project's modules against every clause."""
    clauses: dict[str, dict[str, Any]] = {}

    def check(name: str, test, argument: Any) -> None:
        if name in clauses and not clauses[name]["passed"]:
            return  # Keep the first failure of a clause.
        try:
            clauses[name] = test(argument)
        except Exception as error:
            clauses[name] = _failed(f"the check failed with {type(error).__name__}: {error}"[:300])

    for layout, seed in LAYOUTS:
        try:
            seen = observe(sources, layout, seed)
        except Exception as error:
            failure = _failed(f"the project failed with {type(error).__name__}: {error}"[:300])
            for name in CLAUSES:
                if name not in clauses or clauses[name]["passed"]:
                    clauses[name] = failure
            break
        check("training_samples", _training_samples, seen)
        check("training_labels", _training_labels, seen)
    check("restart", _restart, sources)
    failed = [name for name in CLAUSES if not clauses[name]["passed"]]
    return {
        "schema_version": 1,
        "task": "pipeline_contract",
        "contract_version": CONTRACT_VERSION,
        "family": family,
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
            }
        )
    return rows


def cases_text(rows: list[dict[str, Any]]) -> str:
    lines = []
    for row in rows:
        verdict = "as expected" if row["as_expected"] else f"EXPECTED {row['expected'] or 'none'}"
        lines.append(f"{row['name']:<40} {', '.join(row['failed']) or 'passes':<36} {verdict}")
        lines.extend(f"    {detail}" for detail in row["details"])
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
