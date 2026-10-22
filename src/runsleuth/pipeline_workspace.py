"""Run an agent's training project under RunSleuth monitoring.

agent_workspace runs one script whose training loop belongs to the harness. Here the
project owns its pipeline. It is a folder of modules whose train.py defines main(run),
and main builds its own datasets and loaders, trains, evaluates, writes and reads its
checkpoints and hands over the weights it delivers. The harness owns what the project
is given and what is recorded:

- the data, as Samples whose reads are recorded: which samples the project read while
  training each epoch, in which order, and with which group and label;
- the epoch budget, and a preemption: with interrupt_after_epochs the first attempt is
  stopped as it starts the next epoch, and main is called again in freshly loaded
  modules with the same checkpoints, as after a restart;
- the checkpoints, kept in memory, with every write and read and a hash of each state,
  and what the model and optimizer hold when an attempt starts training;
- the monitoring of RunMonitor, a hash of the model state after each of the first
  optimizer steps and after every epoch, and the labels the loss was given;
- an evaluation of the delivered weights on evaluation splits the project never sees.

The project's modules are checked and executed as agent_workspace checks a script, and
they may import each other. This guards against accidents and plain misuse; it is not
a sandbox.

    python -m runsleuth.pipeline_workspace run --project P --output runs/p \
        --data synthetic --interrupt-after-epochs 1
    python -m runsleuth.pipeline_workspace check --output artifacts/pipeline_tasks/check-1 \
        --data camelyon --config configs/camelyon17_clean.json

check trains every version of pipeline_tasks for the same budget, and the healthy
project once more with a restart, and diagnoses each run against the healthy one.
"""

import argparse
import hashlib
import io
import json
import math
import operator
import re
import subprocess
import sys
import traceback
import types
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, get_worker_info

from runsleuth import pipeline_tasks
from runsleuth.agent_verification import compare_trajectories
from runsleuth.agent_workspace import (
    ALLOWED_IMPORTS,
    HASHED_STEPS,
    OUTPUT_TAIL,
    ScriptRefused,
    check_script,
    clean_environment,
    restricted_builtins,
)
from runsleuth.camelyon_config import CamelyonConfig
from runsleuth.camelyon_data import NativeRGBPatch, load_camelyon_mirror, select_stratified_indices
from runsleuth.optimizer_probe import state_dict_sha256
from runsleuth.pipeline_tasks import HEALTHY
from runsleuth.run_monitor import RunMonitor
from runsleuth.signature_matching import diagnose, extract_evidence, load_library
from runsleuth.train import resolve_device, seed_everything

ENTRY_MODULE = "train"
ENTRY_FUNCTION = "main"
MAX_MODULES = 8
MODULE_NAME = re.compile(r"[a-z][a-z0-9_]*\Z")
CHECKPOINT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
EVALUATION_BATCH_SIZE = 128
# The check's second run of the healthy project, restarted after its first epoch.
RESTARTED = "healthy_restarted"


def project_sources(directory: Path) -> dict[str, str]:
    """The project's modules by name: every .py file directly in its folder."""
    return {
        path.stem: path.read_text(encoding="utf-8") for path in sorted(Path(directory).glob("*.py"))
    }


def check_project(sources: Mapping[str, str]) -> list[str]:
    """Reasons to refuse a project, module by module; empty when it may run."""
    problems = []
    if ENTRY_MODULE not in sources:
        problems.append(f"the project has no {ENTRY_MODULE}.py")
    if len(sources) > MAX_MODULES:
        problems.append(f"the project has more than {MAX_MODULES} modules")
    names = frozenset(sources)
    for name, source in sources.items():
        if not MODULE_NAME.match(name):
            problems.append(f"{name}.py: module names are lowercase letters, digits and _")
        elif name in ALLOWED_IMPORTS or name in sys.stdlib_module_names:
            problems.append(f"{name}.py: the name of a library module may not be reused")
        problems.extend(f"{name}.py: {problem}" for problem in check_script(source, names))
    return problems


def load_project(sources: Mapping[str, str]) -> dict[str, types.ModuleType]:
    """Check a project and execute its entry module with restricted builtins.

    A module runs when it is first imported. Modules are never added to sys.modules, so
    every call loads them afresh and two projects never share one.
    """
    problems = check_project(sources)
    if problems:
        raise ScriptRefused(problems)
    modules: dict[str, types.ModuleType] = {}
    guarded_import = restricted_builtins()["__import__"]

    def project_import(name, globals=None, locals=None, fromlist=(), level=0):
        if not level and name in sources:
            return load(name)
        return guarded_import(name, globals, locals, fromlist, level)

    def load(name: str) -> types.ModuleType:
        if name not in modules:
            module = types.ModuleType(name)
            vars(module)["__builtins__"] = {
                **restricted_builtins(),
                "__import__": project_import,
            }
            # Registered before it runs, as Python does, so that imports may be circular.
            modules[name] = module
            exec(compile(sources[name], f"{name}.py", "exec"), vars(module))
        return modules[name]

    entry = load(ENTRY_MODULE)
    if not callable(vars(entry).get(ENTRY_FUNCTION)):
        raise ScriptRefused([f"{ENTRY_MODULE}.py must define {ENTRY_FUNCTION}(run)"])
    return modules


class Samples:
    """Images in [0, 1] (channel, height, width) with one row of columns per sample.

    samples[i] is (image, label) and samples.image(i) the image alone; both reads are
    recorded when the samples have a reader. The columns, at least labels and for
    Camelyon17 hospitals, hold one value per sample and are read without a record.
    Reads must happen in the process that runs the project (DataLoader num_workers=0),
    where the harness sees them. What records them is private: the static check refuses
    any attribute with a leading underscore.
    """

    def __init__(
        self,
        load: Callable[[int], torch.Tensor],
        columns: Mapping[str, Iterable[Any]],
        *,
        reader: Callable[[int], None] | None = None,
    ) -> None:
        values = {name: tuple(column) for name, column in columns.items()}
        if "labels" not in values or len({len(column) for column in values.values()}) != 1:
            raise ValueError("Samples need labels and columns of equal length")
        self._load, self._columns, self._reader = load, values, reader
        self._size = len(values["labels"])
        for name, column in values.items():
            if hasattr(self, name):
                raise ValueError(f"a column may not be named {name!r}")
            setattr(self, name, column)

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        position = self._position(index)
        return self.image(position), self.labels[position]

    def _position(self, index) -> int:
        position = operator.index(index)
        if position < 0:
            position += self._size
        if not 0 <= position < self._size:
            raise IndexError(f"sample {index} is out of range for {self._size} samples")
        return position

    def image(self, index) -> torch.Tensor:
        position = self._position(index)
        if self._reader is not None:
            if get_worker_info() is not None:
                raise RuntimeError("read samples in the main process: use num_workers=0")
            self._reader(position)
        return self._load(position)

    def _recorded(self, reader: Callable[[int], None] | None) -> "Samples":
        """The same samples with another reader."""
        return Samples(self._load, self._columns, reader=reader)


@dataclass(frozen=True)
class PipelineData:
    """The samples of one run.

    train and validation are handed to the project. evaluation holds the splits the
    harness evaluates the delivered weights on, never handed over. group names the
    column that holdouts hold out (hospitals), and holdouts says per evaluation split
    which groups it evaluates on and whether it holds them out from training; the run
    adds how many training samples were read from them.
    """

    train: Samples
    validation: Samples
    evaluation: dict[str, Samples]
    group: str
    holdouts: dict[str, dict[str, Any]]
    record: dict[str, Any]


# build(seed) -> PipelineData
DataBuilder = Callable[[int], PipelineData]


class Preempted(BaseException):
    """Raised into a project when its first attempt is stopped.

    It derives from BaseException, which a project cannot name, so except Exception
    does not hold the run up; a bare except that swallows it does not either, since
    every later call of the attempt raises it again.
    """


def state_sha256(state: Any) -> str:
    """Hash a checkpoint: tensors by dtype, shape and bytes, other values by repr."""
    digest = hashlib.sha256()

    def visit(path: str, value: Any) -> None:
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous().reshape(-1)
            data = tensor.view(torch.uint8).numpy().tobytes()
            digest.update(f"{path}|{tensor.dtype}|{tuple(value.shape)}|{len(data)}|".encode())
            digest.update(data)
        elif isinstance(value, Mapping):
            for key in sorted(value, key=repr):
                visit(f"{path}/{key!r}", value[key])
        elif isinstance(value, list | tuple):
            for position, item in enumerate(value):
                visit(f"{path}[{position}]", item)
        else:
            digest.update(f"{path}|{value!r}|".encode())

    visit("", state)
    return digest.hexdigest()


def _is_state_dict(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and bool(value)
        and all(isinstance(item, torch.Tensor) for item in value.values())
    )


def _model_states(state: Any) -> dict[str, str]:
    """Hashes of the model-like states in a checkpoint: the state itself or its entries."""
    if _is_state_dict(state):
        return {"": state_dict_sha256(state)}
    if not isinstance(state, Mapping):
        return {}
    return {
        str(key): state_dict_sha256(value) for key, value in state.items() if _is_state_dict(value)
    }


def _counts(values: Iterable[Any]) -> dict[str, int]:
    return {str(key): count for key, count in sorted(Counter(values).items())}


def _digest(values: list[int]) -> str:
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


class Checkpoints:
    """The project's checkpoints: write(name, state), read(name) and name in checkpoints.

    A state is anything torch.load reads back with weights_only=True: tensors, numbers,
    strings and lists, tuples and dicts of them, such as state_dict()s. Read states are
    on the CPU. Checkpoints outlive a preemption.
    """

    def __init__(self, recorder: "Recorder") -> None:
        self._recorder = recorder
        self._files: dict[str, bytes] = {}

    def write(self, name: str, state: Any) -> None:
        self._recorder.check_attempt()
        if not isinstance(name, str) or not CHECKPOINT_NAME.match(name):
            raise ValueError(f"checkpoint names are letters, digits, '_', '.' and '-': {name!r}")
        buffer = io.BytesIO()
        torch.save(state, buffer)
        self._files[name] = buffer.getvalue()
        self._recorder.checkpoint_event("write", name, state)

    def read(self, name: str) -> Any:
        self._recorder.check_attempt()
        if name not in self._files:
            raise KeyError(f"there is no checkpoint named {name!r}")
        state = torch.load(io.BytesIO(self._files[name]), map_location="cpu", weights_only=True)
        self._recorder.checkpoint_event("read", name, state)
        return state

    def __contains__(self, name: object) -> bool:
        return name in self._files


class Recorder:
    """What the harness records of one run, across its attempts."""

    def __init__(
        self,
        run_dir: Path,
        data: PipelineData,
        *,
        epochs: int,
        seed: int,
        interrupt_after_epochs: int | None,
        hashed_steps: int,
        head: str,
    ) -> None:
        self.run_dir, self.data, self.head = Path(run_dir), data, head
        self.interrupt_after_epochs, self.hashed_steps = interrupt_after_epochs, hashed_steps
        self.monitor: RunMonitor | None = None
        self.model: nn.Module | None = None
        self.optimizer: Optimizer | None = None
        self.handles: list[Any] = []
        self.attempt = 0
        self.attempt_epochs = 0
        self.preempted = False
        self.epoch: int | None = None
        self.last_output: Any = None
        self.targets: list[int] = []
        self.reads: dict[str, list[tuple[int, int | None, int]]] = {"train": [], "validation": []}
        self.epochs: list[dict[str, Any]] = []
        self.checkpoints: list[dict[str, Any]] = []
        self.attempts: list[dict[str, Any]] = []
        self.tracked: list[dict[str, Any]] = []
        self.delivered: dict[str, torch.Tensor] | None = None
        self.delivery: dict[str, Any] | None = None
        self.workspace: dict[str, Any] = {
            "project_sha256": None,
            "seed": seed,
            "epochs": epochs,
            "max_train_batches": None,
            "interrupt_after_epochs": interrupt_after_epochs,
            "data": data.record,
            "initial_state_sha256": None,
            "hashed_steps": hashed_steps,
            "step_state_sha256": [],
            "epoch_state_sha256": [],
            "final_state_sha256": None,
        }

    def reader(self, split: str) -> Callable[[int], None]:
        def read(position: int) -> None:
            self.check_attempt()
            self.reads[split].append((self.attempt, self.epoch, position))

        return read

    def start_attempt(self) -> None:
        self.remove_hooks()
        self.attempt += 1
        self.attempt_epochs, self.preempted, self.epoch = 0, False, None

    def end_attempt(self, outcome: str) -> None:
        self.remove_hooks()
        self.attempts.append(
            {"attempt": self.attempt, "outcome": outcome, "epochs_trained": self.attempt_epochs}
        )

    def check_attempt(self) -> None:
        if self.preempted:
            raise Preempted

    def remove_hooks(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def track(self, model: nn.Module, optimizer: Optimizer) -> None:
        self.check_attempt()
        if not isinstance(model, nn.Module) or not isinstance(optimizer, Optimizer):
            raise TypeError("run.track takes the model and its torch.optim optimizer")
        if self.epoch is not None:
            raise RuntimeError("call run.track outside run.epoch")
        if self.monitor is None:
            self.monitor = RunMonitor(model, optimizer, self.run_dir, head=self.head)
            self.workspace["initial_state_sha256"] = state_dict_sha256(model.state_dict())
        else:
            # A restarted attempt builds a new model; the monitor follows it.
            self.monitor.model, self.monitor.optimizer = model, optimizer
        self.remove_hooks()
        self.model, self.optimizer = model, optimizer
        # What the project starts training from: after a restart, what it restored.
        states = [state for state in optimizer.state.values() if state]
        self.tracked.append(
            {
                "attempt": self.attempt,
                "after_epoch": self.last_epoch(),
                "model_same_as": self.same_as(state_dict_sha256(model.state_dict())),
                "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
                "optimizer_parameters_with_state": len(states),
                "optimizer_steps": max(
                    (float(state["step"]) for state in states if "step" in state), default=0.0
                ),
            }
        )
        monitor, steps = self.monitor, self.workspace["step_state_sha256"]

        def observe_inputs(_module, inputs) -> None:
            if inputs and isinstance(inputs[0], torch.Tensor) and inputs[0].dim() >= 2:
                if self.epoch is not None and torch.is_grad_enabled():
                    monitor.observe_batch("train", inputs[0])
                elif not torch.is_grad_enabled():
                    # Outside training the project evaluates on its validation split.
                    monitor.observe_batch("id_eval", inputs[0])

        def keep_output(_module, _inputs, output) -> None:
            self.last_output = output

        def observe_loss(module, inputs) -> None:
            if self.epoch is None or module is model or len(inputs) < 2:
                return
            if inputs[0] is self.last_output and isinstance(inputs[1], torch.Tensor):
                labels = [int(label) for label in inputs[1].detach().flatten().tolist()]
                self.targets.extend(labels)
                seen = set(monitor.report["label_classes"].get("train", [])).union(labels)
                monitor.report["label_classes"]["train"] = sorted(seen)

        def hash_step(*_) -> None:
            if len(steps) < self.hashed_steps:
                steps.append(state_dict_sha256(model.state_dict()))

        self.handles = [
            model.register_forward_pre_hook(observe_inputs),
            model.register_forward_hook(keep_output),
            # The loss is the project's own module; it is found by taking the model's output.
            nn.modules.module.register_module_forward_pre_hook(observe_loss),
            optimizer.register_step_post_hook(hash_step),
        ]

    @contextmanager
    def train_epoch(self, number: int):
        self.check_attempt()
        if (
            self.attempt == 1
            and self.interrupt_after_epochs is not None
            and self.attempt_epochs >= self.interrupt_after_epochs
        ):
            self.preempted = True
            raise Preempted
        if self.monitor is None:
            raise RuntimeError("call run.track(model, optimizer) before the first run.epoch")
        if self.epoch is not None:
            raise RuntimeError("run.epoch cannot be nested")
        if type(number) is not int:
            raise TypeError("run.epoch takes the epoch's number, 1 for the first")
        self.epoch, self.targets = number, []
        start = len(self.reads["train"])
        try:
            with self.monitor.epoch():
                yield
        finally:
            self.epoch = None
        self.attempt_epochs += 1
        self.workspace["epoch_state_sha256"].append(state_dict_sha256(self.model.state_dict()))
        self.monitor.report["parameter_group_epochs"][-1].update(
            attempt=self.attempt, project_epoch=number
        )
        positions = [position for _, _, position in self.reads["train"][start:]]
        labels = self.data.train.labels
        matching = None
        if len(self.targets) == len(positions):
            matching = sum(
                target == labels[position]
                for target, position in zip(self.targets, positions, strict=True)
            )
        self.epochs.append(
            {
                "attempt": self.attempt,
                "epoch": number,
                **self.read_summary(positions),
                "order_sha256": _digest(positions),
                "loss_targets": len(self.targets),
                "loss_targets_matching_read_labels": matching,
            }
        )

    def read_summary(self, positions: list[int]) -> dict[str, Any]:
        groups, labels = getattr(self.data.train, self.data.group), self.data.train.labels
        return {
            "reads": len(positions),
            "distinct": len(set(positions)),
            "reads_by_group": _counts(groups[position] for position in positions),
            "reads_by_label": _counts(labels[position] for position in positions),
        }

    def log(self, metrics: dict[str, Any]) -> None:
        self.check_attempt()
        if self.monitor is None or not self.monitor.report["parameter_group_epochs"]:
            raise RuntimeError("call run.log after an epoch")
        values = {}
        for name, value in metrics.items():
            number = float(value)
            values[name] = number if math.isfinite(number) else None
        row = self.monitor.report["parameter_group_epochs"][-1]
        row["project_metrics"] = {**row.get("project_metrics", {}), **values}

    def checkpoint_event(self, action: str, name: str, state: Any) -> None:
        self.checkpoints.append(
            {
                "attempt": self.attempt,
                "after_epoch": self.last_epoch(),
                "during_epoch": self.epoch,
                "action": action,
                "name": name,
                "state_sha256": state_sha256(state),
                "model_states": _model_states(state),
            }
        )

    def last_epoch(self) -> int | None:
        return self.epochs[-1]["epoch"] if self.epochs else None

    def deliver(self, state: Any) -> None:
        self.check_attempt()
        if self.delivery is not None:
            raise RuntimeError("run.deliver may be called once")
        if not _is_state_dict(state):
            raise TypeError("run.deliver takes a state_dict of the model")
        self.delivered = {name: tensor.detach().cpu().clone() for name, tensor in state.items()}
        sha256 = state_dict_sha256(self.delivered)
        self.delivery = {
            "attempt": self.attempt,
            "after_epoch": self.last_epoch(),
            "state_sha256": sha256,
            "same_as": self.same_as(sha256),
            "evaluation": None,
        }

    def same_as(self, sha256: str) -> list[str]:
        """The model states of this run that have this hash: its initial state, its epochs."""
        same_as = [
            f"epoch {row['epoch']} (attempt {row['attempt']})"
            for row, epoch_sha256 in zip(
                self.epochs, self.workspace["epoch_state_sha256"], strict=True
            )
            if epoch_sha256 == sha256
        ]
        if sha256 == self.workspace["initial_state_sha256"]:
            same_as.insert(0, "initial state")
        return same_as


class PipelineRun:
    """What main(run) is given.

    run.device, run.seed, run.epochs: where to train, the run's seed, and how many epochs
    the run trains in all, counted across a restart.
    run.data.train, run.data.validation: Samples to train on and to select by.
    run.checkpoints: Checkpoints that outlive a restart.
    run.track(model, optimizer): call once both exist, before the first epoch.
    with run.epoch(number): wrap the training of one epoch, numbered from 1.
    run.log(**metrics): the project's validation metrics of the epoch just trained.
    run.deliver(state_dict): the model weights the run delivers, once.
    """

    def __init__(
        self,
        recorder: Recorder,
        checkpoints: Checkpoints,
        *,
        device: torch.device,
        seed: int,
        epochs: int,
        train: Samples,
        validation: Samples,
    ) -> None:
        self.device, self.seed, self.epochs = device, seed, epochs
        self.data = types.SimpleNamespace(train=train, validation=validation)
        self.checkpoints = checkpoints
        self._recorder = recorder

    def track(self, model: nn.Module, optimizer: Optimizer) -> None:
        self._recorder.track(model, optimizer)

    def epoch(self, number: int):
        return self._recorder.train_epoch(number)

    def log(self, **metrics: float) -> None:
        self._recorder.log(metrics)

    def deliver(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        self._recorder.deliver(state_dict)


@torch.inference_mode()
def evaluate_weights(
    model: nn.Module, splits: Mapping[str, Samples], device: torch.device
) -> dict[str, float]:
    """Loss and accuracy per split for inputs normalized with the ImageNet statistics."""
    model.eval()
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    loss_function = nn.CrossEntropyLoss(reduction="sum")
    metrics = {}
    for split, samples in splits.items():
        loss, correct, count = 0.0, 0, 0
        for images, labels in DataLoader(samples, batch_size=EVALUATION_BATCH_SIZE):
            images, labels = images.to(device), labels.to(device)
            logits = model((images - mean) / std)
            loss += float(loss_function(logits, labels))
            correct += int((logits.argmax(dim=1) == labels).sum())
            count += len(labels)
        metrics[f"{split}_validation_loss"] = loss / count
        metrics[f"{split}_validation_accuracy"] = correct / count
    return metrics


def _holdouts(recorder: Recorder) -> dict[str, dict[str, Any]]:
    """The data's holdouts with the training samples read from the held-out groups."""
    data = recorder.data
    groups = getattr(data.train, data.group)
    trained = {position for _, epoch, position in recorder.reads["train"] if epoch is not None}
    holdouts = {}
    for split, holdout in data.holdouts.items():
        holdouts[split] = dict(holdout)
        if holdout.get("held_out_by"):
            held_out = set(holdout["evaluation_groups"])
            holdouts[split]["training_samples_in_evaluation_groups"] = sum(
                groups[position] in held_out for position in trained
            )
    return holdouts


def _training_reads(recorder: Recorder) -> dict[str, Any]:
    """Every training sample read in the run, while training and outside an epoch."""
    reads = recorder.reads["train"]
    trained = sorted({position for _, epoch, position in reads if epoch is not None})
    groups = getattr(recorder.data.train, recorder.data.group)
    return {
        "candidates": len(recorder.data.train),
        "candidates_by_group": _counts(groups),
        "read_while_training": len(trained),
        "read_while_training_by_group": _counts(groups[position] for position in trained),
        "reads_outside_epochs": sum(epoch is None for _, epoch, _ in reads),
        "validation_reads": len(recorder.reads["validation"]),
    }


def run_project(
    project: Path,
    run_dir: Path,
    build_data: DataBuilder,
    *,
    epochs: int,
    seed: int,
    device: torch.device,
    interrupt_after_epochs: int | None = None,
    hashed_steps: int = HASHED_STEPS,
    head: str = "fc",
) -> Path:
    """Run a project for a fixed budget, perhaps with a restart; return the report's path.

    The report is RunMonitor's, with the workspace record agent_workspace writes (so
    agent_verification can compare the hashed states of two runs), evaluation_holdouts
    from the samples actually read while training, and a pipeline record: the
    attempts, what the model and optimizer held when each attempt tracked them (after
    a restart, what was restored), every epoch's reads, the checkpoint events and the
    delivery. final_metrics
    are the harness's evaluation of the delivered weights; the project's own metrics
    are in each epoch's project_metrics.

    A refused project raises ScriptRefused and leaves no report. A project that fails
    after it was loaded still yields one that says so.
    """
    sources = project_sources(project)
    problems = check_project(sources)
    if problems:
        raise ScriptRefused(problems)
    run_dir = Path(run_dir)
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} exists; a run needs a new directory")
    data = build_data(seed)
    recorder = Recorder(
        run_dir,
        data,
        epochs=epochs,
        seed=seed,
        interrupt_after_epochs=interrupt_after_epochs,
        hashed_steps=hashed_steps,
        head=head,
    )
    recorder.workspace["project_sha256"] = state_sha256(dict(sources))
    checkpoints = Checkpoints(recorder)
    failure = None
    for _ in range(1 if interrupt_after_epochs is None else 2):
        recorder.start_attempt()
        seed_everything(seed)
        run = PipelineRun(
            recorder,
            checkpoints,
            device=device,
            seed=seed,
            epochs=epochs,
            train=data.train._recorded(recorder.reader("train")),
            validation=data.validation._recorded(recorder.reader("validation")),
        )
        outcome = "completed"
        try:
            # Fresh modules, as in a restarted process.
            vars(load_project(sources)[ENTRY_MODULE])[ENTRY_FUNCTION](run)
            if recorder.preempted:
                outcome = "preempted"
        except Preempted:
            outcome = "preempted"
        except Exception as error:
            outcome = "failed"
            failure = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc()[-OUTPUT_TAIL:],
            }
        recorder.end_attempt(outcome)
        if outcome != "preempted":
            break
    monitor = recorder.monitor
    if monitor is None:
        run_dir.mkdir(parents=True)
        report: dict[str, Any] = {
            "schema_version": 1,
            "task": "runsleuth_monitored_run",
            "status": "running",
            "error": None,
            "parameter_group_epochs": [],
            "final_metrics": {},
            "input_statistics": {},
            "label_classes": {},
        }
    else:
        report = monitor.report
    if failure is not None and report["status"] == "running":
        report["status"] = "failed"
        report["error"] = {key: failure[key] for key in ("type", "message")}
    if failure is not None:
        recorder.workspace["traceback"] = failure["traceback"]
    if monitor is not None:
        monitor.close()
    elif report["status"] == "running":
        report["status"] = "failed"
        report["error"] = {"type": "RuntimeError", "message": "the project never called run.track"}
    if report["status"] == "completed":
        recorder.workspace["final_state_sha256"] = state_dict_sha256(recorder.model.state_dict())
        if recorder.delivered is None:
            report["status"] = "failed"
            report["error"] = {"type": "RuntimeError", "message": "the run delivered no weights"}
    if report["status"] == "completed":
        try:
            recorder.model.load_state_dict(recorder.delivered)
            metrics = evaluate_weights(recorder.model, data.evaluation, device)
        except Exception as error:
            report["status"] = "failed"
            report["error"] = {"type": type(error).__name__, "message": str(error)}
            recorder.delivery["evaluation"] = {"error": report["error"]}
        else:
            if all(math.isfinite(value) for value in metrics.values()):
                recorder.delivery["evaluation"] = metrics
                report["final_metrics"] = {"epoch": recorder.last_epoch(), **metrics}
            else:
                report["status"] = "diverged"
                report["divergence"] = {
                    "epoch": recorder.last_epoch(),
                    "message": "the delivered weights give metrics that are not finite",
                }
    report["workspace"] = recorder.workspace
    report["evaluation_holdouts"] = _holdouts(recorder)
    report["pipeline"] = {
        "attempts": recorder.attempts,
        "tracked": recorder.tracked,
        "training_reads": _training_reads(recorder),
        "epochs": recorder.epochs,
        "checkpoints": recorder.checkpoints,
        "delivery": recorder.delivery,
    }
    report_path = run_dir / "run_report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report_path


def synthetic_pipeline_data(seed: int) -> PipelineData:
    """Random patches from four hospitals: a smoke test of the harness, not a benchmark.

    The training candidates come in hospital order, 0, 1, 3 and 4, eight each; hospital
    1 is the out-of-distribution evaluation hospital. Validation and in-distribution
    evaluation share their patches, as with Camelyon17.
    """
    generator = torch.Generator().manual_seed(seed)

    def patches(count: int, hospital: int, brightness: float = 1.0) -> dict[str, Any]:
        # Labels without a period, so that reading the wrong sample shows in its label.
        labels = torch.randint(0, 2, (count,), generator=generator)
        noise = torch.rand(count, 3, 16, 16, generator=generator)
        images = (noise * 0.5 + labels.view(-1, 1, 1, 1) * 0.5) * brightness
        return {"images": images, "labels": labels.tolist(), "hospitals": [hospital] * count}

    def samples(parts: list[dict[str, Any]]) -> Samples:
        images = torch.cat([part["images"] for part in parts])
        columns = {
            name: [value for part in parts for value in part[name]]
            for name in ("labels", "hospitals")
        }
        return Samples(lambda position: images[position], columns)

    candidates = samples(
        [patches(8, hospital, 0.8 if hospital == 1 else 1.0) for hospital in (0, 1, 3, 4)]
    )
    validation = samples([patches(4, hospital) for hospital in (0, 3, 4)])
    return PipelineData(
        train=candidates,
        validation=validation,
        evaluation={"id": validation, "ood": samples([patches(12, 1, brightness=0.8)])},
        group="hospitals",
        holdouts={
            "id": {"held_out_by": None, "evaluation_groups": [0, 3, 4]},
            "ood": {"held_out_by": "hospital", "evaluation_groups": [1]},
        },
        record={"source": "synthetic", "candidates": len(candidates)},
    )


def camelyon_pipeline_data(config: CamelyonConfig) -> DataBuilder:
    """Camelyon17 patches for projects, in the order of their slides.

    The training candidates are the baseline's training subset (official training split,
    hospitals 0, 3 and 4) and spare patches of the validation hospital 1 that evaluation
    does not use, as many as their share of the pool, so a project that trains on
    hospitals 0, 3 and 4 trains on exactly the baseline's subset. Slide order puts the
    hospitals in the order 0, 1, 3, 4, as the WILDS metadata lists them. Validation is
    the baseline's in-distribution subset; evaluation adds its out-of-distribution one.
    """

    def build(seed: int) -> PipelineData:
        from torchvision import transforms

        dataset = load_camelyon_mirror(config, download=False)
        transform = transforms.Compose([NativeRGBPatch(), transforms.ToTensor()])
        labels = [int(value) for value in dataset.y_array.tolist()]
        hospitals, slides = (
            [int(value) for value in dataset.metadata_array[:, column].tolist()]
            for column in map(dataset.metadata_fields.index, ("hospital", "slide"))
        )
        official = {
            split: [int(index) for index in dataset.get_subset(split, transform).indices]
            for split in ("train", "id_val", "val")
        }

        def select(indices: list[int], cap: int | None) -> list[int]:
            return select_stratified_indices(
                indices,
                [labels[index] for index in indices],
                [hospitals[index] for index in indices],
                cap,
                seed=config.subset_seed,
            )

        evaluation = {
            "id": select(official["id_val"], config.max_id_val_samples),
            "ood": select(official["val"], config.max_ood_val_samples),
        }
        evaluated = set(evaluation["id"]) | set(evaluation["ood"])
        spare = [index for index in official["val"] if index not in evaluated]
        training = select(official["train"], config.max_train_samples)
        share = max(1, len(training) * len(spare) // len(official["train"]))
        candidates = sorted(training + select(spare, share), key=lambda i: (slides[i], i))

        def samples(indices: list[int]) -> Samples:
            subset = dataset.get_patches(indices, transform)
            return Samples(
                lambda position: subset[position][0],
                {
                    "labels": [labels[index] for index in indices],
                    "hospitals": [hospitals[index] for index in indices],
                },
            )

        validation = samples(evaluation["id"])
        return PipelineData(
            train=samples(candidates),
            validation=validation,
            evaluation={"id": validation, "ood": samples(evaluation["ood"])},
            group="hospitals",
            holdouts={
                # WILDS draws in-distribution validation from the training slides by design.
                "id": {"held_out_by": None, "evaluation_groups": [0, 3, 4]},
                "ood": {"held_out_by": "hospital", "evaluation_groups": [1]},
            },
            record={
                "source": "camelyon17",
                "provenance": dataset.provenance,
                "subset_seed": config.subset_seed,
                "candidates": len(candidates),
                "candidate_indices_sha256": _digest(candidates),
                "evaluation_samples": {split: len(ids) for split, ids in evaluation.items()},
                "evaluation_indices_sha256": {
                    split: _digest(ids) for split, ids in evaluation.items()
                },
            },
        )

    return build


DATA_SOURCES = ("synthetic", "camelyon")


def data_builder(
    source: str, config: Path | None = None, max_eval_samples: int | None = None
) -> DataBuilder:
    """The data source named on the command line; Camelyon17 needs a config.

    max_eval_samples caps each Camelyon17 evaluation split, for quick runs.
    """
    if source == "synthetic":
        return synthetic_pipeline_data
    if source != "camelyon":
        raise ValueError(f"Unknown data source {source!r}; choose from {', '.join(DATA_SOURCES)}")
    if config is None:
        raise ValueError(
            "Camelyon17 data needs --config, for example configs/camelyon17_clean.json"
        )
    loaded = CamelyonConfig.load(Path(config))
    if max_eval_samples is not None:
        loaded = replace(
            loaded, max_id_val_samples=max_eval_samples, max_ood_val_samples=max_eval_samples
        )
    return camelyon_pipeline_data(loaded)


def run_in_subprocess(
    project: Path,
    run_dir: Path,
    *,
    data: str,
    epochs: int,
    seed: int,
    config: Path | None = None,
    device: str = "auto",
    interrupt_after_epochs: int | None = None,
    max_eval_samples: int | None = None,
    timeout: float = 3600.0,
) -> dict[str, Any]:
    """Run a project in a child process and return its outcome.

    status is refused, timeout or crashed, or the run report's status (completed,
    failed, diverged). report is the run report's path when one was written, and
    output is the end of what the child printed.
    """
    outcome = {"status": "crashed", "problems": [], "report": None, "error": None, "output": ""}
    problems = check_project(project_sources(project))
    if problems:
        return {**outcome, "status": "refused", "problems": problems}
    command = [
        sys.executable,
        "-m",
        "runsleuth.pipeline_workspace",
        "run",
        "--project",
        str(Path(project).resolve()),
        "--output",
        str(Path(run_dir).resolve()),
        "--data",
        data,
        "--epochs",
        str(epochs),
        "--seed",
        str(seed),
        "--device",
        device,
    ]
    if config is not None:
        command += ["--config", str(Path(config).resolve())]
    if interrupt_after_epochs is not None:
        command += ["--interrupt-after-epochs", str(interrupt_after_epochs)]
    if max_eval_samples is not None:
        command += ["--max-eval-samples", str(max_eval_samples)]
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
        return {**outcome, "status": "timeout", "output": f"no result within {timeout:g} seconds"}
    outcome["output"] = (completed.stdout + completed.stderr)[-OUTPUT_TAIL:]
    try:
        reported = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        return outcome
    return {**outcome, **reported}


def run_evidence(report: dict[str, Any]) -> dict[str, Any]:
    """What a run's pipeline record says about its training data and its delivery.

    held_out_training_samples is the most training samples read from groups that an
    evaluation split holds out; labels_matching is the share of the loss's targets that
    are the labels of the samples read, or None where they could not be paired.
    """
    pipeline = report.get("pipeline") or {}
    epochs = pipeline.get("epochs") or []
    held_out = [
        holdout["training_samples_in_evaluation_groups"]
        for holdout in (report.get("evaluation_holdouts") or {}).values()
        if holdout.get("held_out_by")
    ]
    matching = [epoch["loss_targets_matching_read_labels"] for epoch in epochs]
    targets = sum(epoch["loss_targets"] for epoch in epochs)
    return {
        "held_out_training_samples": max(held_out) if held_out else None,
        "labels_matching": None if None in matching or not targets else sum(matching) / targets,
        "delivered": (pipeline.get("delivery") or {}).get("same_as"),
    }


def check_versions(
    output_dir: Path,
    *,
    data: str,
    epochs: int,
    seed: int,
    config: Path | None = None,
    device: str = "auto",
    max_eval_samples: int | None = None,
    timeout: float = 3600.0,
    versions: tuple[str, ...] | None = None,
    family: str = "camelyon_pipeline",
    interrupt_after_epochs: int | None = 1,
    run: Callable[..., dict[str, Any]] = run_in_subprocess,
    progress: Callable[[str], None] = lambda _line: None,
) -> dict[str, Any]:
    """Train every version for the same budget and diagnose each run against the healthy one.

    The healthy project trains uninterrupted: it is the reference. Every other run is
    stopped after interrupt_after_epochs epochs and restarted, as the statement allows,
    so that a project that cannot resume shows it; the healthy project runs so once
    more (healthy_restarted) and is as_expected only if it then trains exactly the
    uninterrupted run. Any other version is as_expected when its run completes and is
    diagnosed with its signature, or with no_known_fault for a correct version or a
    fault the library has no signature for. trajectory compares the hashed model states
    with the healthy run's (see agent_verification). The check stops if the healthy run
    does not complete, and is written to check_report.json in output_dir, which must
    be new. With interrupt_after_epochs None nothing is restarted.
    """
    names = pipeline_tasks.family(family).names if versions is None else versions
    if HEALTHY not in names:
        raise ValueError("The healthy project is the reference and must be checked")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True)
    library, reports, rows = load_library(), {}, []
    restarted = (RESTARTED,) if interrupt_after_epochs is not None else ()
    ordered = (HEALTHY, *restarted, *(name for name in names if name != HEALTHY))
    for number, name in enumerate(ordered, start=1):
        progress(f"[{number}/{len(ordered)}] training {name}")
        source = HEALTHY if name == RESTARTED else name
        project = pipeline_tasks.materialize(source, output_dir / "projects" / name, family)
        outcome = run(
            project,
            output_dir / "runs" / name,
            data=data,
            config=config,
            epochs=epochs,
            seed=seed,
            device=device,
            interrupt_after_epochs=None if name == HEALTHY else interrupt_after_epochs,
            max_eval_samples=max_eval_samples,
            timeout=timeout,
        )
        described = None if source == HEALTHY else pipeline_tasks.version(source, family)
        row = {
            "version": name,
            "kind": "restart" if name == RESTARTED else pipeline_tasks.kind(source, family),
            "task": described and described.task,
            "split": described and described.split,
            "expected": pipeline_tasks.expected_diagnosis(source, family),
            "status": outcome["status"],
            "diagnosis": None,
            "as_expected": False,
            "trajectory": None,
            "metrics": None,
            "evidence": None,
            "report": outcome["report"],
            "output": None if outcome["status"] == "completed" else outcome["output"],
        }
        if outcome["report"] is not None:
            reports[name] = json.loads(Path(outcome["report"]).read_text(encoding="utf-8"))
        if name in reports and HEALTHY in reports:
            report, healthy = reports[name], reports[HEALTHY]
            row["diagnosis"] = diagnose(extract_evidence(report, healthy), library)["diagnosis"]
            row["trajectory"] = compare_trajectories(report, healthy)
            row["as_expected"] = (
                outcome["status"] == "completed" and row["diagnosis"] == row["expected"]
            )
            if name == RESTARTED:
                row["as_expected"] &= row["trajectory"]["outcome"] == "identical_full_run"
            row["metrics"] = {
                name: value for name, value in report["final_metrics"].items() if name != "epoch"
            }
            row["evidence"] = run_evidence(report)
        rows.append(row)
        if name == HEALTHY and outcome["status"] != "completed":
            # Without the reference no other version can be diagnosed.
            break
    result = {
        "schema_version": 1,
        "task": "pipeline_check",
        "settings": {
            "family": family,
            "data": data,
            "config": None if config is None else str(config),
            "epochs": epochs,
            "seed": seed,
            "max_eval_samples": max_eval_samples,
            "interrupt_after_epochs": interrupt_after_epochs,
        },
        "versions_as_expected": sum(row["as_expected"] for row in rows),
        "versions": rows,
    }
    (output_dir / "check_report.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def _replay(trajectory: dict[str, Any] | None) -> str:
    """Where a run's hashed states leave the healthy run's, for the check table."""
    if trajectory is None or trajectory["outcome"] == "not_comparable":
        return "-"
    if trajectory["outcome"] != "diverged":
        return "identical" if trajectory["outcome"] == "identical_full_run" else "checked steps"
    where = trajectory["first_difference"]
    place = where["at"].replace("_", " ")
    return place if where["index"] is None else f"{place} {where['index']}"


def check_text(result: dict[str, Any]) -> str:
    """The check as a table, one line per version, then what failed runs printed.

    held out is the most training samples read from a held-out group, labels the share
    of the loss's targets that are the labels of the samples read.
    """
    lines = [
        f"{'version':<24}{'split':<13}{'status':<11}{'diagnosis':<32}{'ok':<5}{'replay':<13}"
        f"{'held out':>9}{'labels':>8}  id acc  ood acc"
    ]
    for row in result["versions"]:
        metrics, evidence = row["metrics"] or {}, row["evidence"] or {}
        accuracies = "  ".join(
            f"{metrics[name]:.4f}" if name in metrics else "   -  "
            for name in ("id_validation_accuracy", "ood_validation_accuracy")
        )
        held_out = evidence.get("held_out_training_samples")
        labels = evidence.get("labels_matching")
        lines.append(
            f"{row['version']:<24}{row['split'] or '-':<13}{row['status']:<11}"
            f"{row['diagnosis'] or '-':<32}{'yes' if row['as_expected'] else 'NO':<5}"
            f"{_replay(row['trajectory']):<13}{'-' if held_out is None else held_out:>9}"
            f"{'-' if labels is None else f'{labels:.0%}':>8}  {accuracies}"
        )
    lines.append(f"{result['versions_as_expected']}/{len(result['versions'])} versions as expected")
    for row in result["versions"]:
        if row.get("output"):
            lines.append(f"\n{row['version']} ({row['status']}) printed:\n{row['output'].strip()}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an agent's training project.")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run a project under monitoring.")
    run.add_argument("--project", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True, help="New run directory.")
    run.add_argument("--epochs", type=int, default=2)
    run.add_argument(
        "--interrupt-after-epochs",
        type=int,
        help="Stop the first attempt after this many epochs and restart the project.",
    )
    check = commands.add_parser(
        "check", help="Train every version and diagnose each run against the healthy one."
    )
    check.add_argument("--output", type=Path, required=True, help="New directory for the check.")
    check.add_argument("--timeout", type=float, default=3600.0, help="Seconds per version.")
    check.add_argument(
        "--versions", nargs="+", help="Versions to check, all if unset; healthy must be among them."
    )
    check.add_argument(
        "--family", choices=tuple(pipeline_tasks.FAMILIES), default="camelyon_pipeline"
    )
    # Three epochs, so that a schedule that is not restored shows after a restart.
    check.add_argument("--epochs", type=int, default=3)
    check.add_argument(
        "--interrupt-after-epochs",
        type=int,
        default=1,
        help="Restart every run but the healthy reference after this many epochs (0: none).",
    )
    for command in (run, check):
        command.add_argument("--data", choices=DATA_SOURCES, required=True)
        command.add_argument("--config", type=Path, help="Camelyon17 config file.")
        command.add_argument("--seed", type=int, default=7)
        command.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
        command.add_argument(
            "--max-eval-samples", type=int, help="Cap each Camelyon17 evaluation split."
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one project, or check every version.

    For run, the last line printed is the outcome as JSON.
    """
    args = build_parser().parse_args(argv)
    if args.command == "check":
        result = check_versions(
            args.output,
            data=args.data,
            config=args.config,
            epochs=args.epochs,
            seed=args.seed,
            device=args.device,
            max_eval_samples=args.max_eval_samples,
            timeout=args.timeout,
            versions=None if args.versions is None else tuple(args.versions),
            family=args.family,
            interrupt_after_epochs=args.interrupt_after_epochs or None,
            progress=lambda line: print(line, flush=True),
        )
        print(check_text(result))
        return 0 if result["versions_as_expected"] == len(result["versions"]) else 1
    outcome = {"status": "failed", "problems": [], "report": None, "error": None}
    try:
        report = run_project(
            args.project,
            args.output,
            data_builder(args.data, args.config, args.max_eval_samples),
            epochs=args.epochs,
            seed=args.seed,
            device=resolve_device(args.device),
            interrupt_after_epochs=args.interrupt_after_epochs,
        )
        status = json.loads(report.read_text(encoding="utf-8"))["status"]
        outcome.update(status=status, report=str(report))
    except ScriptRefused as refusal:
        outcome.update(status="refused", problems=refusal.problems)
    except Exception as error:
        traceback.print_exc()
        outcome["error"] = {"type": type(error).__name__, "message": str(error)}
    print(json.dumps(outcome))
    return 0 if outcome["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
