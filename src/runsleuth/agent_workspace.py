"""Run an agent workspace's training script under RunSleuth monitoring.

The script (train.py, see agent_tasks) only defines the pipeline. This harness owns
what an agent must not change: the data, the epoch budget, the monitoring, a hash of
the model state after each of the first optimizer steps and after every epoch, and
the evaluation.

A script is checked before it runs and executed with restricted builtins: imports
outside an allowlist and file, network or process access are refused. This guards
against accidents and plain misuse; it is not a sandbox. The script still runs as
this user, so run_in_subprocess adds a time limit and removes credentials from the
environment.

    python -m runsleuth.agent_workspace run --workspace W --output runs/w --data synthetic
    python -m runsleuth.agent_workspace check --output artifacts/agent_tasks/check-1 \
        --data camelyon --config configs/camelyon17_clean.json --max-train-batches 200

check trains every task of agent_tasks for the same budget and diagnoses each run
against the healthy one, which shows whether the faults are visible in short runs.
"""

import argparse
import ast
import builtins
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import traceback
import types
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from runsleuth import agent_tasks
from runsleuth.agent_tasks import HEALTHY, SCRIPT_NAME, TASKS
from runsleuth.agent_verification import compare_trajectories
from runsleuth.camelyon_config import CamelyonConfig
from runsleuth.camelyon_data import (
    NativeRGBPatch,
    PairDataset,
    load_camelyon_mirror,
    seed_worker,
    select_stratified_indices,
)
from runsleuth.optimizer_probe import state_dict_sha256
from runsleuth.run_monitor import RunMonitor
from runsleuth.signature_matching import diagnose, extract_evidence, load_library
from runsleuth.train import resolve_device, seed_everything

ALLOWED_IMPORTS = frozenset(
    {"torch", "torchvision", "math", "typing", "functools", "itertools", "collections"}
)
DENIED_NAMES = frozenset(
    "open exec eval compile input breakpoint globals locals vars getattr setattr delattr "
    "exit quit".split()
)
# Attribute and imported names that read or write files, reach the network or start
# processes. Names with a leading underscore are refused as well, except __init__.
DENIED_ATTRIBUTES = frozenset(
    "load save hub jit io datasets onnx tensorboard save_image write_png write_jpeg "
    "cpp_extension collect_env model_zoo serialization multiprocessing distributed package "
    "from_file tofile load_url load_state_dict_from_url download_url_to_file system "
    "popen".split()
)
ALLOWED_BUILTINS = tuple(
    "abs all any bool callable classmethod dict divmod enumerate filter float frozenset "
    "hasattr int isinstance issubclass iter len list map max min next object pow print "
    "property range repr reversed round set slice sorted staticmethod str sum super tuple "
    "type zip ArithmeticError AssertionError AttributeError Exception IndexError KeyError "
    "NotImplementedError RuntimeError StopIteration TypeError ValueError ZeroDivisionError "
    "__build_class__".split()
)
REQUIRED_FUNCTIONS = (
    "use_for_training",
    "prepare_training_batch",
    "prepare_evaluation_batch",
    "build",
    "train_one_epoch",
)
MAX_SOURCE_CHARACTERS = 20_000
HASHED_STEPS = 32
OUTPUT_TAIL = 4000
SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.IGNORECASE)


class ScriptRefused(ValueError):
    """A workspace script that the harness will not run; problems says why."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = list(problems)


def _denied_attribute(name: str) -> bool:
    return name in DENIED_ATTRIBUTES or (name.startswith("_") and name != "__init__")


def check_script(source: str) -> list[str]:
    """Reasons to refuse a script, in line order; empty when it may run."""
    if len(source) > MAX_SOURCE_CHARACTERS:
        return [f"the script is longer than {MAX_SOURCE_CHARACTERS} characters"]
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return [f"line {error.lineno}: syntax error: {error.msg}"]
    problems = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            members = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif node.level:
                problems.add((node.lineno, "relative imports are not allowed"))
                continue
            else:
                modules = [node.module]
                members = [alias.name for alias in node.names]
            for module in modules:
                root, *rest = module.split(".")
                if root not in ALLOWED_IMPORTS:
                    problems.add((node.lineno, f"import of {module!r} is not allowed"))
                members.extend(rest)
            for member in members:
                if member == "*" or _denied_attribute(member):
                    problems.add((node.lineno, f"import of {member!r} is not allowed"))
        elif isinstance(node, ast.Name):
            if node.id in DENIED_NAMES or (node.id.startswith("__") and node.id != "__name__"):
                problems.add((node.lineno, f"{node.id!r} is not allowed"))
        elif isinstance(node, ast.Attribute) and _denied_attribute(node.attr):
            problems.add((node.lineno, f"attribute {node.attr!r} is not allowed"))
    return [f"line {line}: {reason}" for line, reason in sorted(problems)]


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name.split(".")[0] not in ALLOWED_IMPORTS:
        raise ImportError(f"import of {name!r} is not allowed in a workspace script")
    return builtins.__import__(name, globals, locals, fromlist, level)


def restricted_builtins() -> dict[str, Any]:
    """The builtins a workspace script runs with: no file access, guarded imports."""
    return {
        **{name: getattr(builtins, name) for name in ALLOWED_BUILTINS},
        "__import__": _guarded_import,
    }


def load_script(path: Path) -> types.ModuleType:
    """Check a workspace script and execute it as a module with restricted builtins."""
    source = Path(path).read_text(encoding="utf-8")
    problems = check_script(source)
    if problems:
        raise ScriptRefused(problems)
    module = types.ModuleType("workspace_train")
    vars(module)["__builtins__"] = restricted_builtins()
    exec(compile(source, str(path), "exec"), vars(module))
    missing = [name for name in REQUIRED_FUNCTIONS if not callable(vars(module).get(name))]
    if missing:
        raise ScriptRefused([f"the script must define {name}()" for name in missing])
    return module


@dataclass(frozen=True)
class TaskData:
    """Patches as RGB tensors in [0, 1]; the script normalizes them itself.

    Evaluation split names become <split>_validation_accuracy and _loss. holdouts says,
    per evaluation split, which group it holds out (held_out_by, or None) and how many
    training samples come from the groups it evaluates on; it becomes the run report's
    evaluation_holdouts.
    """

    train_loader: Any
    evaluation_loaders: dict[str, Any]
    holdouts: dict[str, dict[str, Any]]
    record: dict[str, Any]


DataBuilder = Callable[[Callable[[int], bool], torch.device, int], TaskData]


def synthetic_task_data(
    use_for_training: Callable[[int], bool], device: torch.device, seed: int
) -> TaskData:
    """Random patches from four hospitals: a smoke test of the harness, not a benchmark.

    Hospitals 0, 3 and 4 supply training and in-distribution validation patches.
    Hospital 1 supplies the out-of-distribution validation patches, and further
    patches that reach training only if the script admits that hospital.
    """
    generator = torch.Generator().manual_seed(seed)

    def patches(count: int, brightness: float = 1.0):
        labels = torch.arange(count) % 2
        noise = torch.rand(count, 3, 16, 16, generator=generator)
        return (noise * 0.5 + labels.view(-1, 1, 1, 1) * 0.5) * brightness, labels

    def loader(parts, *, shuffle=False):
        images = torch.cat([part[0] for part in parts])
        labels = torch.cat([part[1] for part in parts])
        return DataLoader(
            TensorDataset(images, labels),
            batch_size=4,
            shuffle=shuffle,
            generator=torch.Generator().manual_seed(seed),
        )

    candidates = {hospital: patches(8) for hospital in (0, 3, 4)}
    candidates[1] = patches(8, brightness=0.8)
    admitted = sorted(hospital for hospital in candidates if use_for_training(hospital))
    if not admitted:
        raise ValueError("use_for_training admits no hospital")
    return TaskData(
        train_loader=loader([candidates[hospital] for hospital in admitted], shuffle=True),
        evaluation_loaders={
            "id": loader([patches(4) for _ in (0, 3, 4)]),
            "ood": loader([patches(12, brightness=0.8)]),
        },
        holdouts={
            "id": {"held_out_by": None, "evaluation_groups": [0, 3, 4]},
            "ood": {
                "held_out_by": "hospital",
                "evaluation_groups": [1],
                "training_samples_in_evaluation_groups": 8 if 1 in admitted else 0,
            },
        },
        record={
            "source": "synthetic",
            "training_hospitals": admitted,
            "training_samples": 8 * len(admitted),
        },
    )


def camelyon_task_data(config: CamelyonConfig) -> DataBuilder:
    """Camelyon17 patches for workspace scripts, selected as the baseline selects them.

    Evaluation uses the baseline's in-distribution and out-of-distribution validation
    subsets. Training patches are drawn from the hospitals the script admits: the
    official training split, plus those patches of the validation hospital that
    evaluation does not use. A script that admits only the training hospitals thus
    trains on exactly the baseline's subset, and one that admits the validation
    hospital trains on its slides without ever seeing an evaluation patch itself.
    """

    def build(use_for_training: Callable[[int], bool], device: torch.device, seed: int):
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
        candidates = official["train"] + [i for i in official["val"] if i not in evaluated]
        admitted = {
            hospital
            for hospital in sorted({hospitals[index] for index in candidates})
            if use_for_training(hospital)
        }
        candidates = [index for index in candidates if hospitals[index] in admitted]
        if not candidates:
            raise ValueError("use_for_training admits no hospital with patches")
        training = select(candidates, config.max_train_samples)

        def loader(indices: list[int], *, shuffle: bool = False):
            return DataLoader(
                PairDataset(dataset.get_patches(indices, transform), range(len(indices))),
                batch_size=config.batch_size,
                shuffle=shuffle,
                num_workers=config.num_workers,
                pin_memory=device.type == "cuda",
                generator=torch.Generator().manual_seed(seed),
                worker_init_fn=seed_worker if config.num_workers else None,
            )

        def groups(indices: list[int], values: list[int]) -> list[int]:
            return sorted({values[index] for index in indices})

        def digest(indices: list[int]) -> str:
            return hashlib.sha256(json.dumps(indices, separators=(",", ":")).encode()).hexdigest()

        ood_hospitals = groups(evaluation["ood"], hospitals)
        training_slides = set(groups(training, slides))
        holdouts = {
            # WILDS draws in-distribution validation from the training slides by design.
            "id": {"held_out_by": None, "evaluation_groups": groups(evaluation["id"], hospitals)},
            "ood": {
                "held_out_by": "hospital",
                "evaluation_groups": ood_hospitals,
                "training_samples_in_evaluation_groups": sum(
                    hospitals[index] in ood_hospitals for index in training
                ),
            },
        }
        for split, indices in evaluation.items():
            holdouts[split]["slides_shared_with_training"] = len(
                training_slides.intersection(groups(indices, slides))
            )
        return TaskData(
            train_loader=loader(training, shuffle=True),
            evaluation_loaders={split: loader(indices) for split, indices in evaluation.items()},
            holdouts=holdouts,
            record={
                "source": "camelyon17",
                "provenance": dataset.provenance,
                "subset_seed": config.subset_seed,
                "batch_size": config.batch_size,
                "training_hospitals": groups(training, hospitals),
                "training_samples": len(training),
                "training_indices_sha256": digest(training),
                "evaluation_samples": {split: len(ids) for split, ids in evaluation.items()},
                "evaluation_indices_sha256": {
                    split: digest(ids) for split, ids in evaluation.items()
                },
            },
        )

    return build


DATA_SOURCES = ("synthetic", "camelyon")


def data_builder(source: str, config: Path | None = None) -> DataBuilder:
    """The data source named on the command line; Camelyon17 needs its config file."""
    if source == "synthetic":
        return synthetic_task_data
    if source != "camelyon":
        raise ValueError(f"Unknown data source {source!r}; choose from {', '.join(DATA_SOURCES)}")
    if config is None:
        raise ValueError(
            "Camelyon17 data needs --config, for example configs/camelyon17_clean.json"
        )
    return camelyon_task_data(CamelyonConfig.load(Path(config)))


def run_workspace(
    workspace: Path,
    run_dir: Path,
    build_data: DataBuilder,
    *,
    epochs: int,
    seed: int,
    device: torch.device,
    head: str = "fc",
    max_train_batches: int | None = None,
    stop_after_epochs: int | None = None,
    hashed_steps: int = HASHED_STEPS,
) -> Path:
    """Train the workspace script for a fixed budget; return the run report's path.

    stop_after_epochs ends the run early without changing what the script is told:
    it still builds its schedule for epochs epochs, so the epochs it does run replay
    the start of a full run, and no final state is recorded.

    Once training has started, a script that fails or diverges still yields a report
    that says so. A refused script raises ScriptRefused, and an error while the script
    builds its model propagates; neither leaves a report.
    """
    script = Path(workspace) / SCRIPT_NAME
    module = load_script(script)
    data = build_data(module.use_for_training, device, seed)
    steps_per_epoch = len(data.train_loader)
    if max_train_batches is not None:
        steps_per_epoch = min(steps_per_epoch, max_train_batches)
    seed_everything(seed)
    model, optimizer, scheduler = module.build(device, epochs, steps_per_epoch)
    loss_function = nn.CrossEntropyLoss()
    monitor = RunMonitor(model, optimizer, run_dir, head=head)
    record = {
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "seed": seed,
        "epochs": epochs,
        "max_train_batches": max_train_batches,
        "steps_per_epoch": steps_per_epoch,
        "stop_after_epochs": stop_after_epochs,
        "data": data.record,
        "initialization_metrics": None,
        "initial_state_sha256": state_dict_sha256(model.state_dict()),
        "hashed_steps": hashed_steps,
        "step_state_sha256": [],
        "epoch_state_sha256": [],
        "final_state_sha256": None,
    }
    monitor.report["workspace"] = record
    monitor.report["evaluation_holdouts"] = data.holdouts
    phase = {"training": False, "labels": None}

    def observe_training_inputs(_module, inputs) -> None:
        if phase["training"]:
            monitor.observe_batch("train", inputs[0], phase["labels"])

    def hash_step(*_) -> None:
        if len(record["step_state_sha256"]) < hashed_steps:
            record["step_state_sha256"].append(state_dict_sha256(model.state_dict()))

    def batches():
        for index, (images, labels) in enumerate(data.train_loader):
            if max_train_batches is not None and index >= max_train_batches:
                break
            phase["labels"] = labels
            yield images.to(device), labels.to(device)

    @torch.inference_mode()
    def evaluate() -> dict[str, float]:
        model.eval()
        metrics = {}
        for split, loader in data.evaluation_loaders.items():
            loss, correct, count = 0.0, 0, 0
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                inputs = module.prepare_evaluation_batch(images)
                monitor.observe_batch(f"{split}_eval", inputs, labels)
                logits = model(inputs)
                loss += float(loss_function(logits, labels)) * len(labels)
                correct += int((logits.argmax(dim=1) == labels).sum())
                count += len(labels)
            metrics[f"{split}_validation_loss"] = loss / count
            metrics[f"{split}_validation_accuracy"] = correct / count
        return metrics

    def finite(metrics: dict[str, float]) -> bool:
        return all(math.isfinite(value) for value in metrics.values())

    handles = [
        model.register_forward_pre_hook(observe_training_inputs),
        optimizer.register_step_post_hook(hash_step),
    ]
    try:
        with monitor:
            initial = evaluate()
            record["initialization_metrics"] = initial if finite(initial) else None
            for epoch in range(1, epochs + 1):
                if stop_after_epochs is not None and epoch > stop_after_epochs:
                    break
                phase["training"] = True
                with monitor.epoch():
                    module.train_one_epoch(model, batches(), optimizer, scheduler, loss_function)
                phase["training"] = False
                record["epoch_state_sha256"].append(state_dict_sha256(model.state_dict()))
                metrics = evaluate()
                if not finite(metrics):
                    monitor.report["status"] = "diverged"
                    monitor.report["divergence"] = {
                        "epoch": epoch,
                        "message": "validation metrics are not finite",
                    }
                    break
                monitor.log(**metrics)
            else:
                record["final_state_sha256"] = state_dict_sha256(model.state_dict())
    except Exception:
        # RunMonitor has recorded the failure; keep where in the script it happened.
        record["traceback"] = traceback.format_exc()[-OUTPUT_TAIL:]
    finally:
        for handle in handles:
            handle.remove()
    # The monitor closes a failed epoch before this harness can add to the report.
    monitor.report_path.write_text(
        json.dumps(monitor.report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return monitor.report_path


def clean_environment(environment: dict[str, str] | None = None) -> dict[str, str]:
    """A copy of the environment without variables that look like credentials."""
    source = os.environ if environment is None else environment
    cleaned = {name: value for name, value in source.items() if not SECRET_NAME.search(name)}
    return {**cleaned, "PYTHONIOENCODING": "utf-8"}


def run_in_subprocess(
    workspace: Path,
    run_dir: Path,
    *,
    data: str,
    epochs: int,
    seed: int,
    config: Path | None = None,
    device: str = "auto",
    max_train_batches: int | None = None,
    stop_after_epochs: int | None = None,
    timeout: float = 3600.0,
) -> dict[str, Any]:
    """Run a workspace in a child process and return its outcome.

    status is refused, timeout or crashed, or the run report's status (completed,
    failed, diverged). report is the run report's path when one was written, and
    output is the end of what the child printed.
    """
    outcome = {"status": "crashed", "problems": [], "report": None, "error": None, "output": ""}
    problems = check_script((Path(workspace) / SCRIPT_NAME).read_text(encoding="utf-8"))
    if problems:
        return {**outcome, "status": "refused", "problems": problems}
    command = [
        sys.executable,
        "-m",
        "runsleuth.agent_workspace",
        "run",
        "--workspace",
        str(Path(workspace).resolve()),
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
    if max_train_batches is not None:
        command += ["--max-train-batches", str(max_train_batches)]
    if stop_after_epochs is not None:
        command += ["--stop-after-epochs", str(stop_after_epochs)]
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


def check_tasks(
    output_dir: Path,
    *,
    data: str,
    epochs: int,
    seed: int,
    config: Path | None = None,
    device: str = "auto",
    max_train_batches: int | None = None,
    timeout: float = 3600.0,
    tasks: tuple[str, ...] = TASKS,
    run: Callable[..., dict[str, Any]] = run_in_subprocess,
    progress: Callable[[str], None] = lambda _line: None,
) -> dict[str, Any]:
    """Train every task for the same budget and diagnose each run against the healthy one.

    A task is as_expected when its run completes and the diagnosis is its fault's
    signature, or no_known_fault for a correct script. trajectory compares the model
    states the run hashed with the healthy run's (see agent_verification). The check
    stops if the healthy run does not complete. The result is also written to
    check_report.json in output_dir, which must be new.
    """
    if HEALTHY not in tasks:
        raise ValueError("The healthy task is the reference and must be checked")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True)
    library, reports, rows = load_library(), {}, []
    ordered = (HEALTHY, *(name for name in tasks if name != HEALTHY))
    for number, task in enumerate(ordered, start=1):
        progress(f"[{number}/{len(ordered)}] training {task}")
        workspace = agent_tasks.materialize(task, output_dir / "workspaces" / task)
        outcome = run(
            workspace,
            output_dir / "runs" / task,
            data=data,
            config=config,
            epochs=epochs,
            seed=seed,
            device=device,
            max_train_batches=max_train_batches,
            timeout=timeout,
        )
        row = {
            "task": task,
            "expected": agent_tasks.expected_diagnosis(task),
            "status": outcome["status"],
            "diagnosis": None,
            "as_expected": False,
            "trajectory": None,
            "metrics": None,
            "report": outcome["report"],
            "output": None if outcome["status"] == "completed" else outcome["output"],
        }
        if outcome["report"] is not None:
            reports[task] = json.loads(Path(outcome["report"]).read_text(encoding="utf-8"))
        if task in reports and HEALTHY in reports:
            report, healthy = reports[task], reports[HEALTHY]
            row["diagnosis"] = diagnose(extract_evidence(report, healthy), library)["diagnosis"]
            row["as_expected"] = (
                outcome["status"] == "completed" and row["diagnosis"] == row["expected"]
            )
            row["trajectory"] = compare_trajectories(report, healthy)
            row["metrics"] = {
                name: value for name, value in report["final_metrics"].items() if name != "epoch"
            }
        rows.append(row)
        if task == HEALTHY and outcome["status"] != "completed":
            # Without the reference no other task can be diagnosed.
            break
    result = {
        "schema_version": 1,
        "task": "agent_task_check",
        "settings": {
            "data": data,
            "config": None if config is None else str(config),
            "epochs": epochs,
            "seed": seed,
            "max_train_batches": max_train_batches,
        },
        "tasks_as_expected": sum(row["as_expected"] for row in rows),
        "tasks": rows,
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
    """The check as a table, one line per task, then what failed tasks printed.

    replay is identical when every hashed model state matches the healthy run's, or
    names the first state that differs.
    """
    lines = [
        f"{'task':<40}{'status':<11}{'diagnosis':<36}{'ok':<5}{'replay':<15}id acc  ood acc",
    ]
    for row in result["tasks"]:
        metrics = row["metrics"] or {}
        accuracies = "  ".join(
            f"{metrics[name]:.4f}" if name in metrics else "   -  "
            for name in ("id_validation_accuracy", "ood_validation_accuracy")
        )
        lines.append(
            f"{row['task']:<40}{row['status']:<11}{row['diagnosis'] or '-':<36}"
            f"{'yes' if row['as_expected'] else 'NO':<5}{_replay(row['trajectory']):<15}"
            f"{accuracies}"
        )
    lines.append(f"{result['tasks_as_expected']}/{len(result['tasks'])} tasks as expected")
    for row in result["tasks"]:
        if row.get("output"):
            lines.append(f"\n{row['task']} ({row['status']}) printed:\n{row['output'].strip()}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an agent workspace's training script.")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Train a workspace script under monitoring.")
    run.add_argument("--workspace", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True, help="New run directory.")
    run.add_argument(
        "--stop-after-epochs", type=int, help="End early; the schedule keeps its full length."
    )
    check = commands.add_parser(
        "check", help="Train every task and diagnose each run against the healthy one."
    )
    check.add_argument("--output", type=Path, required=True, help="New directory for the check.")
    check.add_argument("--timeout", type=float, default=3600.0, help="Seconds per task.")
    check.add_argument(
        "--tasks",
        nargs="+",
        choices=TASKS,
        default=list(TASKS),
        help="Tasks to check; healthy is the reference and must be among them.",
    )
    for command in (run, check):
        command.add_argument("--data", choices=DATA_SOURCES, required=True)
        command.add_argument("--config", type=Path, help="Camelyon17 config file.")
        command.add_argument("--epochs", type=int, default=3)
        command.add_argument("--seed", type=int, default=7)
        command.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
        command.add_argument("--max-train-batches", type=int, help="Batches per epoch.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one workspace, or check every task.

    For run, the last line printed is the outcome as JSON.
    """
    args = build_parser().parse_args(argv)
    if args.command == "check":
        result = check_tasks(
            args.output,
            data=args.data,
            config=args.config,
            epochs=args.epochs,
            seed=args.seed,
            device=args.device,
            max_train_batches=args.max_train_batches,
            timeout=args.timeout,
            tasks=tuple(args.tasks),
            progress=lambda line: print(line, flush=True),
        )
        print(check_text(result))
        return 0 if result["tasks_as_expected"] == len(result["tasks"]) else 1
    outcome = {"status": "failed", "problems": [], "report": None, "error": None}
    try:
        report = run_workspace(
            args.workspace,
            args.output,
            data_builder(args.data, args.config),
            epochs=args.epochs,
            seed=args.seed,
            device=resolve_device(args.device),
            max_train_batches=args.max_train_batches,
            stop_after_epochs=args.stop_after_epochs,
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
