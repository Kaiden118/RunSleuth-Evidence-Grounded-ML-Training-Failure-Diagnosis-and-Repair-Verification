"""Run an agent workspace's training script under RunSleuth monitoring.

The script (train.py, see agent_tasks) only defines the pipeline. This harness owns
what an agent must not change: the data, the epoch budget, the monitoring, a hash of
the model state after each of the first optimizer steps, and the evaluation.

A script is checked before it runs and executed with restricted builtins: imports
outside an allowlist and file, network or process access are refused. This guards
against accidents and plain misuse; it is not a sandbox. The script still runs as
this user, so run_in_subprocess adds a time limit and removes credentials from the
environment.

    python -m runsleuth.agent_workspace run --workspace W --output runs/w --data synthetic
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

from runsleuth.agent_tasks import SCRIPT_NAME
from runsleuth.optimizer_probe import state_dict_sha256
from runsleuth.run_monitor import RunMonitor
from runsleuth.train import resolve_device, seed_everything

ALLOWED_IMPORTS = frozenset(
    {"torch", "torchvision", "math", "typing", "functools", "itertools", "collections"}
)
DENIED_NAMES = frozenset(
    {
        "open",
        "exec",
        "eval",
        "compile",
        "input",
        "breakpoint",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "exit",
        "quit",
    }
)
# Attribute and imported names that read or write files, reach the network or start
# processes. Names with a leading underscore are refused as well, except __init__.
DENIED_ATTRIBUTES = frozenset(
    {
        "load",
        "save",
        "hub",
        "jit",
        "io",
        "datasets",
        "onnx",
        "tensorboard",
        "save_image",
        "write_png",
        "write_jpeg",
        "cpp_extension",
        "collect_env",
        "model_zoo",
        "serialization",
        "multiprocessing",
        "distributed",
        "package",
        "from_file",
        "tofile",
        "load_url",
        "load_state_dict_from_url",
        "download_url_to_file",
        "system",
        "popen",
    }
)
ALLOWED_BUILTINS = (
    "abs",
    "all",
    "any",
    "bool",
    "callable",
    "classmethod",
    "dict",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "frozenset",
    "hasattr",
    "int",
    "isinstance",
    "issubclass",
    "iter",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "object",
    "pow",
    "print",
    "property",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "staticmethod",
    "str",
    "sum",
    "super",
    "tuple",
    "type",
    "zip",
    "ArithmeticError",
    "AssertionError",
    "AttributeError",
    "Exception",
    "IndexError",
    "KeyError",
    "NotImplementedError",
    "RuntimeError",
    "StopIteration",
    "TypeError",
    "ValueError",
    "ZeroDivisionError",
    "__build_class__",
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

    Evaluation split names become <split>_validation_accuracy and _loss.
    """

    train_loader: Any
    evaluation_loaders: dict[str, Any]
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
        record={
            "source": "synthetic",
            "training_hospitals": admitted,
            "training_samples": 8 * len(admitted),
            "evaluation_hospitals": {"id": [0, 3, 4], "ood": [1]},
        },
    )


DATA_SOURCES: dict[str, DataBuilder] = {"synthetic": synthetic_task_data}


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
    hashed_steps: int = HASHED_STEPS,
) -> Path:
    """Train the workspace script for a fixed budget; return the run report's path.

    Once training has started, a script that fails or diverges still yields a report
    that says so. A refused script raises ScriptRefused, and an error while the script
    builds its model propagates; neither leaves a report.
    """
    script = Path(workspace) / SCRIPT_NAME
    module = load_script(script)
    data = build_data(module.use_for_training, device, seed)
    seed_everything(seed)
    model, optimizer, scheduler = module.build(device, epochs)
    loss_function = nn.CrossEntropyLoss()
    monitor = RunMonitor(model, optimizer, run_dir, head=head)
    record = {
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "seed": seed,
        "epochs": epochs,
        "max_train_batches": max_train_batches,
        "data": data.record,
        "initialization_metrics": None,
        "initial_state_sha256": state_dict_sha256(model.state_dict()),
        "hashed_steps": hashed_steps,
        "step_state_sha256": [],
        "final_state_sha256": None,
    }
    monitor.report["workspace"] = record
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
                phase["training"] = True
                with monitor.epoch():
                    module.train_one_epoch(model, batches(), optimizer, scheduler, loss_function)
                phase["training"] = False
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
    device: str = "auto",
    max_train_batches: int | None = None,
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
    if max_train_batches is not None:
        command += ["--max-train-batches", str(max_train_batches)]
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an agent workspace's training script.")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Train a workspace script under monitoring.")
    run.add_argument("--workspace", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True, help="New run directory.")
    run.add_argument("--data", choices=sorted(DATA_SOURCES), required=True)
    run.add_argument("--epochs", type=int, default=3)
    run.add_argument("--seed", type=int, default=7)
    run.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    run.add_argument("--max-train-batches", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one workspace; the last line printed is the outcome as JSON."""
    args = build_parser().parse_args(argv)
    outcome = {"status": "failed", "problems": [], "report": None, "error": None}
    try:
        report = run_workspace(
            args.workspace,
            args.output,
            DATA_SOURCES[args.data],
            epochs=args.epochs,
            seed=args.seed,
            device=resolve_device(args.device),
            max_train_batches=args.max_train_batches,
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
