"""Checks of a training program written from scratch, made from outside the program.

A program (standalone_task) is run on one seed straight through, straight through once
more, and stopped after an epoch and resumed, once for every epoch it can be stopped
after. From the files the runs leave behind, and from the checker's own evaluation of
every epoch's weights, three things the statement asks for are decided with no
reference implementation:

- repeat: the same run made twice writes the same weights.
- restart: a stopped and resumed run trains as the uninterrupted one. What the log says
  of the epochs and the learning rates is compared first: it does not depend on
  arithmetic, so it decides even for a program that does not repeat itself. The
  weights after the last epoch decide only for a program that repeats itself;
  otherwise the requirement is undecided, not violated.
- delivery: the delivered weights are those of the earliest epoch with the highest
  validation accuracy, decided for the uninterrupted run and for every resumed one. The
  checker counts the correct validation samples of every epoch itself. Where its
  ranking and the one in the program's log name different epochs, delivery is
  undecided: a program is not blamed for what may be a difference in arithmetic.

A verdict is about these runs. "holds" says that the requirement held on them, never
that the program is correct.

    python -m runsleuth.standalone_check run --programs artifacts/standalone/programs \\
        --data artifacts/standalone/data --out artifacts/standalone/checks --work D:/work
    python -m runsleuth.standalone_check report artifacts/standalone/checks

A program is screened before it runs, and runs without the variables that look like
credentials. The screen refuses what a training program has no use for; it is not a
sandbox.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from runsleuth import standalone_programs, standalone_task
from runsleuth.agent_workspace import OUTPUT_TAIL, clean_environment
from runsleuth.optimizer_probe import state_dict_sha256

# The epochs and the first seed of RunSleuth v2's delivery probes
# (evaluations/verifier_v2_probes.json). A run of them takes seconds, so a program is
# stopped after every epoch but the last in turn: whenever its best epoch is not the
# last, one of the stops lies after it.
EPOCHS, SEED = 5, 101
TIMEOUT = 1800.0
BATCH_SIZE = 32
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CHECK_FILE = "check.json"
MAX_SOURCE_CHARACTERS = 40_000
# refused: the screen; runtime_failure: an uninterrupted run did not complete;
# unsupported_interface: a run completed without the files the statement asks for;
# resume_failure: the uninterrupted runs are fine and a stop or a resume did not
# complete; decided: every run can be used.
CATEGORIES = (
    "refused",
    "runtime_failure",
    "unsupported_interface",
    "resume_failure",
    "decided",
)
VIOLATIONS = ("violated", "violated_tie", "no_epoch")
ALLOWED_IMPORTS = frozenset(
    "argparse collections contextlib copy dataclasses datetime functools gc glob hashlib io "
    "itertools json logging math numpy os pathlib random re shutil sys time torch torchvision "
    "typing warnings".split()
)
ALLOWED_MODULES = frozenset({"urllib.parse"})
DENIED_NAMES = frozenset("exec eval compile input breakpoint __import__".split())
# Names that start processes, fetch from the network or remove directory trees.
DENIED_ATTRIBUTES = frozenset(
    "system popen startfile execl execle execlp execv execve execvp spawnl spawnle spawnlp "
    "spawnv spawnve spawnvp kill rmtree load_url load_state_dict_from_url "
    "download_url_to_file urlopen urlretrieve".split()
)


def screen(source: str) -> list[str]:
    """Reasons not to run a program, in line order; empty when it may run."""
    if len(source) > MAX_SOURCE_CHARACTERS:
        return [f"the program is longer than {MAX_SOURCE_CHARACTERS} characters"]
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return [f"line {error.lineno}: syntax error: {error.msg}"]
    problems = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            if isinstance(node, ast.ImportFrom) and node.level:
                problems.add((node.lineno, "relative imports are not allowed"))
                continue
            modules = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module]
            )
            for module in modules:
                if module.split(".")[0] not in ALLOWED_IMPORTS and module not in ALLOWED_MODULES:
                    problems.add((node.lineno, f"import of {module!r} is not allowed"))
            for alias in node.names:
                if alias.name in DENIED_ATTRIBUTES:
                    problems.add((node.lineno, f"import of {alias.name!r} is not allowed"))
        elif isinstance(node, ast.Name) and node.id in DENIED_NAMES:
            problems.add((node.lineno, f"{node.id!r} is not allowed"))
        elif isinstance(node, ast.Attribute) and node.attr in DENIED_ATTRIBUTES:
            problems.add((node.lineno, f"attribute {node.attr!r} is not allowed"))
    return [f"line {line}: {reason}" for line, reason in sorted(problems)]


def build_model() -> nn.Module:
    """The statement's network, without weights: the checker loads the program's."""
    from torchvision.models import resnet18

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def call(
    program: Path,
    data: Path,
    out: Path,
    *,
    seed: int,
    epochs: int,
    stop_after: int | None = None,
    resume: bool = False,
    device: str = "cuda",
    timeout: float = TIMEOUT,
) -> dict[str, Any]:
    """Call the program once, as the statement says it is called."""
    arguments = ["--seed", str(seed), "--epochs", str(epochs), "--device", device]
    if stop_after is not None:
        arguments += ["--stop-after", str(stop_after)]
    if resume:
        arguments.append("--resume")
    command = [
        sys.executable,
        str(Path(program).resolve()),
        "--data",
        str(Path(data).resolve()),
        "--out",
        str(Path(out).resolve()),
        *arguments,
    ]
    record: dict[str, Any] = {"arguments": arguments, "status": "crashed", "returncode": None}
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=clean_environment(),
            cwd=out,
        )
    except subprocess.TimeoutExpired:
        record.update(status="timeout", output=f"no result within {timeout:g} seconds")
    else:
        record.update(
            status="completed" if completed.returncode == 0 else "crashed",
            returncode=completed.returncode,
            output=(completed.stdout + completed.stderr)[-OUTPUT_TAIL:],
        )
    record["seconds"] = round(time.monotonic() - started, 1)
    return record


def run_trajectory(
    program: Path, data: Path, out: Path, *, stop_after: int | None = None, **options
) -> dict[str, Any]:
    """The calls of one trajectory into an empty directory: one call, or with stop_after
    a call that stops and one that resumes, and what the stop left behind."""
    out.mkdir(parents=True)
    if stop_after is None:
        calls = [call(program, data, out, **options)]
        return {"calls": calls, "status": calls[0]["status"]}
    calls = [call(program, data, out, stop_after=stop_after, **options)]
    left = sorted(path.name for path in out.iterdir())
    if calls[0]["status"] == "completed":
        calls.append(call(program, data, out, resume=True, **options))
    return {"calls": calls, "status": calls[-1]["status"], "files_after_stop": left}


def load_state(path: Path, model: nn.Module) -> dict[str, torch.Tensor]:
    """A state_dict the program wrote, read without running anything it pickled."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("not a state_dict")
    model.load_state_dict(state)
    return state


def evaluate(
    model: nn.Module,
    state: dict[str, torch.Tensor],
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    """The correct validation samples and the mean loss of a state, as the statement
    asks a program to measure them: evaluation mode, batches of 32 in the order given."""
    model.load_state_dict(state)
    model.to(device).eval()
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    loss_function = nn.CrossEntropyLoss(reduction="sum")
    loss, correct = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(labels), BATCH_SIZE):
            batch = images[start : start + BATCH_SIZE].to(device).float() / 255
            targets = labels[start : start + BATCH_SIZE].to(device)
            logits = model((batch - mean) / std)
            loss += float(loss_function(logits, targets))
            correct += int((logits.argmax(dim=1) == targets).sum())
    return {"correct": correct, "loss": loss / len(labels)}


def _difference(first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]) -> float | None:
    """The largest absolute difference between two states; None when their shapes differ."""
    if first.keys() != second.keys():
        return None
    largest = 0.0
    for name, tensor in first.items():
        if tensor.shape != second[name].shape:
            return None
        if tensor.numel():
            gap = (tensor.double() - second[name].double()).abs().max()
            largest = max(largest, float(gap))
    return largest


def _log(out: Path) -> tuple[list[dict[str, Any]], list[str]]:
    path, entries, problems = out / standalone_task.LOG, [], []
    if not path.is_file():
        return [], [f"{standalone_task.LOG} is missing"]
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            problems.append(f"{standalone_task.LOG} line {number} is not JSON")
            continue
        missing = (
            [f for f in standalone_task.LOG_FIELDS if f not in entry]
            if isinstance(entry, dict)
            else list(standalone_task.LOG_FIELDS)
        )
        if missing:
            problems.append(f"{standalone_task.LOG} line {number} lacks {', '.join(missing)}")
        else:
            entries.append({field: entry[field] for field in standalone_task.LOG_FIELDS})
    return entries, problems


def read_run(
    out: Path,
    epochs: int,
    model: nn.Module,
    measure: Callable[[dict[str, torch.Tensor]], dict[str, Any]],
) -> dict[str, Any]:
    """What a finished trajectory left behind: its weights, measured, and its log.

    problems lists what the statement asks for and the checker could not use.
    last_state holds the weights after the last epoch for the comparisons and is not
    part of the record.
    """
    log, problems = _log(out)
    names = [standalone_task.EPOCH_WEIGHTS.format(epoch=n) for n in range(1, epochs + 1)]
    measured: dict[str, dict[str, Any]] = {}
    last_state = None
    for name in (*names, standalone_task.DELIVERED_WEIGHTS):
        if not (out / name).is_file():
            problems.append(f"{name} is missing")
            continue
        try:
            state = load_state(out / name, model)
        except Exception as error:  # whatever a program may have written instead
            problems.append(f"{name} is not the network's state_dict: {type(error).__name__}")
            continue
        measured[name] = {"sha256": state_dict_sha256(state), **measure(state)}
        if name == names[-1]:
            last_state = state
    # A resumed run that logs an epoch twice says something about its restart; the log
    # is then no ranking to compare with, and delivery is decided without it.
    logged = [entry["epoch"] for entry in log]
    log_order = (
        None
        if logged == list(range(1, epochs + 1))
        else f"the log's epochs are {logged}, not 1 to {epochs}"
    )
    return {
        "problems": problems,
        "log": log,
        "log_order": log_order,
        "epochs": [measured.get(name) for name in names],
        "delivered": measured.get(standalone_task.DELIVERED_WEIGHTS),
        "last_state": last_state,
    }


def _earliest_best(values: list[float]) -> int:
    """The epoch, from 1, of the earliest highest value."""
    return values.index(max(values)) + 1


def delivery_verdict(run: dict[str, Any], stop_after: int | None = None) -> dict[str, Any]:
    """Whether a run delivered its earliest best epoch.

    status is holds; violated, when an epoch with fewer correct samples was delivered;
    violated_tie, a later epoch with as many; no_epoch, weights that are no epoch's;
    undecided, when the delivered epoch is the best by the program's own log, which
    ranks another epoch first than the checker does.
    With stop_after, history says where the best epoch lies and whether the stricter
    event occurred: the best before the stop beats every later epoch, and the best of
    the later epochs was delivered.
    """
    correct = [epoch["correct"] for epoch in run["epochs"]]
    hashes = [epoch["sha256"] for epoch in run["epochs"]]
    best = _earliest_best(correct)
    delivered = [n for n, sha in enumerate(hashes, start=1) if sha == run["delivered"]["sha256"]]
    verdict: dict[str, Any] = {
        "correct": correct,
        "best_epoch": best,
        "delivered_epochs": delivered,
        "delivered_correct": run["delivered"]["correct"],
        "best_is_last": best == len(correct),
    }
    reported = None
    if run["log_order"] is None:
        accuracies = [entry["validation_accuracy"] for entry in run["log"]]
        if all(isinstance(value, int | float) for value in accuracies):
            reported = _earliest_best(accuracies)
    verdict["reported_best_epoch"] = reported
    if not delivered:
        status = "no_epoch"
    elif best in delivered:
        status = "holds"
    elif reported is not None and reported != best and reported in delivered:
        status = "undecided"
    else:
        status = "violated" if correct[delivered[0] - 1] < correct[best - 1] else "violated_tie"
    verdict["status"] = status
    if stop_after is not None:
        before, after = correct[:stop_after], correct[stop_after:]
        later_best = stop_after + _earliest_best(after)
        verdict["history"] = {
            "best_before_stop": best <= stop_after,
            "strictly": max(before) > max(after),
            "later_best_epoch": later_best,
            "later_best_delivered": max(before) > max(after) and later_best in delivered,
        }
    return verdict


def _first_difference(first: dict[str, Any], second: dict[str, Any]) -> int | None:
    """The first epoch whose weights differ between two runs."""
    pairs = zip(first["epochs"], second["epochs"], strict=True)
    return next((n for n, (a, b) in enumerate(pairs, start=1) if a["sha256"] != b["sha256"]), None)


def _learning_rates_agree(first: dict[str, Any], second: dict[str, Any]) -> bool:
    """Whether two logs give the same learning rates, up to the last digits of a float."""
    rates = [[entry["learning_rate"] for entry in run["log"]] for run in (first, second)]
    if len(rates[0]) != len(rates[1]):
        return False
    if not all(isinstance(rate, int | float) for rate in rates[0] + rates[1]):
        return rates[0] == rates[1]
    return all(
        math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-15)
        for a, b in zip(rates[0], rates[1], strict=True)
    )


def repeat_verdict(straight: dict[str, Any], repeat: dict[str, Any]) -> dict[str, Any]:
    """Whether the same run made twice wrote the same weights after its last epoch."""
    same = straight["epochs"][-1]["sha256"] == repeat["epochs"][-1]["sha256"]
    return {
        "status": "holds" if same else "violated",
        "first_difference": _first_difference(straight, repeat),
        "largest_difference": _difference(straight["last_state"], repeat["last_state"]),
        "learning_rates_agree": _learning_rates_agree(straight, repeat),
    }


def restart_verdict(
    straight: dict[str, Any], resumed: dict[str, Any], repeat: dict[str, Any]
) -> dict[str, Any]:
    """Whether a resumed run trained as the uninterrupted one.

    witness names what shows a violation. log_epochs: the uninterrupted run logged
    every epoch once and the resumed run did not. learning_rates: the two logs give
    other learning rates, while the two uninterrupted runs gave the same. weights: the
    weights after the last epoch differ, for a program that repeats itself. Without a
    witness the status is undecided for a program that does not repeat itself, and
    holds otherwise.
    """
    rates_agree = _learning_rates_agree(straight, resumed)
    verdict: dict[str, Any] = {
        "first_difference": _first_difference(straight, resumed),
        "largest_difference": _difference(straight["last_state"], resumed["last_state"]),
        "learning_rates_agree": rates_agree,
        "witness": None,
    }
    same = straight["epochs"][-1]["sha256"] == resumed["epochs"][-1]["sha256"]
    if straight["log_order"] is None and resumed["log_order"] is not None:
        verdict.update(status="violated", witness="log_epochs")
    elif repeat["learning_rates_agree"] and not rates_agree:
        verdict.update(status="violated", witness="learning_rates")
    elif repeat["status"] != "holds":
        verdict["status"] = "undecided"
    elif not same:
        verdict.update(status="violated", witness="weights")
    else:
        verdict["status"] = "holds"
    return verdict


def _overall(statuses: list[str]) -> str | None:
    """One status for several stops: any violation, else any undecided, else holds."""
    if not statuses:
        return None
    if any(status in VIOLATIONS for status in statuses):
        return "violated"
    return "undecided" if "undecided" in statuses else "holds"


def check_program(
    program: Path,
    data: Path,
    work: Path,
    *,
    seed: int = SEED,
    epochs: int = EPOCHS,
    stops: Sequence[int] | None = None,
    device: str = "cuda",
    timeout: float = TIMEOUT,
    model_factory: Callable[[], nn.Module] = build_model,
    keep_weights: bool = False,
) -> dict[str, Any]:
    """Run a program's trajectories in work and decide repeat, restart and delivery.

    stops are the epochs the program is stopped after, every epoch but the last when
    not given. work must not exist: the checker makes it and, unless keep_weights,
    removes it afterwards, having measured and hashed every state the program wrote.
    The trajectories end at the first that fails or leaves files the checker cannot
    use: after an uninterrupted run there is nothing to compare with, and a program
    that cannot resume after one stop is not asked to after another.
    """
    if Path(work).exists():
        raise FileExistsError(f"{work} exists; remove it or choose another work directory")
    stops = list(range(1, epochs)) if stops is None else list(stops)
    source = Path(program).read_text(encoding="utf-8")
    record: dict[str, Any] = {
        "schema_version": 2,
        "task": "standalone_check",
        "program_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "seed": seed,
        "epochs": epochs,
        "stops": stops,
        "device": device,
        "torch": torch.__version__,
        "screen": screen(source),
        "trajectories": {},
        "repeat": None,
        "restart": {"status": None, "stops": {}},
        "delivery": {"straight": None, "resumed": {}},
    }
    started = time.monotonic()
    if record["screen"]:
        return {**record, "category": "refused", "seconds": 0.0}
    arrays = {name: np.load(Path(data) / f"{name}.npy") for name in standalone_task.ARRAYS[2:]}
    images = torch.from_numpy(arrays["validation_images"])
    labels = torch.from_numpy(arrays["validation_labels"])
    record["validation_samples"] = len(labels)
    target, model, measured = torch.device(device), model_factory(), {}

    def measure(state: dict[str, torch.Tensor]) -> dict[str, Any]:
        sha = state_dict_sha256(state)
        if sha not in measured:
            measured[sha] = evaluate(model, state, images, labels, target)
        return measured[sha]

    runs: dict[str, dict[str, Any]] = {}
    plan = [("straight", None), ("repeat", None), *((f"resumed-{k}", k) for k in stops)]
    options = {"seed": seed, "epochs": epochs, "device": device, "timeout": timeout}
    try:
        for name, stop in plan:
            trajectory = run_trajectory(
                program, data, Path(work) / name, stop_after=stop, **options
            )
            if trajectory["status"] == "completed":
                runs[name] = read_run(Path(work) / name, epochs, model, measure)
                trajectory.update(
                    {key: runs[name][key] for key in ("problems", "log", "log_order")},
                    epochs=runs[name]["epochs"],
                    delivered=runs[name]["delivered"],
                )
            record["trajectories"][name] = trajectory
            if name not in runs or runs[name]["problems"]:
                break
        usable = {name: run for name, run in runs.items() if not run["problems"]}
        if "straight" in usable:
            record["delivery"]["straight"] = delivery_verdict(usable["straight"])
        if "repeat" in usable:
            record["repeat"] = repeat_verdict(usable["straight"], usable["repeat"])
            for stop in stops:
                run = usable.get(f"resumed-{stop}")
                if run is not None:
                    record["restart"]["stops"][str(stop)] = restart_verdict(
                        usable["straight"], run, record["repeat"]
                    )
                    record["delivery"]["resumed"][str(stop)] = delivery_verdict(run, stop)
    finally:
        if not keep_weights:
            shutil.rmtree(work, ignore_errors=True)
    record["restart"]["status"] = _overall(
        [verdict["status"] for verdict in record["restart"]["stops"].values()]
    )
    failed = [
        name for name, body in record["trajectories"].items() if body["status"] != "completed"
    ]
    if any(not name.startswith("resumed") for name in failed):
        category = "runtime_failure"
    elif any(run["problems"] for run in runs.values()):
        category = "unsupported_interface"
    elif failed:
        category = "resume_failure"
    else:
        category = "decided"
    return {**record, "category": category, "seconds": round(time.monotonic() - started, 1)}


def check_attempts(
    programs: Path,
    data: Path,
    output: Path,
    work: Path,
    *,
    progress: Callable[[str], None] = print,
    **options,
) -> list[dict[str, Any]]:
    """Check every written program under programs that output has no check of yet."""
    seed, written = options.get("seed", SEED), []
    for attempt in standalone_programs.read_attempts(programs):
        if attempt["status"] != "written":
            continue
        directory = Path(output) / attempt["id"] / f"seed-{seed}"
        if (directory / CHECK_FILE).is_file():
            continue
        program = (
            standalone_programs.attempt_directory(
                programs, attempt["model_name"], attempt["dataset"], attempt["attempt"]
            )
            / standalone_task.PROGRAM
        )
        record = {
            "id": attempt["id"],
            "dataset": attempt["dataset"],
            "model_name": attempt["model_name"],
            "attempt": attempt["attempt"],
            "statement_version": attempt.get("statement_version", 1),
            "generation": attempt.get("generation"),
            "claim": attempt["claim"],
            **check_program(
                program,
                Path(data) / attempt["dataset"],
                Path(work) / f"{attempt['id']}-{seed}",
                **options,
            ),
        }
        directory.mkdir(parents=True, exist_ok=True)
        (directory / CHECK_FILE).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        written.append(record)
        progress(line(record))
    return written


def read_checks(output: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in Path(output).glob(f"*/seed-*/{CHECK_FILE}")
    ]
    return sorted(records, key=lambda r: (r["model_name"], r["dataset"], r["attempt"], r["seed"]))


def findings(record: dict[str, Any]) -> list[str]:
    """What a check found against a program, in the statement's terms; empty when
    nothing was found on its runs."""
    found = []
    if record["category"] in ("refused", "runtime_failure", "unsupported_interface"):
        return [record["category"]]
    if record["category"] == "resume_failure":
        found.append("resume_failure")
    if record["repeat"] and record["repeat"]["status"] == "violated":
        found.append("repeat")
    witnesses = sorted(
        {v["witness"] for v in record["restart"]["stops"].values() if v["status"] == "violated"}
    )
    if witnesses:
        found.append(f"restart ({', '.join(witnesses)})")
    delivered = [record["delivery"]["straight"], *record["delivery"]["resumed"].values()]
    if any(verdict and verdict["status"] in VIOLATIONS for verdict in delivered):
        found.append("delivery")
    return found


def line(record: dict[str, Any]) -> str:
    """One program on one seed: its category, the verdicts it has, and its claim."""
    name = f"{record['model_name']} {record['dataset']} {record['attempt']} seed {record['seed']}"
    if record["category"] in ("refused", "runtime_failure", "unsupported_interface"):
        why = record["screen"][:1] or [
            f"{trajectory}: {' '.join(c['status'] for c in body['calls'])}"
            + (f"; {body['problems'][0]}" if body.get("problems") else "")
            for trajectory, body in record["trajectories"].items()
            if body["status"] != "completed" or body.get("problems")
        ]
        return f"{name}: {record['category']} ({'; '.join(why)}); claims {record['claim']}"

    def stops(verdicts: dict[str, dict[str, Any]]) -> str:
        return " ".join(f"{stop}:{verdict['status']}" for stop, verdict in verdicts.items()) or "-"

    failed = [
        f"{trajectory}: {' '.join(c['status'] for c in body['calls'])}"
        for trajectory, body in record["trajectories"].items()
        if body["status"] != "completed"
    ]
    straight = record["delivery"]["straight"]
    return (
        f"{name}: {record['category']}"
        + (f" ({'; '.join(failed)})" if failed else "")
        + f"; repeat {record['repeat']['status']}"
        + f"; restart {record['restart']['status']} [{stops(record['restart']['stops'])}]"
        + f"; delivery {straight['status']}, resumed [{stops(record['delivery']['resumed'])}]"
        + f"; correct {straight['correct']} of {record['validation_samples']}"
        + f"; found: {', '.join(findings(record)) or 'nothing'}"
        + f"; claims {record['claim']}; {record['seconds']:.0f} s"
    )


def report_text(records: list[dict[str, Any]]) -> str:
    """Every check, then what the checks say about whether the study can be made."""
    if not records:
        return "no checks"
    ran = [r for r in records if r["category"] in ("resume_failure", "decided")]
    categories = Counter(record["category"] for record in records)
    resumed = [v for r in ran for v in r["delivery"]["resumed"].values()]
    claimed = [r for r in records if r["claim"] == "meets"]
    seconds = sum(record["seconds"] for record in records)
    return "\n".join(
        [
            *(line(record) for record in records),
            "",
            f"{len(records)} checks: "
            + ", ".join(f"{categories[name]} {name}" for name in CATEGORIES if categories[name]),
            f"repeat: {dict(Counter(r['repeat']['status'] for r in ran))}",
            f"restart: {dict(Counter(str(r['restart']['status']) for r in ran))}",
            "delivery, uninterrupted: "
            f"{dict(Counter(r['delivery']['straight']['status'] for r in ran))}",
            f"delivery, resumed runs: {dict(Counter(v['status'] for v in resumed))}",
            "best is not the last epoch (uninterrupted): "
            f"{sum(not r['delivery']['straight']['best_is_last'] for r in ran)} of {len(ran)}",
            "resumed runs whose best epoch before the stop beats every later one: "
            f"{sum(v['history']['strictly'] for v in resumed)} of {len(resumed)}, "
            f"of which the later best was delivered in "
            f"{sum(v['history']['later_best_delivered'] for v in resumed)}",
            f"claims to meet every requirement: {len(claimed)} of {len(records)}, "
            f"of which something was found in {sum(bool(findings(r)) for r in claimed)}",
            f"{seconds / 60:.0f} minutes in all, {seconds / len(records):.0f} s a check",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Checks of training programs from outside.")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Check every written program not yet checked.")
    run.add_argument("--programs", type=Path, required=True, help="standalone_programs' output.")
    run.add_argument("--data", type=Path, required=True, help="The directory of --data folders.")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--work", type=Path, required=True, help="Where the programs write.")
    run.add_argument("--seed", type=int, default=SEED)
    run.add_argument("--epochs", type=int, default=EPOCHS)
    run.add_argument(
        "--stops", type=int, nargs="+", help="Epochs to stop after; all but the last if unset."
    )
    run.add_argument("--device", default="cuda")
    run.add_argument("--timeout", type=float, default=TIMEOUT, help="Seconds per call.")
    run.add_argument("--keep-weights", action="store_true", help="Keep what the programs wrote.")
    report = commands.add_parser("report", help="Print the checks in an output directory.")
    report.add_argument("out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        check_attempts(
            args.programs,
            args.data,
            args.out,
            args.work,
            seed=args.seed,
            epochs=args.epochs,
            stops=args.stops,
            device=args.device,
            timeout=args.timeout,
            keep_weights=args.keep_weights,
        )
        print()
    print(report_text(read_checks(args.out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
