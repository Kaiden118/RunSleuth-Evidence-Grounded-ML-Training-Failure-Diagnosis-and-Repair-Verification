"""Contract tests: does a script do what the task statement says?

TASK.md (agent_tasks.STATEMENT) is the public specification an agent works from.
This module turns each of its requirements into a check that the agent never sees
and that runs only after the agent has finished:

    training_hospitals      1. which hospitals are trained on
    input_normalization     2. what training and evaluation do to the patches
    model                   3. a pretrained ResNet18 with a new two-output final layer
    all_parameters_trained  4. every parameter is trainable, optimized and changes
    training_mode           4. the network is in training mode whenever it is trained
    optimizer               5. AdamW, its learning rate and weight decay, a step per batch
    learning_rate_schedule  6. a cosine over the run, updated per epoch or per batch

The checks call the script's functions the way the harness does: they build the model
and train it for a few epochs of a few small batches on the CPU, putting the model in
evaluation mode between epochs as the harness's evaluation does. They require only
what the statement states, so a correct script may be written differently from the
healthy one.

Passing every check is this benchmark's working definition of a script that meets its
specification. It is not a proof that the script is correct: a check can be wrong, and
the statement does not cover everything a training script can get wrong. Reviews
(agent_review) audit the checks, and the trained model's metrics are recorded apart.

The checks share no code with the verifier. They use the harness only to load a
script under its guards and to build a training set, and never read run telemetry,
evidence or signatures.

    python -m runsleuth.agent_contract run --script workspace/train.py
    python -m runsleuth.agent_contract check
"""

import argparse
import json
import math
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from runsleuth import agent_tasks
from runsleuth.agent_tasks import FAULTS, HEALTHY, TASKS
from runsleuth.agent_workspace import (
    ScriptRefused,
    check_script,
    clean_environment,
    load_script,
    load_source,
    synthetic_task_data,
)

CONTRACT_VERSION = 1
CLAUSES = (
    "training_hospitals",
    "input_normalization",
    "model",
    "all_parameters_trained",
    "training_mode",
    "optimizer",
    "learning_rate_schedule",
)
HOSPITALS = range(5)
TRAINING_HOSPITALS = [0, 3, 4]
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
LEARNING_RATE, WEIGHT_DECAY = 1e-4, 1e-4
# Epochs and batches per epoch of the short runs. Two different shapes, so that a
# schedule written for one fixed length does not pass by coincidence.
RUNS = ((3, 4), (2, 3))
BATCH_SIZE = 4
PATCH_SIZE = 96
TOLERANCE = 1e-6


def _passed() -> dict[str, Any]:
    return {"passed": True, "detail": ""}


def _failed(detail: str) -> dict[str, Any]:
    return {"passed": False, "detail": detail}


def _some(names: list[str]) -> str:
    return ", ".join(names[:3]) + (f" and {len(names) - 3} more" if len(names) > 3 else "")


def _training_hospitals(module: types.ModuleType) -> dict[str, Any]:
    admitted = [hospital for hospital in HOSPITALS if module.use_for_training(hospital)]
    if admitted != TRAINING_HOSPITALS:
        return _failed(f"trains on hospitals {admitted}, not on {TRAINING_HOSPITALS}")
    # The harness builds its training set from this function alone; look at that set too.
    used = synthetic_task_data(module.use_for_training, torch.device("cpu"), 0).record
    if used["training_hospitals"] != TRAINING_HOSPITALS:
        return _failed(f"the harness's training set holds hospitals {used['training_hospitals']}")
    return _passed()


def _input_normalization(module: types.ModuleType, patch_size: int) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(0)
    images = torch.rand(BATCH_SIZE, 3, patch_size, patch_size, generator=generator)
    expected = (images - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)
    for name in ("prepare_training_batch", "prepare_evaluation_batch"):
        prepared = getattr(module, name)(images.clone())
        same_shape = tuple(prepared.shape) == tuple(expected.shape)
        if not same_shape or not torch.allclose(prepared, expected, atol=TOLERANCE):
            return _failed(f"{name} does not return (patches - mean) / std")
    return _passed()


def build(module: types.ModuleType, epochs: int, steps: int) -> tuple[Any, Any, Any, dict]:
    """Build the script's model as the harness does, and note what it is made of."""
    torch.manual_seed(0)
    model, optimizer, scheduler = module.build(torch.device("cpu"), epochs, steps)
    made = {
        "built": {name: value.clone() for name, value in model.state_dict().items()},
        "final_layer": list(model.named_children())[-1][0],
    }
    return model, optimizer, scheduler, made


def observe(module: types.ModuleType, epochs: int, steps: int, patch_size: int) -> dict[str, Any]:
    """Train the script's model as the harness does, recording what happens."""
    model, optimizer, scheduler, made = build(module, epochs, steps)
    seen: dict[str, Any] = {
        **made,
        "epochs": epochs,
        "steps": steps,
        "optimizer": type(optimizer),
        "learning_rates": [],  # one list per epoch, one entry per step: a rate per group
        "weight_decays": set(),
        "frozen": set(),
        "held": {},  # parameter name -> how often the optimizer holds it, where not once
        "outside_training_mode": {},  # module name -> first epoch
        "forward_passes": 0,
    }
    start = {name: value.detach().clone() for name, value in model.named_parameters()}

    def before_step(stepped, _arguments, _keywords) -> None:
        groups = stepped.param_groups
        seen["learning_rates"][-1].append([group["lr"] for group in groups])
        seen["weight_decays"].update(group.get("weight_decay") for group in groups)
        held = [id(parameter) for group in groups for parameter in group["params"]]
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                seen["frozen"].add(name)
            if held.count(id(parameter)) != 1:
                seen["held"][name] = held.count(id(parameter))

    def before_forward(_model, _inputs) -> None:
        seen["forward_passes"] += 1
        for name, part in model.named_modules():
            if not part.training:
                seen["outside_training_mode"].setdefault(name or "the network", seen["epoch"])

    handles = [
        optimizer.register_step_pre_hook(before_step),
        model.register_forward_pre_hook(before_forward),
    ]
    generator = torch.Generator().manual_seed(0)
    loss_function = nn.CrossEntropyLoss()
    try:
        for epoch in range(1, epochs + 1):
            seen["epoch"] = epoch
            model.eval()  # The harness evaluates before the first epoch and after each.
            batches = [
                (
                    torch.rand(BATCH_SIZE, 3, patch_size, patch_size, generator=generator),
                    torch.arange(BATCH_SIZE) % 2,
                )
                for _ in range(steps)
            ]
            seen["learning_rates"].append([])
            module.train_one_epoch(model, iter(batches), optimizer, scheduler, loss_function)
    finally:
        for handle in handles:
            handle.remove()
    changed = {
        name for name, value in model.named_parameters() if not torch.equal(value, start[name])
    }
    seen["changed"] = changed
    seen["parameters"] = list(start)
    return seen


def _model(seen: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    built, expected = seen["built"], reference["built"]
    shapes = {name: tuple(value.shape) for name, value in built.items()}
    expected_shapes = {name: tuple(value.shape) for name, value in expected.items()}
    if shapes != expected_shapes:
        different = sorted(
            name
            for name in shapes.keys() | expected_shapes.keys()
            if shapes.get(name) != expected_shapes.get(name)
        )
        name = different[0]
        return _failed(
            f"{name} has shape {shapes.get(name)}, not {expected_shapes.get(name)}"
            + (f"; {len(different) - 1} more entries differ" if len(different) > 1 else "")
        )
    final_layer = reference["final_layer"] + "."
    altered = [
        name
        for name, value in built.items()
        if not name.startswith(final_layer) and not torch.equal(value, expected[name])
    ]
    if altered:
        return _failed(f"the pretrained values of {_some(altered)} were not kept")
    return _passed()


def _all_parameters_trained(seen: dict[str, Any]) -> dict[str, Any]:
    if seen["frozen"]:
        return _failed(f"{_some(sorted(seen['frozen']))} cannot be trained (requires_grad is off)")
    if seen["held"]:
        name, count = sorted(seen["held"].items())[0]
        holds = f"holds {name} {count} times" if count else f"does not hold {name}"
        others = f" (and {len(seen['held']) - 1} more)" if len(seen["held"]) > 1 else ""
        return _failed(f"the optimizer {holds}{others}")
    final_layer = seen["final_layer"] + "."
    for part, names in (
        ("the final layer", [n for n in seen["parameters"] if n.startswith(final_layer)]),
        (
            "the rest of the network",
            [n for n in seen["parameters"] if not n.startswith(final_layer)],
        ),
    ):
        if names and not seen["changed"].intersection(names):
            return _failed(f"{part} did not change during training")
    return _passed()


def _training_mode(seen: dict[str, Any]) -> dict[str, Any]:
    if not seen["forward_passes"]:
        return _failed("no forward pass of the network was seen during training")
    if seen["outside_training_mode"]:
        name, epoch = sorted(seen["outside_training_mode"].items(), key=lambda item: item[1])[0]
        return _failed(f"{name} was in evaluation mode while training in epoch {epoch}")
    return _passed()


def _optimizer(seen: dict[str, Any]) -> dict[str, Any]:
    if not issubclass(seen["optimizer"], torch.optim.AdamW):
        return _failed(f"the optimizer is {seen['optimizer'].__name__}, not AdamW")
    counts = [len(epoch) for epoch in seen["learning_rates"]]
    if counts != [seen["steps"]] * seen["epochs"]:
        return _failed(
            f"optimizer steps per epoch were {counts}, with {seen['steps']} batches each"
        )
    first = seen["learning_rates"][0][0]
    if not all(math.isclose(rate, LEARNING_RATE, rel_tol=TOLERANCE) for rate in first):
        return _failed(f"the first step uses learning rate {_some([f'{r:g}' for r in first])}")
    if not all(
        isinstance(decay, float) and math.isclose(decay, WEIGHT_DECAY, rel_tol=TOLERANCE)
        for decay in seen["weight_decays"]
    ):
        decays = sorted(str(decay) for decay in seen["weight_decays"])
        return _failed(f"weight decay is {_some(decays)}, not {WEIGHT_DECAY:g}")
    return _passed()


def _learning_rate_schedule(seen: dict[str, Any]) -> dict[str, Any]:
    """The rates relative to the first one, against a cosine over the whole run."""
    epochs, steps = seen["epochs"], seen["steps"]
    rates = [
        (epoch, step) for epoch, of_epoch in enumerate(seen["learning_rates"]) for step in of_epoch
    ]
    if not rates or any(rate <= 0 for rate in rates[0][1]):
        return _failed("the first optimizer step has no positive learning rate")
    first = rates[0][1]

    def follows(expected) -> bool:
        return all(
            len(step) == len(first)
            and all(
                math.isclose(rate / start, expected(index, epoch), abs_tol=TOLERANCE)
                for rate, start in zip(step, first, strict=True)
            )
            for index, (epoch, step) in enumerate(rates)
        )

    def cosine(progress: float) -> float:
        return (1 + math.cos(math.pi * progress)) / 2

    per_epoch = follows(lambda _index, epoch: cosine(epoch / epochs))
    per_batch = follows(lambda index, _epoch: cosine(index / (epochs * steps)))
    if per_epoch or per_batch:
        return _passed()
    starts = [of_epoch[0][0] / first[0] for of_epoch in seen["learning_rates"] if of_epoch]
    return _failed(
        f"over {epochs} epochs of {steps} batches, the learning rate at the start of each "
        f"epoch was {', '.join(f'{rate:.4g}' for rate in starts)} of the first; a cosine over "
        f"the run gives {', '.join(f'{cosine(e / epochs):.4g}' for e in range(epochs))}"
    )


def contract_result(
    module: types.ModuleType, reference: types.ModuleType, *, patch_size: int = PATCH_SIZE
) -> dict[str, Any]:
    """Check a loaded script against every clause; reference is the healthy script."""
    clauses: dict[str, dict[str, Any]] = {}

    def check(name: str, test, *arguments) -> None:
        if name in clauses and not clauses[name]["passed"]:
            return  # Keep the first failure of a clause.
        try:
            clauses[name] = test(*arguments)
        except Exception as error:
            clauses[name] = _failed(f"the check failed with {type(error).__name__}: {error}"[:300])

    check("training_hospitals", _training_hospitals, module)
    check("input_normalization", _input_normalization, module, patch_size)
    for epochs, steps in RUNS:
        try:
            seen = observe(module, epochs, steps, patch_size)
            expected = build(reference, epochs, steps)[3]
        except Exception as error:
            failure = _failed(f"training failed with {type(error).__name__}: {error}"[:300])
            for name in CLAUSES[2:]:
                if name not in clauses or clauses[name]["passed"]:
                    clauses[name] = failure
            break
        check("model", _model, seen, expected)
        check("all_parameters_trained", _all_parameters_trained, seen)
        check("training_mode", _training_mode, seen)
        check("optimizer", _optimizer, seen)
        check("learning_rate_schedule", _learning_rate_schedule, seen)
    failed = [name for name in CLAUSES if not clauses[name]["passed"]]
    return {
        "schema_version": 1,
        "task": "agent_contract",
        "contract_version": CONTRACT_VERSION,
        "status": "failed" if failed else "passed",
        "failed": failed,
        "clauses": {name: clauses[name] for name in CLAUSES},
    }


def _refused(problems: list[str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": "agent_contract",
        "contract_version": CONTRACT_VERSION,
        "status": "refused",
        "failed": list(CLAUSES),
        "clauses": {},
        "problems": problems,
    }


def run_contract(
    script: Path, reference: Path | None = None, *, patch_size: int = PATCH_SIZE
) -> dict[str, Any]:
    """Check a script file in this process; a script the harness refuses fails."""
    try:
        module = load_script(script)
    except ScriptRefused as refusal:
        return _refused(refusal.problems)
    healthy = (
        load_source(agent_tasks.task_source(HEALTHY))
        if reference is None
        else load_script(reference)
    )
    return contract_result(module, healthy, patch_size=patch_size)


def run_contract_in_subprocess(
    script: Path,
    reference: Path | None = None,
    *,
    patch_size: int = PATCH_SIZE,
    timeout: float = 900.0,
) -> dict[str, Any]:
    """Check a script in a child process without credentials; status may be crashed or timeout."""
    problems = check_script(Path(script).read_text(encoding="utf-8"))
    if problems:
        return _refused(problems)
    command = [sys.executable, "-m", "runsleuth.agent_contract", "run"]
    command += ["--script", str(Path(script).resolve()), "--patch-size", str(patch_size)]
    if reference is not None:
        command += ["--reference", str(Path(reference).resolve())]
    unfinished = {**_refused([]), "status": "crashed"}
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
    """A script, as edits of the healthy one, and the clauses it should fail."""

    fails: tuple[str, ...]
    summary: str
    edits: tuple[tuple[str, str], ...]


FAULT_CLAUSES = {
    "stale_head": "all_parameters_trained",
    "frozen_head": "all_parameters_trained",
    "high_learning_rate": "optimizer",
    "train_eval_normalization_mismatch": "input_normalization",
    "imagenet_head_kept": "model",
    "train_in_eval_mode": "training_mode",
    "scheduler_stepped_per_batch": "learning_rate_schedule",
    "validation_hospital_in_training": "training_hospitals",
    "schedule_ends_an_epoch_early": "learning_rate_schedule",
}
_HEAD = "    model.fc = nn.Linear(model.fc.in_features, 2)\n"
_ADAMW = "torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)"
_COSINE = "torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)"
_LAMBDA = "torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: "
_TRAIN = "    model.train()\n    for images, labels in batches:\n"
_LOOP = "    for images, labels in batches:\n"
# Scripts beyond the tasks themselves, to see that the checks neither accept only the
# healthy script's wording nor reject only the faults they were written against.
CASES = {
    "hospitals_compared_one_by_one": Case(
        (),
        "The two held-out hospitals are excluded by two comparisons.",
        (
            (
                "hospital not in (VALIDATION_HOSPITAL, TEST_HOSPITAL)",
                "hospital != VALIDATION_HOSPITAL and hospital != TEST_HOSPITAL",
            ),
        ),
    ),
    "normalization_written_out": Case(
        (),
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
    "head_moved_to_the_device_twice": Case(
        (),
        "The new head is moved to the device, then the whole model is.",
        ((_HEAD, _HEAD.replace("2)", "2).to(device)")),),
    ),
    "two_parameter_groups": Case(
        (),
        "The head and the rest are separate optimizer groups with the same settings.",
        (
            (
                _ADAMW,
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
    "cosine_written_as_a_formula": Case(
        (),
        "The per-epoch cosine is a LambdaLR instead of CosineAnnealingLR.",
        (
            ("import torch\n", "import math\n\nimport torch\n"),
            (_COSINE, _LAMBDA + "0.5 * (1 + math.cos(math.pi * epoch / epochs)))"),
        ),
    ),
    "training_mode_set_for_every_batch": Case(
        (),
        "model.train() is called inside the loop instead of before it.",
        ((_TRAIN, _LOOP + "        model.train()\n"),),
    ),
    "schedule_period_in_steps_stepped_per_epoch": Case(
        ("learning_rate_schedule",),
        "The period counts batches but the schedule is stepped per epoch, so the rate "
        "hardly falls. An agent wrote this in the first pilot.",
        (("T_max=epochs)", "T_max=epochs * steps_per_epoch)"),),
    ),
    "linear_decay": Case(
        ("learning_rate_schedule",),
        "The rate falls steadily to zero, but along a line instead of a cosine.",
        ((_COSINE, _LAMBDA + "1 - epoch / epochs)"),),
    ),
    "same_wrong_normalization_twice": Case(
        ("input_normalization",),
        "Training and evaluation agree with each other but not with the ImageNet statistics.",
        (
            ("MEAN = (0.485, 0.456, 0.406)", "MEAN = (0.5, 0.5, 0.5)"),
            ("STD = (0.229, 0.224, 0.225)", "STD = (0.5, 0.5, 0.5)"),
        ),
    ),
    "head_bias_left_frozen": Case(
        ("all_parameters_trained",),
        "A partial repair of a frozen head: its weight trains, its bias does not.",
        ((_HEAD, _HEAD + "    model.fc.bias.requires_grad_(False)\n"),),
    ),
    "head_frozen_from_the_second_epoch": Case(
        ("all_parameters_trained",),
        "The head trains in the first epoch only.",
        (
            (
                _TRAIN,
                "    model.train()\n"
                "    if scheduler.last_epoch >= 1:\n"
                "        model.fc.requires_grad_(False)\n" + _LOOP,
            ),
        ),
    ),
    "training_mode_set_only_when_built": Case(
        ("training_mode",),
        "model.train() is called once in build(), so the harness's evaluation undoes it.",
        (
            (_TRAIN, _LOOP),
            ("    model.to(device)\n", "    model.to(device)\n    model.train()\n"),
        ),
    ),
    "training_mode_lost_after_the_first_epoch": Case(
        ("training_mode",),
        "model.train() is called in the first epoch only.",
        ((_TRAIN, "    if scheduler.last_epoch == 0:\n        model.train()\n" + _LOOP),),
    ),
    "leak_repaired_and_a_training_hospital_dropped": Case(
        ("training_hospitals",),
        "Hospitals 1 and 2 are excluded, and so is hospital 0.",
        (("hospital not in (VALIDATION_HOSPITAL, TEST_HOSPITAL)", "hospital in (3, 4)"),),
    ),
    "learning_rate_halved": Case(
        ("optimizer",),
        "A repair that also changed the learning rate.",
        (("LEARNING_RATE = 1e-4", "LEARNING_RATE = 5e-5"),),
    ),
    "weight_decay_dropped": Case(
        ("optimizer",),
        "A repair that also removed weight decay.",
        (("weight_decay=WEIGHT_DECAY", "weight_decay=0.0"),),
    ),
    "sgd_instead_of_adamw": Case(
        ("optimizer",), "The optimizer is SGD.", (("torch.optim.AdamW", "torch.optim.SGD"),)
    ),
    "two_faults_at_once": Case(
        ("training_hospitals", "learning_rate_schedule"),
        "The validation hospital is trained on and the schedule ends an epoch early.",
        (
            *FAULTS["validation_hospital_in_training"].edits,
            *FAULTS["schedule_ends_an_epoch_early"].edits,
        ),
    ),
}


def expected_failures(name: str) -> tuple[str, ...]:
    """The clauses a task or case should fail; none for a correct script."""
    if name in CASES:
        return CASES[name].fails
    return (FAULT_CLAUSES[name],) if name in FAULTS else ()


def case_source(name: str) -> str:
    """The script of a task or of a case."""
    if name in TASKS:
        return agent_tasks.task_source(name)
    source = agent_tasks.healthy_source()
    for old, new in CASES[name].edits:
        if source.count(old) != 1:
            raise ValueError(f"{name}: every edit must match the script exactly once")
        source = source.replace(old, new)
    return source


def check_cases(*, patch_size: int = PATCH_SIZE, sources=case_source) -> list[dict[str, Any]]:
    """Run the contract on every task and case and compare with what each should fail."""
    reference = load_source(sources(HEALTHY))
    rows = []
    for name in (*TASKS, *CASES):
        result = contract_result(load_source(sources(name)), reference, patch_size=patch_size)
        expected = list(expected_failures(name))
        rows.append(
            {
                "name": name,
                "kind": "task" if name in TASKS else "case",
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
        lines.append(f"{row['name']:<46} {', '.join(row['failed']) or 'passes':<42} {verdict}")
        lines.extend(f"    {detail}" for detail in row["details"])
    agreed = sum(row["as_expected"] for row in rows)
    correct = sum(not row["expected"] for row in rows)
    lines.append(
        f"{agreed}/{len(rows)} scripts as expected "
        f"({correct} that should pass, {len(rows) - correct} that should fail)"
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Contract tests for agent scripts.")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Check one script; prints the result as JSON.")
    run.add_argument("--script", type=Path, required=True)
    run.add_argument("--reference", type=Path, help="The healthy script; the template if unset.")
    check = commands.add_parser("check", help="Check every task and case against expectations.")
    for command in (run, check):
        command.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        result = run_contract(args.script, args.reference, patch_size=args.patch_size)
        print(json.dumps(result))
        return 0
    rows = check_cases(patch_size=args.patch_size)
    print(cases_text(rows))
    return 0 if all(row["as_expected"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
