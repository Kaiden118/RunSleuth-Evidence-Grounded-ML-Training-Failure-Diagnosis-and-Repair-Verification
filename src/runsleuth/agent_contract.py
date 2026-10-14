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

Each task family (agent_tasks) has its statement, so its clauses differ where the
statements do. The ham10000 family's first clause is training_sources, and it has one
more, held_out_lesions: held_out_key gives the images of one lesion one key and the
images of different lesions different keys.

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
from runsleuth.agent_tasks import FAULTS, HEALTHY
from runsleuth.agent_workspace import (
    ScriptRefused,
    check_script,
    clean_environment,
    load_script,
    load_source,
    synthetic_task_data,
)

CONTRACT_VERSION = 2
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
# The clauses that need the model trained; a script that cannot be trained fails them.
TRAINED_CLAUSES = CLAUSES[2:]
HAM_CLAUSES = ("training_sources", "held_out_lesions", *CLAUSES[1:])


@dataclass(frozen=True, slots=True)
class Spec:
    """Where a family's statement asks for something other than the camelyon one."""

    clauses: tuple[str, ...]
    groups: range
    training_groups: list[int]
    group_name: str
    patch_size: int
    # Whether the harness's synthetic training set numbers groups as this family does.
    harness_check: bool


SPECS = {
    "camelyon": Spec(CLAUSES, HOSPITALS, TRAINING_HOSPITALS, "hospitals", 96, True),
    "ham10000": Spec(HAM_CLAUSES, range(4), [0, 3], "sources", 128, False),
}
# Images and lesions as HAM10000 names them: two images of one lesion, and other lesions.
SAME_LESION = (("ISIC_0024306", "HAM_0000118"), ("ISIC_0024307", "HAM_0000118"))
OTHER_LESIONS = (("ISIC_0024308", "HAM_0002730"), ("ISIC_0024309", "HAM_0001466"))
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
LEARNING_RATE, WEIGHT_DECAY = 1e-4, 1e-2
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


def _training_groups(module: types.ModuleType, spec: Spec) -> dict[str, Any]:
    admitted = [group for group in spec.groups if module.use_for_training(group)]
    if admitted != spec.training_groups:
        return _failed(f"trains on {spec.group_name} {admitted}, not on {spec.training_groups}")
    if spec.harness_check:
        # The harness builds its training set from this function alone; look at it too.
        used = synthetic_task_data(module.use_for_training, torch.device("cpu"), 0).record
        if used["training_hospitals"] != spec.training_groups:
            return _failed(
                f"the harness's training set holds hospitals {used['training_hospitals']}"
            )
    return _passed()


def _held_out_lesions(module: types.ModuleType) -> dict[str, Any]:
    keys = {pair: module.held_out_key(*pair) for pair in (*SAME_LESION, *OTHER_LESIONS)}
    first, second = (keys[pair] for pair in SAME_LESION)
    if first != second:
        return _failed("two images of one lesion get different keys")
    if len(set(keys.values())) != 1 + len(OTHER_LESIONS):
        return _failed("images of different lesions share a key")
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
    module: types.ModuleType,
    reference: types.ModuleType,
    *,
    patch_size: int | None = None,
    family: str = "camelyon",
) -> dict[str, Any]:
    """Check a loaded script against every clause; reference is the healthy script."""
    spec = SPECS[family]
    patch_size = patch_size or spec.patch_size
    clauses: dict[str, dict[str, Any]] = {}

    def check(name: str, test, *arguments) -> None:
        if name in clauses and not clauses[name]["passed"]:
            return  # Keep the first failure of a clause.
        try:
            clauses[name] = test(*arguments)
        except Exception as error:
            clauses[name] = _failed(f"the check failed with {type(error).__name__}: {error}"[:300])

    check(spec.clauses[0], _training_groups, module, spec)
    if "held_out_lesions" in spec.clauses:
        check("held_out_lesions", _held_out_lesions, module)
    check("input_normalization", _input_normalization, module, patch_size)
    for epochs, steps in RUNS:
        try:
            seen = observe(module, epochs, steps, patch_size)
            expected = build(reference, epochs, steps)[3]
        except Exception as error:
            failure = _failed(f"training failed with {type(error).__name__}: {error}"[:300])
            for name in TRAINED_CLAUSES:
                if name not in clauses or clauses[name]["passed"]:
                    clauses[name] = failure
            break
        check("model", _model, seen, expected)
        check("all_parameters_trained", _all_parameters_trained, seen)
        check("training_mode", _training_mode, seen)
        check("optimizer", _optimizer, seen)
        check("learning_rate_schedule", _learning_rate_schedule, seen)
    failed = [name for name in spec.clauses if not clauses[name]["passed"]]
    return {
        "schema_version": 1,
        "task": "agent_contract",
        "contract_version": CONTRACT_VERSION,
        "family": family,
        "status": "failed" if failed else "passed",
        "failed": failed,
        "clauses": {name: clauses[name] for name in spec.clauses},
    }


def _refused(problems: list[str], family: str = "camelyon") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": "agent_contract",
        "contract_version": CONTRACT_VERSION,
        "family": family,
        "status": "refused",
        "failed": list(SPECS[family].clauses),
        "clauses": {},
        "problems": problems,
    }


def run_contract(
    script: Path,
    reference: Path | None = None,
    *,
    patch_size: int | None = None,
    family: str = "camelyon",
) -> dict[str, Any]:
    """Check a script file in this process; a script the harness refuses fails."""
    try:
        module = load_script(script)
    except ScriptRefused as refusal:
        return _refused(refusal.problems, family)
    healthy = (
        load_source(agent_tasks.task_source(HEALTHY, family))
        if reference is None
        else load_script(reference)
    )
    return contract_result(module, healthy, patch_size=patch_size, family=family)


def run_contract_in_subprocess(
    script: Path,
    reference: Path | None = None,
    *,
    patch_size: int | None = None,
    family: str = "camelyon",
    timeout: float = 900.0,
) -> dict[str, Any]:
    """Check a script in a child process without credentials; status may be crashed or timeout."""
    problems = check_script(Path(script).read_text(encoding="utf-8"))
    if problems:
        return _refused(problems, family)
    command = [sys.executable, "-m", "runsleuth.agent_contract", "run"]
    command += ["--script", str(Path(script).resolve()), "--family", family]
    if patch_size is not None:
        command += ["--patch-size", str(patch_size)]
    if reference is not None:
        command += ["--reference", str(Path(reference).resolve())]
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
    "schedule_period_in_steps_stepped_per_epoch": "learning_rate_schedule",
    "learning_rate_halved": "optimizer",
    "same_wrong_normalization_twice": "input_normalization",
    "head_bias_left_frozen": "all_parameters_trained",
    "head_frozen_from_the_second_epoch": "all_parameters_trained",
    "weight_decay_dropped": "optimizer",
    "training_hospital_left_out": "training_hospitals",
}
_HEAD = "    model.fc = nn.Linear(model.fc.in_features, 2)\n"
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
    "head_moved_to_the_device_twice": Case(
        (),
        "The new head is moved to the device, then the whole model is.",
        ((_HEAD, _HEAD.replace("2)", "2).to(device)")),),
    ),
    "training_mode_set_for_every_batch": Case(
        (),
        "model.train() is called inside the loop instead of before it.",
        ((_TRAIN, _LOOP + "        model.train()\n"),),
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
    "linear_decay": Case(
        ("learning_rate_schedule",),
        "The rate falls to zero along a line instead of a cosine. Over two epochs the two "
        "are the same schedule, which is why the checks also train for three.",
        (
            (
                "torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)",
                "torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1 - epoch / epochs)",
            ),
        ),
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


_SOURCE_LINE = "source not in (VALIDATION_SOURCE, TEST_SOURCE)"
_GENERAL_CASES = (
    "head_moved_to_the_device_twice",
    "training_mode_set_for_every_batch",
    "training_mode_set_only_when_built",
    "training_mode_lost_after_the_first_epoch",
    "sgd_instead_of_adamw",
    "linear_decay",
)
HAM_FAULT_CLAUSES = {
    **{
        name: clause
        for name, clause in FAULT_CLAUSES.items()
        if name in agent_tasks.family("ham10000").all_faults
    },
    "validation_source_in_training": "training_sources",
    "held_out_by_image": "held_out_lesions",
    "training_source_left_out": "training_sources",
}
HAM_CASES = {
    **{name: CASES[name] for name in _GENERAL_CASES},
    "sources_listed_one_by_one": Case(
        (),
        "The training sources are listed instead of the held-out ones.",
        ((_SOURCE_LINE, "source in (0, 3)"),),
    ),
    "lesion_key_with_a_prefix": Case(
        (),
        "The key is built from the lesion, with a prefix of its own.",
        (("    return lesion_id\n", '    return "lesion " + lesion_id\n'),),
    ),
    "one_key_for_every_lesion": Case(
        ("held_out_lesions",),
        "Every image shares one key, so every image is held out with the validation set.",
        (("    return lesion_id\n", '    return "held out"\n'),),
    ),
    "two_leaks_at_once": Case(
        ("training_sources", "held_out_lesions"),
        "The validation source is trained on and images are held out one by one.",
        (
            *agent_tasks.family("ham10000").faults["validation_source_in_training"].edits,
            *agent_tasks.family("ham10000").faults["held_out_by_image"].edits,
        ),
    ),
}
FAULT_CLAUSES_BY_FAMILY = {"camelyon": FAULT_CLAUSES, "ham10000": HAM_FAULT_CLAUSES}
CASES_BY_FAMILY = {"camelyon": CASES, "ham10000": HAM_CASES}


def expected_failures(name: str, family: str = "camelyon") -> tuple[str, ...]:
    """The clauses a task or case should fail; none for a correct script."""
    if name in CASES_BY_FAMILY[family]:
        return CASES_BY_FAMILY[family][name].fails
    faulty = agent_tasks.is_faulty(name, family)
    return (FAULT_CLAUSES_BY_FAMILY[family][name],) if faulty else ()


def case_source(name: str, family: str = "camelyon") -> str:
    """The script of a task or of a case."""
    if name in agent_tasks.family(family).tasks:
        return agent_tasks.task_source(name, family)
    source = agent_tasks.healthy_source(family)
    for old, new in CASES_BY_FAMILY[family][name].edits:
        if source.count(old) != 1:
            raise ValueError(f"{name}: every edit must match the script exactly once")
        source = source.replace(old, new)
    return source


def check_cases(
    *, patch_size: int | None = None, sources=None, family: str = "camelyon"
) -> list[dict[str, Any]]:
    """Run the contract on every task and case and compare with what each should fail."""
    sources = sources or (lambda name: case_source(name, family))
    tasks = agent_tasks.family(family).tasks
    reference = load_source(sources(HEALTHY))
    rows = []
    for name in (*tasks, *CASES_BY_FAMILY[family]):
        result = contract_result(
            load_source(sources(name)), reference, patch_size=patch_size, family=family
        )
        expected = list(expected_failures(name, family))
        rows.append(
            {
                "name": name,
                "kind": "task" if name in tasks else "case",
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
        command.add_argument("--patch-size", type=int, help="The family's size if unset.")
        command.add_argument("--family", choices=tuple(SPECS), default="camelyon")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        result = run_contract(
            args.script, args.reference, patch_size=args.patch_size, family=args.family
        )
        print(json.dumps(result))
        return 0
    rows = check_cases(patch_size=args.patch_size, family=args.family)
    print(cases_text(rows))
    return 0 if all(row["as_expected"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
