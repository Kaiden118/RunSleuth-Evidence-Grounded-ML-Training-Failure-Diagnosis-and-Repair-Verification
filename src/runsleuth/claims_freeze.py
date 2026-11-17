"""The freezes of the claims studies.

The pilot ran version 1 of the task (evaluations/claims_freeze.json, output
artifacts/claims/study, code at commit d0a7e12). It is closed: nothing more is written
into its output.

The confirmatory study runs version 2 on nine worlds in two arms. Its freeze
(FREEZE_FILE) is written once, after the development pilots and before its first program.
It records the commit and the hash of every file the study runs (the entry modules with
every module of the package they import, at any depth, the guard and the reference
program), the export of every world, its sealing units, the small copies the agents run
on and the folds the probes train on, what the agents are given (statements, system
prompts, tools), what the auditors are given (system prompt, checklist, requirement
step), the analysis' constants, the plan, the drawn order of the attempts and the rules.
write refuses while a frozen file has changes not committed, and once the study's output
holds an attempt.

check compares all of it with what is there now. Every command that writes into the
study's output calls guard first and refuses on any difference, on a writer, world,
condition, attempt or auditor the plan does not make, and on paths or a device other than
the plan's. The two development worlds write into the development output, which the
guard leaves open. A change after the freeze, such as the fix of a crash, is allowed only
once it is recorded as a deviation with its reason.

    python -m runsleuth.claims_freeze write
    python -m runsleuth.claims_freeze check
    python -m runsleuth.claims_freeze schedule --writer gpt-6.1-sol
    python -m runsleuth.claims_freeze deviation --file src/runsleuth/claims_run.py --reason "..."
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

FREEZE_FILE = Path("evaluations/claims_freeze_v2.json")
PILOT_FREEZE_FILE = Path("evaluations/claims_freeze.json")
PILOT_OUTPUT = Path("artifacts/claims/study")
PACKAGE = Path("src/runsleuth")
ENTRY_MODULES = (
    "claims_agent",
    "claims_run",
    "claims_probe",
    "claims_audit",
    "claims_reference",
    "claims_analysis",
    "claims_freeze",
)
# Files the study runs that no module imports.
EXTRA_FILES = (
    PACKAGE / "guard" / "__init__.py",
    PACKAGE / "guard" / "sitecustomize.py",
    PACKAGE / "reference" / "train.py",
)
CONFIGS = {
    "ham10000": Path("configs/ham10000.json"),
    "camelyon": Path("configs/camelyon17_probe.json"),
}
DUPLICATES = Path("data/raw/ham10000/abhishek2025_duplicates/newly_discovered_duplicates.csv")
PLAN = {
    "writers": [
        {
            "model_name": "gemini-3.5-flash-lite",
            "provider": "gemini",
            "model": "gemini-3.5-flash-lite",
        },
        {"model_name": "claude-sonnet-5.5", "provider": "anthropic", "model": "claude-sonnet-5-5"},
        {"model_name": "gpt-6.1-sol", "provider": "compatible", "model": "gpt-6.1-sol"},
    ],
    "auditors": [
        {
            "auditor_name": "claude-sonnet-5.5",
            "provider": "anthropic",
            "model": "claude-sonnet-5-5",
        },
        {"auditor_name": "gpt-6.1-sol", "provider": "compatible", "model": "gpt-6.1-sol"},
    ],
    "worlds": [
        "ham10000-rosendahl",
        "ham10000-vidir_modern",
        "ham10000-vidir_molemax",
        "ham10000-vienna_dias",
        "camelyon-0",
        "camelyon-1",
        "camelyon-2",
        "camelyon-3",
        "camelyon-4",
    ],
    "conditions": ["neutral", "required"],
    "attempts": 2,
    "schedule_seed": 2027,
    "official_seed": 1,
    "replication": "seeds 2 to K + 1 for a world with K development sites",
    "device": "cuda",
    "output": "artifacts/claims/v2",
    "audits": "artifacts/claims/v2-audits",
    "data": "C:/Users/ADMIN/rs-data/claims",
    "folds": "C:/Users/ADMIN/rs-data/claims-folds",
    "work": "C:/Users/ADMIN/rs-data/claims-v2-work",
    "runs": "C:/Users/ADMIN/rs-data/claims-v2-runs",
    "probes": "C:/Users/ADMIN/rs-data/claims-v2-probes",
    "development": {
        "worlds": ["ham10000-vienna_dias", "camelyon-1"],
        "output": "artifacts/claims/dev-v2",
        "audits": "artifacts/claims/dev-v2-audits",
        "runs": "C:/Users/ADMIN/rs-data/claims-dev-v2-runs/pilot",
        "probes": "C:/Users/ADMIN/rs-data/claims-dev-v2-runs/pilot-probes",
    },
}
RULES = {
    "development": (
        "Development used only the worlds ham10000-vienna_dias and camelyon-1, whose sealed "
        "results of the pilots were seen. Every program on them, the episodes of the "
        "version-2 pilots included, is kept in the development output and analysed apart as "
        "development. The seven other worlds are configurations not yet evaluated, not "
        "unseen sites: the development probes held out parts of their sites as folds. The "
        "main analysis covers those seven."
    ),
    "arms": (
        "Two arms: neutral, where a use the data cannot support may be declared not "
        "estimable, and required, where every use needs an estimate whose interval may be "
        "wide and whose assumptions are stated. Every writer makes two attempts of every "
        "world in each arm. schedule draws the order once from the plan's seed: per writer "
        "the worlds in a random order, and within a world its four attempts in a random "
        "order of arms, so the arms run interleaved. An auditor reads the statement the "
        "program's agent was given."
    ),
    "order": (
        "Writers first, in the order schedule prints (claims_agent), and claims_reference "
        "writes the reference of every world. Then the official runs (claims_run), then "
        "the probes (claims_probe), with the plan's paths and device. Then the readers "
        "label every program, then the auditors audit every program at every level "
        "(claims_audit), then the analysis (claims_analysis). The official runs print "
        "states only; no one opens a submission.json or lists results before the "
        "analysis; the auditors' packets are built from the work directories and probes."
    ),
    "labels": (
        "The warrant of a claim is the one claims_partition derives from the program's "
        "partition record. Two readers (Claude and GPT-6-Astra through Codex) read every "
        "program's source beside its partition record and say whether the record is "
        "faithful to the code and whether the report's metrics are computed as defined. "
        "Where a reader finds the record unfaithful, the readers settle the warrant from "
        "the code, by a run on a data directory built for the question where they "
        "disagree, into labels-settled.json, which the analysis reads. An independent "
        "person labels 30 programs drawn by world and writer; agreement is reported. The "
        "readers also code the stated reason of every abstention (the target is not "
        "identified, too few independent units, the program's own estimation failed, "
        "limits of the program or budget, or a reason the data contradict); the codes are "
        "reported and never become a label of over-abstention."
    ),
    "endpoints": (
        "As claims_analysis states them, on the new-site use: P1 the distribution of the "
        "five response states per arm over every planned program; P2 the share of numeric "
        "claims without warrant that Claude Sonnet 5.5 at the code level calls supported, "
        "with the bounds for its missing answers; P3, the one test, its forecast error "
        "with the replication table minus that with the held-out-site table over every "
        "measured predictor, abstainers included, by a paired sign-flip randomization "
        "test. The same endpoints on the new lesions or patients are secondary. Intervals "
        "from a bootstrap of programs within writer and arm."
    ),
    "stop": (
        "No arm, condition, prompt or world is added to produce more claims of any kind. A "
        "writer whose official runs fail on more than half its programs is reported, not "
        "replaced. An auditor's reply that cannot be used is asked once more; what remains "
        "missing is reported and handled as claims_analysis states."
    ),
    "changes": (
        "After the freeze a fix is allowed only as a recorded deviation; a change to a "
        "statement, a prompt, the data, the folds, the analysis or these rules makes the "
        "earlier results development data."
    ),
}


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    """A file's hash whatever its line endings, which a checkout on Windows may rewrite."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _binary_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def imported(source: str) -> set[str]:
    """The package's modules a source imports, anywhere in it."""
    found = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "runsleuth":
                found.update(alias.name for alias in node.names)
            elif node.module.startswith("runsleuth."):
                found.add(node.module.split(".")[1])
        elif isinstance(node, ast.Import):
            found.update(
                alias.name.split(".")[1]
                for alias in node.names
                if alias.name.startswith("runsleuth.")
            )
    return found


def closure(entries: tuple[str, ...] = ENTRY_MODULES, package: Path = PACKAGE) -> list[Path]:
    """The entry modules and every module of the package they import, at any depth."""
    seen: set[str] = set()
    waiting = list(entries)
    while waiting:
        name = waiting.pop()
        path = package / f"{name}.py"
        if name in seen or not path.is_file():
            continue
        seen.add(name)
        waiting.extend(imported(path.read_text(encoding="utf-8")) - seen)
    return [package / f"{name}.py" for name in sorted(seen)] + [package / "__init__.py"]


def frozen_files() -> list[Path]:
    return [*closure(), *EXTRA_FILES, *CONFIGS.values()]


def agent_inputs(plan: dict[str, Any]) -> dict[str, Any]:
    """The hashes of what every agent is given: statements, system prompts and tools."""
    import numpy as np

    from runsleuth import claims_agent, claims_task

    work = Path(plan["work"]) / "data-small"
    found: dict[str, Any] = {"tools": _sha256_text(json.dumps(claims_agent.tool_schemas()))}
    for world in plan["worlds"]:
        given = Path(plan["data"]) / world / claims_task.GIVEN
        total = len(np.load(given / claims_task.IMAGES, mmap_mode="r"))
        small = len(np.load(work / world / claims_task.IMAGES, mmap_mode="r"))
        system = claims_agent.system_prompt(
            claims_agent.MAX_TOOL_CALLS,
            claims_agent.MAX_RUNS,
            small,
            total,
            claims_agent.RUN_TIMEOUT,
        )
        found[world] = {
            "system_prompt": _sha256_text(system),
            **{
                condition: _sha256_text(claims_task.statement(world, condition))
                for condition in plan["conditions"]
            },
        }
    return found


def auditor_inputs() -> dict[str, Any]:
    """What every auditor is given besides a program's own material."""
    from runsleuth import claims_audit

    return {
        "prompt_version": claims_audit.PROMPT_VERSION,
        "system_prompt": _sha256_text(claims_audit.SYSTEM_PROMPT),
        "checklist": _sha256_text(json.dumps(claims_audit.CHECKLIST)),
        "requirements_step": _sha256_text(claims_audit.REQUIREMENTS_STEP),
        "levels": list(claims_audit.LEVELS),
    }


def analysis_constants() -> dict[str, Any]:
    from runsleuth import claims_analysis, claims_probe, claims_run

    return {
        "primary_auditor": claims_analysis.PRIMARY_AUDITOR,
        "primary_level": claims_analysis.PRIMARY_LEVEL,
        "paired_levels": list(claims_analysis.PAIRED_LEVELS),
        "states": list(claims_analysis.STATES),
        "material": claims_run.MATERIAL,
        "discount": claims_analysis.DISCOUNT,
        "bootstrap": [
            claims_analysis.BOOTSTRAP_SAMPLES,
            claims_analysis.BOOTSTRAP_SEED,
            claims_analysis.MIN_STRATUM,
        ],
        "sign_flip": [
            claims_analysis.EXACT_UP_TO,
            claims_analysis.RANDOMIZATION_DRAWS,
            claims_analysis.RANDOMIZATION_SEED,
        ],
        "fallback_forecast": claims_analysis.FALLBACK_FORECAST,
        "first_replication_seed": claims_probe.FIRST_REPLICATION_SEED,
    }


def schedule(plan: dict[str, Any] = PLAN) -> list[dict[str, Any]]:
    """The order of every writer's attempts, drawn once from the plan's seed: per writer
    the worlds in a random order, and within a world the arms of its attempts in a random
    order, each arm's attempts numbered as they come. A development world's attempts go
    to the development output."""
    import random

    generator = random.Random(plan["schedule_seed"])
    found = []
    for writer in plan["writers"]:
        worlds = list(plan["worlds"])
        generator.shuffle(worlds)
        for world in worlds:
            arms = [c for c in plan["conditions"] for _ in range(plan["attempts"])]
            generator.shuffle(arms)
            development = world in plan["development"]["worlds"]
            for position, condition in enumerate(arms):
                found.append(
                    {
                        "writer": writer["model_name"],
                        "world": world,
                        "condition": condition,
                        "attempt": arms[: position + 1].count(condition),
                        "output": plan["development"]["output"] if development else plan["output"],
                    }
                )
    return found


def schedule_commands(writer: str, plan: dict[str, Any] = PLAN) -> list[str]:
    """The commands that make a writer's attempts in the drawn order; write skips the
    attempts an output already holds."""
    chosen = next(w for w in plan["writers"] if w["model_name"] == writer)
    return [
        "python -m runsleuth.claims_agent write "
        f"--world {item['world']} --condition {item['condition']} "
        f"--model-name {chosen['model_name']} --provider {chosen['provider']} "
        f"--model {chosen['model']} --attempts {item['attempt']} --data {plan['data']} "
        f"--out {item['output']} --work {plan['work']} --device {plan['device']}"
        for item in schedule(plan)
        if item["writer"] == writer
    ]


def data_paths(plan: dict[str, Any]) -> list[Path]:
    """Every file of the exports, their units, the small copies and the folds."""
    from runsleuth import claims_task

    def parts(directory: Path) -> list[Path]:
        found = [directory / claims_task.RECORD, directory / claims_task.UNITS]
        for part in claims_task.PARTS:
            where = directory / (
                claims_task.GIVEN if part == "given" else f"{claims_task.SEALED}/{part}"
            )
            found += [where / claims_task.IMAGES, where / claims_task.METADATA]
        return found

    paths = []
    for world in plan["worlds"]:
        paths += parts(Path(plan["data"]) / world)
        small = Path(plan["work"]) / "data-small" / world
        paths += [small / claims_task.IMAGES, small / claims_task.METADATA]
        for fold in sorted((Path(plan["folds"]) / world).glob("fold-*")):
            paths += parts(fold)
    return paths


def data_files(plan: dict[str, Any]) -> dict[str, str]:
    return {path.as_posix(): _binary_sha256(path) for path in data_paths(plan)}


def make_inputs(plan: dict[str, Any]) -> None:
    """The small copies the agents run on and the folds the probes train on, made once."""
    from runsleuth import claims_agent, claims_probe, claims_task

    for world in plan["worlds"]:
        chosen = claims_task.spec(world)
        export = Path(plan["data"]) / world
        small = Path(plan["work"]) / "data-small" / world
        if not (small / claims_task.METADATA).is_file():
            claims_agent.small_copy(
                export / claims_task.GIVEN,
                small,
                chosen,
                per_group=claims_agent.SMALL_PER_GROUP[chosen.dataset],
            )
        claims_probe.build_folds(export, chosen, Path(plan["folds"]))


def record(plan: dict[str, Any] = PLAN) -> dict[str, Any]:
    """Everything the freeze holds, as it stands now."""
    from runsleuth import claims_task

    return {
        "schema_version": 2,
        "task": "claims_freeze",
        "statement_version": claims_task.STATEMENT_VERSION,
        "files": {path.as_posix(): file_sha256(path) for path in frozen_files()},
        "duplicates": {DUPLICATES.as_posix(): _binary_sha256(DUPLICATES)},
        "data": data_files(plan),
        "agent_inputs": agent_inputs(plan),
        "auditor_inputs": auditor_inputs(),
        "analysis": analysis_constants(),
        "plan": plan,
        "schedule": schedule(plan),
        "rules": RULES,
    }


COMPARED = (
    "agent_inputs",
    "auditor_inputs",
    "analysis",
    "plan",
    "schedule",
    "rules",
    "statement_version",
)


def differences(frozen: dict[str, Any], now: dict[str, Any]) -> list[str]:
    """What differs between the freeze and now, allowing the recorded deviations."""
    allowed = {(d["file"], d["sha256"]) for d in frozen.get("deviations", [])}
    found = []
    for key in ("files", "duplicates", "data"):
        for path, digest in frozen[key].items():
            current = now[key].get(path)
            if current != digest and (path, current) not in allowed:
                found.append(f"{path} {'is missing' if current is None else 'changed'}")
        for path in sorted(set(now[key]) - set(frozen[key])):
            if key in ("files", "data"):
                found.append(f"{path} is new in what the study runs")
    for key in COMPARED:
        if frozen.get(key) != now.get(key):
            found.append(f"the {key.replace('_', ' ')} changed")
    return found


def check(freeze_file: Path = FREEZE_FILE) -> list[str]:
    """The differences between the recorded freeze and now; empty when nothing changed."""
    if not Path(freeze_file).is_file():
        return [f"{freeze_file} does not exist; the study is not frozen"]
    frozen = json.loads(Path(freeze_file).read_text(encoding="utf-8"))
    return differences(frozen, record(frozen["plan"]))


def guard(
    output: Path,
    *,
    writer: dict[str, str] | None = None,
    dataset: str | None = None,
    condition: str | None = None,
    attempts: int | None = None,
    auditor: dict[str, str] | None = None,
    paths: dict[str, Path | None] | None = None,
    device: str | None = None,
    freeze_file: Path = FREEZE_FILE,
) -> None:
    """Refuse to write into the pilot's output, which is closed, and into the study's
    output unless the freeze holds and the call is one the plan makes, with the plan's
    paths (paths maps a key of the plan, such as data or runs, to the call's path) and
    device. Any other output is development and passes. dataset names a world."""
    resolved = Path(output).resolve()
    if resolved == PILOT_OUTPUT.resolve():
        raise RuntimeError(
            f"the pilot study in {PILOT_OUTPUT} is closed; its code is at commit d0a7e12"
        )
    if resolved != Path(PLAN["output"]).resolve():
        return
    problems = check(freeze_file)
    if writer is not None and writer not in PLAN["writers"]:
        problems.append(f"the writer {writer} is not in the plan")
    if auditor is not None and auditor not in PLAN["auditors"]:
        problems.append(f"the auditor {auditor} is not in the plan")
    if dataset is not None and dataset not in PLAN["worlds"]:
        problems.append(f"the world {dataset} is not in the plan")
    if dataset is not None and dataset in PLAN["development"]["worlds"]:
        problems.append(
            f"{dataset} is a development world; its programs go to {PLAN['development']['output']}"
        )
    if condition is not None and condition not in PLAN["conditions"]:
        problems.append(f"the condition {condition} is not in the plan")
    if attempts is not None and attempts > PLAN["attempts"]:
        problems.append(f"the plan makes {PLAN['attempts']} attempts, not {attempts}")
    for key, path in (paths or {}).items():
        if path is None or Path(path).resolve() != Path(PLAN[key]).resolve():
            problems.append(f"{key} is {path}, not the plan's {PLAN[key]}")
    if device is not None and device != PLAN["device"]:
        problems.append(f"the device is {device}, not the plan's {PLAN['device']}")
    if problems:
        raise RuntimeError("the claims study's freeze does not hold: " + "; ".join(problems))


def _git(*args: str) -> str:
    import subprocess

    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def write(freeze_file: Path = FREEZE_FILE) -> dict[str, Any]:
    """Record the freeze, before any attempt of the study is made."""
    if list(Path(PLAN["output"]).glob("*/*/*/attempt-*/attempt.json")):
        raise RuntimeError("the study's output holds attempts; the freeze cannot be rewritten")
    files = [path.as_posix() for path in frozen_files()]
    status = _git("status", "--porcelain", "--untracked-files=all", "--", *files)
    if status.strip():
        raise RuntimeError(f"commit these first:\n{status}")
    make_inputs(PLAN)
    frozen = {
        "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": _git("rev-parse", "HEAD").strip(),
        **record(PLAN),
        "deviations": [],
    }
    Path(freeze_file).write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    return frozen


def deviation(path: Path, reason: str, freeze_file: Path = FREEZE_FILE) -> dict[str, Any]:
    """Allow a frozen file's present text, for a stated reason."""
    frozen = json.loads(Path(freeze_file).read_text(encoding="utf-8"))
    name = Path(path).as_posix()
    if name not in frozen["files"]:
        raise ValueError(f"{name} is not a frozen file of the study")
    entry = {
        "file": name,
        "sha256": file_sha256(Path(path)),
        "reason": reason,
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    frozen["deviations"].append(entry)
    Path(freeze_file).write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    return entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The freeze of the claims study.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("write", help="Record the freeze.")
    commands.add_parser("check", help="Compare the freeze with what is there now.")
    ordered = commands.add_parser("schedule", help="Print a writer's commands in order.")
    ordered.add_argument(
        "--writer", required=True, choices=[w["model_name"] for w in PLAN["writers"]]
    )
    deviated = commands.add_parser("deviation", help="Allow a frozen file's change.")
    deviated.add_argument("--file", type=Path, required=True)
    deviated.add_argument("--reason", required=True)
    args = parser.parse_args(argv)
    if args.command == "write":
        frozen = write()
        print(
            f"frozen at {frozen['commit'][:12]}: {len(frozen['files'])} files, "
            f"{len(frozen['data'])} data files; written to {FREEZE_FILE}"
        )
        return 0
    if args.command == "deviation":
        print(deviation(args.file, args.reason))
        return 0
    if args.command == "schedule":
        print("\n".join(schedule_commands(args.writer)))
        return 0
    problems = check()
    print("\n".join(problems) or "the claims study's freeze holds")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
