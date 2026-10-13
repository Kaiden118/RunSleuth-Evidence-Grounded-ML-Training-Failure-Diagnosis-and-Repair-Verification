"""The agent study: every task under every condition, on training seeds drawn once.

    python -m runsleuth.agent_study draw --count 5
    python -m runsleuth.agent_study prepare
    python -m runsleuth.agent_study run --name flash-lite --provider gemini
    python -m runsleuth.agent_study summarize --name flash-lite

draw picks training seeds that nothing has used and records them with the commit and
with everything a run must not change afterwards: the task statement and scripts, the
conditions and what they show, the agent's prompt, tools and limits, the budgets, the
contract tests, the signature library, the repair gate and the data configuration.
Commit the draw before training, so the history shows that it came first.

prepare trains the healthy script and every task's script for every seed; no model is
asked. run puts one model through the study, into a directory of its own, and can be
repeated to resume. Both refuse to start when something frozen has changed since the
draw. All models share one store of training runs, so a script is trained once per
seed whichever model wrote it.

The pilots (agent_episodes pilot) were development; nothing of them is a result.
"""

import argparse
import hashlib
import json
import secrets
import subprocess
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from runsleuth import agent_episodes, agent_loop, agent_review, agent_tasks
from runsleuth.agent_episodes import CONDITIONS, FULL, PRESENTATION_VERSION, QUICK, RunStore
from runsleuth.agent_tasks import TASKS

DRAW_PATH = Path("evaluations/agent_study_draw.json")
STUDY_DIRECTORY = Path("artifacts/agent_study")
CONFIG_PATH = Path("configs/camelyon17_clean.json")
EVALUATIONS = Path("evaluations")
MAX_TOOL_CALLS, MAX_RUNS = 15, 3
SEED_LIMIT = 2**31
# Seeds of development runs that no seeds file records.
DEVELOPMENT_SEEDS = (7, 11, 42, 2026)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], capture_output=True, text=True, check=True
    ).stdout.strip()


def used_seeds(directory: Path = EVALUATIONS) -> list[int]:
    """Every training seed a development run or an earlier draw has used."""
    seeds = set(DEVELOPMENT_SEEDS)
    for path in sorted(Path(directory).glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(record, dict):
            for key in ("seeds", "excluded_seeds"):
                seeds.update(seed for seed in record.get(key) or [] if isinstance(seed, int))
    return sorted(seeds)


def frozen_components(config: Path = CONFIG_PATH) -> dict[str, Any]:
    """What decides the study's results; none of it may change after the draw."""
    from runsleuth import agent_contract, agent_verification
    from runsleuth.camelyon_optimizer_training import REPAIR_POLICY
    from runsleuth.camelyon_variant_runner import REPAIR_GATE
    from runsleuth.signature_matching import LIBRARY_PATH

    scripts = {task: _sha256(agent_tasks.task_source(task)) for task in TASKS}
    prompts = [
        agent_loop.system_prompt(allow_runs, MAX_TOOL_CALLS, MAX_RUNS)
        for allow_runs in (False, True)
    ]
    return {
        "task_statement_sha256": _sha256(agent_tasks.STATEMENT),
        "tasks": list(TASKS),
        "task_scripts_sha256": _sha256(json.dumps(scripts, sort_keys=True)),
        "conditions": {name: asdict(condition) for name, condition in CONDITIONS.items()},
        "presentation_version": PRESENTATION_VERSION,
        "agent": {
            "prompt_version": agent_loop.PROMPT_VERSION,
            "system_prompts_sha256": _sha256("\n".join(prompts)),
            "tools_sha256": _sha256(json.dumps(agent_loop.tool_schemas(True), sort_keys=True)),
            "max_tool_calls": MAX_TOOL_CALLS,
            "max_runs": MAX_RUNS,
        },
        "budgets": {"quick": asdict(QUICK), "full": asdict(FULL)},
        "contract": {
            "version": agent_contract.CONTRACT_VERSION,
            "source_sha256": _file_sha256(agent_contract.__file__),
        },
        "failure_signatures_sha256": _file_sha256(LIBRARY_PATH),
        "verification_source_sha256": _file_sha256(agent_verification.__file__),
        "repair_gate": dict(REPAIR_GATE),
        "repair_policy": dict(REPAIR_POLICY),
        "data": {"source": "camelyon", "config_sha256": _file_sha256(config)},
    }


def draw_seeds(
    count: int,
    path: Path = DRAW_PATH,
    *,
    config: Path = CONFIG_PATH,
    evaluations: Path = EVALUATIONS,
    randbelow: Callable[[int], int] = secrets.randbelow,
    git: Callable[..., str] = _git,
) -> dict[str, Any]:
    """Draw the study's training seeds once and record what is frozen with them."""
    if isinstance(count, bool) or not 2 <= count <= 10:
        raise ValueError("Draw two to ten seeds")
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already holds a draw; a study has one draw")
    if git("status", "--porcelain", "--", "src", "configs"):
        raise RuntimeError("Commit the source changes first, so the draw records the exact code")
    excluded = used_seeds(evaluations)
    seeds: list[int] = []
    while len(seeds) < count:
        seed = randbelow(SEED_LIMIT)
        if seed not in excluded and seed not in seeds:
            seeds.append(seed)
    record = {
        "schema_version": 1,
        "task": "agent_study_draw",
        "drawn_at": datetime.now(UTC).isoformat(),
        "method": f"secrets.randbelow({SEED_LIMIT}), skipping seeds already used",
        "excluded_seeds": excluded,
        "seeds": seeds,
        "code_commit": git("rev-parse", "HEAD"),
        "frozen": frozen_components(config),
        "episodes_per_model": len(TASKS) * len(CONDITIONS) * count,
        "scope": "training seeds never used in development; one episode per task, "
        "condition and seed for each model",
    }
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    return record


def read_draw(path: Path = DRAW_PATH, *, config: Path = CONFIG_PATH) -> dict[str, Any]:
    """The draw, after checking that nothing frozen has changed since."""
    draw = json.loads(Path(path).read_text(encoding="utf-8"))
    current = frozen_components(config)
    changed = sorted(name for name in draw["frozen"] if draw["frozen"][name] != current.get(name))
    changed += sorted(name for name in current if name not in draw["frozen"])
    if changed:
        raise RuntimeError(
            f"Frozen since the draw of {draw['drawn_at'][:10]} and changed: {', '.join(changed)}"
        )
    return draw


def study_store(
    directory: Path = STUDY_DIRECTORY,
    *,
    data: str = "camelyon",
    config: Path | None = CONFIG_PATH,
    device: str = "auto",
    runner: Callable[..., dict[str, Any]] | None = None,
) -> RunStore:
    """The store of training runs that every model of the study shares."""
    # The store's own seed is never trained on; every run is of a drawn seed.
    return RunStore(
        Path(directory) / "runs", data=data, config=config, seed=0, device=device, runner=runner
    )


def prepare_study(
    draw: dict[str, Any],
    store: RunStore,
    *,
    progress: Callable[[str], None] = lambda _line: None,
    **budgets: Any,
) -> list[dict[str, Any]]:
    """Train the runs every episode needs, for every seed of the draw."""
    rows = []
    for seed in draw["seeds"]:
        progress(f"seed {seed}")
        rows += [
            {**row, "seed": seed}
            for row in agent_episodes.prepare_runs(
                store.for_seed(seed), tuple(TASKS), progress=progress, **budgets
            )
        ]
    return rows


def run_study(
    name: str,
    client: Any,
    model: str,
    draw: dict[str, Any],
    store: RunStore,
    *,
    directory: Path = STUDY_DIRECTORY,
    **options: Any,
) -> dict[str, Any]:
    """Put one model through the study: every task, condition and seed, once."""
    return agent_episodes.run_pilot(
        Path(directory) / name,
        client,
        model,
        store,
        tasks=tuple(TASKS),
        conditions=tuple(CONDITIONS),
        seeds=tuple(draw["seeds"]),
        max_tool_calls=MAX_TOOL_CALLS,
        max_runs=MAX_RUNS,
        **options,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    draw = commands.add_parser("draw", help="Draw the training seeds and record what is frozen.")
    draw.add_argument("--count", type=int, default=5)
    prepare = commands.add_parser("prepare", help="Train the runs every episode needs.")
    run = commands.add_parser("run", help="Put one model through the study.")
    run.add_argument("--provider", default="gemini", help="gemini, ollama, openai or compatible.")
    run.add_argument("--model", help="Defaults to the provider's environment variable.")
    run.add_argument("--pause", type=float, default=4.0, help="Seconds before a model call.")
    run.add_argument("--error-wait", type=float, default=60.0, help="Seconds after an error.")
    summarize = commands.add_parser("summarize", help="Summarize one model's episodes.")
    for command in (run, summarize):
        command.add_argument("--name", required=True, help="The model's directory in the study.")
    for command in (prepare, run):
        command.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    for command in (draw, prepare, run):
        command.add_argument("--draw", type=Path, default=DRAW_PATH, help="The draw's file.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    def show(line: str) -> None:
        print(line, flush=True)

    if args.command == "draw":
        record = draw_seeds(args.count, args.draw)
        show(f"seeds={record['seeds']} commit={record['code_commit']}")
        show(f"{record['episodes_per_model']} episodes per model; commit {args.draw} now")
        return 0
    reviews = (
        agent_review.load_reviews(agent_review.REVIEWS_PATH)
        if agent_review.REVIEWS_PATH.is_file()
        else {}
    )
    if args.command == "summarize":
        episodes = agent_episodes.load_episodes(STUDY_DIRECTORY / args.name)
        show(agent_episodes.summary_text(agent_episodes.summarize(episodes, reviews)))
        return 0
    draw, store = read_draw(args.draw), study_store(device=args.device)
    if args.command == "prepare":
        rows = prepare_study(draw, store, progress=show)
        return 0 if all(row["status"] == "completed" for row in rows) else 1
    from runsleuth.llm_client import create_llm_client

    client, model = create_llm_client(args.provider, args.model)
    summary = run_study(
        args.name,
        client,
        model,
        draw,
        store,
        provider=args.provider,
        reviews=reviews,
        progress=show,
        pause=args.pause,
        error_wait=args.error_wait,
        max_api_errors=5,
    )
    show(agent_episodes.summary_text(summary))
    if summary["provider_failures"]:
        show("Stopped at a provider failure. The same command resumes the study.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
