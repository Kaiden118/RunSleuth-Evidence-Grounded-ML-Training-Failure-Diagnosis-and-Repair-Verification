"""Episodes of the agent study: one task, one condition, one model, scored afterwards.

An episode gives an agent (agent_loop) a task workspace (agent_tasks) under one of
five conditions, each adding one thing to what the agent can use:

    C0  the script and the task statement; no runs
    C1  bounded training runs, with their metrics and errors
    C2  C1 and the raw mechanism evidence of each run, without any verdict
    C3  C2 and the matcher's diagnosis against a healthy reference run
    C4  C3 and the verifier's decision on each run

Afterwards the final script is trained on a fuller budget, and three things are
recorded apart: what the agent claimed; whether the script is semantically correct,
decided from the source alone (the same program as the healthy script, or an
untouched script that was correct or faulty to begin with; anything else is left for
review); and what each acceptance criterion decides (agent_verification). The
verifier never labels correctness.

Runs are cached by script, budget and seed, so the healthy reference and scripts an
agent left unchanged are trained once.

    python -m runsleuth.agent_episodes pilot --output artifacts/agent_pilot/pilot-1 \\
        --data camelyon --config configs/camelyon17_clean.json
    python -m runsleuth.agent_episodes summarize artifacts/agent_pilot/pilot-1
"""

import argparse
import ast
import hashlib
import json
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from runsleuth import agent_tasks
from runsleuth.agent_loop import run_agent
from runsleuth.agent_tasks import FAULTS, HEALTHY, SCRIPT_NAME, TASKS
from runsleuth.agent_verification import trajectory_text, verify_repair_run
from runsleuth.signature_matching import diagnose, extract_evidence, load_library


@dataclass(frozen=True)
class Condition:
    """What an agent may use beyond reading and editing the script."""

    runs: bool
    evidence: bool
    diagnosis: bool
    verification: bool


CONDITIONS = {
    "C0": Condition(runs=False, evidence=False, diagnosis=False, verification=False),
    "C1": Condition(runs=True, evidence=False, diagnosis=False, verification=False),
    "C2": Condition(runs=True, evidence=True, diagnosis=False, verification=False),
    "C3": Condition(runs=True, evidence=True, diagnosis=True, verification=False),
    "C4": Condition(runs=True, evidence=True, diagnosis=True, verification=True),
}


@dataclass(frozen=True)
class Budget:
    """How long a run trains and on how many validation patches it is evaluated."""

    epochs: int
    max_train_batches: int | None
    max_eval_samples: int | None


# Runs an agent asks for are short, with a small validation subset; the final script
# is scored on more training and on the full validation subsets.
QUICK = Budget(epochs=2, max_train_batches=100, max_eval_samples=1000)
FULL = Budget(epochs=2, max_train_batches=200, max_eval_samples=None)
PILOT_TASKS = (
    "stale_head",
    "validation_hospital_in_training",
    "schedule_ends_an_epoch_early",
    HEALTHY,
)
OUTPUT_TAIL = 1500


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class RunStore:
    """Train scripts through the harness, once per script, budget and seed."""

    def __init__(
        self,
        root: Path,
        *,
        data: str,
        config: Path | None = None,
        seed: int = 7,
        device: str = "auto",
        timeout: float = 3600.0,
        runner: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        if runner is None:
            from runsleuth.agent_workspace import run_in_subprocess as runner
        self.root, self.data, self.config = Path(root), data, config
        self.seed, self.device, self.timeout, self.runner = seed, device, timeout, runner
        self.trained = 0

    def run(self, source: str, budget: Budget) -> dict[str, Any]:
        """The outcome of training a script on a budget: status, problems, output, report."""
        settings = {
            "script": _sha256(source),
            "budget": asdict(budget),
            "seed": self.seed,
            "data": self.data,
            "config": None if self.config is None else _sha256(Path(self.config).read_text()),
        }
        directory = self.root / _sha256(json.dumps(settings, sort_keys=True))[:20]
        saved = directory / "outcome.json"
        if saved.is_file():
            return json.loads(saved.read_text(encoding="utf-8"))
        if directory.exists():
            shutil.rmtree(directory)  # An interrupted run left no outcome.
        workspace = directory / "workspace"
        workspace.mkdir(parents=True)
        (workspace / SCRIPT_NAME).write_text(source, encoding="utf-8", newline="\n")
        outcome = self.runner(
            workspace,
            directory / "run",
            data=self.data,
            config=self.config,
            epochs=budget.epochs,
            seed=self.seed,
            device=self.device,
            max_train_batches=budget.max_train_batches,
            max_eval_samples=budget.max_eval_samples,
            timeout=self.timeout,
        )
        self.trained += 1
        report = outcome.get("report")
        result = {
            "settings": settings,
            "status": outcome["status"],
            "problems": outcome.get("problems") or [],
            "error": outcome.get("error"),
            "output": (outcome.get("output") or "")[-OUTPUT_TAIL:],
            "report": None if report is None else json.loads(Path(report).read_text("utf-8")),
        }
        saved.write_text(json.dumps(result, allow_nan=False) + "\n", encoding="utf-8")
        return result


def _rounded(value: Any) -> Any:
    """Shorten floats so that a run's summary stays small in the agent's context."""
    if isinstance(value, float):
        return float(f"{value:.5g}")
    if isinstance(value, dict):
        return {key: _rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rounded(item) for item in value]
    return value


def present(
    outcome: dict[str, Any],
    condition: Condition,
    *,
    reference: Callable[[], dict | None],
    before: Callable[[], dict | None],
    library: dict,
) -> dict[str, Any]:
    """What an agent is shown of one training run under a condition.

    reference() and before() return the run reports of the healthy script and of the
    task's original script on the same budget; they are only called when needed.
    """
    shown: dict[str, Any] = {"status": outcome["status"]}
    report = outcome["report"]
    if outcome["problems"]:
        shown["problems"] = outcome["problems"]
    if report is None:
        if outcome["error"]:
            shown["error"] = outcome["error"]
        if outcome["output"]:
            shown["output"] = outcome["output"]
        return shown
    if report.get("error"):
        shown["error"] = report["error"]
        shown["traceback"] = (report["workspace"].get("traceback") or "")[-OUTPUT_TAIL:]
    shown["epochs"] = [
        {"epoch": row["epoch"], **row.get("metrics", {})}
        for row in report["parameter_group_epochs"]
    ]
    if condition.evidence:
        shown["evidence"] = {
            name: value
            for name, value in extract_evidence(report).items()
            if value is not None and not name.startswith("reference")
        }
    healthy = reference() if condition.diagnosis or condition.verification else None
    if condition.diagnosis:
        found = diagnose(extract_evidence(report, healthy), library)
        signature = next(
            (item for item in library["signatures"] if item["id"] == found["diagnosis"]), None
        )
        shown["diagnosis"] = {"diagnosis": found["diagnosis"]}
        if signature is not None:
            shown["diagnosis"].update(
                meaning=signature["description"], repair=signature["repair"]["description"]
            )
    if condition.verification:
        if report["status"] != "completed" or healthy is None:
            shown["verification"] = {
                "decision": "rejected",
                "reasons": ["the run or its healthy reference did not complete"],
            }
        else:
            verified = verify_repair_run(report, healthy, before())
            shown["verification"] = {
                "decision": verified["decision"],
                "reasons": verified["reasons"],
                "trajectory": trajectory_text(verified["trajectory"]),
            }
    return _rounded(shown)


def same_program(left: str, right: str) -> bool:
    """Whether two sources are the same program, ignoring comments and layout."""
    try:
        return ast.dump(ast.parse(left)) == ast.dump(ast.parse(right))
    except SyntaxError:
        return False


def semantic_label(task: str, initial: str, final: str, healthy: str) -> tuple[bool | None, str]:
    """Whether the final script is correct, from the source alone; None needs review."""
    if same_program(final, healthy):
        return True, "same_program_as_healthy_script"
    if same_program(final, initial):
        return task not in FAULTS, "unchanged_script"
    return None, "needs_review"


def run_episode(
    client: Any,
    model: str,
    task: str,
    condition: str,
    directory: Path,
    store: RunStore,
    *,
    quick: Budget = QUICK,
    full: Budget = FULL,
    **agent_options: Any,
) -> dict[str, Any]:
    """Run one agent on one task under one condition, score it, and save episode.json."""
    rules, library = CONDITIONS[condition], load_library()
    directory = Path(directory)
    workspace = agent_tasks.materialize(task, directory / "workspace")
    script = workspace / SCRIPT_NAME
    initial, healthy = agent_tasks.task_source(task), agent_tasks.task_source(HEALTHY)

    def train(_workspace: Path) -> dict[str, Any]:
        return present(
            store.run(script.read_text(encoding="utf-8"), quick),
            rules,
            reference=lambda: store.run(healthy, quick)["report"],
            before=lambda: store.run(initial, quick)["report"],
            library=library,
        )

    agent = run_agent(
        client, model, workspace, run_training=train if rules.runs else None, **agent_options
    )
    final = script.read_text(encoding="utf-8")
    correct, basis = semantic_label(task, initial, final, healthy)
    outcome, reference = store.run(final, full), store.run(healthy, full)["report"]
    report, verification = outcome["report"], None
    if report is not None and report["status"] == "completed" and reference is not None:
        verified = verify_repair_run(report, reference, store.run(initial, full)["report"])
        verification = {
            "decision": verified["decision"],
            "reasons": verified["reasons"],
            "re_diagnosis": verified["re_diagnosis"],
            "criteria": verified["criteria"],
            "trajectory": verified["trajectory"],
        }
    claim = agent["claim"]
    episode = {
        "schema_version": 1,
        "task": "agent_episode",
        "repair_task": task,
        "condition": condition,
        "model": model,
        "task_is_correct": task not in FAULTS,
        "expected_diagnosis": agent_tasks.expected_diagnosis(task),
        "budgets": {"quick": asdict(quick), "full": asdict(full)},
        "claimed_correct": None if claim is None else claim["training_is_correct"],
        "semantically_correct": correct,
        "semantic_basis": basis,
        "final_run_status": outcome["status"],
        "final_metrics": None if report is None else report.get("final_metrics"),
        "verification": verification,
        "final_script": final,
        "agent": agent,
    }
    (directory / "episode.json").write_text(
        json.dumps(episode, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return episode


def summarize(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts per condition. A false claim says a script trains correctly when it does not."""
    rows = {}
    for condition in CONDITIONS:
        group = [episode for episode in episodes if episode["condition"] == condition]
        if not group:
            continue
        verified = [episode["verification"] for episode in group if episode["verification"]]

        def count(test: Callable[[dict], bool], group: list = group) -> int:
            return sum(bool(test(episode)) for episode in group)

        rows[condition] = {
            "episodes": len(group),
            "finished": count(lambda e: e["agent"]["status"] == "finished"),
            "claimed_correct": count(lambda e: e["claimed_correct"] is True),
            "semantically_correct": count(lambda e: e["semantically_correct"] is True),
            "semantically_wrong": count(lambda e: e["semantically_correct"] is False),
            "needs_review": count(lambda e: e["semantically_correct"] is None),
            "false_claims": count(
                lambda e: e["claimed_correct"] is True and e["semantically_correct"] is False
            ),
            "false_alarms": count(
                lambda e: e["claimed_correct"] is False and e["semantically_correct"] is True
            ),
            "edited_a_correct_script": count(
                lambda e: e["task_is_correct"] and e["agent"]["script_changed"]
            ),
            "verifier_accepted": sum(item["decision"] == "accepted" for item in verified),
            "training_runs": sum(episode["agent"]["training_runs"] for episode in group),
            "input_tokens": sum(episode["agent"]["input_tokens"] for episode in group),
            "output_tokens": sum(episode["agent"]["output_tokens"] for episode in group),
        }
    return {
        "schema_version": 1,
        "task": "agent_episode_summary",
        "scope": "development pilot; injected faults, one seed, small samples",
        "episodes": len(episodes),
        "conditions": rows,
    }


def summary_text(summary: dict[str, Any]) -> str:
    columns = (
        ("episodes", "episodes"),
        ("finished", "finished"),
        ("claimed_correct", "claims ok"),
        ("semantically_correct", "correct"),
        ("semantically_wrong", "wrong"),
        ("needs_review", "review"),
        ("false_claims", "false claims"),
        ("false_alarms", "false alarms"),
        ("verifier_accepted", "verifier ok"),
        ("training_runs", "runs"),
    )
    lines = ["condition  " + "  ".join(f"{title:>12}" for _, title in columns)]
    for condition, row in summary["conditions"].items():
        lines.append(f"{condition:<9}  " + "  ".join(f"{row[key]:>12}" for key, _ in columns))
    return "\n".join(lines)


def load_episodes(directory: Path) -> list[dict[str, Any]]:
    paths = sorted(Path(directory).glob("episodes/*/episode.json"))
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def run_pilot(
    output_dir: Path,
    client: Any,
    model: str,
    store: RunStore,
    *,
    tasks: tuple[str, ...] = PILOT_TASKS,
    conditions: tuple[str, ...] = tuple(CONDITIONS),
    repetitions: int = 3,
    progress: Callable[[str], None] = lambda _line: None,
    **options: Any,
) -> dict[str, Any]:
    """Run every task under every condition, resuming after an interruption.

    Episodes that already have an episode.json are kept. The summary is written to
    pilot_report.json in output_dir.
    """
    output_dir = Path(output_dir)
    plan = [
        (task, condition, repetition)
        for task in tasks
        for condition in conditions
        for repetition in range(1, repetitions + 1)
    ]
    for number, (task, condition, repetition) in enumerate(plan, start=1):
        directory = output_dir / "episodes" / f"{task}-{condition}-{repetition}"
        if (directory / "episode.json").is_file():
            continue
        if directory.exists():
            shutil.rmtree(directory)  # An interrupted episode is run again.
        progress(f"[{number}/{len(plan)}] {task} {condition} repetition {repetition}")
        episode = run_episode(client, model, task, condition, directory, store, **options)
        progress(
            f"    agent {episode['agent']['status']}, claimed correct: "
            f"{episode['claimed_correct']}, semantically correct: {episode['semantically_correct']}"
        )
    summary = summarize(load_episodes(output_dir))
    (output_dir / "pilot_report.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Episodes of the agent study.")
    commands = parser.add_subparsers(dest="command", required=True)
    pilot = commands.add_parser("pilot", help="Run tasks under conditions with one model.")
    pilot.add_argument("--output", type=Path, required=True, help="Directory; reused to resume.")
    pilot.add_argument("--data", choices=("synthetic", "camelyon"), required=True)
    pilot.add_argument("--config", type=Path, help="Camelyon17 config file.")
    pilot.add_argument("--provider", default="gemini", help="gemini, ollama or openai.")
    pilot.add_argument("--model", help="Defaults to the provider's environment variable.")
    pilot.add_argument("--tasks", nargs="+", choices=TASKS, default=list(PILOT_TASKS))
    pilot.add_argument(
        "--conditions", nargs="+", choices=tuple(CONDITIONS), default=list(CONDITIONS)
    )
    pilot.add_argument("--repetitions", type=int, default=3)
    pilot.add_argument("--seed", type=int, default=7)
    pilot.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    pilot.add_argument("--pause", type=float, default=4.0, help="Seconds before a model call.")
    pilot.add_argument("--error-wait", type=float, default=60.0, help="Seconds after an error.")
    show = commands.add_parser("summarize", help="Summarize the episodes of a pilot directory.")
    show.add_argument("directory", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "summarize":
        print(summary_text(summarize(load_episodes(args.directory))))
        return 0
    from runsleuth.llm_client import create_llm_client

    client, model = create_llm_client(args.provider, args.model)
    store = RunStore(
        args.output / "runs",
        data=args.data,
        config=args.config,
        seed=args.seed,
        device=args.device,
    )
    summary = run_pilot(
        args.output,
        client,
        model,
        store,
        tasks=tuple(args.tasks),
        conditions=tuple(args.conditions),
        repetitions=args.repetitions,
        progress=lambda line: print(line, flush=True),
        pause=args.pause,
        error_wait=args.error_wait,
        max_api_errors=5,
    )
    print(summary_text(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
