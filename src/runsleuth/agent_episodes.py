"""Episodes of the agent study: one task, one condition, one model, scored afterwards.

An episode gives an agent (agent_loop) a task workspace (agent_tasks) under one of
five conditions, each adding one thing to what the agent can use:

    C0  the script and the task statement; no runs
    C1  bounded training runs, with their metrics and errors
    C2  C1 and the raw mechanism evidence of each run, without any verdict
    C3  C2 and the matcher's diagnosis against a healthy reference run
    C4  C3 and the verifier's decision on each run, with its triage: whether an
        accepted run is the reference's run or differs from it, and where

Afterwards the final script is trained on a fuller budget, and three things are
recorded apart: what the agent claimed; whether the script meets the task statement,
decided by contract tests the agent never sees (agent_contract, kept in contract.json
beside episode.json); and what each acceptance criterion decides
(agent_verification). The verifier never labels a script. Two cross-checks audit the
contract tests without overriding them: what the source alone settles (the same
program as the healthy script, or an untouched script), and reviews (agent_review).

Both later records can be made again for episodes that exist, without asking a model
or training: score runs the contract tests, and verify applies the current verifier
to the runs in the store (verification.json beside episode.json).

Runs are cached by script, budget and seed, so the healthy reference and scripts an
agent left unchanged are trained once.

    python -m runsleuth.agent_episodes prepare --output artifacts/agent_pilot/pilot-1 \\
        --data camelyon --config configs/camelyon17_clean.json
    python -m runsleuth.agent_episodes pilot --output artifacts/agent_pilot/pilot-1 \\
        --data camelyon --config configs/camelyon17_clean.json
    python -m runsleuth.agent_episodes score artifacts/agent_pilot/pilot-1
    python -m runsleuth.agent_episodes verify --output artifacts/agent_pilot/pilot-1 \\
        --data camelyon --config configs/camelyon17_clean.json
    python -m runsleuth.agent_episodes review artifacts/agent_pilot/pilot-1
    python -m runsleuth.agent_episodes summarize artifacts/agent_pilot/pilot-1
"""

import argparse
import ast
import hashlib
import json
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from runsleuth import agent_review, agent_tasks
from runsleuth.agent_loop import run_agent
from runsleuth.agent_tasks import FAULTS, HEALTHY, SCRIPT_NAME, TASKS
from runsleuth.agent_verification import (
    TRIAGE,
    TRIAGE_MEANING,
    trajectory_text,
    verify_repair_run,
)
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
# Raised when a condition shows an agent something else than before.
PRESENTATION_VERSION = 2


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

    def run(self, source: str, budget: Budget, *, train: bool = True) -> dict[str, Any] | None:
        """The outcome of training a script on a budget: status, problems, output, report.

        With train off, nothing is trained: an outcome the store does not hold is None.
        """
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
        if not train:
            return None
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
                "triage": "reject",
            }
        else:
            verified = verify_repair_run(report, healthy, before())
            shown["verification"] = {
                "decision": verified["decision"],
                "reasons": verified["reasons"],
                "trajectory": trajectory_text(verified["trajectory"]),
                "triage": verified["triage"],
                "meaning": TRIAGE_MEANING[verified["triage"]],
                "differences_from_reference": verified["differences"],
            }
    return _rounded(shown)


def same_program(left: str, right: str) -> bool:
    """Whether two sources are the same program, ignoring comments and layout."""
    try:
        return ast.dump(ast.parse(left)) == ast.dump(ast.parse(right))
    except SyntaxError:
        return False


def semantic_label(task: str, initial: str, final: str, healthy: str) -> tuple[bool | None, str]:
    """Whether the final script is correct, where the source alone settles it."""
    if agent_review.program_sha256(final) is None:
        return False, "does_not_parse"
    if same_program(final, healthy):
        return True, "same_program_as_healthy_script"
    if same_program(final, initial):
        return task not in FAULTS, "unchanged_script"
    return None, "needs_review"


VERIFICATION_FIELDS = (
    "decision",
    "reasons",
    "re_diagnosis",
    "criteria",
    "trajectory",
    "triage",
    "differences",
)


def final_verification(
    store: RunStore, final: str, healthy: str, initial: str, budget: Budget, *, train: bool = True
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Train a final script on a budget and verify it against the healthy script.

    Returns the final run's outcome and the verification record, which is None when the
    final run or the reference did not complete. With train off, the runs must be in
    the store already; LookupError names the first that is not.
    """

    def run(name: str, source: str) -> dict[str, Any]:
        outcome = store.run(source, budget, train=train)
        if outcome is None:
            raise LookupError(f"the run of the {name} script is not in the store")
        return outcome

    outcome, reference = run("final", final), run("healthy", healthy)["report"]
    report = outcome["report"]
    if report is None or report["status"] != "completed" or reference is None:
        return outcome, None
    verified = verify_repair_run(report, reference, run("task's", initial)["report"])
    return outcome, {name: verified[name] for name in VERIFICATION_FIELDS}


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
    contract: Callable[[Path], dict[str, Any]] | None = None,
    **agent_options: Any,
) -> dict[str, Any]:
    """Run one agent on one task under one condition and save episode.json.

    The final script is then scored by the contract tests (score_episode); the episode
    is returned as load_episode reads it.
    """
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
    outcome, verification = final_verification(store, final, healthy, initial, full)
    report = outcome["report"]
    claim = agent["claim"]
    episode = {
        "schema_version": 1,
        "task": "agent_episode",
        "repair_task": task,
        "condition": condition,
        "model": model,
        "task_is_correct": task not in FAULTS,
        "expected_diagnosis": agent_tasks.expected_diagnosis(task),
        "statement_sha256": hashlib.sha256(agent_tasks.STATEMENT.encode("utf-8")).hexdigest(),
        "presentation_version": PRESENTATION_VERSION,
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
    score_episode(directory, contract or score_script)
    return load_episode(directory)


def score_script(script: Path) -> dict[str, Any]:
    """The contract tests' result for a script, from a child process."""
    from runsleuth.agent_contract import run_contract_in_subprocess

    return run_contract_in_subprocess(script)


def score_episode(
    directory: Path, contract: Callable[[Path], dict[str, Any]] = score_script
) -> dict[str, Any]:
    """Run the contract tests on an episode's final script and save contract.json."""
    directory = Path(directory)
    episode = json.loads((directory / "episode.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as temporary:
        script = Path(temporary) / SCRIPT_NAME
        script.write_text(episode["final_script"], encoding="utf-8", newline="\n")
        result = contract(script)
    (directory / "contract.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def load_episode(directory: Path) -> dict[str, Any]:
    """An episode with its name and, if it was scored, the contract tests' result."""
    directory = Path(directory)
    episode = json.loads((directory / "episode.json").read_text(encoding="utf-8"))
    scored, verified = directory / "contract.json", directory / "verification.json"
    contract = json.loads(scored.read_text(encoding="utf-8")) if scored.is_file() else None
    if verified.is_file():  # The verifier was applied again after the episode was run.
        episode["verification"] = json.loads(verified.read_text(encoding="utf-8"))["verification"]
    return {**episode, "name": directory.name, "contract": contract}


def triage(episode: dict[str, Any]) -> str | None:
    """accept, review or reject; None for an episode verified before triage existed."""
    verification = episode["verification"]
    if verification is None:
        return "reject"  # The final script did not run to the end: nothing to accept.
    return verification.get("triage")


def meets_contract(episode: dict[str, Any]) -> bool | None:
    """Whether the final script passed the contract tests; None if they gave no result."""
    status = (episode.get("contract") or {}).get("status")
    return {"passed": True, "failed": False, "refused": False}.get(status)


def provider_failed(episode: dict[str, Any]) -> bool:
    """The model could not be asked, so the episode shows nothing about the agent."""
    return episode["agent"]["status"] == "api_error"


CRITERIA = (
    "agent_claim",
    "before_and_after_metrics",
    "reference_metrics",
    "mechanism_check",
    "mechanism_check_and_reference_metrics",
    "checks_and_same_run_as_reference",
)


def acceptances(episode: dict[str, Any]) -> dict[str, bool]:
    """Which acceptance criteria accept the final script; one that cannot decide does not."""
    computed = (episode["verification"] or {}).get("criteria", {})
    accepted = {name: computed.get(name) is True for name in CRITERIA[1:-1]}
    return {
        "agent_claim": episode["claimed_correct"] is True,
        **accepted,
        # The verifier's acceptance without a reviewer: triage accept.
        CRITERIA[-1]: triage(episode) == "accept",
    }


def summarize(
    episodes: list[dict[str, Any]], reviews: dict[str, dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Counts per condition and per acceptance criterion.

    A script is labelled by the contract tests alone; one they gave no result for is
    neither passing nor failing. A false claim says a script trains correctly when it
    fails them. Episodes the provider failed are counted apart and left out of
    everything else. checks says how often the contract tests agree with what the
    source alone settles and with reviews (agent_review.load_reviews); neither
    changes a label.
    """
    reviews = reviews or {}
    failures = sum(provider_failed(episode) for episode in episodes)
    episodes = [
        {**episode, "meets_contract": meets_contract(episode)}
        for episode in episodes
        if not provider_failed(episode)
    ]
    criteria = {}
    for name in CRITERIA:
        accepted = [episode for episode in episodes if acceptances(episode)[name]]
        criteria[name] = {
            "accepted": len(accepted),
            "accepted_failing": sum(e["meets_contract"] is False for e in accepted),
            "rejected_passing": sum(
                e["meets_contract"] is True and not acceptances(e)[name] for e in episodes
            ),
        }
    rows = {}
    for condition in CONDITIONS:
        group = [episode for episode in episodes if episode["condition"] == condition]
        if not group:
            continue

        def count(test: Callable[[dict], bool], group: list = group) -> int:
            return sum(bool(test(episode)) for episode in group)

        def accepted(episode: dict) -> bool:
            return (episode["verification"] or {}).get("decision") == "accepted"

        rows[condition] = {
            "episodes": len(group),
            "finished": count(lambda e: e["agent"]["status"] == "finished"),
            "claimed_correct": count(lambda e: e["claimed_correct"] is True),
            "contract_passed": count(lambda e: e["meets_contract"] is True),
            "contract_failed": count(lambda e: e["meets_contract"] is False),
            "unscored": count(lambda e: e["meets_contract"] is None),
            "false_claims": count(
                lambda e: e["claimed_correct"] is True and e["meets_contract"] is False
            ),
            "false_alarms": count(
                lambda e: e["claimed_correct"] is False and e["meets_contract"] is True
            ),
            "edited_a_correct_script": count(
                lambda e: e["task_is_correct"] and e["agent"]["script_changed"]
            ),
            "verifier_accepted": count(accepted),
            "verifier_accepted_failing": count(
                lambda e: accepted(e) and e["meets_contract"] is False
            ),
            "verifier_rejected_passing": count(
                lambda e: not accepted(e) and e["meets_contract"] is True
            ),
            "training_runs": sum(episode["agent"]["training_runs"] for episode in group),
            "input_tokens": sum(episode["agent"]["input_tokens"] for episode in group),
            "output_tokens": sum(episode["agent"]["output_tokens"] for episode in group),
        }
    triaged = {
        name: {
            "episodes": sum(triage(e) == name for e in episodes),
            "contract_passed": sum(
                triage(e) == name and e["meets_contract"] is True for e in episodes
            ),
            "contract_failed": sum(
                triage(e) == name and e["meets_contract"] is False for e in episodes
            ),
        }
        for name in TRIAGE
    }
    scored = [episode for episode in episodes if episode["meets_contract"] is not None]
    by_source = [e for e in scored if e["semantically_correct"] is not None]
    by_review = [
        (e, reviews[program])
        for e in scored
        if (program := agent_review.program_sha256(e["final_script"])) in reviews
    ]
    failed_clauses: dict[str, int] = {}
    for episode in scored:
        for clause in episode["contract"]["failed"]:
            failed_clauses[clause] = failed_clauses.get(clause, 0) + 1
    return {
        "schema_version": 2,
        "task": "agent_episode_summary",
        "scope": "development pilot; injected faults, one seed, small samples",
        "episodes": len(episodes),
        "provider_failures": failures,
        "unscored": len(episodes) - len(scored),
        "contract_versions": sorted({e["contract"]["contract_version"] for e in scored}),
        "failed_clauses": failed_clauses,
        "conditions": rows,
        "criteria": criteria,
        "triage": triaged,
        "untriaged": sum(triage(episode) is None for episode in episodes),
        "checks": {
            "source": {
                "episodes": len(by_source),
                "disagree": [
                    e.get("name", "")
                    for e in by_source
                    if e["semantically_correct"] != e["meets_contract"]
                ],
            },
            "reviews": {
                "episodes": len(by_review),
                "blind_to_verifier": sum(review["blind_to_verifier"] for _, review in by_review),
                "disagree": [
                    e.get("name", "")
                    for e, review in by_review
                    if review["correct"] != e["meets_contract"]
                ],
            },
        },
    }


def summary_text(summary: dict[str, Any]) -> str:
    columns = (
        ("episodes", "episodes"),
        ("finished", "finished"),
        ("claimed_correct", "claims ok"),
        ("contract_passed", "passes"),
        ("contract_failed", "fails"),
        ("unscored", "unscored"),
        ("false_claims", "false claims"),
        ("false_alarms", "false alarms"),
        ("verifier_accepted", "verifier ok"),
        ("verifier_accepted_failing", "ok but fails"),
        ("verifier_rejected_passing", "no but passes"),
        ("training_runs", "runs"),
    )
    lines = ["condition  " + "  ".join(f"{title:>13}" for _, title in columns)]
    for condition, row in summary["conditions"].items():
        lines.append(f"{condition:<9}  " + "  ".join(f"{row[key]:>13}" for key, _ in columns))
    counts = (
        ("accepted", "accepted"),
        ("accepted_failing", "ok but fails"),
        ("rejected_passing", "no but passes"),
    )
    lines += ["", f"{'acceptance criterion':<38}" + "  ".join(f"{t:>13}" for _, t in counts)]
    for name, row in summary["criteria"].items():
        lines.append(
            f"{name.replace('_', ' '):<38}" + "  ".join(f"{row[key]:>13}" for key, _ in counts)
        )
    lines += ["", f"{'verifier triage':<38}" + f"{'episodes':>13}  {'passes':>13}  {'fails':>13}"]
    for name, row in summary["triage"].items():
        lines.append(
            f"{name:<38}{row['episodes']:>13}  {row['contract_passed']:>13}  "
            f"{row['contract_failed']:>13}"
        )
    if summary["untriaged"]:
        lines.append(
            f"{summary['untriaged']} episodes were verified before triage existed; "
            "the verify command applies the current verifier to them."
        )
    versions = ", ".join(str(version) for version in summary["contract_versions"]) or "none"
    clauses = ", ".join(f"{name} {count}" for name, count in summary["failed_clauses"].items())
    lines += [
        "",
        f"passes and fails are the contract tests' (version {versions}); "
        f"{summary['unscored']} episodes have no result.",
        f"failed clauses: {clauses or 'none'}",
    ]
    for title, check in (
        ("what the source alone settles", summary["checks"]["source"]),
        ("reviews", summary["checks"]["reviews"]),
    ):
        disagree = check["disagree"]
        line = f"{title}: {check['episodes'] - len(disagree)} of {check['episodes']} episodes agree"
        if "blind_to_verifier" in check:
            line += f" ({check['blind_to_verifier']} reviewed blind to the verifier)"
        lines.append(line + (f"; disagree: {', '.join(disagree)}" if disagree else ""))
    return "\n".join(lines)


def load_episodes(directory: Path) -> list[dict[str, Any]]:
    paths = sorted(Path(directory).glob("episodes/*/episode.json"))
    return [load_episode(path.parent) for path in paths]


def score_episodes(
    directory: Path,
    contract: Callable[[Path], dict[str, Any]] = score_script,
    *,
    redo: bool = False,
    progress: Callable[[str], None] = lambda _line: None,
) -> int:
    """Score the episodes of a pilot directory that have no result yet, or all of them.

    Episodes that end in the same script text share one run of the tests.
    """
    results: dict[str, dict[str, Any]] = {}

    def once(script: Path) -> dict[str, Any]:
        source = script.read_text(encoding="utf-8")
        if source not in results:
            results[source] = contract(script)
        return results[source]

    scored = 0
    for path in sorted(Path(directory).glob("episodes/*/episode.json")):
        if redo or not (path.parent / "contract.json").is_file():
            result = score_episode(path.parent, once)
            progress(f"{path.parent.name}: {result['status']} {', '.join(result['failed'])}")
            scored += 1
    return scored


def verify_episodes(
    directory: Path,
    store: RunStore,
    *,
    progress: Callable[[str], None] = lambda _line: None,
) -> int:
    """Apply the current verifier to the episodes of a pilot directory; train nothing.

    Each episode's final script, its task's script and the healthy script must have
    their full-budget runs in the store, as they do after the episode was run. The
    result goes to verification.json, which load_episode prefers over the record made
    when the episode was run. Returns how many episodes were verified.
    """
    healthy, verified = agent_tasks.task_source(HEALTHY), 0
    for path in sorted(Path(directory).glob("episodes/*/episode.json")):
        episode, name = json.loads(path.read_text(encoding="utf-8")), path.parent.name
        initial = agent_tasks.task_source(episode["repair_task"])
        if _sha256(initial) != episode["agent"]["initial_script_sha256"]:
            progress(f"{name}: skipped, the task's script has changed since the episode")
            continue
        try:
            _, verification = final_verification(
                store,
                episode["final_script"],
                healthy,
                initial,
                Budget(**episode["budgets"]["full"]),
                train=False,
            )
        except LookupError as missing:
            progress(f"{name}: skipped, {missing}")
            continue
        record = {
            "schema_version": 1,
            "task": "agent_episode_verification",
            "verification": verification,
        }
        (path.parent / "verification.json").write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        progress(f"{name}: {'no verification' if verification is None else verification['triage']}")
        verified += 1
    return verified


def run_pilot(
    output_dir: Path,
    client: Any,
    model: str,
    store: RunStore,
    *,
    tasks: tuple[str, ...] = PILOT_TASKS,
    conditions: tuple[str, ...] = tuple(CONDITIONS),
    repetitions: int = 3,
    reviews: dict[str, dict[str, Any]] | None = None,
    progress: Callable[[str], None] = lambda _line: None,
    **options: Any,
) -> dict[str, Any]:
    """Run every task under every condition, resuming after an interruption.

    Episodes that already have an episode.json are kept, except those the provider
    failed, which are run again. A provider failure stops the pilot, since the next
    episode would meet it too. The summary is written to pilot_report.json in output_dir.
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
        saved = directory / "episode.json"
        if saved.is_file() and not provider_failed(json.loads(saved.read_text(encoding="utf-8"))):
            continue
        if directory.exists():
            shutil.rmtree(directory)  # An interrupted or failed episode is run again.
        progress(f"[{number}/{len(plan)}] {task} {condition} repetition {repetition}")
        episode = run_episode(client, model, task, condition, directory, store, **options)
        if provider_failed(episode):
            progress(f"    stopped, the provider failed: {episode['agent']['api_errors'][-1]}")
            break
        progress(
            f"    agent {episode['agent']['status']}, claimed correct: "
            f"{episode['claimed_correct']}, contract tests: {episode['contract']['status']}"
        )
    summary = summarize(load_episodes(output_dir), reviews)
    (output_dir / "pilot_report.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary


def prepare_runs(
    store: RunStore,
    tasks: tuple[str, ...] = PILOT_TASKS,
    *,
    quick: Budget = QUICK,
    full: Budget = FULL,
    progress: Callable[[str], None] = lambda _line: None,
) -> list[dict[str, Any]]:
    """Train the healthy script and every task's original script on both budgets.

    Every episode needs these runs and none needs a model for them, so they can be
    trained ahead of the agents; episodes then find them in the store.
    """
    sources = {HEALTHY: agent_tasks.task_source(HEALTHY)}
    sources.update({task: agent_tasks.task_source(task) for task in tasks})
    rows = []
    for task, source in sources.items():
        for name, budget in (("quick", quick), ("full", full)):
            started = time.perf_counter()
            outcome = store.run(source, budget)
            seconds = round(time.perf_counter() - started, 1)
            report = outcome["report"]
            rows.append(
                {
                    "task": task,
                    "budget": name,
                    "status": outcome["status"],
                    "seconds": seconds,
                    "final_metrics": None if report is None else report.get("final_metrics"),
                }
            )
            progress(f"{task} ({name}): {outcome['status']} in {seconds:.0f} s")
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Episodes of the agent study.")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare", help="Train the runs every episode needs; no model is called."
    )
    pilot = commands.add_parser("pilot", help="Run tasks under conditions with one model.")
    verify = commands.add_parser(
        "verify", help="Apply the current verifier to a pilot's episodes; trains nothing."
    )
    for command in (prepare, pilot, verify):
        command.add_argument(
            "--output", type=Path, required=True, help="Directory; reused to resume."
        )
        command.add_argument("--data", choices=("synthetic", "camelyon"), required=True)
        command.add_argument("--config", type=Path, help="Camelyon17 config file.")
        command.add_argument("--seed", type=int, default=7)
        command.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    for command in (prepare, pilot):
        command.add_argument("--tasks", nargs="+", choices=TASKS, default=list(PILOT_TASKS))
    pilot.add_argument("--provider", default="gemini", help="gemini, ollama or openai.")
    pilot.add_argument("--model", help="Defaults to the provider's environment variable.")
    pilot.add_argument(
        "--conditions", nargs="+", choices=tuple(CONDITIONS), default=list(CONDITIONS)
    )
    pilot.add_argument("--repetitions", type=int, default=3)
    pilot.add_argument("--pause", type=float, default=4.0, help="Seconds before a model call.")
    pilot.add_argument("--error-wait", type=float, default=60.0, help="Seconds after an error.")
    show = commands.add_parser("summarize", help="Summarize the episodes of a pilot directory.")
    review = commands.add_parser(
        "review", help="Show the scripts of a pilot directory that still need a review."
    )
    score = commands.add_parser(
        "score", help="Run the contract tests on the final scripts of a pilot directory."
    )
    score.add_argument("--redo", action="store_true", help="Score scripts that have a result.")
    for command in (show, review, score):
        command.add_argument("directory", type=Path)
    for command in (pilot, show, review):
        command.add_argument(
            "--reviews",
            type=Path,
            help=f"Reviewed labels; {agent_review.REVIEWS_PATH.as_posix()} if it exists.",
        )
    return parser


def _reviews(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None and not agent_review.REVIEWS_PATH.is_file():
        return {}
    return agent_review.load_reviews(path or agent_review.REVIEWS_PATH)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "summarize":
        print(summary_text(summarize(load_episodes(args.directory), _reviews(args.reviews))))
        return 0
    if args.command == "score":
        score_episodes(
            args.directory, redo=args.redo, progress=lambda line: print(line, flush=True)
        )
        return 0
    if args.command == "review":
        programs = agent_review.pending(load_episodes(args.directory), _reviews(args.reviews))
        print(agent_review.pending_text(programs))
        return 0
    store = RunStore(
        args.output / "runs",
        data=args.data,
        config=args.config,
        seed=args.seed,
        device=args.device,
    )
    if args.command == "verify":
        verify_episodes(args.output, store, progress=lambda line: print(line, flush=True))
        return 0
    if args.command == "prepare":
        rows = prepare_runs(store, tuple(args.tasks), progress=lambda line: print(line, flush=True))
        return 0 if all(row["status"] == "completed" for row in rows) else 1
    from runsleuth.llm_client import create_llm_client

    client, model = create_llm_client(args.provider, args.model)
    summary = run_pilot(
        args.output,
        client,
        model,
        store,
        tasks=tuple(args.tasks),
        conditions=tuple(args.conditions),
        repetitions=args.repetitions,
        reviews=_reviews(args.reviews),
        progress=lambda line: print(line, flush=True),
        pause=args.pause,
        error_wait=args.error_wait,
        max_api_errors=5,
    )
    print(summary_text(summary))
    if summary["provider_failures"]:
        print("Stopped at a provider failure. The same command resumes the pilot.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
