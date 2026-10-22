"""The study's analysis, written before its episodes exist.

A script is labelled by the contract tests alone. Rates are computed per task first
and then averaged over tasks, so a task counts once however many seeds it ran on and
whichever kind it is. Intervals are percentile bootstraps that resample tasks. The
planned comparisons are between conditions, paired by task, with a sign-flip test on
the per-task differences:

    C0 against C2   does running the script and seeing raw evidence help or hurt?
    C2 against C3   what does a verdict add to the same evidence?

The third planned comparison is of acceptance criteria on the final scripts: what the
verifier's two-way decision accepts although it fails the contract tests, and what
triage does with exactly those scripts and with the ones that pass.

Added on 2026-10-14, after the draw, while the first model was running and before any
of its outcomes were analysed, and reported apart from what was planned:

    C1 against C2   what does raw evidence add to the runs' metrics?
    C3 against C4   what does the verdict on the agent's own runs add?

and the verifier judged on its own: each criterion's false acceptances, the failing
scripts among those it accepts, how many it accepts without a reviewer and how many it
sends to one, on the final scripts of C0 to C3 and per condition. C4 is left out of the
pooled figures because its agent saw the verifier's verdict on its own runs, so its
scripts are no fair test of that verifier. For the failing scripts, the verifier's
diagnosis is counted apart from its decision: a script can be rejected for the wrong
reason.

    python -m runsleuth.agent_analysis artifacts/agent_study/flash-lite
"""

import argparse
import itertools
import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

from runsleuth import agent_episodes, agent_tasks
from runsleuth.agent_episodes import CONDITIONS, CRITERIA
from runsleuth.agent_verification import TRIAGE

PLANNED_COMPARISONS = (("C0", "C2"), ("C2", "C3"))
AMENDED = "2026-10-14"
SECONDARY_COMPARISONS = (("C1", "C2"), ("C3", "C4"))
VERIFIER_CONDITIONS = ("C0", "C1", "C2", "C3")
DIAGNOSES = ("the_task_fault", "another_fault", "no_known_fault", "not_diagnosed")
FAULT_GROUPS = {
    "all_faults": ("listed_fault", "unlisted_fault"),
    "listed_faults": ("listed_fault",),
    "unlisted_faults": ("unlisted_fault",),
}
CORRECT_KINDS = ("healthy", "lookalike")
# What an episode on a faulty task, and on a correct one, can come to.
FAULT_OUTCOMES: dict[str, Callable[[dict], bool]] = {
    "repaired": lambda row: row["passes"],
    "false_claim": lambda row: row["claims_correct"] and not row["passes"],
}
CORRECT_OUTCOMES: dict[str, Callable[[dict], bool]] = {
    "left_correct": lambda row: row["passes"],
    "edited": lambda row: row["edited"],
    "false_alarm": lambda row: row["claims_incorrect"] and row["passes"],
}
RESAMPLES = 10_000
# Up to this many tasks every pattern of signs is counted; beyond it, patterns are sampled.
EXACT_TASKS = 16
SAMPLED_PATTERNS = 100_000


def rows(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per episode that has a label; the provider's failures are left out."""
    labelled = []
    for episode in episodes:
        passes = agent_episodes.meets_contract(episode)
        if agent_episodes.provider_failed(episode) or passes is None:
            continue
        labelled.append(
            {
                "task": episode["repair_task"],
                "kind": episode.get("task_kind")
                or agent_tasks.kind(episode["repair_task"], episode.get("family", "camelyon")),
                "condition": episode["condition"],
                "passes": passes,
                "claims_correct": episode["claimed_correct"] is True,
                "claims_incorrect": episode["claimed_correct"] is False,
                "edited": episode["agent"]["script_changed"],
                "triage": agent_episodes.triage(episode),
                "accepted_by": agent_episodes.acceptances(episode),
                "diagnosis": _diagnosis(episode),
            }
        )
    return labelled


def _diagnosis(episode: dict[str, Any]) -> str:
    """What the verifier diagnosed in the final script, against the task's own fault."""
    found = (episode["verification"] or {}).get("re_diagnosis")
    if found is None:
        return "not_diagnosed"
    if found == agent_tasks.NO_FAULT:
        return "no_known_fault"
    return "the_task_fault" if found == episode.get("expected_diagnosis") else "another_fault"


def task_rates(
    data: list[dict[str, Any]],
    condition: str,
    kinds: tuple[str, ...],
    outcome: Callable[[dict], bool],
) -> dict[str, float]:
    """For each task of these kinds, the share of its episodes with an outcome."""
    seen: dict[str, list[bool]] = {}
    for row in data:
        if row["condition"] == condition and row["kind"] in kinds:
            seen.setdefault(row["task"], []).append(bool(outcome(row)))
    return {task: sum(values) / len(values) for task, values in seen.items()}


def interval(values: list[float], rng: random.Random) -> dict[str, Any]:
    """The mean over tasks with a 95% interval from resampling the tasks."""
    count = len(values)
    if not count:
        return {"tasks": 0, "mean": None, "low": None, "high": None}
    means = sorted(sum(rng.choices(values, k=count)) / count for _ in range(RESAMPLES))
    return {
        "tasks": count,
        "mean": sum(values) / count,
        "low": means[int(0.025 * RESAMPLES)],
        "high": means[int(0.975 * RESAMPLES) - 1],
    }


def sign_flip_p(differences: list[float], rng: random.Random) -> float | None:
    """Two-sided: how often flipping the signs of the differences gives a sum as large."""
    count = len(differences)
    if not count:
        return None
    observed = abs(sum(differences)) - 1e-12
    if count <= EXACT_TASKS:
        patterns: Any = itertools.product((1, -1), repeat=count)
        total = 2**count
    else:
        patterns = (rng.choices((1, -1), k=count) for _ in range(SAMPLED_PATTERNS))
        total = SAMPLED_PATTERNS
    as_large = sum(
        abs(sum(sign * value for sign, value in zip(signs, differences, strict=True))) >= observed
        for signs in patterns
    )
    return as_large / total


def compare(
    data: list[dict[str, Any]], pairs: tuple[tuple[str, str], ...], rng: random.Random
) -> list[dict[str, Any]]:
    """Each pair of conditions on every faulty task both ran, paired by task."""
    comparisons = []
    for first, second in pairs:
        for name, test in FAULT_OUTCOMES.items():
            before = task_rates(data, first, FAULT_GROUPS["all_faults"], test)
            after = task_rates(data, second, FAULT_GROUPS["all_faults"], test)
            differences = [after[task] - before[task] for task in sorted(before.keys() & after)]
            comparisons.append(
                {
                    "from": first,
                    "to": second,
                    "outcome": name,
                    "difference": interval(differences, rng),
                    "p_value": sign_flip_p(differences, rng),
                }
            )
    return comparisons


def decisions(scripts: list[dict[str, Any]], criterion: str) -> dict[str, list[int]]:
    """What a criterion does with these final scripts, each as [how many, out of how many]."""
    accepted = [row for row in scripts if row["accepted_by"][criterion]]
    wrongly = sum(not row["passes"] for row in accepted)
    reviewed = criterion == CRITERIA[-1]  # Only triage hands scripts to a reviewer.
    return {
        "false_acceptance": [wrongly, sum(not row["passes"] for row in scripts)],
        "failing_among_accepted": [wrongly, len(accepted)],
        "accepted_without_review": [len(accepted), len(scripts)],
        "sent_to_review": [
            sum(reviewed and row["triage"] == "review" for row in scripts),
            len(scripts),
        ],
    }


def verifier(data: list[dict[str, Any]], conditions: list[str]) -> dict[str, Any]:
    """The acceptance criteria and the verifier's diagnosis, judged on the final scripts."""
    pooled = [row for row in data if row["condition"] in VERIFIER_CONDITIONS]
    failing = [row for row in pooled if not row["passes"]]
    return {
        "added": AMENDED,
        "pooled_conditions": [name for name in VERIFIER_CONDITIONS if name in conditions],
        "pooled": {name: decisions(pooled, name) for name in CRITERIA},
        "by_condition": {
            condition: {
                name: decisions([row for row in data if row["condition"] == condition], name)
                for name in CRITERIA
            }
            for condition in conditions
        },
        "failing_scripts_by_diagnosis": {
            diagnosis: {
                decision: sum(
                    row["diagnosis"] == diagnosis and row["triage"] == decision for row in failing
                )
                for decision in TRIAGE
            }
            for diagnosis in DIAGNOSES
        },
    }


def _counts(data: list[dict[str, Any]], test: Callable[[dict], bool]) -> dict[str, list[int]]:
    """[how many satisfy the test, out of how many], in all and per kind of task."""
    counts = {"all": [sum(bool(test(row)) for row in data), len(data)]}
    for kind in sorted({row["kind"] for row in data}):
        of_kind = [row for row in data if row["kind"] == kind]
        counts[kind] = [sum(bool(test(row)) for row in of_kind), len(of_kind)]
    return counts


def analyze(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Everything the study reports for one model, from its episodes."""
    data, rng = rows(episodes), random.Random(0)
    conditions = [name for name in CONDITIONS if any(row["condition"] == name for row in data)]

    def by_condition(kinds: tuple[str, ...], outcomes: dict[str, Callable]) -> dict[str, Any]:
        return {
            condition: {
                name: interval(list(task_rates(data, condition, kinds, test).values()), rng)
                for name, test in outcomes.items()
            }
            for condition in conditions
        }

    comparisons = compare(data, PLANNED_COMPARISONS, rng)
    failing = [row for row in data if not row["passes"]]
    passing = [row for row in data if row["passes"]]
    return {
        "schema_version": 2,
        "task": "agent_study_analysis",
        "models": sorted({episode["model"] for episode in episodes}),
        "episodes": len(data),
        "episodes_without_a_label": len(episodes) - len(data),
        "tasks": {
            kind: len({row["task"] for row in data if row["kind"] == kind})
            for kind in ("healthy", "lookalike", "listed_fault", "unlisted_fault")
        },
        "faulty_tasks": {
            group: by_condition(kinds, FAULT_OUTCOMES) for group, kinds in FAULT_GROUPS.items()
        },
        "correct_tasks": by_condition(CORRECT_KINDS, CORRECT_OUTCOMES),
        "planned_comparisons": comparisons,
        "criteria": {
            name: {
                "accepts_failing": _counts(
                    failing, lambda row, name=name: row["accepted_by"][name]
                ),
                "rejects_passing": _counts(
                    passing, lambda row, name=name: not row["accepted_by"][name]
                ),
            }
            for name in CRITERIA
        },
        "triage": {
            label: {
                decision: _counts(scripts, lambda row, decision=decision: row["triage"] == decision)
                for decision in TRIAGE
            }
            for label, scripts in (("failing", failing), ("passing", passing))
        },
        "secondary_comparisons": {
            "added": AMENDED,
            "comparisons": compare(data, SECONDARY_COMPARISONS, rng),
        },
        "verifier": verifier(data, conditions),
    }


def _share(value: dict[str, Any]) -> str:
    if value["mean"] is None:
        return "-"
    return f"{value['mean']:.2f} [{value['low']:.2f}, {value['high']:.2f}]"


def _of(count: list[int]) -> str:
    return f"{count[0]}/{count[1]}"


def _comparison_lines(comparisons: list[dict[str, Any]]) -> list[str]:
    lines = []
    for comparison in comparisons:
        p_value = comparison["p_value"]
        lines.append(
            f"  {comparison['to']} minus {comparison['from']}, {comparison['outcome']}: "
            f"{_share(comparison['difference'])}, sign-flip p = "
            + ("-" if p_value is None else f"{p_value:.3f}")
        )
    return lines


def analysis_text(result: dict[str, Any]) -> str:
    tasks = result["tasks"]
    lines = [
        f"model: {', '.join(result['models']) or 'none'}; {result['episodes']} episodes with a "
        f"label, {result['episodes_without_a_label']} without",
        f"tasks: {tasks['listed_fault']} faults with a signature, {tasks['unlisted_fault']} "
        f"without, {tasks['lookalike']} look-alikes, {tasks['healthy']} healthy",
        "shares are means over tasks with 95% intervals from resampling tasks",
    ]
    for group, table in result["faulty_tasks"].items():
        lines += ["", f"{group.replace('_', ' ')}: repaired | false claim"]
        for condition, row in table.items():
            lines.append(
                f"  {condition}  {_share(row['repaired']):<22} {_share(row['false_claim'])}"
            )
    lines += ["", "correct tasks: left correct | edited | false alarm"]
    for condition, row in result["correct_tasks"].items():
        lines.append(
            f"  {condition}  {_share(row['left_correct']):<22} {_share(row['edited']):<22} "
            f"{_share(row['false_alarm'])}"
        )
    lines += ["", "planned comparisons, over all faults and paired by task:"]
    lines += _comparison_lines(result["planned_comparisons"])
    lines += ["", f"{'acceptance criterion':<40} accepts a failing script | rejects a passing one"]
    for name, row in result["criteria"].items():
        accepts, rejects = row["accepts_failing"], row["rejects_passing"]
        unlisted = accepts.get("unlisted_fault", [0, 0])
        lines.append(
            f"  {name.replace('_', ' '):<38} {_of(accepts['all']):<8} "
            f"(without a signature {_of(unlisted)}) | {_of(rejects['all'])}"
        )
    lines += ["", "triage of the final scripts: accept | review | reject"]
    for label, row in result["triage"].items():
        lines.append(
            f"  {label:<8} " + " | ".join(_of(row[decision]["all"]) for decision in TRIAGE)
        )
    secondary = result["secondary_comparisons"]
    lines += ["", f"secondary comparisons, added {secondary['added']} after the draw:"]
    lines += _comparison_lines(secondary["comparisons"])
    judged = result["verifier"]
    lines += [
        "",
        f"the verifier on the final scripts of {', '.join(judged['pooled_conditions']) or '-'} "
        f"(added {judged['added']}):",
        f"  {'criterion':<38} false acceptance | failing among accepted | "
        "accepted without review | sent to review",
    ]
    for name, row in judged["pooled"].items():
        lines.append(
            f"  {name.replace('_', ' '):<38} {_of(row['false_acceptance']):<16} "
            f"{_of(row['failing_among_accepted']):<22} {_of(row['accepted_without_review']):<23} "
            f"{_of(row['sent_to_review'])}"
        )
    lines += ["", "failing scripts by the verifier's diagnosis: accept | review | reject"]
    for diagnosis, row in judged["failing_scripts_by_diagnosis"].items():
        lines.append(
            f"  {diagnosis.replace('_', ' '):<16} "
            + " | ".join(str(row[decision]) for decision in TRIAGE)
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directories", type=Path, nargs="+", help="One directory per model.")
    args = parser.parse_args(argv)
    for directory in args.directories:
        result = analyze(agent_episodes.load_episodes(directory))
        (directory / "analysis.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(analysis_text(result), end="\n\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
