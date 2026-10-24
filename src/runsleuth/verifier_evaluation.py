"""The verifiers judged on the challenge set: their decisions on every candidate, and how
often they are right, with denominators.

decide applies the verifiers that need no key (the calibrated metric gate, stage one's
gate and RunSleuth v1) to every candidate's run against the healthy run on its seed and
writes decisions.json. review asks the LLM reviewer about every candidate, with the
user's key; its prompts are built here from the candidate's modules, the project it
started from and the two runs (llm_reviewer.user_prompt), never from anything the hidden
tests said. score reads the labels and every verifier's decisions and counts them.

The project a candidate started from is the version its agent was given; for a version
handed in as it is, the version itself (no changes); for a partial repair, its fault.
An evidence-removed candidate is judged on its parent's run report without the records
its transform names (challenge_set.remove_evidence), and has its parent's label.

The counts are by candidate and by case: the candidates that start from one version
are not independent, so every case (starting version) is listed with the decisions on
its candidates. Rates follow the plan: false acceptance (accepted wrong / wrong), wrong
among accepted (which depends on how many candidates are wrong), correct acceptance and
false rejection (of the correct ones), the share sent to review and automatic coverage;
and the same again without the candidates that could not be run. The intervals of
false acceptance, false rejection and the review share come from a bootstrap over
cases, which draws a case's candidates together (BOOTSTRAP); a partial repair counts
in its fault's case. A verifier that never errs in a kind, or always does, has every
draw agree; its interval then reaches the exact bound with the cases as units.

    python -m runsleuth.verifier_evaluation decide --split development
    python -m runsleuth.verifier_evaluation review --split development \\
        --name claude-sonnet-5.5 --provider anthropic --model claude-sonnet-5-5
    python -m runsleuth.verifier_evaluation score --split development
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from runsleuth import candidates as candidate_set
from runsleuth import challenge_set, llm_reviewer, metric_gate, pipeline_tasks, verifier_v1
from runsleuth.challenge_set import Candidate
from runsleuth.llm_client import PROVIDERS, create_llm_client
from runsleuth.signature_matching import load_library

ROOT = candidate_set.ROOT
DECISIONS = ("accept", "review", "reject")
KEYLESS = ("metric_gate", "stage_one_gate", "runsleuth_v1")
# RunSleuth v2's results (verifier_v2), by the order its checks were chosen in.
V2_ORDERS = {"gap": "runsleuth_v2", "fixed": "runsleuth_v2_fixed_order"}
# The bootstrap over cases behind every interval, fixed at the freeze.
BOOTSTRAP = {"resamples": 2000, "seed": 0, "level": 0.95}
INTERVAL_RATES = ("false_acceptance", "false_rejection", "review_share")


def _read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _split_dir(split: str, root: Path) -> Path:
    return Path(root) / split


def run_record(outcome: dict[str, Any] | None) -> dict[str, Any] | None:
    """A run's report, or what the harness said of a run that wrote none."""
    if outcome is None:
        return None
    if outcome.get("report"):
        return _read(outcome["report"])
    return {key: outcome.get(key) for key in ("status", "problems", "error")}


def candidate_run(candidate: Candidate, runs: dict[str, Any]) -> dict[str, Any] | None:
    """The run a candidate is judged on."""
    if candidate.kind == "evidence_removed":
        parent = run_record(runs["runs"].get(candidate.parent))
        return (
            None if parent is None else challenge_set.remove_evidence(parent, candidate.transform)
        )
    return run_record(runs["runs"].get(candidate.id))


def reference_run(candidate: Candidate, runs: dict[str, Any]) -> dict[str, Any]:
    return run_record(runs["references"][f"{candidate.family}/{candidate.seed}"])


def start_sources(candidate: Candidate) -> dict[str, str]:
    """The modules of the project a candidate started from."""
    if candidate.kind == "partial_repair":
        repair = challenge_set.PARTIAL_REPAIRS[candidate.start]
        return pipeline_tasks.version_sources(repair.fault, repair.family)
    return pipeline_tasks.version_sources(candidate.start, candidate.family)


def judged(split: str, root: Path = ROOT) -> list[tuple[Candidate, dict[str, Any]]]:
    """The candidates of a split that have a run to judge, with it."""
    runs = _read(_split_dir(split, root) / "runs.json")
    found = []
    for candidate in candidate_set.candidates(split):
        record = candidate_run(candidate, runs)
        if record is not None:
            found.append((candidate, record))
    return found


def decide(split: str, root: Path = ROOT) -> dict[str, Any]:
    """The keyless verifiers' decision on every candidate with a run; decisions.json."""
    runs = _read(_split_dir(split, root) / "runs.json")
    library = load_library()
    verdicts: dict[str, dict[str, str]] = {name: {} for name in KEYLESS}
    details = {}
    for candidate, record in judged(split, root):
        reference = reference_run(candidate, runs)
        gate = metric_gate.load_gate(candidate.family)
        verdicts["metric_gate"][candidate.id] = metric_gate.decide(record, reference, gate)[
            "decision"
        ]
        verdicts["stage_one_gate"][candidate.id] = metric_gate.decide_stage_one(record, reference)
        result = verifier_v1.verify(record, reference, gate, library)
        verdicts["runsleuth_v1"][candidate.id] = result["triage"]
        details[candidate.id] = {
            "re_diagnosis": result["re_diagnosis"],
            "differences": result["differences"],
            "gate_reasons": result["gate"]["reasons"],
        }
    record = {"schema_version": 1, "task": "verifier_decisions", "verifiers": verdicts}
    record["runsleuth_v1_details"] = details
    _write(_split_dir(split, root) / "decisions.json", record)
    return record


def review_items(split: str, root: Path = ROOT) -> list[dict[str, Any]]:
    """The LLM reviewer's prompt for every candidate with a run."""
    runs = _read(_split_dir(split, root) / "runs.json")
    items = []
    for candidate, record in judged(split, root):
        sources = candidate_set.candidate_sources(candidate, root)
        if candidate.kind == "evidence_removed":
            parent = next(c for c in candidate_set.candidates(split) if c.id == candidate.parent)
            sources = candidate_set.candidate_sources(parent, root)
        prompt = llm_reviewer.user_prompt(
            pipeline_tasks.family(candidate.family).statement,
            start_sources(candidate),
            sources,
            record,
            reference_run(candidate, runs),
        )
        items.append({"id": candidate.id, "prompt": prompt})
    return items


def reviews_dir(split: str, name: str, root: Path = ROOT) -> Path:
    return _split_dir(split, root) / "reviews" / name


def reviewer_decisions(split: str, name: str, root: Path = ROOT) -> dict[str, str]:
    """The LLM reviewer's decision on every candidate it has reviewed, API errors aside."""
    return {
        record["id"]: record["decision"]
        for record in llm_reviewer.read_reviews(reviews_dir(split, name, root))
        if record["status"] in ("completed", "invalid_output")
    }


def vendor_groups(name: str) -> dict[str, Any]:
    """For a verifier with an LLM reviewer in it, the agent candidates whose model its
    reviewer's maker made, and the others."""
    if "llm_reviewer/" not in name:
        return {}
    vendor = challenge_set.VENDORS.get(name.split("llm_reviewer/", 1)[1])

    def same(candidate: Candidate) -> bool:
        return challenge_set.VENDORS.get(candidate.model) == vendor

    return {
        "reviewer_same_vendor": lambda c, _l: c.origin == "agent" and same(c),
        "reviewer_other_vendor": lambda c, _l: c.origin == "agent" and not same(c),
    }


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "count": numerator,
        "of": denominator,
        "rate": round(numerator / denominator, 4) if denominator else None,
    }


def rates(rows: list[tuple[str, str]]) -> dict[str, Any]:
    """The plan's rates from (label, decision) pairs."""
    table = Counter(rows)
    wrong = sum(count for (label, _), count in table.items() if label == "wrong")
    correct = sum(count for (label, _), count in table.items() if label == "correct")
    accepted = table[("wrong", "accept")] + table[("correct", "accept")]
    reviewed = table[("wrong", "review")] + table[("correct", "review")]
    return {
        "table": {
            label: {decision: table[(label, decision)] for decision in DECISIONS}
            for label in ("wrong", "correct")
        },
        "false_acceptance": _rate(table[("wrong", "accept")], wrong),
        "wrong_among_accepted": _rate(table[("wrong", "accept")], accepted),
        "correct_acceptance": _rate(table[("correct", "accept")], correct),
        "false_rejection": _rate(table[("correct", "reject")], correct),
        "review_share": _rate(reviewed, len(rows)),
        "automatic_coverage": _rate(len(rows) - reviewed, len(rows)),
    }


def case_of(candidate: Candidate) -> str:
    """The case a candidate is drawn with: its starting version, or a partial repair's
    fault."""
    start = candidate.start
    if candidate.kind == "partial_repair":
        start = challenge_set.PARTIAL_REPAIRS[start].fault
    return f"{candidate.family}/{start}"


def _quantile(values: list[float], q: float) -> float:
    position = q * (len(values) - 1)
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def _interval_counts(table: Counter) -> dict[str, tuple[int, int]]:
    """The numerator and denominator of every rate in INTERVAL_RATES."""
    wrong = sum(table[("wrong", decision)] for decision in DECISIONS)
    correct = sum(table[("correct", decision)] for decision in DECISIONS)
    reviewed = table[("wrong", "review")] + table[("correct", "review")]
    return {
        "false_acceptance": (table[("wrong", "accept")], wrong),
        "false_rejection": (table[("correct", "reject")], correct),
        "review_share": (reviewed, wrong + correct),
    }


def intervals(
    rows: list[tuple[str, str, str]], settings: dict[str, Any] = BOOTSTRAP
) -> dict[str, Any]:
    """Intervals of the rates in INTERVAL_RATES from (case, label, decision) rows, by
    drawing cases with all their candidates; a draw without a rate's denominator does
    not count for that rate. When none or all of a rate's candidates count, every draw
    agrees, and the interval reaches the exact (Clopper-Pearson) bound for none or all
    of the cases that have the rate's denominator."""
    by_case: dict[str, Counter] = defaultdict(Counter)
    for case, label, decision in rows:
        by_case[case][(label, decision)] += 1
    cases = [by_case[case] for case in sorted(by_case)]
    generator = random.Random(settings["seed"])
    drawn: dict[str, list[float]] = {name: [] for name in INTERVAL_RATES}
    for _ in range(settings["resamples"]):
        table = Counter()
        for _case in cases:
            table.update(generator.choice(cases))
        for name, (numerator, denominator) in _interval_counts(table).items():
            if denominator:
                drawn[name].append(numerator / denominator)
    observed = _interval_counts(sum(cases, Counter()))
    by_case_counts = [_interval_counts(case) for case in cases]
    tail = (1 - settings["level"]) / 2
    found: dict[str, Any] = {}
    for name, values in drawn.items():
        if not values:
            found[name] = None
            continue
        values.sort()
        low, high = _quantile(values, tail), _quantile(values, 1 - tail)
        numerator, denominator = observed[name]
        units = sum(1 for counts in by_case_counts if counts[name][1])
        if numerator == 0:
            high = max(high, 1 - tail ** (1 / units))
        elif numerator == denominator:
            low = min(low, tail ** (1 / units))
        found[name] = {"low": round(low, 4), "high": round(high, 4), "draws": len(values)}
    found["cases"] = len(cases)
    return found


def score(split: str, root: Path = ROOT, reviewers: tuple[str, ...] = ()) -> dict[str, Any]:
    """Every verifier's rates on a split, overall and by group, and its decisions by case."""
    directory = _split_dir(split, root)
    labels = _read(directory / "labels.json")["labels"]
    verdicts = dict(_read(directory / "decisions.json")["verifiers"])
    for name in reviewers:
        verdicts[f"llm_reviewer/{name}"] = reviewer_decisions(split, name, root)
    v2_costs = {}
    for order, name in V2_ORDERS.items():
        path = directory / f"v2-{order}.json"
        if path.exists():
            decided = _read(path)["candidates"]
            verdicts[name] = {cid: found["decision"] for cid, found in decided.items()}
            v2_costs[name] = v2_cost([found["cost"] for found in decided.values()])
    for name in reviewers:
        if "runsleuth_v2" in verdicts:
            verdicts[f"runsleuth_v2+llm_reviewer/{name}"] = combined(
                verdicts["runsleuth_v2"], verdicts[f"llm_reviewer/{name}"]
            )
    planned = {candidate.id: candidate for candidate in candidate_set.candidates(split)}
    groups = {
        "all": lambda _c, _l: True,
        "agent": lambda c, _l: c.origin == "agent",
        "human": lambda c, _l: c.origin == "human",
        "program_transform": lambda c, _l: c.origin == "program_transform",
        "runnable": lambda _c, label: label["outcome"] != "not_runnable",
        **{family: (lambda c, _l, f=family: c.family == f) for family in challenge_set.FAMILIES},
    }
    result: dict[str, Any] = {"schema_version": 1, "task": "verifier_scores", "split": split}
    result["labels"] = dict(Counter(label["label"] for label in labels.values()))
    result["verifiers"] = {}
    result["v2_cost"] = v2_costs
    for name, decided in verdicts.items():
        judged_ids = [cid for cid in decided if cid in labels and cid in planned]
        by_group = {}
        for group, member in {**groups, **vendor_groups(name)}.items():
            rows = [
                (case_of(planned[cid]), labels[cid]["label"], decided[cid])
                for cid in judged_ids
                if member(planned[cid], labels[cid])
            ]
            if rows:
                by_group[group] = rates([(label, decision) for _, label, decision in rows])
                by_group[group]["intervals"] = intervals(rows)
        cases: dict[str, Counter] = defaultdict(Counter)
        for cid in judged_ids:
            candidate = planned[cid]
            cases[f"{candidate.family}/{candidate.start}"][
                (labels[cid]["label"], decided[cid])
            ] += 1
        result["verifiers"][name] = {
            "judged": len(judged_ids),
            "groups": by_group,
            "cases": {
                case: {f"{label} {decision}": count for (label, decision), count in counts.items()}
                for case, counts in sorted(cases.items())
            },
        }
    _write(directory / "scores.json", result)
    return result


def _cell(rate: dict[str, Any]) -> str:
    return f"{rate['count']}/{rate['of']}"


def combined(v2: dict[str, str], reviewer: dict[str, str]) -> dict[str, str]:
    """The rule fixed before any test result: v2's violation rejects; v2 and the
    reviewer accepting accepts; anything else goes to review."""
    found = {}
    for cid in v2.keys() & reviewer.keys():
        if v2[cid] == "reject":
            found[cid] = "reject"
        else:
            both = v2[cid] == "accept" and reviewer[cid] == "accept"
            found[cid] = "accept" if both else "review"
    return found


def v2_cost(costs: list[dict[str, Any]]) -> dict[str, Any]:
    """What v2's checks cost over the candidates, as each would pay alone."""
    total = Counter()
    for cost in costs:
        total.update(cost)
    count = len(costs) or 1
    return {
        "candidates": len(costs),
        "runs": total["runs"],
        "steps": total["steps"],
        "seconds": round(total["seconds"], 1),
        "seconds_per_candidate": round(total["seconds"] / count, 1),
        "runs_per_candidate": round(total["runs"] / count, 2),
    }


def _interval(found: dict[str, Any] | None) -> str:
    return "-" if found is None else f"[{found['low']:.3f}, {found['high']:.3f}]"


def score_text(result: dict[str, Any]) -> str:
    """One line per verifier and group, the agent candidates' intervals, then each
    case's decisions."""
    lines = [f"{result['split']}: labels {result['labels']}"]
    header = (
        f"{'verifier':<30} {'group':<18} {'false acc':>9} {'wrong/acc':>9} "
        f"{'corr acc':>9} {'false rej':>9} {'review':>8}"
    )
    lines.append(header)
    for name, scored in result["verifiers"].items():
        for group, rated in scored["groups"].items():
            lines.append(
                f"{name:<30} {group:<18} {_cell(rated['false_acceptance']):>9} "
                f"{_cell(rated['wrong_among_accepted']):>9} "
                f"{_cell(rated['correct_acceptance']):>9} "
                f"{_cell(rated['false_rejection']):>9} {_cell(rated['review_share']):>8}"
            )
    for name, cost in result.get("v2_cost", {}).items():
        lines.append(
            f"{name} cost: {cost['runs']} probe runs, {cost['seconds']} s for "
            f"{cost['candidates']} candidates ({cost['runs_per_candidate']} runs, "
            f"{cost['seconds_per_candidate']} s each)"
        )
    lines.append(f"{BOOTSTRAP['level']:.0%} intervals on agent candidates, bootstrap over cases")
    for name, scored in result["verifiers"].items():
        found = scored["groups"].get("agent", {}).get("intervals")
        if found:
            lines.append(
                f"{name:<30} false acc {_interval(found['false_acceptance'])}, "
                f"false rej {_interval(found['false_rejection'])}, "
                f"review {_interval(found['review_share'])} ({found['cases']} cases)"
            )
    lines.append("decisions by case (label decision: candidates)")
    for name, scored in result["verifiers"].items():
        lines.append(f"  {name}")
        for case, counts in scored["cases"].items():
            shown = ", ".join(f"{key}: {count}" for key, count in sorted(counts.items()))
            lines.append(f"    {case:<55} {shown}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="The verifiers judged on the challenge set.")
    commands = parser.add_subparsers(dest="command", required=True)
    decided = commands.add_parser("decide", help="Apply the verifiers that need no key.")
    reviewed = commands.add_parser("review", help="Ask the LLM reviewer about every candidate.")
    reviewed.add_argument("--name", required=True, help="The reviewer's name, for its folder.")
    reviewed.add_argument("--provider", choices=PROVIDERS, required=True)
    reviewed.add_argument("--model", help="The provider's model, if not its environment variable.")
    scored = commands.add_parser("score", help="Count every verifier's decisions against labels.")
    scored.add_argument("--reviewers", nargs="*", default=(), help="Reviewer names to include.")
    for command in (decided, reviewed, scored):
        command.add_argument("--split", choices=challenge_set.SPLITS, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "decide":
        decided = decide(args.split)
        print({name: dict(Counter(found.values())) for name, found in decided["verifiers"].items()})
    elif args.command == "review":
        client, model = create_llm_client(args.provider, args.model)
        llm_reviewer.review_all(
            review_items(args.split),
            reviews_dir(args.split, args.name),
            client,
            model,
            provider=args.provider,
        )
        print(
            llm_reviewer.summary_text(llm_reviewer.read_reviews(reviews_dir(args.split, args.name)))
        )
    else:
        print(score_text(score(args.split, reviewers=tuple(args.reviewers))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
