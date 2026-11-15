"""Frozen v2 and an LLM reviewer in sequence: v2 runs its checks and decides what it
can, and the reviewer decides what v2 would send to people, given v2's evidence.

v2 accepts, rejects or sends a candidate to review. Here a candidate v2 sends to review
goes to the reviewer instead, asked as the screening's input c (matched_screening): the
statement, the files, the changes and the observation packet of every run v2 read
(matched_evidence), with the bounded question. What the reviewer cannot decide still
goes to people. The candidates are the stored ones of a split; nothing is trained, and
one review is asked per candidate.

The score sets three verifiers side by side, over every candidate and over the agents'
repairs: v2 alone, the reviewer alone on v2's review set, and v2 then the reviewer. It
reads the labels; nothing the reviewer is given does.

    python -m runsleuth.matched_cascade run --name claude-sonnet-5.5 \\
        --provider anthropic --model claude-sonnet-5-5
    python -m runsleuth.matched_cascade score --name claude-sonnet-5.5
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

from runsleuth import matched_evidence, matched_screening
from runsleuth.llm_client import PROVIDERS, create_llm_client

INPUT = "c"
ROOT = matched_screening.ROOT / "cascade"


def cascade_items(split: str = "test", order: str = "gap") -> dict[str, dict[str, Any]]:
    """Input c for every stored candidate v2 sent to review.

    Raises ValueError when a candidate's packet does not keep v2's result.
    """
    from runsleuth import candidates as candidate_set
    from runsleuth import pipeline_tasks, verifier_evaluation

    root = candidate_set.ROOT / split
    decided = json.loads((root / f"v2-{order}.json").read_text(encoding="utf-8"))["candidates"]
    names = [name for name, result in decided.items() if result["decision"] == "review"]
    trips = matched_evidence.stored_round_trips(split, names, order)
    by_id = {candidate.id: candidate for candidate in candidate_set.candidates(split)}
    items = {}
    for name in names:
        candidate, trip = by_id[name], trips[name]
        if trip["stored"] != trip["packet_result"]:
            raise ValueError(f"the packet of {name} does not keep v2's result")
        prompt = matched_screening._sources_prompt(
            pipeline_tasks.family(candidate.family).statement,
            verifier_evaluation.start_sources(candidate),
            candidate_set.candidate_sources(candidate),
        ) + (f"## Records of the training runs\n\n{trip['packet']}\n")
        items[name] = {"inputs": {INPUT: (matched_screening.BOUNDED_SYSTEM_PROMPT, prompt)}}
    return items


def decisions(records: list[dict[str, Any]]) -> dict[str, str]:
    """The reviewer's decision per candidate; a failed call decides nothing."""
    return {r["item"]: r["decision"] for r in records if r["status"] != "api_error"}


def combined(v2: dict[str, str], reviewer: dict[str, str]) -> dict[str, str]:
    """v2's decision, and the reviewer's where v2 sent a candidate to review; review where
    the reviewer has not answered."""
    return {
        name: reviewer.get(name, "review") if decision == "review" else decision
        for name, decision in v2.items()
    }


def counts(decided: dict[str, str], labels: dict[str, str], names: set[str]) -> dict[str, Any]:
    """Wrong accepted, correct accepted, correct rejected and sent to review, over names."""
    wrong = [n for n in names if labels[n] == "wrong"]
    right = [n for n in names if labels[n] == "correct"]
    return {
        "wrong_accepted": (sum(decided[n] == "accept" for n in wrong), len(wrong)),
        "correct_accepted": (sum(decided[n] == "accept" for n in right), len(right)),
        "correct_rejected": (sum(decided[n] == "reject" for n in right), len(right)),
        "review": (sum(decided[n] == "review" for n in names), len(names)),
    }


def score(
    v2: dict[str, str],
    reviewer: dict[str, str],
    labels: dict[str, str],
    agents: set[str],
) -> dict[str, dict[str, Any]]:
    """The three verifiers over every candidate and over the agents' repairs."""
    sent = {name for name, decision in v2.items() if decision == "review"}
    together = combined(v2, reviewer)
    alone = {name: reviewer.get(name, "review") for name in sent}
    everything = set(v2)
    return {
        "v2 alone": {"all": counts(v2, labels, everything), "agents": counts(v2, labels, agents)},
        "reviewer on v2's review set": {
            "all": counts(alone, labels, sent),
            "agents": counts(alone, labels, sent & agents),
        },
        "v2 then the reviewer": {
            "all": counts(together, labels, everything),
            "agents": counts(together, labels, agents),
        },
    }


def score_text(table: dict[str, dict[str, Any]], records: list[dict[str, Any]]) -> str:
    def cell(pair: tuple[int, int]) -> str:
        return f"{pair[0]}/{pair[1]}"

    lines = [
        f"{'verifier':30s} {'group':7s} {'wrong accepted':>15s} {'correct accepted':>17s} "
        f"{'correct rejected':>17s} {'review':>8s}"
    ]
    for verifier, groups in table.items():
        for group, found in groups.items():
            lines.append(
                f"{verifier:30s} {group:7s} {cell(found['wrong_accepted']):>15s} "
                f"{cell(found['correct_accepted']):>17s} {cell(found['correct_rejected']):>17s} "
                f"{cell(found['review']):>8s}"
            )
    calls = [
        call for record in records if record["status"] != "api_error" for call in record["calls"]
    ]
    lines.append(
        f"{len(decisions(records))} reviews, {len(calls)} calls: "
        f"{sum(c['input_tokens'] for c in calls)} input and "
        f"{sum(c['output_tokens'] for c in calls)} output tokens; "
        f"{dict(Counter(record['status'] for record in records))}"
    )
    return "\n".join(lines)


def stored_score(name: str, split: str = "test", order: str = "gap") -> str:
    """The score of a reviewer's cascade on the stored candidates of a split."""
    from runsleuth import candidates as candidate_set

    root = candidate_set.ROOT / split
    decided = json.loads((root / f"v2-{order}.json").read_text(encoding="utf-8"))["candidates"]
    labels = json.loads((root / "labels.json").read_text(encoding="utf-8"))["labels"]
    agents = {c.id for c in candidate_set.candidates(split) if c.origin == "agent"}
    records = matched_screening.read_reviews(ROOT / name)
    table = score(
        {n: r["decision"] for n, r in decided.items()},
        decisions(records),
        {n: entry["label"] for n, entry in labels.items()},
        agents & set(decided),
    )
    return score_text(table, records)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v2, then an LLM reviewer with v2's evidence.")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Ask the reviewer about v2's review set.")
    run.add_argument("--name", required=True, help="The reviewer's name, a directory.")
    run.add_argument("--provider", choices=PROVIDERS, required=True)
    run.add_argument("--model", help="The model, if not the provider's environment variable.")
    run.add_argument("--limit", type=int, help="At most this many calls, as a price probe.")
    scored = commands.add_parser("score", help="The three verifiers against the labels.")
    scored.add_argument("--name", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        items = cascade_items()
        client, model = create_llm_client(args.provider, args.model)
        matched_screening.run_asks(
            items,
            [(name, INPUT, 1) for name in items],
            ROOT / args.name,
            client,
            model,
            provider=args.provider,
            limit=args.limit,
        )
        print()
    print(stored_score(args.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
