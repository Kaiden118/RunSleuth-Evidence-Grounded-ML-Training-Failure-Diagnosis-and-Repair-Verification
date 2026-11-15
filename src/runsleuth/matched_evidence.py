"""The evidence frozen v2 decided a stored candidate on, as text another reader can be given.

v2 decided every stored candidate from the record of its run, the healthy project's run
on the same seed, and the probe runs it chose, all kept under
artifacts/candidates/<split> (runs.json, v2_runs, v2_probes). This module

- replays v2 on a candidate from those stored runs alone: a probe run that is not
  stored is refused, so nothing is trained;
- keeps of every run report the fields v2 reads (project) and names every state hash
  briefly, the same hash by the same name across the candidate's runs, since v2
  compares hashes only for equality (relabel);
- writes the runs as an observation packet: a heading per run that says which run it
  is, then the run's fields as JSON. The packet holds observations only: no part
  status, finding, check name or decision of v2's.

A packet is checked by a round trip: parsed back and given to frozen v2 in place of the
stored runs, it must give every part the status and the candidate the decision that v2
gave from the stored runs. A packet that passes holds everything v2 decided on.

    python -m runsleuth.matched_evidence check --split test
    python -m runsleuth.matched_evidence packet --split test --id <candidate id>
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from runsleuth import verifier_v2

# The fields of a run report v2 reads (verifier_v2, agent_verification.compare_trajectories):
# True keeps a field whole, a dict keeps those of its fields, a list keeps those fields
# of every item, "*" those fields of every value.
_GROUP = {"min_tensors_with_gradient": True, "parameter_tensors": True}
_AUDIT = {
    "missing_trainable_names": True,
    "duplicate_parameter_occurrences": True,
    "foreign_parameter_tensors": True,
}
_STATISTICS = {"channel_mean": True, "channel_std": True}
READ = {
    "status": True,
    "problems": True,
    "error": True,
    "head_output_units": True,
    "optimizer": True,
    "trainability": {"*": {"frozen_names": True}},
    "final_trainability": {"*": {"frozen_names": True}},
    "optimizer_audit": _AUDIT,
    "final_optimizer_audit": _AUDIT,
    "input_statistics": {"train": _STATISTICS, "id_eval": _STATISTICS},
    "final_metrics": {"id_validation_accuracy": True, "id_validation_loss": True},
    "parameter_group_epochs": [
        {
            "learning_rate": True,
            "optimizer_steps": True,
            "training_forwards": True,
            "eval_mode_training_forwards": True,
            "project_metrics": True,
            "head": _GROUP,
            "backbone": _GROUP,
        }
    ],
    "pipeline": {
        "epochs": [
            {
                "epoch": True,
                "attempt": True,
                "evaluation": True,
                "reads": True,
                "distinct": True,
                "reads_by_group": True,
                "loss_targets": True,
                "loss_targets_matching_read_labels": True,
            }
        ],
        "delivery": {"same_as": True},
        "training_reads": {"candidates": True, "candidates_by_group": True},
    },
    "workspace": {
        "seed": True,
        "epochs": True,
        "max_train_batches": True,
        "hashed_steps": True,
        "interrupt_after_epochs": True,
        "initial_state_sha256": True,
        "step_state_sha256": True,
        "epoch_state_sha256": True,
        "final_state_sha256": True,
    },
}
STATE_FIELDS = (
    "initial_state_sha256",
    "step_state_sha256",
    "epoch_state_sha256",
    "final_state_sha256",
)
_HEADING = re.compile(r"^## (candidate|reference|uninterrupted|repeat|delivery-\d+): ", re.M)
_BLOCK = re.compile(r"```json\n(.*?)\n```", re.S)
LEGEND = """\
The runs below are records of training runs, one per heading. Their fields:
- workspace: the run's seed, its epochs in all, the epoch after which it was stopped
  and restarted (interrupt_after_epochs; null if it was not), and the network's state
  at the start, after the first optimizer steps, after every epoch and at the end. A
  state is given by a name: equal names are identical weights, in every run below.
- parameter_group_epochs: one entry per epoch trained, in order: the learning rate the
  epoch trained with (first, last, how often it changed), optimizer steps, training
  forwards, training forwards in evaluation mode, what the project reported with
  run.log (project_metrics), and per parameter group how many tensors it has and how
  many had a gradient at least.
- pipeline.epochs: per epoch trained, the attempt (1 before a restart, 2 after), the
  training samples read and how many of them were distinct, reads per group (hospital or
  source), loss targets and how many of them were the labels of the samples read, and,
  where the harness evaluated the epoch's weights itself, its loss and accuracy
  (evaluation).
- pipeline.delivery.same_as: the epochs whose weights the delivered weights equal.
- final_metrics: the harness's own loss and accuracy of the delivered weights.
- trainability, optimizer_audit: parameters that were frozen, or that the optimizer
  did not hold exactly once. input_statistics: per channel mean and deviation of the
  network's inputs in training and validation.
"""


def project(value: Any, keep: Any = None) -> Any:
    """The fields of a run report that v2 reads; a field the report lacks stays absent."""
    keep = READ if keep is None else keep
    if keep is True or value is None:
        return copy.deepcopy(value)
    if isinstance(keep, list):
        return [project(item, keep[0]) for item in value] if isinstance(value, list) else value
    if not isinstance(value, dict):
        return copy.deepcopy(value)
    if "*" in keep:
        return {name: project(item, keep["*"]) for name, item in value.items()}
    return {name: project(value[name], sub) for name, sub in keep.items() if name in value}


def relabel(reports: Iterable[dict[str, Any] | None]) -> list[dict[str, Any] | None]:
    """Copies of the reports with every state hash named S1, S2, ... in the order the
    hashes first appear, the same hash by the same name in every report."""
    names: dict[str, str] = {}

    def name(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        if value not in names:
            names[value] = f"S{len(names) + 1}"
        return names[value]

    renamed = []
    for report in reports:
        report = copy.deepcopy(report)
        workspace = (report or {}).get("workspace")
        if isinstance(workspace, dict):
            for field in STATE_FIELDS:
                if isinstance(workspace.get(field), list):
                    workspace[field] = [name(value) for value in workspace[field]]
                elif field in workspace:
                    workspace[field] = name(workspace[field])
        renamed.append(report)
    return renamed


class Recorded(verifier_v2.Probes):
    """v2's probes, keeping every run v2 asked for and the report it read, by role."""

    def __init__(self, train, probe_train, settings) -> None:
        super().__init__(train, probe_train, settings)
        self.runs: dict[str, dict[str, Any] | None] = {}

    def run(self, kind, sources, family, seed, epochs, tag=""):
        report, cost = super().run(kind, sources, family, seed, epochs, tag)
        self.runs[role(kind, seed, tag)] = report
        return report, cost


class Replayed(verifier_v2.Probes):
    """v2's probes answered from given reports by role; a run not given is an error."""

    def __init__(self, runs: dict[str, dict[str, Any] | None], settings) -> None:
        super().__init__(None, None, settings)
        self.runs = runs

    def run(self, kind, sources, family, seed, epochs, tag=""):
        name = role(kind, seed, tag)
        if name not in self.runs:
            raise KeyError(f"v2 asked for the {name} run, which the evidence lacks")
        return copy.deepcopy(self.runs[name]), {"runs": 1, "steps": 0, "seconds": 0.0}


def role(kind: str, seed: int, tag: str = "") -> str:
    """The name of a probe run: uninterrupted, repeat (its determinism control) or
    delivery-<probe seed>."""
    if kind == "uninterrupted":
        return "repeat" if tag else "uninterrupted"
    return f"delivery-{seed}"


def describe(name: str, report: dict[str, Any] | None, settings: dict[str, Any]) -> str:
    """What a run is, in words that say nothing of what it shows."""
    if name == "candidate":
        workspace = (report or {}).get("workspace") or {}
        stop = workspace.get("interrupt_after_epochs")
        restart = f", stopped after epoch {stop} and restarted" if stop else ""
        return (
            f"the candidate's own run, seed {workspace.get('seed')}, "
            f"{workspace.get('epochs')} epochs{restart}"
        )
    if name == "reference":
        return "the healthy project's run with the same seed, epochs and stop"
    if name == "uninterrupted":
        return (
            "the candidate run again with the same seed and epochs, not stopped, the harness "
            "evaluating every epoch's weights itself"
        )
    if name == "repeat":
        return "the same uninterrupted run made once more"
    delivery = settings["delivery"]
    return (
        f"the candidate on probe seed {name.removeprefix('delivery-')}: "
        f"{delivery['epochs']} epochs on a smaller training set, stopped after epoch "
        f"{delivery['interrupt_after_epochs']} and restarted, the harness evaluating every "
        f"epoch's weights itself on {delivery['max_eval_samples']} validation samples"
    )


def packet(
    run: dict[str, Any] | None,
    reference: dict[str, Any] | None,
    probes: dict[str, dict[str, Any] | None],
    settings: dict[str, Any],
) -> str:
    """The observation packet of a candidate: the legend, then every run."""
    names = ["candidate", "reference", *probes]
    reports = relabel(
        project(report) if report is not None else None
        for report in (run, reference, *probes.values())
    )
    parts = [LEGEND]
    for name, report in zip(names, reports, strict=True):
        parts.append(
            f"## {name}: {describe(name, report, settings)}\n\n"
            f"```json\n{json.dumps(report, indent=1, sort_keys=True)}\n```\n"
        )
    return "\n".join(parts)


def parse_packet(text: str) -> dict[str, dict[str, Any] | None]:
    """The runs of a packet by name, as v2 is given them."""
    headings = list(_HEADING.finditer(text))
    runs = {}
    for number, heading in enumerate(headings):
        end = headings[number + 1].start() if number + 1 < len(headings) else len(text)
        block = _BLOCK.search(text, heading.end(), end)
        if block is None:
            raise ValueError(f"the {heading.group(1)} run has no JSON block")
        runs[heading.group(1)] = json.loads(block.group(1))
    if "candidate" not in runs or "reference" not in runs:
        raise ValueError("a packet needs the candidate's run and the reference's")
    return runs


def summary(result: dict[str, Any]) -> dict[str, Any]:
    """What a round trip must keep: the decision, every part's status, the checks run."""
    return {
        "decision": result["decision"],
        "parts": {part: body["status"] for part, body in result["parts"].items()},
        "checks": result["checks"],
    }


def round_trip(
    sources: dict[str, str],
    family: str,
    seed: int,
    run: dict[str, Any] | None,
    reference: dict[str, Any],
    probes: Recorded,
    *,
    order: str = "gap",
) -> dict[str, Any]:
    """v2 on the stored runs, then on the runs of their packet; the packet and both
    results' summaries."""
    first = verifier_v2.verify(sources, family, seed, run, reference, probes, order=order)
    text = packet(run, reference, probes.runs, probes.settings)
    parsed = parse_packet(text)
    replayed = Replayed(
        {name: report for name, report in parsed.items() if name not in ("candidate", "reference")},
        probes.settings,
    )
    second = verifier_v2.verify(
        sources, family, seed, parsed["candidate"], parsed["reference"], replayed, order=order
    )
    return {"packet": text, "stored": summary(first), "packet_result": summary(second)}


def _refused(*_args, **_kwargs) -> dict[str, Any]:
    raise RuntimeError("this run is not stored; the replay trains nothing")


def stored_round_trips(
    split: str, ids: Iterable[str] | None = None, order: str = "gap"
) -> dict[str, dict[str, Any]]:
    """The round trip of every stored candidate of a split, or of those named, as v2's
    command reads them (verifier_v2.main), with every probe run taken from the store."""
    from runsleuth import candidates as candidate_set
    from runsleuth.challenge_set import remove_evidence

    settings = verifier_v2.load_settings()
    root = candidate_set.ROOT / split
    runs = json.loads((root / "runs.json").read_text(encoding="utf-8"))
    data = {family: (source, None) for family, (source, _) in candidate_set.DATA.items()}

    def runner(folder: str):
        return candidate_set.Runner(root / folder, data=data, run=_refused).train

    def read(outcome: dict | None) -> dict | None:
        if outcome is None:
            return None
        if outcome.get("report"):
            return json.loads(Path(outcome["report"]).read_text(encoding="utf-8"))
        return {key: outcome.get(key) for key in ("status", "problems", "error")}

    wanted = None if ids is None else set(ids)
    results = {}
    for candidate in candidate_set.candidates(split):
        if wanted is not None and candidate.id not in wanted:
            continue
        sources = candidate_set.candidate_sources(candidate)
        outcome = runs["runs"].get(candidate.parent or candidate.id)
        if sources is None or outcome is None:
            continue
        run = read(outcome)
        if candidate.kind == "evidence_removed" and run is not None:
            run = remove_evidence(run, candidate.transform)
        reference = read(runs["references"][f"{candidate.family}/{candidate.seed}"])
        probes = Recorded(runner("v2_runs"), runner("v2_probes"), settings)
        results[candidate.id] = round_trip(
            sources, candidate.family, candidate.seed, run, reference, probes, order=order
        )
    return results


def check_text(results: dict[str, dict[str, Any]], decided: dict[str, Any]) -> str:
    """How many round trips kept v2's results, and against the stored decisions."""
    lines = []
    same_as_stored = same_round_trip = 0
    sizes = []
    for name, result in results.items():
        stored = decided.get(name)
        replay_ok = stored is not None and summary(stored) == result["stored"]
        trip_ok = result["stored"] == result["packet_result"]
        same_as_stored += replay_ok
        same_round_trip += trip_ok
        sizes.append(len(result["packet"]))
        if not (replay_ok and trip_ok):
            lines.append(
                f"{name}: replay {'=' if replay_ok else '!='} stored, packet "
                f"{'=' if trip_ok else '!='} replay"
            )
    lines.append(
        f"{len(results)} candidates: the replay gives the stored result for {same_as_stored}, "
        f"the packet the replay's for {same_round_trip}; packets of "
        f"{min(sizes, default=0)} to {max(sizes, default=0)} characters"
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v2's evidence as an observation packet.")
    commands = parser.add_subparsers(dest="command", required=True)
    checked = commands.add_parser("check", help="Round-trip the packets of stored candidates.")
    shown = commands.add_parser("packet", help="Print one stored candidate's packet.")
    for command in (checked, shown):
        command.add_argument("--split", default="test")
        command.add_argument("--order", choices=verifier_v2.ORDERS, default="gap")
    checked.add_argument("--id", nargs="*", dest="ids", help="Only these candidates.")
    shown.add_argument("--id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from runsleuth import candidates as candidate_set

    if args.command == "packet":
        (result,) = stored_round_trips(args.split, [args.id], args.order).values()
        print(result["packet"], end="")
        return 0
    results = stored_round_trips(args.split, args.ids, args.order)
    stored = candidate_set.ROOT / args.split / f"v2-{args.order}.json"
    decided = json.loads(stored.read_text(encoding="utf-8"))["candidates"]
    print(check_text(results, decided))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
