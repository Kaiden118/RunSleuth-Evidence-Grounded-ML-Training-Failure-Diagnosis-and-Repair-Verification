"""The verification challenge set: its candidates, their split, origin and label.

A candidate is a project handed in for acceptance: one an agent handed in at the end of
an episode, a version of pipeline_tasks handed in as it is, a partial repair written for
a test fault, or the run of a version with part of its evidence removed. A verifier is
measured on whether it accepts the correct candidates and rejects the wrong ones.

The rules, settled with the user on 2026-10-22 before any verifier was developed on
the set:

- Label. The frozen hidden tests (pipeline_contract) alone decide: a candidate that
  passes every clause is correct, one that fails any is wrong. A candidate the harness
  refuses, or whose contract run crashes or times out, is wrong and also not_runnable,
  so that results can be given without those as well. Clauses that pass with a note
  (a part the contract could not test) leave the label as it is and mark it partially
  tested. The agent's own verdict is kept beside the label and never changes it, nor
  does the blind human review, which audits the labels.
- Split. Independence is by fault: everything made from a starting version (agent
  repairs, seeds, evidence-removed runs) belongs to that version's split. The healthy
  project is development material. Contract cases certify the hidden tests and belong
  to no split.
- Candidates. Agents start from every fault and correct variant of a split, the
  control excepted: in the test split five models (MODELS), three repetitions each,
  one test seed per repetition; in the development split Gemini Flash-Lite alone on
  seed 7.
  The versions of a split are handed in as they are as well; in the test split, so
  are six partial repairs, written before any verifier was developed, and two
  evidence-removed runs of every version. Partial repairs and evidence-removed runs
  are reported apart from agent repairs; the main results rest on the latter.
- Seeds. The three test seeds are drawn at random when the set is frozen, from seeds
  no earlier run has used, and recorded; development runs use seed 7 only.
- Models. Settled on 2026-10-23 after the smoke tests: GPT-6.1 Sol writes candidates
  in place of GPT-6 Luna, and Claude Sonnet 5.5 writes none, so that the two LLM
  reviewers (REVIEWERS), Gemini 4 Argon and Claude Sonnet 5.5, never judge a candidate
  of their own model; only Argon judges some of its maker's (Flash-Lite's), and those
  are scored apart (VENDORS). Claude Opus 5.5 reviewed the development candidates as
  well and judged them as Argon did, at about seven times the cost, so Sonnet took
  its place.

    python -m runsleuth.challenge_set plan --split test
"""

import argparse
import copy
import json
import secrets
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

from runsleuth import pipeline_contract, pipeline_tasks
from runsleuth.agent_study import used_seeds
from runsleuth.heldout import USED_SEEDS
from runsleuth.pipeline_tasks import HEALTHY, SPLITS

FAMILIES = ("camelyon_pipeline", "ham10000_pipeline")
MODELS = (
    "gemini-3.5-flash-lite",
    "gpt-6.1-sol",
    "deepseek-v4.1-flash-1017",
    "muse-spark-1.3",
    "qwen3.8-27b",
)
# The LLM reviewers, a baseline verifier; none of them writes candidates.
REVIEWERS = ("gemini-4-argon", "claude-sonnet-5.5")
# Who made each model, so that a reviewer judging its maker's candidates shows apart.
VENDORS = {
    "gemini-3.5-flash-lite": "google",
    "gpt-6.1-sol": "openai",
    "deepseek-v4.1-flash-1017": "deepseek",
    "muse-spark-1.3": "meta",
    "qwen3.8-27b": "alibaba",
    "gemini-4-argon": "google",
    "claude-sonnet-5.5": "anthropic",
    "claude-opus-5.5": "anthropic",
}
DEVELOPMENT_MODELS = ("gemini-3.5-flash-lite",)
REPETITIONS = 3
DEVELOPMENT_SEED = 7
TEST_SEED_COUNT = 3
SEED_LIMIT = 2**31
SEEDS_FILE = Path("evaluations/challenge_set_seeds.json")
ORIGINS = ("agent", "human", "program_transform")
# The records of a run report's pipeline section that each transform removes.
TRANSFORMS = {
    "no_restart_record": ("attempts", "tracked", "checkpoints"),
    "no_data_reads": ("training_reads", "epochs"),
}
NOT_RUNNABLE = ("refused", "crashed", "timeout")


@dataclass(frozen=True, slots=True)
class PartialRepair:
    """A repair of a test fault that mends part of it, as edits (module, old, new) of
    the fault's project, and the clauses the hidden tests should fail it on."""

    family: str
    fault: str
    summary: str
    edits: tuple[tuple[str, str, str], ...]
    fails: tuple[str, ...]


PARTIAL_REPAIRS = {
    "labels_cut_to_length": PartialRepair(
        "camelyon_pipeline",
        "labels_not_filtered",
        "The label list is cut to as many labels as patches were kept, the first ones "
        "instead of those at the kept indices.",
        (
            (
                "data",
                "        self.labels = list(patches.labels)\n",
                "        self.labels = list(patches.labels)[: len(self.indices)]\n",
            ),
        ),
        ("training_labels",),
    ),
    # Restoring the optimizer in place of the schedule would be the development fault
    # scheduler_not_restored, which a test candidate must not repeat.
    "optimizer_learning_rates_restored": PartialRepair(
        "camelyon_pipeline",
        "optimizer_not_restored",
        "A restart restores the learning rates of the optimizer's parameter groups from "
        "the checkpoint but not its moments, which the schedule's restore already did.",
        (
            (
                "train",
                '        model.load_state_dict(state["model"])\n',
                '        model.load_state_dict(state["model"])\n'
                '        saved = state["optimizer"]["param_groups"]\n'
                "        for group, kept in zip(optimizer.param_groups, saved):\n"
                '            group["lr"] = kept["lr"]\n',
            ),
        ),
        ("restart",),
    ),
    "best_accuracy_restored_ties_to_the_later": PartialRepair(
        "camelyon_pipeline",
        "best_accuracy_not_restored",
        "A restart restores the best accuracy, but an epoch as good as the best replaces "
        "it, so a tie goes to the later epoch.",
        (
            (
                "train",
                '        start = state["epoch"] + 1\n',
                '        start, best_accuracy = state["epoch"] + 1, state["best_accuracy"]\n',
            ),
            (
                "train",
                '        if metrics["validation_accuracy"] > best_accuracy:\n',
                '        if metrics["validation_accuracy"] >= best_accuracy:\n',
            ),
        ),
        ("delivery",),
    ),
    "generator_seeded_by_the_start_epoch": PartialRepair(
        "ham10000_pipeline",
        "shuffle_generator_made_once",
        "The generator is still made once per attempt, now seeded by the run's seed and "
        "the epoch the attempt starts at, so a restarted run draws other batches.",
        (
            (
                "train",
                "    generator = torch.Generator().manual_seed(run.seed)\n",
                "    generator = torch.Generator().manual_seed(1000 * run.seed + start)\n",
            ),
        ),
        ("restart",),
    ),
    "flipped_again_on_return": PartialRepair(
        "ham10000_pipeline",
        "flipped_image_cached",
        "Every image is flipped anew when it is returned, but the cache still holds the "
        "flipped image, so the flips a restarted run draws differ.",
        (
            (
                "data",
                "        return self.cache[key], self.labels[position]\n",
                "        return flip(self.cache[key], self.generator), self.labels[position]\n",
            ),
        ),
        ("restart",),
    ),
    "akiec_in_place_of_bcc": PartialRepair(
        "ham10000_pipeline",
        "akiec_left_out_of_training",
        "akiec is added to the trained malignant diagnoses in place of bcc.",
        (("config", 'MALIGNANT = ("mel", "bcc")', 'MALIGNANT = ("mel", "akiec")'),),
        ("training_labels",),
    ),
}


def partial_sources(name: str) -> dict[str, str]:
    """The modules of a partial repair: its fault's, with its edits."""
    repair = PARTIAL_REPAIRS[name]
    sources = pipeline_tasks.version_sources(repair.fault, repair.family)
    for module, old, new in repair.edits:
        if sources[module].count(old) != 1:
            raise ValueError(f"{name}: every edit must match its module exactly once")
        sources[module] = sources[module].replace(old, new)
    return sources


def split_of(name: str, family: str) -> str:
    """The split of a starting version or partial repair; contract cases have none."""
    if name in PARTIAL_REPAIRS:
        return split_of(PARTIAL_REPAIRS[name].fault, PARTIAL_REPAIRS[name].family)
    if name == HEALTHY:
        return "development"
    if name in pipeline_contract.CASES[family]:
        raise ValueError(f"{name} is a contract case: it certifies the hidden tests, in no split")
    return pipeline_tasks.version(name, family).split


def starting_points(split: str) -> list[tuple[str, str]]:
    """(family, version) for every project agents start from in a split: its faults and
    correct variants, not its control."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; choose from {', '.join(SPLITS)}")
    return [
        (family, name)
        for family in FAMILIES
        for name, item in pipeline_tasks.family(family).versions.items()
        if item.split == split and not item.control
    ]


@dataclass(frozen=True, slots=True)
class Candidate:
    """A planned candidate. kind is repair (by an agent), version (as it is),
    partial_repair or evidence_removed; seed is the training seed of its run, None
    while the test seeds are not drawn; parent is the candidate a transform removes
    evidence from."""

    id: str
    family: str
    task: str
    start: str
    split: str
    origin: str
    kind: str
    model: str | None = None
    repetition: int | None = None
    seed: int | None = None
    transform: str | None = None
    parent: str | None = None


def _task(family: str, name: str) -> str:
    if name in PARTIAL_REPAIRS:
        name = PARTIAL_REPAIRS[name].fault
    return pipeline_tasks.version(name, family).task


def plan(split: str, seeds: tuple[int, ...] | None = None) -> list[Candidate]:
    """Every candidate of a split. seeds are the test seeds, one per repetition."""
    if split == "development":
        models, seeds = DEVELOPMENT_MODELS, (DEVELOPMENT_SEED,) * REPETITIONS
    else:
        models = MODELS
        if seeds is not None and len(seeds) != REPETITIONS:
            raise ValueError(f"The test split takes {REPETITIONS} seeds, one per repetition")
        seeds = seeds or (None,) * REPETITIONS
    candidates = [
        Candidate(
            f"{family}/{name}/{model}/r{repetition}",
            family,
            _task(family, name),
            name,
            split,
            "agent",
            "repair",
            model=model,
            repetition=repetition,
            seed=seeds[repetition - 1],
        )
        for family, name in starting_points(split)
        for model in models
        for repetition in range(1, REPETITIONS + 1)
    ]
    versions = [
        Candidate(
            f"{family}/{name}/as_is",
            family,
            item.task,
            name,
            split,
            "human",
            "version",
            seed=seeds[0],
        )
        for family in FAMILIES
        for name, item in pipeline_tasks.family(family).versions.items()
        if item.split == split
    ]
    candidates.extend(versions)
    if split == "test":
        candidates.extend(
            Candidate(
                f"{repair.family}/{name}",
                repair.family,
                _task(repair.family, name),
                name,
                split,
                "human",
                "partial_repair",
                seed=seeds[0],
            )
            for name, repair in PARTIAL_REPAIRS.items()
        )
        candidates.extend(
            Candidate(
                f"{version.id}/{transform}",
                version.family,
                version.task,
                version.start,
                split,
                "program_transform",
                "evidence_removed",
                seed=version.seed,
                transform=transform,
                parent=version.id,
            )
            for version in versions
            for transform in TRANSFORMS
        )
    return candidates


def remove_evidence(report: dict[str, Any], transform: str) -> dict[str, Any]:
    """A copy of a run report without the pipeline records a transform names."""
    if transform not in TRANSFORMS:
        raise ValueError(f"Unknown transform {transform!r}; choose from {', '.join(TRANSFORMS)}")
    removed = copy.deepcopy(report)
    for key in TRANSFORMS[transform]:
        (removed.get("pipeline") or {}).pop(key, None)
    return removed


def label(result: dict[str, Any], family: str) -> dict[str, Any]:
    """The label a contract result gives a candidate of a family.

    Refuses a result of another contract version or family state, so that labels of
    different versions are never mixed.
    """
    current = (pipeline_contract.CONTRACT_VERSION, pipeline_tasks.family_sha256(family))
    found = (result.get("contract_version"), result.get("family_sha256"))
    if result.get("family") != family or found != current:
        raise ValueError(
            f"The result is of {result.get('family')} at contract version {found[0]} and "
            f"family {str(found[1])[:12]}; labels come from {family} at version "
            f"{current[0]} and family {current[1][:12]}"
        )
    status = result["status"]
    if status not in ("passed", "failed", *NOT_RUNNABLE):
        raise ValueError(f"Unknown contract status {status!r}")
    outcome = "not_runnable" if status in NOT_RUNNABLE else status
    notes = [
        f"{clause}: {checked['detail']}"
        for clause, checked in result.get("clauses", {}).items()
        if checked["passed"] and checked["detail"]
    ]
    return {
        "label": "correct" if outcome == "passed" else "wrong",
        "outcome": outcome,
        "contract_status": status,
        "failed_clauses": list(result["failed"]) if outcome == "failed" else [],
        "problems": list(result.get("problems", [])),
        "partially_tested": bool(notes),
        "notes": notes,
        "contract_version": current[0],
        "family_sha256": current[1],
    }


def _commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def draw_test_seeds(path: Path = SEEDS_FILE) -> list[int]:
    """Draw the test seeds at random from seeds no earlier run used, and record them.

    Refuses to draw again once they are recorded, and to draw before the freeze is
    committed with no frozen file changed since.
    """
    from runsleuth import freeze

    if Path(path).exists():
        raise FileExistsError(f"{path} already records the test seeds")
    freeze.check()
    if not freeze.committed(freeze.FREEZE_FILE):
        raise RuntimeError(f"Commit {freeze.FREEZE_FILE} before the test seeds are drawn")
    used = set(used_seeds()) | set(USED_SEEDS) | {DEVELOPMENT_SEED}
    seeds: list[int] = []
    while len(seeds) < TEST_SEED_COUNT:
        seed = secrets.randbelow(SEED_LIMIT)
        if seed not in used and seed not in seeds:
            seeds.append(seed)
    record = {
        "schema_version": 1,
        "task": "challenge_set_seeds",
        "seeds": seeds,
        "excluded_seeds": sorted(used),
        "drawn_on": date.today().isoformat(),
        "commit": _commit(),
        "freeze_sha256": freeze._file_sha256(freeze.FREEZE_FILE),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return seeds


def plan_text(candidates: list[Candidate]) -> str:
    """How many candidates of each kind a plan has, by family."""
    kinds = Counter((candidate.kind, candidate.family) for candidate in candidates)
    lines = [f"{len(candidates)} candidates"]
    for kind in ("repair", "version", "partial_repair", "evidence_removed"):
        counts = [kinds[(kind, family)] for family in FAMILIES]
        if any(counts):
            by_family = ", ".join(
                f"{family} {count}" for family, count in zip(FAMILIES, counts, strict=True)
            )
            lines.append(f"  {kind:<17} {sum(counts):>4}  ({by_family})")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="The verification challenge set.")
    commands = parser.add_subparsers(dest="command", required=True)
    shown = commands.add_parser("plan", help="Count, or list as JSON, the candidates of a split.")
    shown.add_argument("--split", choices=SPLITS, required=True)
    shown.add_argument("--json", action="store_true", help="Print every candidate as JSON.")
    commands.add_parser("draw-seeds", help=f"Draw the test seeds once, into {SEEDS_FILE}.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "draw-seeds":
        print(draw_test_seeds())
        return 0
    seeds = None
    if args.split == "test" and SEEDS_FILE.exists():
        seeds = tuple(json.loads(SEEDS_FILE.read_text(encoding="utf-8"))["seeds"])
    candidates = plan(args.split, seeds)
    if args.json:
        print(json.dumps([asdict(candidate) for candidate in candidates], indent=2))
    else:
        print(plan_text(candidates))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
