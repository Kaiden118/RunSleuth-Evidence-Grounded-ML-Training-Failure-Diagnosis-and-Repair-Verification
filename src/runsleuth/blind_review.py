"""The blind review that audits the hidden tests' labels.

Labels come from the hidden tests alone; two annotators review the candidates' code to
check them, and change none (evaluations/freeze.json). A label belongs to a project's
code, so the review is of every distinct project a split's candidates hand in (units):
an evidence-removed candidate hands in its parent's, and candidates whose modules agree
hand in one. Settled with the user on 2026-10-26:

- Rounds. A pilot of ten development units, five wrong and five correct, measures how
  long an item takes and lets the annotators practise; the test round is every unit of
  the test split, each reviewed by both annotators.
- Packets. An annotator sees the task statement of each dataset without the agent's
  instructions, the reference project, and for every item its changes against the
  reference and its full modules. Items carry a neutral id in an order of their own for
  each annotator; nothing names the candidate, its model, the version it started from,
  its label or any verifier's decision. The key that maps ids to candidates stays out
  of the packets, and packets are written once.
- Answers. One row per item: correct, wrong or unsure; for a wrong item the numbers of
  the requirements it breaks; one sentence of reason; the minutes it took.
- Analysis. Agreement between the annotators before any discussion, raw and as Cohen's
  kappa, with unsure as its own category and again without the items either called
  unsure; then the annotators settle their disagreements together (consensus), and the
  consensus is compared with the labels, by unit and weighted by candidates. A label
  the consensus disputes is checked and reported with its cause; it is not changed.
- Annotators. The second annotator is blind. The author reviews as well and knows the
  tasks, which the report states.

    python -m runsleuth.blind_review packets --round pilot --annotators author second
    python -m runsleuth.blind_review packets --round test --annotators author second
    python -m runsleuth.blind_review score --round test --answers A.csv B.csv \\
        [--consensus C.csv]
"""

import argparse
import csv
import difflib
import hashlib
import json
import random
import secrets
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from runsleuth import candidates as candidate_set
from runsleuth import pipeline_tasks
from runsleuth.pipeline_tasks import HEALTHY

ROOT = Path("artifacts/blind_review")
ROUNDS = {"pilot": "development", "test": "test"}
PILOT = {"wrong": 5, "correct": 5}
VERDICTS = ("correct", "wrong", "unsure")
DATASETS = {"camelyon_pipeline": "camelyon17", "ham10000_pipeline": "ham10000"}
COLUMNS = ("id", "dataset", "verdict", "violated", "reason", "minutes")
INSTRUCTIONS = """\
# Blind review

Each item is a small PyTorch project that fine-tunes a ResNet18 on medical images. Decide
whether it does everything its task statement requires.

1. Read the statement of the item's dataset (`statement_<dataset>.md`) and the reference
   project (`reference/<dataset>/`), which meets every requirement.
2. For each item in `items/`, read its changes against the reference and, where needed,
   its full modules. The reference is one correct way to meet the statement, not the
   only one: a change that breaks no requirement leaves the item correct.
3. Fill one row of `answers.csv` per item:
   - `verdict`: `correct`, `wrong` or `unsure`;
   - `violated`: for a wrong item, the numbers of the requirements it breaks, separated
     by `;` (for example `1;5`);
   - `reason`: one sentence;
   - `minutes`: how long the item took.

Judge from the code alone: do not run it or search for it. Do not discuss the items
with the other annotator until both of you have finished.
"""


def _write(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _sources_sha256(family: str, sources: dict[str, str]) -> str:
    return hashlib.sha256(
        json.dumps({"family": family, "sources": sources}, sort_keys=True).encode()
    ).hexdigest()


def units(split: str, root: Path = candidate_set.ROOT) -> list[dict[str, Any]]:
    """The distinct projects a split's candidates hand in, with their label and the
    candidates that hand each in; an evidence-removed candidate counts with its parent."""
    labels = json.loads((Path(root) / split / "labels.json").read_text(encoding="utf-8"))
    labels = labels["labels"]
    planned = candidate_set.candidates(split)
    found: dict[str, dict[str, Any]] = {}
    unit_of = {}
    for candidate in planned:
        if candidate.kind == "evidence_removed":
            continue
        sources = candidate_set.candidate_sources(candidate, root)
        if sources is None or candidate.id not in labels:
            continue
        digest = _sources_sha256(candidate.family, sources)
        unit = found.setdefault(
            digest,
            {
                "sha256": digest,
                "family": candidate.family,
                "sources": sources,
                "label": labels[candidate.id]["label"],
                "candidates": [],
            },
        )
        if unit["label"] != labels[candidate.id]["label"]:
            raise ValueError(f"Candidates with the same code have other labels: {digest}")
        unit["candidates"].append(candidate.id)
        unit_of[candidate.id] = digest
    for candidate in planned:
        if candidate.kind == "evidence_removed" and candidate.parent in unit_of:
            found[unit_of[candidate.parent]]["candidates"].append(candidate.id)
    return sorted(found.values(), key=lambda unit: unit["sha256"])


def statement(family: str) -> str:
    """The task statement as the annotators read it: without the agent's instructions."""
    text = pipeline_tasks.family(family).statement
    return text.split("\n## Your job", 1)[0].rstrip() + "\n"


def item_text(item_id: str, unit: dict[str, Any]) -> str:
    """An item: its changes against the reference project and its full modules."""
    dataset = DATASETS[unit["family"]]
    reference = pipeline_tasks.version_sources(HEALTHY, unit["family"])
    diff = "".join(
        "".join(
            difflib.unified_diff(
                reference[module].splitlines(keepends=True),
                unit["sources"][module].splitlines(keepends=True),
                f"reference/{module}.py",
                f"{item_id}/{module}.py",
            )
        )
        for module in reference
    )
    changes = f"```diff\n{diff}```" if diff else "No changes against the reference."
    modules = "\n\n".join(
        f"### {module}.py\n\n```python\n{unit['sources'][module].rstrip()}\n```"
        for module in reference
    )
    return (
        f"# Item {item_id}\n\n"
        f"Dataset: {dataset} (`statement_{dataset}.md`, `reference/{dataset}/`)\n\n"
        f"## Changes against the reference\n\n{changes}\n\n## Modules\n\n{modules}\n"
    )


def write_packets(
    round_name: str,
    annotators: list[str],
    *,
    root: Path = ROOT,
    candidates_root: Path = candidate_set.ROOT,
    seed: int | None = None,
) -> dict[str, Any]:
    """Write one packet per annotator and the key, once per round."""
    directory = Path(root) / round_name
    if directory.exists():
        raise FileExistsError(f"{directory} holds this round's packets already")
    if len(set(annotators)) != len(annotators) or len(annotators) < 2:
        raise ValueError("Two or more annotators, each with a name of their own")
    seed = secrets.randbelow(2**31) if seed is None else seed
    generator = random.Random(seed)
    chosen = units(ROUNDS[round_name], candidates_root)
    if round_name == "pilot":
        chosen = [
            unit
            for label, count in PILOT.items()
            for unit in generator.sample([u for u in chosen if u["label"] == label], count)
        ]
    generator.shuffle(chosen)
    prefix = "P" if round_name == "pilot" else "R"
    items = {f"{prefix}{number:02d}": unit for number, unit in enumerate(chosen, 1)}
    for annotator in annotators:
        packet = directory / annotator
        _write(packet / "INSTRUCTIONS.md", INSTRUCTIONS)
        for family, dataset in DATASETS.items():
            _write(packet / f"statement_{dataset}.md", statement(family))
            for module, source in pipeline_tasks.version_sources(HEALTHY, family).items():
                _write(packet / "reference" / dataset / f"{module}.py", source)
        order = list(items)
        random.Random(f"{seed}/{annotator}").shuffle(order)
        for item_id in order:
            _write(packet / "items" / f"{item_id}.md", item_text(item_id, items[item_id]))
        # With a byte-order mark, so that Excel opens and saves the sheet as UTF-8.
        with (packet / "answers.csv").open("w", encoding="utf-8-sig", newline="") as sheet:
            writer = csv.writer(sheet)
            writer.writerow(COLUMNS)
            for item_id in order:
                writer.writerow([item_id, DATASETS[items[item_id]["family"]], "", "", "", ""])
    key = {
        "schema_version": 1,
        "task": "blind_review_key",
        "round": round_name,
        "split": ROUNDS[round_name],
        "written_on": date.today().isoformat(),
        "seed": seed,
        "annotators": annotators,
        "items": {
            item_id: {key: unit[key] for key in ("sha256", "family", "label", "candidates")}
            for item_id, unit in items.items()
        },
    }
    _write(directory / "key.json", json.dumps(key, indent=2) + "\n")
    return key


def read_answers(path: Path, key: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """An annotator's rows by item; every item of the round answered once. A sheet
    saved by Excel on a Chinese Windows may be in GBK."""
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = Path(path).read_text(encoding="gbk")
    rows = list(csv.DictReader(text.splitlines()))
    answers = {}
    for row in rows:
        verdict = (row.get("verdict") or "").strip().lower()
        if verdict not in VERDICTS:
            raise ValueError(f"{path}: item {row.get('id')} has verdict {verdict!r}")
        if row["id"] in answers:
            raise ValueError(f"{path}: item {row['id']} is answered twice")
        violated = [part.strip() for part in (row.get("violated") or "").split(";")]
        minutes = (row.get("minutes") or "").strip()
        answers[row["id"]] = {
            "verdict": verdict,
            "violated": [int(part) for part in violated if part],
            "reason": (row.get("reason") or "").strip(),
            "minutes": float(minutes) if minutes else None,
        }
    missing = sorted(set(key["items"]) - set(answers))
    unknown = sorted(set(answers) - set(key["items"]))
    if missing or unknown:
        raise ValueError(f"{path}: items missing {missing}, unknown {unknown}")
    return answers


def kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Cohen's kappa of two annotators' verdicts; None when chance agreement is total."""
    if not pairs:
        return None
    observed = sum(a == b for a, b in pairs) / len(pairs)
    first, second = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(first[v] * second[v] for v in VERDICTS) / len(pairs) ** 2
    return None if expected == 1 else round((observed - expected) / (1 - expected), 4)


def agreement(first: dict, second: dict) -> dict[str, Any]:
    """How far two annotators agree, before they discuss anything."""
    pairs = [(first[i]["verdict"], second[i]["verdict"]) for i in sorted(first)]
    sure = [(a, b) for a, b in pairs if "unsure" not in (a, b)]
    return {
        "items": len(pairs),
        "agreed": sum(a == b for a, b in pairs),
        "kappa": kappa(pairs),
        "items_without_unsure": len(sure),
        "agreed_without_unsure": sum(a == b for a, b in sure),
        "kappa_without_unsure": kappa(sure),
        "table": {f"{a} / {b}": count for (a, b), count in sorted(Counter(pairs).items())},
        "disagreements": [i for i in sorted(first) if first[i]["verdict"] != second[i]["verdict"]],
    }


def against_labels(consensus: dict, key: dict[str, Any]) -> dict[str, Any]:
    """The settled verdicts against the hidden tests' labels, by unit and weighted by the
    candidates that hand each unit in."""
    by_unit: Counter = Counter()
    by_candidate: Counter = Counter()
    disputed = []
    for item_id, item in sorted(key["items"].items()):
        verdict = consensus[item_id]["verdict"]
        by_unit[(item["label"], verdict)] += 1
        by_candidate[(item["label"], verdict)] += len(item["candidates"])
        if verdict != item["label"]:
            disputed.append(
                {
                    "id": item_id,
                    "label": item["label"],
                    "verdict": verdict,
                    "reason": consensus[item_id]["reason"],
                    "candidates": len(item["candidates"]),
                }
            )
    return {
        "units": {f"{label} / {verdict}": n for (label, verdict), n in sorted(by_unit.items())},
        "candidates": {
            f"{label} / {verdict}": n for (label, verdict), n in sorted(by_candidate.items())
        },
        "agreed_units": sum(n for (label, verdict), n in by_unit.items() if label == verdict),
        "agreed_candidates": sum(
            n for (label, verdict), n in by_candidate.items() if label == verdict
        ),
        "disputed": disputed,
    }


def score(
    round_name: str,
    answers: list[Path],
    consensus: Path | None = None,
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Agreement of every pair of annotators and, once settled, the consensus against
    the labels; written to the round's report.json."""
    directory = Path(root) / round_name
    key = json.loads((directory / "key.json").read_text(encoding="utf-8"))
    # A sheet is named by its file, or by its folder when it kept the name answers.csv.
    names = [
        Path(p).parent.name if Path(p).name == "answers.csv" else Path(p).stem for p in answers
    ]
    if len(set(names)) != len(names):
        raise ValueError(f"Two answer sheets share a name: {names}")
    read = {name: read_answers(path, key) for name, path in zip(names, answers, strict=True)}
    minutes = [
        answer["minutes"]
        for rows in read.values()
        for answer in rows.values()
        if answer["minutes"] is not None
    ]
    report: dict[str, Any] = {
        "schema_version": 1,
        "task": "blind_review_report",
        "round": round_name,
        "items": len(key["items"]),
        "minutes_per_item": round(sum(minutes) / len(minutes), 1) if minutes else None,
        "agreement": {
            f"{a} / {b}": agreement(read[a], read[b])
            for n, a in enumerate(names)
            for b in names[n + 1 :]
        },
    }
    if consensus is not None:
        report["labels"] = against_labels(read_answers(consensus, key), key)
    _write(directory / "report.json", json.dumps(report, indent=2) + "\n")
    return report


def report_text(report: dict[str, Any]) -> str:
    lines = [f"{report['round']}: {report['items']} items, {report['minutes_per_item']} min each"]
    for pair, found in report["agreement"].items():
        lines.append(
            f"{pair}: agreed {found['agreed']}/{found['items']}, kappa {found['kappa']}; "
            f"without unsure {found['agreed_without_unsure']}/{found['items_without_unsure']}, "
            f"kappa {found['kappa_without_unsure']}; disagreements {found['disagreements']}"
        )
    if "labels" in report:
        found = report["labels"]
        lines.append(f"consensus against labels, by unit: {found['units']}")
        lines.append(f"weighted by candidates: {found['candidates']}")
        for item in found["disputed"]:
            lines.append(f"  {item['id']}: label {item['label']}, consensus {item['verdict']}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="The blind review of the labels.")
    commands = parser.add_subparsers(dest="command", required=True)
    written = commands.add_parser("packets", help="Write the round's packets and key once.")
    written.add_argument("--annotators", nargs="+", required=True)
    scored = commands.add_parser("score", help="Agreement, and the consensus against labels.")
    scored.add_argument("--answers", nargs="+", required=True, type=Path)
    scored.add_argument("--consensus", type=Path)
    for command in (written, scored):
        command.add_argument("--round", choices=tuple(ROUNDS), required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "packets":
        key = write_packets(args.round, args.annotators)
        print(f"{len(key['items'])} items for {args.annotators}: {ROOT / args.round}")
    else:
        print(report_text(score(args.round, args.answers, args.consensus)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
