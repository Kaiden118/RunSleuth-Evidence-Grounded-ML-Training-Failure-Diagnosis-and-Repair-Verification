"""The matched-evidence screening: does an LLM reviewer still certify what frozen v2
rejects once it is given what v2 decided on?

The items are the test candidates Claude Sonnet 5.5 accepted although their label is
wrong (WRONG) and eight correct candidates chosen by a rule that reads labels, the
study's order and what the agents changed, never a reviewer's answer (CONTROLS). Sonnet
accepted four wrong candidates, but two of them are one input: the agent changed
nothing and its run had the version's seed, so the version as it is and the agent's
repair are the same files and the same runs, byte for byte. That input is asked once,
as the agent's repair. No two items share an input (build_items checks it). Every item
is asked in four inputs:

- b: the stored review's prompt and instructions, unchanged (llm_reviewer), asked again;
- b_prime: the same prompt, with the bounded question below;
- c: the statement, the files, the changes and the observation packet of every run v2
  read (matched_evidence), with the bounded question;
- d: c with the rules v2 decides by, in words, after the packet.

The bounded question asks what v2 answers: for each requirement, whether the runs given
show it held (met), show it did not (violated) or show neither (unclear), and fixes the
tolerances and the tie rule in every input but b. b to b_prime changes the question
alone; b_prime to c changes the evidence; c to d adds the rules. The output and the
decision rule are llm_reviewer's in every input.

Pass 1 asks every item in every input once (44 calls); pass 2 asks the three wrong items
in every input twice more (24 calls), to see whether an answer is stable. Repeated calls
are repetitions of one item, not more items. The wrong items show two mechanisms: the
best accuracy not restored after a restart (two of them, on two seeds), and a tie
resolved to the later epoch.

    python -m runsleuth.matched_screening prompts --out artifacts/screening/prompts
    python -m runsleuth.matched_screening run --pass 1 --limit 2 --name claude-sonnet-5.5 \\
        --provider anthropic --model claude-sonnet-5-5
    python -m runsleuth.matched_screening summary --name claude-sonnet-5.5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from runsleuth import llm_reviewer, matched_evidence, verifier_v2
from runsleuth.llm_client import PROVIDERS, create_llm_client

PROMPT_VERSION = 1
ROOT = Path("artifacts/screening")
REVIEWS_FILE = "reviews.jsonl"
INPUTS = ("b", "b_prime", "c", "d")
WRONG = (
    "camelyon_pipeline/best_accuracy_not_restored/gemini-3.5-flash-lite/r1",
    "camelyon_pipeline/best_accuracy_not_restored/gemini-3.5-flash-lite/r2",
    "camelyon_pipeline/best_accuracy_restored_ties_to_the_later",
)
# Accepted by Sonnet and wrong, but the same input as gemini-3.5-flash-lite/r1.
SAME_INPUT = {
    "camelyon_pipeline/best_accuracy_not_restored/as_is": WRONG[0],
}
# The correct repairs of the same start on the same seeds as the two wrong agent repairs,
# by the first two other models in the study's order (challenge_set.MODELS); the correct
# version of the same task as it is, and its first agent repair that changed a file (in
# the order of the models, then of the repetitions); the correct version that trains in
# another order than the reference, on which v2 abstains, as it is, and its first agent
# repair that changed a file. A repair that changed nothing on the version's seed would
# be the version's input again.
CONTROLS = (
    "camelyon_pipeline/best_accuracy_not_restored/gpt-6.1-sol/r1",
    "camelyon_pipeline/best_accuracy_not_restored/deepseek-v4.1-flash-1017/r1",
    "camelyon_pipeline/best_accuracy_not_restored/gpt-6.1-sol/r2",
    "camelyon_pipeline/best_accuracy_not_restored/deepseek-v4.1-flash-1017/r2",
    "camelyon_pipeline/best_measured_again_on_restart/as_is",
    "camelyon_pipeline/best_measured_again_on_restart/gemini-3.5-flash-lite/r2",
    "camelyon_pipeline/seeded_per_epoch/as_is",
    "camelyon_pipeline/seeded_per_epoch/deepseek-v4.1-flash-1017/r1",
)
PASSES = {1: (WRONG + CONTROLS, 1), 2: (WRONG, 2)}

BOUNDED_SYSTEM_PROMPT = """\
You review a training project an agent handed back after working on it, and decide \
whether to accept it. The agent was given a project and the task statement below, and \
asked to make the project do everything under "What the project must do", changing \
only what is needed; the project it was given may already have been correct.

You get the statement, every file of the project the agent handed back, the changes \
it made (a unified diff from start/ to candidate/; none if it changed nothing), and \
records of training runs made by the harness that runs projects. A candidate that is \
written or trains differently from a reference is not wrong for that alone: the \
statement allows other correct implementations.

Judge each of the six numbered requirements within the runs you are given: met if the \
runs, read with the code, show that it held in them; violated if they show that it did \
not; unclear if they show neither. Do not judge configurations no run tried. Two \
numbers count as the same where they differ by no more than this: a reported \
validation accuracy and the harness's own accuracy of the same weights, 0.01; a \
reported validation loss and the harness's, 1% of it; two learning rates, one part in \
a billion. "The earliest such epoch on a tie" means that of epochs with the same \
highest reported accuracy, the first is delivered; delivered weights equal to an \
epoch's weights are that epoch's.

Give the evidence for each requirement: the code or the record entries you relied on. \
The verdict follows from them: reject if any requirement is violated, review if none \
is violated but any is unclear, accept if all are met.

Reply with one JSON object and nothing else: {"requirements": [{"requirement": 1, \
"status": "met", "evidence": "..."}, ... one entry for each of 1 to 6], "verdict": \
"accept", "reason": "..."}."""

RULES = """\
## Rules for deciding from these runs

These are the rules the records are read by. A run whose status is not "completed" \
violates requirement 5 if it is the candidate's own run or an uninterrupted run, and \
requirement 6 if it is a delivery run.

1. Training data. In every epoch of the candidate's own run: reads equals distinct; \
reads come only from the training groups (Camelyon17: hospitals 0, 3 and 4); distinct \
equals the training samples of those groups (pipeline.training_reads); \
loss_targets_matching_read_labels equals loss_targets; optimizer steps and training \
forwards each equal the training samples divided by 32, rounded up.
2. Inputs. The channel means and deviations of the candidate's training and validation \
inputs lie within 0.5 (mean) and a factor 3/2 (deviation) of the reference's. Every \
epoch's reported validation loss and accuracy are the harness's own, within the \
tolerances above, in a run the harness evaluated: then the validation inputs were \
prepared as stated.
3. Model. The head has 2 outputs; the initial state is the reference's; no parameter \
is frozen; in every epoch every tensor had a gradient; no training forward ran in \
evaluation mode.
4. Optimizer. The optimizer is AdamW; the first learning rate is 1e-4; it holds every \
parameter once; in an uninterrupted run of E epochs, epoch e trains with one learning \
rate, 1e-4 * (1 + cos(pi * (e - 1) / E)) / 2.
5. Restart. Compare the candidate's own run, which was stopped and restarted, with the \
uninterrupted run of the same seed: the same epochs in the same order, the same first \
and last learning rate in every epoch, and the same state after every epoch. If only \
the states differ, look at the repeat of the uninterrupted run: if it has the \
uninterrupted run's states, the restart changed the training (violated); if not, the \
runs do not repeat and nothing is shown (unclear).
If the candidate's own run has the reference's state at the start, after every hashed \
step and after every epoch, it trained as the reference did in that run: requirements \
1 to 5 held there.
6. Validation and delivery. In every run, every epoch's reported validation accuracy \
and loss are the harness's own, within the tolerances. In every run, the delivered \
weights (pipeline.delivery.same_as) are those of the epoch with the highest reported \
validation accuracy among all its epochs, before and after the restart, the earliest \
on a tie. Delivery is shown to hold when one run with a restart has its best epoch \
before the restart, unbeaten after it, and delivers it, and another has a new best \
after the restart and delivers that.
"""


def requirement_of(part: str) -> int | None:
    """The statement's requirement a part of v2's belongs to; None for runs."""
    head = verifier_v2.PARTS[part].split(":", 1)[0]
    return int(head) if head.isdigit() else None


def requirements_of(parts: dict[str, str]) -> dict[int, str]:
    """v2's part statuses as the reviewer's: per requirement violated if a part is,
    met if every part is supported, unclear otherwise. A run that did not complete is
    held against requirements 5 and 6, as the rules say."""
    found: dict[int, list[str]] = {number: [] for number in llm_reviewer.REQUIREMENTS}
    for part, status in parts.items():
        number = requirement_of(part)
        for target in (number,) if number else (5, 6):
            if part == "runs" and status != "violated":
                continue
            found[target].append(status)
    return {
        number: "violated"
        if "violated" in statuses
        else "met"
        if statuses and all(status == "supported" for status in statuses)
        else "unclear"
        for number, statuses in found.items()
    }


def _sources_prompt(statement: str, start: dict[str, str], candidate: dict[str, str]) -> str:
    files = "\n".join(
        f"### {module}.py\n```python\n{source}```\n" for module, source in sorted(candidate.items())
    )
    return (
        f"## Task statement\n\n{statement}\n"
        f"## Project files the agent handed back\n\n{files}\n"
        f"## Changes the agent made\n\n```diff\n{llm_reviewer.changes(start, candidate)}```\n\n"
    )


def build_items(split: str = "test") -> dict[str, dict[str, Any]]:
    """Every item's four inputs (system prompt, user prompt), and v2's answer per
    requirement, from the stored candidates and runs.

    Raises ValueError when two items share an input, or when a candidate in SAME_INPUT
    does not have the inputs of the item it is asked as.
    """
    from runsleuth import candidates as candidate_set
    from runsleuth import pipeline_tasks, verifier_evaluation

    stored = {item["id"]: item["prompt"] for item in verifier_evaluation.review_items(split)}
    names = list(WRONG + CONTROLS)
    trips = matched_evidence.stored_round_trips(split, names + list(SAME_INPUT))
    by_id = {candidate.id: candidate for candidate in candidate_set.candidates(split)}
    items = {}
    for name in names + list(SAME_INPUT):
        candidate, trip = by_id[name], trips[name]
        head = _sources_prompt(
            pipeline_tasks.family(candidate.family).statement,
            verifier_evaluation.start_sources(candidate),
            candidate_set.candidate_sources(candidate),
        )
        evidence = f"## Records of the training runs\n\n{trip['packet']}\n"
        items[name] = {
            "wrong": name in WRONG,
            "v2": {
                "decision": trip["stored"]["decision"],
                "requirements": requirements_of(trip["stored"]["parts"]),
                "checks": trip["stored"]["checks"],
            },
            "round_trip": trip["stored"] == trip["packet_result"],
            "runs": list(matched_evidence.parse_packet(trip["packet"])),
            "inputs": {
                "b": (llm_reviewer.SYSTEM_PROMPT, stored[name]),
                "b_prime": (BOUNDED_SYSTEM_PROMPT, stored[name]),
                "c": (BOUNDED_SYSTEM_PROMPT, head + evidence),
                "d": (BOUNDED_SYSTEM_PROMPT, head + evidence + "\n" + RULES),
            },
        }
    return distinct(items, SAME_INPUT)


def distinct(
    items: dict[str, dict[str, Any]], same_input: dict[str, str]
) -> dict[str, dict[str, Any]]:
    """The items without those asked as another, each checked to have that other's
    inputs; ValueError when two of the items left share an input."""
    items = dict(items)
    for name, asked_as in same_input.items():
        if items.pop(name)["inputs"] != items[asked_as]["inputs"]:
            raise ValueError(f"{name} does not have the inputs of {asked_as}")
    seen: dict[str, str] = {}
    for name, item in items.items():
        key = _sha256(json.dumps(item["inputs"]))
        if key in seen:
            raise ValueError(f"{name} and {seen[key]} are one input")
        seen[key] = name
    return items


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def write_prompts(items: dict[str, dict[str, Any]], out: Path) -> list[str]:
    """Every input as a text file, for reading, and one line per item."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    lines = []
    for number, (name, item) in enumerate(items.items(), start=1):
        sizes = []
        for which, (system, prompt) in item["inputs"].items():
            path = out / f"{number:02d}-{which}.md"
            path.write_text(f"# System\n\n{system}\n\n# User\n\n{prompt}", encoding="utf-8")
            sizes.append(f"{which} {len(system) + len(prompt)}")
        lines.append(
            f"{number:02d} {'wrong  ' if item['wrong'] else 'correct'} {name}: v2 "
            f"{item['v2']['decision']} {item['v2']['checks']}; round trip "
            f"{'kept' if item['round_trip'] else 'LOST'}; runs {item['runs']}; characters "
            + ", ".join(sizes)
        )
    (out / "items.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def review(client: Any, model: str, system: str, prompt: str) -> dict[str, Any]:
    """One review as llm_reviewer.review makes it, with the given system prompt."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    form = {
        "type": "json_schema",
        "json_schema": {"name": "review", "strict": True, "schema": llm_reviewer.response_schema()},
    }
    record: dict[str, Any] = {
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "system_prompt_sha256": _sha256(system),
        "prompt_sha256": _sha256(prompt),
        "status": "running",
        "model_calls": 0,
        "calls": [],
        "review": None,
        "decision": None,
    }
    started = time.monotonic()
    while record["model_calls"] < llm_reviewer.MAX_MODEL_CALLS:
        record["model_calls"] += 1
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format=form,
                max_tokens=llm_reviewer.MAX_OUTPUT_TOKENS,
            )
        except Exception as error:  # network, quota or provider errors end the review
            record.update(
                status="api_error",
                error={
                    "type": type(error).__name__,
                    "status_code": getattr(error, "status_code", None),
                },
            )
            break
        choice = response.choices[0]
        raw = choice.message.content or ""
        parsed, issues = llm_reviewer.parse_review(raw)
        record["calls"].append(
            {
                **llm_reviewer._usage(response),
                "finish_reason": choice.finish_reason,
                "raw_text": raw,
                "issues": issues,
            }
        )
        if parsed is not None:
            statuses = {entry["requirement"]: entry["status"] for entry in parsed["requirements"]}
            record.update(
                status="completed",
                review=parsed,
                requirements=statuses,
                decision=llm_reviewer.decision_of(statuses.values()),
            )
            break
        messages += [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": "The reply could not be used: " + "; ".join(issues)},
        ]
    else:
        record.update(status="invalid_output", decision="review")
    record["seconds"] = round(time.monotonic() - started, 2)
    return record


def read_reviews(output: Path) -> list[dict[str, Any]]:
    path = Path(output) / REVIEWS_FILE
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def asked(items: dict[str, dict[str, Any]], pass_: int) -> list[tuple[str, str, int]]:
    """The (item, input, repetition) a pass asks, in order."""
    names, times = PASSES[pass_]
    first = 1 if pass_ == 1 else 2
    return [
        (name, which, repetition)
        for repetition in range(first, first + times)
        for name in names
        for which in INPUTS
    ]


def run_pass(
    items: dict[str, dict[str, Any]],
    pass_: int,
    output: Path,
    client: Any,
    model: str,
    *,
    provider: str,
    limit: int | None = None,
    progress: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """Ask what a pass asks and output does not hold yet, at most limit calls; an API
    error is asked again on the next run, and three in a row stop this one."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    done = {
        (record["item"], record["input"], record["repetition"])
        for record in read_reviews(output)
        if record["status"] != "api_error"
    }
    written, errors = [], 0
    for name, which, repetition in asked(items, pass_):
        if (name, which, repetition) in done:
            continue
        if limit is not None and len(written) >= limit:
            break
        system, prompt = items[name]["inputs"][which]
        record = {
            "item": name,
            "input": which,
            "repetition": repetition,
            "provider": provider,
            "reviewed_at": datetime.now(UTC).isoformat(timespec="seconds"),
            **review(client, model, system, prompt),
        }
        with (output / REVIEWS_FILE).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        written.append(record)
        calls = record["calls"]
        progress(
            f"{name} {which} #{repetition}: {record['status']}, {record['decision']}; "
            f"{sum(c['input_tokens'] for c in calls)} input, "
            f"{sum(c['output_tokens'] for c in calls)} output tokens, {record['seconds']:.0f} s"
        )
        errors = errors + 1 if record["status"] == "api_error" else 0
        if errors >= 3:
            progress("stopped after 3 API errors in a row; run again to resume")
            break
    return written


def summary_text(records: list[dict[str, Any]], items: dict[str, dict[str, Any]]) -> str:
    """Per item and input, the decisions of the repetitions and v2's; then the tokens."""
    if not records:
        return "no reviews"
    answered = [record for record in records if record["status"] != "api_error"]
    decisions: dict[tuple[str, str], list[str]] = {}
    for record in answered:
        decisions.setdefault((record["item"], record["input"]), []).append(record["decision"])
    lines = [f"{'item':76s} " + " ".join(f"{which:>9s}" for which in INPUTS) + "   v2"]
    for name, item in items.items():
        cells = [
            ",".join(d[:3] for d in decisions.get((name, which), [])) or "-" for which in INPUTS
        ]
        mark = "wrong  " if item["wrong"] else "correct"
        lines.append(
            f"{mark} {name.removeprefix('camelyon_pipeline/'):68s} "
            + " ".join(f"{cell:>9s}" for cell in cells)
            + f"   {item['v2']['decision']}"
        )
    for which in INPUTS:
        wrong = [d for (n, w), ds in decisions.items() if w == which and n in WRONG for d in ds]
        right = [d for (n, w), ds in decisions.items() if w == which and n in CONTROLS for d in ds]
        lines.append(
            f"{which}: wrong items accepted {wrong.count('accept')} of {len(wrong)}; "
            f"correct items {dict(Counter(right))}"
        )
    calls = [call for record in answered for call in record["calls"]]
    lines.append(
        f"{len(answered)} reviews, {len(calls)} calls: "
        f"{sum(c['input_tokens'] for c in calls)} input and "
        f"{sum(c['output_tokens'] for c in calls)} output tokens; "
        f"{Counter(record['status'] for record in records)}"
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="The matched-evidence screening.")
    commands = parser.add_subparsers(dest="command", required=True)
    shown = commands.add_parser("prompts", help="Write every input to files and check them.")
    shown.add_argument("--out", type=Path, default=ROOT / "prompts")
    asked_ = commands.add_parser("run", help="Ask a pass's reviews not yet asked.")
    asked_.add_argument("--pass", dest="pass_", type=int, choices=sorted(PASSES), required=True)
    asked_.add_argument("--name", required=True, help="The reviewer's name, a directory.")
    asked_.add_argument("--provider", choices=PROVIDERS, required=True)
    asked_.add_argument("--model", help="The model, if not the provider's environment variable.")
    asked_.add_argument("--limit", type=int, help="At most this many calls, as a price probe.")
    summed = commands.add_parser("summary", help="Count a reviewer's answers.")
    summed.add_argument("--name", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    items = build_items("test")
    if args.command == "prompts":
        print("\n".join(write_prompts(items, args.out)))
        return 0
    output = ROOT / args.name
    if args.command == "run":
        client, model = create_llm_client(args.provider, args.model)
        run_pass(items, args.pass_, output, client, model, provider=args.provider, limit=args.limit)
    print(summary_text(read_reviews(output), items))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
