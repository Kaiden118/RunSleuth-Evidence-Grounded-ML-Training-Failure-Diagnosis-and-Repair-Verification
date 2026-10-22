"""The LLM reviewer: a baseline verifier that asks a strong LLM whether a candidate meets
its statement, from the same information RunSleuth has.

The reviewer is given the task statement, the files of the project the agent handed
back, the changes it made to the project it was given, and two run records from the
harness, trimmed of hashes and lists that say nothing to a reader: the candidate's run
and the run of a reference project known to meet the statement, on the same data, seed
and settings. It is not given the hidden tests, labels or contract results, the
reference project's code, the names of the version, task or fault, the model that
wrote the candidate, nor what that model said about it. The candidate is named nowhere
in the prompt, and this module imports neither the contract nor the challenge set.

It answers in JSON: for each of the statement's six requirements met, violated or
unclear, with its evidence, and a verdict. The decision follows from the requirements
by the rule the prompt states: reject if one is violated, review if none is but one is
unclear, accept if all are met; the model's own verdict is recorded beside it. A reply
that cannot be read is answered once with what was wrong with it; a second such reply
is invalid_output, which decides review. One review per candidate, at the provider's
default settings, so repeated runs can differ.

    python -m runsleuth.llm_reviewer pilot --family camelyon_pipeline \\
        --check artifacts/pipeline_tasks/check-6 --output artifacts/reviewer/pilot-1 \\
        --provider gemini
"""

import argparse
import difflib
import hashlib
import json
import time
from collections import Counter
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from runsleuth import pipeline_tasks
from runsleuth.diagnose_run import REASONING_BLOCK, RESPONSE_FORMATS
from runsleuth.llm_client import PROVIDERS, create_llm_client
from runsleuth.pipeline_tasks import HEALTHY

PROMPT_VERSION = 1
REQUIREMENTS = (1, 2, 3, 4, 5, 6)
STATUSES = ("met", "violated", "unclear")
VERDICTS = ("accept", "review", "reject")
MAX_MODEL_CALLS = 2
MAX_OUTPUT_TOKENS = 8000
MAX_CONSECUTIVE_API_ERRORS = 3
REVIEWS_FILE = "reviews.jsonl"
# Run-record entries a reader can do nothing with, and lists of plain values longer
# than LONG_LIST, which the summary replaces with their length.
DROPPED = frozenset(
    {
        "step_state_sha256",
        "provenance",
        "project_sha256",
        "candidate_indices_sha256",
        "evaluation_indices_sha256",
        "state_sha256",
    }
)
LONG_LIST = 12

SYSTEM_PROMPT = """\
You review a training project an agent handed back after working on it, and decide \
whether to accept it. The agent was given a project and the task statement below, and \
asked to make the project do everything under "What the project must do", changing \
only what is needed; the project it was given may already have been correct.

You get the statement, every file of the project the agent handed back, the changes \
it made (a unified diff from start/ to candidate/; none if it changed nothing), and two \
run records from the harness that runs projects: the candidate's run, and the run of a \
reference project known to meet the statement, on the same data, seed and settings. \
The candidate's run was stopped as it started its second epoch and restarted, as the \
statement allows; the reference ran without a stop. A candidate that is written or \
trains differently from the reference is not wrong for that alone: the statement \
allows other correct implementations.

The run records hold what the harness observed:
- final_metrics: the harness's own evaluation of the delivered weights, on the \
in-distribution validation samples (id) and on held-out samples (ood).
- parameter_group_epochs: per trained epoch, its attempt, optimizer steps, forwards in \
evaluation mode, learning rates, update sizes, and project_metrics, what the project \
reported with run.log.
- trainability, optimizer_audit: which parameters can train and which the optimizer holds.
- input_statistics: the channel means and deviations of what the network received.
- workspace: the seed and settings; initial, epoch and final state hashes (equal hashes \
mean identical weights).
- pipeline.attempts: each call of main, and how it ended (preempted or completed).
- pipeline.tracked: what run.track saw in each attempt; model_same_as names the earlier \
weights the model then had.
- pipeline.training_reads and pipeline.epochs: which samples training read, per group \
and label, how often, in which order (order_sha256), and how many of the loss's targets \
were the labels of the samples read (null where projects make their own labels).
- pipeline.checkpoints: every write and read, and whose weights a checkpoint held.
- pipeline.delivery: when the weights were delivered and whose weights they were.

For each of the six numbered requirements, say whether the candidate meets it: met if \
the code and the records support it, violated if they show it is not met, unclear if \
what you were given cannot decide it. Give the evidence for each: the code or the \
record entries you relied on. The verdict follows from them: reject if any requirement \
is violated, review if none is violated but any is unclear, accept if all are met.

Reply with one JSON object and nothing else: {"requirements": [{"requirement": 1, \
"status": "met", "evidence": "..."}, ... one entry for each of 1 to 6], "verdict": \
"accept", "reason": "..."}."""


def response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["requirements", "verdict", "reason"],
        "properties": {
            "requirements": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["requirement", "status", "evidence"],
                    "properties": {
                        # parse_review checks the numbers; some providers take no integer enum.
                        "requirement": {"type": "integer"},
                        "status": {"type": "string", "enum": list(STATUSES)},
                        "evidence": {"type": "string"},
                    },
                },
            },
            "verdict": {"type": "string", "enum": list(VERDICTS)},
            "reason": {"type": "string"},
        },
    }


def _trimmed(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _trimmed(item) for key, item in value.items() if key not in DROPPED}
    if isinstance(value, list):
        if len(value) > LONG_LIST and not any(isinstance(item, (dict, list)) for item in value):
            return f"{len(value)} entries"
        return [_trimmed(item) for item in value]
    if isinstance(value, float):
        return float(f"{value:.6g}")
    return value


def run_summary(report: dict[str, Any]) -> dict[str, Any]:
    """A run record as the reviewer reads it."""
    return _trimmed(report)


def changes(start: dict[str, str], candidate: dict[str, str]) -> str:
    """The unified diff from the project the agent was given to the one it handed back."""
    lines: list[str] = []
    for module in sorted(start.keys() | candidate.keys()):
        lines += difflib.unified_diff(
            start.get(module, "").splitlines(keepends=True),
            candidate.get(module, "").splitlines(keepends=True),
            f"start/{module}.py",
            f"candidate/{module}.py",
        )
    return "".join(lines) or "The agent changed no file.\n"


def user_prompt(
    statement: str,
    start: dict[str, str],
    candidate: dict[str, str],
    candidate_run: dict[str, Any],
    reference_run: dict[str, Any],
) -> str:
    files = "\n".join(
        f"### {module}.py\n```python\n{source}```\n" for module, source in sorted(candidate.items())
    )
    return (
        f"## Task statement\n\n{statement}\n"
        f"## Project files the agent handed back\n\n{files}\n"
        f"## Changes the agent made\n\n```diff\n{changes(start, candidate)}```\n\n"
        f"## Candidate's run record\n\n```json\n{json.dumps(run_summary(candidate_run))}\n```\n\n"
        f"## Reference project's run record\n\n```json\n{json.dumps(run_summary(reference_run))}\n```\n"
    )


def decision_of(statuses: Iterable[str]) -> str:
    statuses = list(statuses)
    if "violated" in statuses:
        return "reject"
    return "review" if "unclear" in statuses else "accept"


def parse_review(text: str) -> tuple[dict[str, Any] | None, list[str]]:
    """The review in a reply, or None and what is wrong with the reply."""
    text = REASONING_BLOCK.sub("", text).strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        review = json.loads(text)
    except json.JSONDecodeError as error:
        return None, [f"the reply is not one JSON object: {error.msg}"]
    if not isinstance(review, dict):
        return None, ["the reply is not one JSON object"]
    issues = []
    entries = review.get("requirements")
    if not isinstance(entries, list):
        return None, ["requirements must be a list of six entries"]
    numbers = [entry.get("requirement") for entry in entries if isinstance(entry, dict)]
    if sorted(numbers) != list(REQUIREMENTS) or len(entries) != len(REQUIREMENTS):
        issues.append("requirements must hold one entry for each requirement 1 to 6")
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("status") not in STATUSES:
            issues.append(f"every status must be one of {', '.join(STATUSES)}")
            break
        if not isinstance(entry.get("evidence"), str):
            issues.append("every requirement needs its evidence as a string")
            break
    if review.get("verdict") not in VERDICTS:
        issues.append(f"verdict must be one of {', '.join(VERDICTS)}")
    if issues:
        return None, issues
    return review, []


def _usage(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", None)
    if cached is None:
        cached = getattr(usage, "prompt_cache_hit_tokens", None)
    details = getattr(usage, "completion_tokens_details", None)
    return {
        "served_model": getattr(response, "model", None),
        "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "cached_input_tokens": cached,
        "reasoning_tokens": getattr(details, "reasoning_tokens", None),
    }


def review(
    client: Any, model: str, prompt: str, *, response_format: str = "json_schema"
) -> dict[str, Any]:
    """Ask for one review; what was asked, answered, decided and spent."""
    if response_format not in RESPONSE_FORMATS:
        raise ValueError(f"response_format must be one of {RESPONSE_FORMATS}")
    request_format = (
        {
            "type": "json_schema",
            "json_schema": {"name": "review", "strict": True, "schema": response_schema()},
        }
        if response_format == "json_schema"
        else {"type": "json_object"}
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    record: dict[str, Any] = {
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "response_format": response_format,
        "status": "running",
        "model_calls": 0,
        "calls": [],
        "review": None,
        "decision": None,
    }
    started = time.monotonic()
    while record["model_calls"] < MAX_MODEL_CALLS:
        record["model_calls"] += 1
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format=request_format,
                max_tokens=MAX_OUTPUT_TOKENS,
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
        parsed, issues = parse_review(raw)
        record["calls"].append(
            {
                **_usage(response),
                "finish_reason": choice.finish_reason,
                "raw_text": raw,
                "issues": issues,
            }
        )
        if parsed is not None:
            decided = decision_of(entry["status"] for entry in parsed["requirements"])
            record.update(
                status="completed",
                review=parsed,
                decision=decided,
                verdict_consistent=parsed["verdict"] == decided,
            )
            break
        messages += [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": "The reply could not be used: " + "; ".join(issues)},
        ]
    else:
        record["status"] = "invalid_output"
    if record["status"] == "invalid_output":
        record["decision"] = "review"
    record["seconds"] = round(time.monotonic() - started, 2)
    return record


def development_items(family: str, check: Path) -> list[dict[str, Any]]:
    """The development versions of a family as they are, with their runs in a pipeline
    check and the healthy run there as reference."""
    runs = Path(check) / "runs"

    def read(name: str) -> dict[str, Any]:
        return json.loads((runs / name / "run_report.json").read_text(encoding="utf-8"))

    reference = read(HEALTHY)
    items = []
    for name, item in pipeline_tasks.family(family).versions.items():
        if item.split != "development":
            continue
        sources = pipeline_tasks.version_sources(name, family)
        items.append(
            {
                "id": f"{family}/{name}/as_is",
                "prompt": user_prompt(
                    pipeline_tasks.family(family).statement, sources, sources, read(name), reference
                ),
            }
        )
    return items


def review_all(
    items: list[dict[str, Any]],
    output: Path,
    client: Any,
    model: str,
    *,
    provider: str,
    response_format: str = "json_schema",
    progress: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """Review every item not yet reviewed in output, appending one record per review.

    A review that ended in an API error is tried again; the run stops after
    MAX_CONSECUTIVE_API_ERRORS of them in a row.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / REVIEWS_FILE
    done = {
        record["id"]
        for record in read_reviews(output)
        if record["status"] in ("completed", "invalid_output")
    }
    errors, written = 0, []
    for item in items:
        if item["id"] in done:
            continue
        record = {
            "id": item["id"],
            "provider": provider,
            "reviewed_at": datetime.now(UTC).isoformat(timespec="seconds"),
            **review(client, model, item["prompt"], response_format=response_format),
        }
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        written.append(record)
        progress(f"{item['id']}: {record['status']}, {record['decision']}")
        errors = errors + 1 if record["status"] == "api_error" else 0
        if errors >= MAX_CONSECUTIVE_API_ERRORS:
            progress(f"stopped after {errors} API errors in a row; run again to resume")
            break
    return written


def read_reviews(output: Path) -> list[dict[str, Any]]:
    """The latest record of every reviewed item in an output directory."""
    path = Path(output) / REVIEWS_FILE
    if not path.exists():
        return []
    latest: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            latest[record["id"]] = record
    return list(latest.values())


def summary_text(records: list[dict[str, Any]]) -> str:
    """Decisions, statuses and tokens of a set of reviews; no labels."""
    lines = [
        f"{record['id']:<60} {record['status']:<15} {record['decision']}" for record in records
    ]
    calls = [call for record in records for call in record["calls"]]
    statuses = Counter(record["status"] for record in records)
    decisions = Counter(record["decision"] for record in records)
    tokens = (
        sum(call["input_tokens"] for call in calls),
        sum(call["output_tokens"] for call in calls),
    )
    served = sorted({call["served_model"] for call in calls if call["served_model"]})
    lines += [
        f"{len(records)} reviews: {dict(statuses)}; decisions {dict(decisions)}",
        f"{len(calls)} calls, {tokens[0]} input and {tokens[1]} output tokens; served {served}",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="The LLM reviewer, a baseline verifier.")
    commands = parser.add_subparsers(dest="command", required=True)
    pilot = commands.add_parser("pilot", help="Review a family's development versions as they are.")
    pilot.add_argument("--family", choices=tuple(pipeline_tasks.FAMILIES), required=True)
    pilot.add_argument("--check", type=Path, required=True, help="A pipeline check on seed 7.")
    pilot.add_argument("--output", type=Path, required=True)
    pilot.add_argument("--provider", choices=PROVIDERS, required=True)
    pilot.add_argument("--model", help="The model, if not the provider's environment variable.")
    pilot.add_argument("--response-format", choices=RESPONSE_FORMATS, default="json_schema")
    summarize = commands.add_parser("summarize", help="Count the reviews in an output directory.")
    summarize.add_argument("output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "summarize":
        print(summary_text(read_reviews(args.output)))
        return 0
    client, model = create_llm_client(args.provider, args.model)
    review_all(
        development_items(args.family, args.check),
        args.output,
        client,
        model,
        provider=args.provider,
        response_format=args.response_format,
    )
    print(summary_text(read_reviews(args.output)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
