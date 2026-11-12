"""Training programs written from scratch by a model: one call, one program.

A model is given the task statement of standalone_task and nothing else. Its reply is
its answer: the program is the reply's only ```python block, and its claim the line that
starts with CLAIM. Nothing is asked twice and nothing is repaired. A reply without a
usable program is an attempt that produced none; it is kept and counted as such. A call
the provider failed is no attempt, and is made again when the command is run again.

    python -m runsleuth.standalone_programs write --dataset camelyon \\
        --model-name gpt-6.1-sol --provider compatible --model gpt-6.1-sol \\
        --attempts 2 --out artifacts/standalone/programs
    python -m runsleuth.standalone_programs list artifacts/standalone/programs

A program written by one call is written by a model, not by an agent: the model ran
nothing before it answered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from runsleuth import standalone_task
from runsleuth.llm_client import PROVIDERS, create_llm_client

PROMPT_VERSION = 1
SYSTEM_PROMPT = "You are an expert machine learning engineer. Do exactly what the task asks."
MAX_OUTPUT_TOKENS = 32000
MAX_CONSECUTIVE_API_ERRORS = 3
ATTEMPT_FILE = "attempt.json"
# written: one program in the reply. The others leave no program: the reply held none,
# held several, or was cut off; api_error is a call that failed and is made again.
STATUSES = ("written", "no_program", "several_programs", "truncated", "api_error")
GENERATION = "one call, no tools"
_FENCE = re.compile(r"```([A-Za-z0-9_+-]*)[ \t]*\r?\n(.*?)```", re.DOTALL)
_CLAIM = re.compile(r"^[\s>*_`]*CLAIM:\s*(.*?)[\s*_`]*$", re.IGNORECASE)
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def attempt_id(model_name: str, dataset: str, attempt: int) -> str:
    """A name for an attempt that tells neither its model nor its number."""
    return _sha256(f"{model_name}/{dataset}/{attempt}")[:12]


def parse_reply(text: str) -> dict[str, Any]:
    """The program and the claim of a reply.

    The program is the only ```python block; with none or several there is no program.
    The claim is the last line outside the blocks that starts with CLAIM: meets,
    does_not_meet, or unparsed when it is neither of the two sentences asked for.
    """
    blocks = [
        body for language, body in _FENCE.findall(text) if language.lower() in ("python", "py")
    ]
    lines = _FENCE.sub("", text).splitlines()
    claim = claim_line = reason = None
    for number, line in enumerate(lines):
        match = _CLAIM.match(line)
        if not match:
            continue
        claim_line, said = line.strip(), match.group(1).lower()
        following = [later.strip() for later in lines[number + 1 :] if later.strip()]
        reason = following[0] if following else None
        if standalone_task.CLAIMS[1] in said:
            claim = "does_not_meet"
        elif standalone_task.CLAIMS[0] in said:
            claim = "meets"
        else:
            claim = "unparsed"
    return {
        "program": blocks[0] if len(blocks) == 1 else None,
        "python_blocks": len(blocks),
        "claim": claim,
        "claim_line": claim_line,
        "claim_reason": reason if claim == "does_not_meet" else None,
    }


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


def write_program(client: Any, model: str, prompt: str) -> dict[str, Any]:
    """Ask once for a program; what was asked, answered and spent."""
    record: dict[str, Any] = {
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "system_prompt_sha256": _sha256(SYSTEM_PROMPT),
        "statement_sha256": _sha256(prompt),
        "generation": GENERATION,
        "status": "api_error",
        "call": None,
        "program": None,
        "program_sha256": None,
        "program_lines": None,
        "python_blocks": 0,
        "claim": None,
        "claim_line": None,
        "claim_reason": None,
    }
    started = time.monotonic()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=MAX_OUTPUT_TOKENS,
        )
    except Exception as error:  # network, quota or provider errors: no attempt was made
        record["error"] = {
            "type": type(error).__name__,
            "status_code": getattr(error, "status_code", None),
        }
    else:
        choice = response.choices[0]
        raw = choice.message.content or ""
        parsed = parse_reply(raw)
        record["call"] = {
            **_usage(response),
            "finish_reason": choice.finish_reason,
            "raw_text": raw,
        }
        if choice.finish_reason == "length":
            status, parsed["program"] = "truncated", None
        elif parsed["program"] is not None:
            status = "written"
        else:
            status = "several_programs" if parsed["python_blocks"] > 1 else "no_program"
        record.update(parsed, status=status)
        if parsed["program"] is not None:
            record["program_sha256"] = _sha256(parsed["program"])
            record["program_lines"] = len(parsed["program"].splitlines())
    record["seconds"] = round(time.monotonic() - started, 2)
    return record


def attempt_directory(output: Path, model_name: str, dataset: str, attempt: int) -> Path:
    return Path(output) / model_name / dataset / f"attempt-{attempt}"


def write_attempts(
    output: Path,
    dataset: str,
    model_name: str,
    client: Any,
    model: str,
    *,
    provider: str,
    attempts: int,
    size: int | None = None,
    pause: float = 0.0,
    progress: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """Make the attempts 1 to attempts that have not been made, one call each.

    Every attempt gets a directory with attempt.json and, when the reply held one
    program, train.py. The run stops after MAX_CONSECUTIVE_API_ERRORS failed calls in
    a row.
    """
    if not _NAME.match(model_name):
        raise ValueError(f"{model_name!r} cannot name a directory")
    prompt = standalone_task.statement(dataset, size)
    errors, written = 0, []
    for attempt in range(1, attempts + 1):
        directory = attempt_directory(output, model_name, dataset, attempt)
        path = directory / ATTEMPT_FILE
        if path.is_file() and json.loads(path.read_text(encoding="utf-8"))["status"] != "api_error":
            continue
        if pause:
            time.sleep(pause)
        record = {
            "schema_version": 1,
            "task": "standalone_program",
            "id": attempt_id(model_name, dataset, attempt),
            "dataset": dataset,
            "model_name": model_name,
            "attempt": attempt,
            "provider": provider,
            "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
            **write_program(client, model, prompt),
        }
        directory.mkdir(parents=True, exist_ok=True)
        program = record.pop("program")
        if program is not None:
            (directory / standalone_task.PROGRAM).write_text(program, encoding="utf-8", newline="")
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        written.append(record)
        progress(f"{model_name} {dataset} attempt {attempt}: {record['status']}")
        errors = errors + 1 if record["status"] == "api_error" else 0
        if errors >= MAX_CONSECUTIVE_API_ERRORS:
            progress(f"stopped after {errors} API errors in a row; run again to resume")
            break
    return written


def read_attempts(output: Path) -> list[dict[str, Any]]:
    """Every attempt under output, by model, dataset and number."""
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in Path(output).glob(f"*/*/attempt-*/{ATTEMPT_FILE}")
    ]
    return sorted(records, key=lambda r: (r["model_name"], r["dataset"], r["attempt"]))


def list_text(records: list[dict[str, Any]]) -> str:
    """One line per attempt, and what the attempts cost in tokens."""
    if not records:
        return "no attempts"
    lines = [
        f"{'model':26s} {'dataset':9s} {'n':>2s} {'status':16s} {'claim':13s} "
        f"{'lines':>5s} {'input':>7s} {'output':>7s} {'seconds':>7s}"
    ]
    for record in records:
        call = record["call"] or {}
        lines.append(
            f"{record['model_name']:26s} {record['dataset']:9s} {record['attempt']:2d} "
            f"{record['status']:16s} {record['claim'] or '-':13s} "
            f"{record['program_lines'] or '-':>5} {call.get('input_tokens', '-'):>7} "
            f"{call.get('output_tokens', '-'):>7} {record['seconds']:7.1f}"
        )
    made = [record for record in records if record["status"] != "api_error"]
    spent = [record["call"] for record in made]
    lines.append(
        f"{len(made)} attempts, {sum(r['status'] == 'written' for r in made)} with a program; "
        f"{sum(call['input_tokens'] for call in spent)} input and "
        f"{sum(call['output_tokens'] for call in spent)} output tokens"
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Training programs written by a model.")
    commands = parser.add_subparsers(dest="command", required=True)
    write = commands.add_parser("write", help="Ask a model for programs, one call each.")
    write.add_argument("--dataset", required=True, choices=sorted(standalone_task.DATASETS))
    write.add_argument("--model-name", required=True, help="The name the attempts are kept by.")
    write.add_argument("--provider", choices=PROVIDERS, required=True)
    write.add_argument("--model", help="The model, if not the provider's environment variable.")
    write.add_argument("--attempts", type=int, default=1, help="Make attempts 1 to this number.")
    write.add_argument("--out", type=Path, required=True)
    write.add_argument("--pause", type=float, default=0.0, help="Seconds before every call.")
    listed = commands.add_parser("list", help="List the attempts in an output directory.")
    listed.add_argument("out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "write":
        client, model = create_llm_client(args.provider, args.model)
        write_attempts(
            args.out,
            args.dataset,
            args.model_name,
            client,
            model,
            provider=args.provider,
            attempts=args.attempts,
            pause=args.pause,
        )
    print(list_text(read_attempts(args.out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
