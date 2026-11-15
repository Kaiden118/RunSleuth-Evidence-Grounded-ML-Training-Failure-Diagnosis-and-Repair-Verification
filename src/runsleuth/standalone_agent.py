"""An LLM agent that writes a training program from scratch and can run it.

The agent is given the task statement of standalone_task and an empty workspace. It
writes train.py through tools and may run it on a small copy of the data, with the
seed, the epochs, the stop and the resume it chooses, into run directories it names. A
run shows what a developer would see: the exit status, the end of the output, the
files in the run's directory with their sizes and the end of its log. The agent ends
by calling finish with its own claim: whether the program does everything the
statement asks. The claim is what the agent said, never a verdict, and the agent sees
no check of standalone_check.

    python -m runsleuth.standalone_agent write --dataset camelyon \\
        --model-name gemini-3.5-flash-lite --provider gemini --model gemini-3.5-flash-lite \\
        --attempts 1 --data artifacts/standalone/data --out artifacts/standalone/agent-programs \\
        --work C:/Users/ADMIN/rs-data/standalone-work/agent
    python -m runsleuth.standalone_agent list artifacts/standalone/agent-programs

Attempts are kept as standalone_programs keeps them, so standalone_check reads both.
Every model call, tool call and result is recorded, with the model the provider says
served each call and the tokens it reports (the provider handling is agent_loop's).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from runsleuth import standalone_check, standalone_programs, standalone_task
from runsleuth.agent_loop import _call_usage, _rejected, _with_provider_content
from runsleuth.llm_client import PROVIDERS, create_llm_client

PROMPT_VERSION = 1
GENERATION = "agent with a run tool"
MAX_TOOL_CALLS = 30
MAX_RUNS = 10
MAX_OUTPUT_TOKENS = 32000
MAX_TEXT_REPLIES = 2
MAX_ERROR_CHARACTERS = 1500
SHOWN_OUTPUT = 2000
SHOWN_LOG_LINES = 6
RUN_TIMEOUT = 300.0
# The small copy of the data the agent's runs train on.
SMALL = {"max_train": 256, "max_validation": 128}
_RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}\Z")


def system_prompt(max_tool_calls: int, max_runs: int, small: dict[str, int], timeout: float) -> str:
    """The instructions every episode starts with; only the budget lines vary."""
    return (
        "You are a machine-learning engineer. Write the program the task statement asks "
        f"for, as the file {standalone_task.PROGRAM} in your workspace.\n"
        f"Work only through the tools. write_file replaces the whole of "
        f"{standalone_task.PROGRAM}; edit_file replaces one exact occurrence of old_text; "
        "read_file returns the file as it stands.\n"
        f"run_program runs {standalone_task.PROGRAM} as the statement says it is called, on "
        f"a small copy of the data ({small['max_train']} training and "
        f"{small['max_validation']} validation samples), with the seed and the epochs you "
        "give and, if you give them, stop_after and resume. It writes into the run "
        "directory you name; a directory keeps its files between calls, so a run can be "
        "stopped and resumed in it, and a new name starts from an empty directory. It "
        "returns the exit status, the end of the output, the files in the directory with "
        f"their sizes and the end of the log. You may call it at most {max_runs} times, and "
        f"a call may take at most {timeout:g} seconds.\n"
        "When you are done, call finish. Set meets_every_requirement to true only if "
        f"{standalone_task.PROGRAM}, as it stands, does everything the statement asks. In "
        "reason, say which requirement it does not meet and why, or write none.\n"
        f"You have at most {max_tool_calls} tool calls in total, finish included."
    )


def tool_schemas() -> list[dict[str, Any]]:
    """The tools offered to the model, in the Chat Completions format."""

    def tool(
        name: str, description: str, properties: dict | None = None, required: list | None = None
    ) -> dict[str, Any]:
        parameters: dict[str, Any] = {"type": "object", "properties": properties or {}}
        if properties:
            parameters["required"] = list(properties) if required is None else required
        return {
            "type": "function",
            "function": {"name": name, "description": description, "parameters": parameters},
        }

    text, whole = {"type": "string"}, {"type": "integer"}
    program = standalone_task.PROGRAM
    return [
        tool("read_file", f"Read {program} as it stands."),
        tool("write_file", f"Replace the whole of {program}.", {"content": text}),
        tool(
            "edit_file",
            f"Replace one exact occurrence of old_text in {program} with new_text.",
            {"old_text": text, "new_text": text},
        ),
        tool(
            "run_program",
            f"Run {program} on the small copy of the data, into a run directory.",
            {
                "run": {"type": "string", "description": "The run directory's name."},
                "seed": whole,
                "epochs": whole,
                "stop_after": whole,
                "resume": {"type": "boolean"},
            },
            ["run", "seed", "epochs"],
        ),
        tool(
            "finish",
            "End the session with your conclusion.",
            {"meets_every_requirement": {"type": "boolean"}, "reason": text},
        ),
    ]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def shown_run(outcome: dict[str, Any], out: Path) -> dict[str, Any]:
    """What the agent sees of a call of its program: nothing a developer would not."""
    shown: dict[str, Any] = {
        "status": outcome["status"],
        "returncode": outcome["returncode"],
        "seconds": outcome["seconds"],
        "output": outcome.get("output", "")[-SHOWN_OUTPUT:],
        "files": sorted(
            [path.name, path.stat().st_size] for path in out.iterdir() if path.is_file()
        ),
    }
    log = out / standalone_task.LOG
    if log.is_file():
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        shown["log_end"] = lines[-SHOWN_LOG_LINES:]
    return shown


def program_runner(
    data: Path, runs: Path, *, device: str = "cuda", timeout: float = RUN_TIMEOUT
) -> Callable[[Path, dict[str, Any]], dict[str, Any]]:
    """run_program for run_agent: calls of the program on data, into runs/<name>.

    A program standalone_check's screen refuses is not run.
    """

    def run(program: Path, arguments: dict[str, Any]) -> dict[str, Any]:
        problems = standalone_check.screen(program.read_text(encoding="utf-8"))
        if problems:
            return {"status": "refused", "problems": problems}
        out = Path(runs) / arguments["run"]
        out.mkdir(parents=True, exist_ok=True)
        outcome = standalone_check.call(
            program,
            data,
            out,
            seed=arguments["seed"],
            epochs=arguments["epochs"],
            stop_after=arguments.get("stop_after"),
            resume=bool(arguments.get("resume")),
            device=device,
            timeout=timeout,
        )
        return shown_run(outcome, out)

    return run


def run_agent(
    client: Any,
    model: str,
    workspace: Path,
    statement: str,
    *,
    run_program: Callable[[Path, dict[str, Any]], dict[str, Any]],
    max_tool_calls: int = MAX_TOOL_CALLS,
    max_runs: int = MAX_RUNS,
    small: dict[str, int] | None = None,
    timeout: float = RUN_TIMEOUT,
    pause: float = 0.0,
    error_wait: float = 30.0,
    max_api_errors: int = 3,
    echo_reasoning: bool = False,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Let the model write a program in workspace until it finishes or a limit stops it.

    run_program(program, arguments) calls the program and returns what the agent may
    see of the call. status is finished, or the limit that ended the episode:
    tool_call_limit, no_tool_call (the model kept answering in text) or api_error.
    claim is the agent's own conclusion, None unless it finished. runs lists the calls
    of the program the agent asked for, so that one can see whether it ever resumed a
    run. pause, error_wait, max_api_errors and echo_reasoning are as in agent_loop.
    """
    program = Path(workspace) / standalone_task.PROGRAM
    system = system_prompt(max_tool_calls, max_runs, small or SMALL, timeout)
    tools = tool_schemas()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": statement},
    ]
    record: dict[str, Any] = {
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "system_prompt_sha256": _sha256(system),
        "statement_version": standalone_task.STATEMENT_VERSION,
        "statement_sha256": _sha256(statement),
        "generation": GENERATION,
        "tools": [tool["function"]["name"] for tool in tools],
        "limits": {
            "max_tool_calls": max_tool_calls,
            "max_runs": max_runs,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "run_timeout": timeout,
        },
        "episode_status": "running",
        "agent_claim": None,
        "model_calls": 0,
        "tool_calls": 0,
        "runs": [],
        "input_tokens": 0,
        "output_tokens": 0,
        # Totals of what the provider reported; None if it never reported them.
        "cached_input_tokens": None,
        "reasoning_tokens": None,
        "served_models": [],
        "echoes_reasoning": echo_reasoning,
        "api_errors": [],
        "steps": [],
    }

    def write(arguments: dict[str, Any]) -> dict[str, Any]:
        content = arguments.get("content")
        if not isinstance(content, str) or not content.strip():
            return {"error": "content must be the text of the program"}
        if len(content) > standalone_check.MAX_SOURCE_CHARACTERS:
            limit = standalone_check.MAX_SOURCE_CHARACTERS
            return {"error": f"the program may have at most {limit} characters"}
        program.write_text(content, encoding="utf-8", newline="\n")
        return {"ok": True, "characters": len(content)}

    def edit(arguments: dict[str, Any]) -> dict[str, Any]:
        old, new = arguments.get("old_text"), arguments.get("new_text")
        if not isinstance(old, str) or not isinstance(new, str) or not old:
            return {"error": "old_text and new_text must be strings, and old_text not empty"}
        if not program.is_file():
            return {"error": f"{standalone_task.PROGRAM} has not been written yet"}
        source = program.read_text(encoding="utf-8")
        if source.count(old) != 1:
            return {"error": f"old_text occurs {source.count(old)} times; it must occur once"}
        return write({"content": source.replace(old, new)})

    def read(_arguments: dict[str, Any]) -> dict[str, Any]:
        if not program.is_file():
            return {"error": f"{standalone_task.PROGRAM} has not been written yet"}
        return {"content": program.read_text(encoding="utf-8")}

    def run(arguments: dict[str, Any]) -> dict[str, Any]:
        name, numbers = arguments.get("run"), ("seed", "epochs", "stop_after")
        given = {key: arguments[key] for key in numbers if arguments.get(key) is not None}
        if not isinstance(name, str) or not _RUN_NAME.match(name):
            return {"error": "run must be a name of letters, digits, - and _"}
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in given.values()):
            return {"error": "seed, epochs and stop_after must be whole numbers, not negative"}
        if "seed" not in given or "epochs" not in given:
            return {"error": "seed and epochs are needed"}
        if not program.is_file():
            return {"error": f"{standalone_task.PROGRAM} has not been written yet"}
        if len(record["runs"]) >= max_runs:
            return {"error": "no runs are left"}
        asked = {"run": name, **given, "resume": arguments.get("resume") is True}
        shown = run_program(program, asked)
        record["runs"].append({**asked, "status": shown["status"]})
        return {**shown, "runs_left": max_runs - len(record["runs"])}

    def finish(arguments: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(arguments.get("meets_every_requirement"), bool):
            return {"error": "meets_every_requirement must be true or false"}
        record["agent_claim"] = {
            "meets_every_requirement": arguments["meets_every_requirement"],
            "reason": str(arguments.get("reason", "")),
        }
        return {"ok": True}

    handlers = {
        "read_file": read,
        "write_file": write,
        "edit_file": edit,
        "run_program": run,
        "finish": finish,
    }
    return converse(
        client,
        model,
        messages,
        tools,
        handlers,
        record,
        finished=lambda name: name == "finish" and record["agent_claim"] is not None,
        max_tool_calls=max_tool_calls,
        pause=pause,
        error_wait=error_wait,
        max_api_errors=max_api_errors,
        echo_reasoning=echo_reasoning,
        sleep=sleep,
    )


def converse(
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]],
    record: dict[str, Any],
    *,
    finished: Callable[[str], bool],
    max_tool_calls: int,
    pause: float = 0.0,
    error_wait: float = 30.0,
    max_api_errors: int = 3,
    echo_reasoning: bool = False,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Talk with the model until it finishes or a limit stops it, and keep the record.

    messages starts with the system prompt and the statement. handlers answer the tool
    calls by name; finished(name) says whether the call just answered ended the episode.
    record holds the counters run_agent starts with; its episode_status is running until
    the episode ends as finished, tool_call_limit, no_tool_call or api_error.
    """
    text_replies = 0
    while record["episode_status"] == "running":
        sleep(pause)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_tokens=MAX_OUTPUT_TOKENS,
            )
        except Exception as error:  # network, quota or provider errors
            status_code = getattr(error, "status_code", None)
            record["api_errors"].append(
                {
                    "type": type(error).__name__,
                    "status_code": status_code,
                    "message": str(error)[:MAX_ERROR_CHARACTERS],
                }
            )
            if _rejected(status_code) or len(record["api_errors"]) >= max_api_errors:
                record["episode_status"] = "api_error"
            else:
                sleep(error_wait)
            continue
        record["model_calls"] += 1
        usage = _call_usage(response)
        record["input_tokens"] += usage["input_tokens"]
        record["output_tokens"] += usage["output_tokens"]
        for name in ("cached_input_tokens", "reasoning_tokens"):
            if usage[name] is not None:
                record[name] = (record[name] or 0) + usage[name]
        served = usage["served_model"]
        if served is not None and served not in record["served_models"]:
            record["served_models"].append(served)
        message = response.choices[0].message
        calls = list(getattr(message, "tool_calls", None) or [])
        step = {
            "content": message.content or "",
            "tool_calls": [],
            "finish_reason": getattr(response.choices[0], "finish_reason", None),
            **usage,
        }
        record["steps"].append(step)
        if not calls:
            text_replies += 1
            if text_replies > MAX_TEXT_REPLIES:
                record["episode_status"] = "no_tool_call"
                break
            messages.append(
                _with_provider_content(
                    {"role": "assistant", "content": message.content or ""},
                    message,
                    reasoning=echo_reasoning,
                )
            )
            messages.append(
                {"role": "user", "content": "Use a tool. Call finish when you are done."}
            )
            continue
        messages.append(
            _with_provider_content(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        _with_provider_content(
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.function.name,
                                    "arguments": call.function.arguments,
                                },
                            },
                            call,
                        )
                        for call in calls
                    ],
                },
                message,
                reasoning=echo_reasoning,
            )
        )
        for call in calls:
            name = call.function.name
            if record["episode_status"] != "running":
                # A finished or exhausted episode still owes every call an answer.
                result = {"error": "the session has ended"}
            else:
                record["tool_calls"] += 1
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                except ValueError:
                    arguments = None
                if not isinstance(arguments, dict):
                    result = {"error": "the arguments were not a JSON object"}
                elif name not in handlers:
                    result = {"error": f"unknown tool {name}"}
                else:
                    result = handlers[name](arguments)
                if finished(name):
                    record["episode_status"] = "finished"
                elif record["tool_calls"] >= max_tool_calls:
                    record["episode_status"] = "tool_call_limit"
                result = {**result, "tool_calls_left": max_tool_calls - record["tool_calls"]}
            step["tool_calls"].append(
                {"name": name, "arguments": call.function.arguments, "result": result}
            )
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)}
            )
    return record


def attempt_id(model_name: str, dataset: str, attempt: int) -> str:
    """A name for an agent's attempt that tells neither its model nor its number."""
    return _sha256(f"agent/{model_name}/{dataset}/{attempt}")[:12]


def write_attempts(
    output: Path,
    dataset: str,
    model_name: str,
    client: Any,
    model: str,
    *,
    provider: str,
    attempts: int,
    data: Path,
    work: Path,
    size: int | None = None,
    device: str = "cuda",
    progress: Callable[[str], None] = print,
    **options,
) -> list[dict[str, Any]]:
    """Make the attempts 1 to attempts that have not been made, one episode each.

    data is the dataset's --data directory; the agent's runs train on a small copy of
    it, made once under work. Every attempt gets a directory with attempt.json and, if
    the agent wrote one, train.py as it stood when the episode ended. The agent's run
    directories live under work and are removed when its episode ends. An episode the
    provider failed is no attempt and is made again when the command is run again.
    """
    if not standalone_programs._NAME.match(model_name):
        raise ValueError(f"{model_name!r} cannot name a directory")
    small = Path(work) / "data-small" / dataset
    if not (small / standalone_task.RECORD).is_file():
        standalone_task.shrink(data, small, **SMALL)
    statement = standalone_task.statement(dataset, size, answer=False)
    written = []
    for attempt in range(1, attempts + 1):
        directory = standalone_programs.attempt_directory(output, model_name, dataset, attempt)
        path = directory / standalone_programs.ATTEMPT_FILE
        if path.is_file() and json.loads(path.read_text(encoding="utf-8"))["status"] != "api_error":
            continue
        identity = attempt_id(model_name, dataset, attempt)
        runs = Path(work) / "runs" / identity
        if runs.exists():
            raise FileExistsError(f"{runs} exists; remove it or choose another work directory")
        directory.mkdir(parents=True, exist_ok=True)
        program = directory / standalone_task.PROGRAM
        if program.is_file():
            program.replace(directory / f"{standalone_task.PROGRAM}.earlier")
        started = time.monotonic()
        try:
            episode = run_agent(
                client,
                model,
                directory,
                statement,
                run_program=program_runner(small, runs, device=device),
                **options,
            )
        finally:
            shutil.rmtree(runs, ignore_errors=True)
        claim = episode["agent_claim"]
        if episode["episode_status"] == "api_error":
            status = "api_error"
        else:
            status = "written" if program.is_file() else "no_program"
        source = program.read_text(encoding="utf-8") if program.is_file() else None
        record = {
            "schema_version": 1,
            "task": "standalone_program",
            "id": identity,
            "dataset": dataset,
            "model_name": model_name,
            "attempt": attempt,
            "provider": provider,
            "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "status": status,
            "claim": None
            if claim is None
            else ("meets" if claim["meets_every_requirement"] else "does_not_meet"),
            "claim_line": None,
            "claim_reason": claim["reason"] if claim else None,
            "program_sha256": _sha256(source) if source is not None else None,
            "program_lines": len(source.splitlines()) if source is not None else None,
            # The totals of the episode, where standalone_programs has its one call.
            "call": {
                "served_model": ", ".join(episode["served_models"]) or None,
                "input_tokens": episode["input_tokens"],
                "output_tokens": episode["output_tokens"],
                "cached_input_tokens": episode["cached_input_tokens"],
                "reasoning_tokens": episode["reasoning_tokens"],
                "finish_reason": None,
                "raw_text": None,
            },
            "seconds": round(time.monotonic() - started, 2),
            **episode,
        }
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        written.append(record)
        progress(episode_line(record))
        if status == "api_error":
            progress("stopped at an API error; run again to resume")
            break
    return written


def episode_line(record: dict[str, Any]) -> str:
    """One episode: how it ended, what the agent ran and claimed, and what it cost."""
    runs = record["runs"]
    resumed = sum(run["resume"] for run in runs)
    return (
        f"{record['model_name']} {record['dataset']} attempt {record['attempt']}: "
        f"{record['episode_status']}, {record['status']}, claims {record['claim']}; "
        f"{record['model_calls']} model calls, {record['tool_calls']} tool calls, "
        f"{len(runs)} runs ({resumed} resumed, "
        f"{sum(run['status'] != 'completed' for run in runs)} not completed); "
        f"{record['input_tokens']} input and {record['output_tokens']} output tokens, "
        f"{record['seconds']:.0f} s"
    )


def list_text(records: list[dict[str, Any]]) -> str:
    episodes = [record for record in records if record.get("generation") == GENERATION]
    return "\n".join(
        [*(episode_line(record) for record in episodes), standalone_programs.list_text(records)]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Training programs written by an agent.")
    commands = parser.add_subparsers(dest="command", required=True)
    write = commands.add_parser("write", help="Let a model write programs, one episode each.")
    write.add_argument("--dataset", required=True, choices=sorted(standalone_task.DATASETS))
    write.add_argument("--model-name", required=True, help="The name the attempts are kept by.")
    write.add_argument("--provider", choices=PROVIDERS, required=True)
    write.add_argument("--model", help="The model, if not the provider's environment variable.")
    write.add_argument("--attempts", type=int, default=1, help="Make attempts 1 to this number.")
    write.add_argument("--data", type=Path, required=True, help="The directory of --data folders.")
    write.add_argument("--out", type=Path, required=True)
    write.add_argument("--work", type=Path, required=True, help="Where the agent's runs write.")
    write.add_argument("--device", default="cuda")
    write.add_argument("--pause", type=float, default=0.0, help="Seconds before every call.")
    write.add_argument("--echo-reasoning", action="store_true", help="As in agent_loop.")
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
            data=args.data / args.dataset,
            work=args.work,
            device=args.device,
            pause=args.pause,
            echo_reasoning=args.echo_reasoning,
        )
        print()
    print(list_text(standalone_programs.read_attempts(args.out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
