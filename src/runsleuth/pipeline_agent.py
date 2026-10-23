"""An LLM agent that reviews and repairs a training project through tools.

The agent is given a project folder (config.py, data.py, train.py and TASK.md, as
pipeline_tasks.materialize writes it) and nothing else: TASK.md and the three modules
come with its first message. It can read the files, edit or rewrite the modules and,
when the caller allows it, ask for a short training run. It ends by calling finish
with its own claim: whether the project now does what TASK.md asks. The claim is what
the agent said, never a verdict.

A short run trains the project in the pipeline harness as a candidate's run is trained,
stopped once as it starts its second epoch and restarted, for a short budget
(SHORT_RUN), and shows the agent what a developer running it would see (shown_run):
whether it completed, the end of its error if not, the epochs each attempt trained,
what the project reported with run.log, whether it delivered weights, and the end of
its output. It shows neither the harness's own evaluation of the delivered weights nor
any record a verifier reads, so candidates are written without the verifiers'
information.

Every model call, tool call and result is recorded, with the model the provider says
served each call, the tokens it reports and why the reply ended (finish_reason,
"length" at the output limit), as stage one's loop records them (the provider
handling is agent_loop's).
"""

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from runsleuth.agent_loop import _call_usage, _rejected, _with_provider_content
from runsleuth.agent_workspace import MAX_SOURCE_CHARACTERS
from runsleuth.pipeline_tasks import STATEMENT_NAME

PROMPT_VERSION = 1
MODULE_FILES = ("config.py", "data.py", "train.py")
FILES = (STATEMENT_NAME, *MODULE_FILES)
MAX_TOOL_CALLS = 20
MAX_RUNS = 3
# Every model gets the same limit. In the smoke tests Muse Spark spent all of 8,000
# on thinking in six replies and DeepSeek reached 7,967; a reply is cut only where a
# model would think longer than this.
MAX_OUTPUT_TOKENS = 32000
MAX_TEXT_REPLIES = 2
MAX_ERROR_CHARACTERS = 1500
SHOWN_OUTPUT = 2000
# A short run: two epochs, the first attempt stopped after the first, and evaluation
# capped, as the candidate's own run is trained for longer.
SHORT_RUN = {"epochs": 2, "interrupt_after_epochs": 1, "max_eval_samples": 1000}


def system_prompt(allow_runs: bool, max_tool_calls: int, max_runs: int) -> str:
    """The instructions every episode starts with; only the budget lines vary."""
    running = (
        "run_training trains the project as it stands for a short budget, as the harness "
        "runs it: stopped once as it starts its second epoch and started again. It returns "
        "what the run showed: whether it completed, the epochs each attempt trained, what "
        "the project reported with run.log, whether it delivered weights, and the end of "
        f"its output. You may call it at most {max_runs} times."
        if allow_runs
        else "You cannot run the project in this session; judge it by reading it."
    )
    return (
        "You are a machine-learning engineer reviewing a PyTorch training project.\n"
        f"{STATEMENT_NAME} says what the project must do; its modules are "
        f"{', '.join(MODULE_FILES)}. Decide whether the project does all of it. If it does "
        "not, fix it with the smallest changes that make it do so; if it does, leave it "
        "alone.\n"
        f"Work only through the tools. {running}\n"
        "When you are done, call finish. Set project_is_correct to true only if the project, "
        f"as it stands after your changes, does everything {STATEMENT_NAME} asks. In problem, "
        "name what was wrong, or write none. Keep summary to a few sentences.\n"
        f"You have at most {max_tool_calls} tool calls in total, finish included."
    )


def tool_schemas(allow_runs: bool) -> list[dict[str, Any]]:
    """The tools offered to the model, in the Chat Completions format."""

    def tool(name: str, description: str, properties: dict | None = None) -> dict[str, Any]:
        parameters: dict[str, Any] = {"type": "object", "properties": properties or {}}
        if properties:
            parameters["required"] = list(properties)
        return {
            "type": "function",
            "function": {"name": name, "description": description, "parameters": parameters},
        }

    text = {"type": "string"}
    module = {"type": "string", "enum": list(MODULE_FILES)}
    tools = [
        tool(
            "read_file",
            "Read a file of the project.",
            {"name": {"type": "string", "enum": list(FILES)}},
        ),
        tool(
            "edit_file",
            "Replace one exact occurrence of old_text in a module with new_text.",
            {"name": module, "old_text": text, "new_text": text},
        ),
        tool("write_file", "Replace the whole of a module.", {"name": module, "content": text}),
    ]
    if allow_runs:
        tools.append(tool("run_training", "Train the project as it stands for a short budget."))
    tools.append(
        tool(
            "finish",
            "End the session with your conclusion.",
            {"project_is_correct": {"type": "boolean"}, "problem": text, "summary": text},
        )
    )
    return tools


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def first_message(project: Path) -> str:
    """TASK.md and every module, as the episode's first user message."""
    project = Path(project)
    parts = [f"{STATEMENT_NAME}:\n\n{(project / STATEMENT_NAME).read_text(encoding='utf-8')}"]
    parts += [
        f"{name}:\n\n```python\n{(project / name).read_text(encoding='utf-8')}```"
        for name in MODULE_FILES
    ]
    return "\n".join(parts)


def shown_run(outcome: dict[str, Any]) -> dict[str, Any]:
    """What a short run shows the agent, from pipeline_workspace.run_in_subprocess's
    outcome and the run report it names."""
    shown: dict[str, Any] = {"status": outcome["status"]}
    if outcome.get("problems"):
        shown["problems"] = list(outcome["problems"])
    report = None
    if outcome.get("report"):
        report = json.loads(Path(outcome["report"]).read_text(encoding="utf-8"))
    if report is not None:
        pipeline = report.get("pipeline") or {}
        shown["attempts"] = [
            {
                "attempt": attempt["attempt"],
                "epochs_trained": attempt["epochs_trained"],
                "ended": "stopped by the harness as it started its next epoch"
                if attempt["outcome"] == "preempted"
                else attempt["outcome"],
            }
            for attempt in pipeline.get("attempts") or []
        ]
        shown["logged"] = [
            {
                "attempt": row.get("attempt"),
                "epoch": row.get("project_epoch"),
                **(row.get("project_metrics") or {}),
            }
            for row in report.get("parameter_group_epochs") or []
        ]
        shown["delivered_weights"] = pipeline.get("delivery") is not None
        if report.get("error"):
            shown["error"] = report["error"]
    elif outcome.get("error"):
        shown["error"] = outcome["error"]
    # The last line the child prints is the harness's own outcome, not the project's.
    lines = (outcome.get("output") or "").rstrip("\n").split("\n")
    if lines and lines[-1].startswith("{"):
        lines = lines[:-1]
    shown["output"] = "\n".join(lines)[-SHOWN_OUTPUT:]
    return shown


def run_agent(
    client: Any,
    model: str,
    project: Path,
    *,
    run_training: Callable[[Path], dict[str, Any]] | None = None,
    max_tool_calls: int = MAX_TOOL_CALLS,
    max_runs: int = MAX_RUNS,
    pause: float = 0.0,
    error_wait: float = 30.0,
    max_api_errors: int = 3,
    echo_reasoning: bool = False,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Let the model work on a project until it finishes or a limit stops it.

    run_training(project) trains the project and returns what the agent may see of the
    run (shown_run); without it the agent can only read and edit. status is finished,
    or the limit that ended the episode: tool_call_limit, no_tool_call (the model kept
    answering in text) or api_error. claim is the agent's own conclusion, None unless it
    finished. pause, error_wait, max_api_errors and echo_reasoning are as in agent_loop.
    """
    project = Path(project)
    allow_runs = run_training is not None
    system = system_prompt(allow_runs, max_tool_calls, max_runs)
    tools = tool_schemas(allow_runs)
    initial = {name: (project / name).read_text(encoding="utf-8") for name in MODULE_FILES}
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": first_message(project)},
    ]
    record: dict[str, Any] = {
        "schema_version": 1,
        "task": "pipeline_agent",
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "system_prompt_sha256": _sha256(system),
        "tools": [tool["function"]["name"] for tool in tools],
        "limits": {
            "max_tool_calls": max_tool_calls,
            "max_runs": max_runs,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        },
        "status": "running",
        "claim": None,
        "model_calls": 0,
        "tool_calls": 0,
        "training_runs": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        # Totals of what the provider reported; None if it never reported them.
        "cached_input_tokens": None,
        "reasoning_tokens": None,
        "served_models": [],
        "echoes_reasoning": echo_reasoning,
        "api_errors": [],
        "initial_sha256": {name: _sha256(source) for name, source in initial.items()},
        "final_sha256": None,
        "changed_modules": [],
        "steps": [],
    }

    def module(arguments: dict[str, Any]) -> Path | None:
        name = arguments.get("name")
        return project / name if name in MODULE_FILES else None

    def write(arguments: dict[str, Any]) -> dict[str, Any]:
        path, content = module(arguments), arguments.get("content")
        if path is None:
            return {"error": f"only {', '.join(MODULE_FILES)} can be written"}
        if not isinstance(content, str) or not content.strip():
            return {"error": "content must be the new text of the module"}
        if len(content) > MAX_SOURCE_CHARACTERS:
            return {"error": f"a module may have at most {MAX_SOURCE_CHARACTERS} characters"}
        path.write_text(content, encoding="utf-8", newline="\n")
        return {"ok": True, "characters": len(content)}

    def edit(arguments: dict[str, Any]) -> dict[str, Any]:
        path = module(arguments)
        old, new = arguments.get("old_text"), arguments.get("new_text")
        if path is None:
            return {"error": f"only {', '.join(MODULE_FILES)} can be edited"}
        if not isinstance(old, str) or not isinstance(new, str) or not old:
            return {"error": "old_text and new_text must be strings, and old_text not empty"}
        source = path.read_text(encoding="utf-8")
        if source.count(old) != 1:
            return {"error": f"old_text occurs {source.count(old)} times; it must occur once"}
        return write({"name": arguments["name"], "content": source.replace(old, new)})

    def read(arguments: dict[str, Any]) -> dict[str, Any]:
        if arguments.get("name") not in FILES:
            return {"error": f"only {', '.join(FILES)} can be read"}
        return {"content": (project / arguments["name"]).read_text(encoding="utf-8")}

    def train(_arguments: dict[str, Any]) -> dict[str, Any]:
        if not allow_runs:
            return {"error": "the project cannot be run in this session"}
        if record["training_runs"] >= max_runs:
            return {"error": "no training runs are left"}
        record["training_runs"] += 1
        return run_training(project)

    def finish(arguments: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(arguments.get("project_is_correct"), bool):
            return {"error": "project_is_correct must be true or false"}
        record["claim"] = {
            "project_is_correct": arguments["project_is_correct"],
            "problem": str(arguments.get("problem", "")),
            "summary": str(arguments.get("summary", "")),
        }
        return {"ok": True}

    handlers = {
        "read_file": read,
        "edit_file": edit,
        "write_file": write,
        "run_training": train,
        "finish": finish,
    }
    text_replies = 0
    while record["status"] == "running":
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
                record["status"] = "api_error"
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
        if (
            usage["served_model"] is not None
            and usage["served_model"] not in record["served_models"]
        ):
            record["served_models"].append(usage["served_model"])
        message = response.choices[0].message
        calls = list(getattr(message, "tool_calls", None) or [])
        finish = getattr(response.choices[0], "finish_reason", None)
        # Anthropic reports no reasoning tokens; its thinking shows as blocks.
        blocks = getattr(message, "extra_content", None)
        thinking = (
            sum(
                isinstance(block, dict) and block.get("type") in ("thinking", "redacted_thinking")
                for block in blocks
            )
            if isinstance(blocks, list)
            else None
        )
        step = {
            "content": message.content or "",
            "tool_calls": [],
            "finish_reason": finish,
            "thinking_blocks": thinking,
            **usage,
        }
        record["steps"].append(step)
        if not calls:
            text_replies += 1
            if text_replies > MAX_TEXT_REPLIES:
                record["status"] = "no_tool_call"
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
            if record["status"] != "running":
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
                if name == "finish" and record["claim"] is not None:
                    record["status"] = "finished"
                elif record["tool_calls"] >= max_tool_calls:
                    record["status"] = "tool_call_limit"
                result = {**result, "tool_calls_left": max_tool_calls - record["tool_calls"]}
            step["tool_calls"].append(
                {"name": name, "arguments": call.function.arguments, "result": result}
            )
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)}
            )
    final = {name: (project / name).read_text(encoding="utf-8") for name in MODULE_FILES}
    record["final_sha256"] = {name: _sha256(source) for name, source in final.items()}
    record["changed_modules"] = [name for name in MODULE_FILES if final[name] != initial[name]]
    return record
