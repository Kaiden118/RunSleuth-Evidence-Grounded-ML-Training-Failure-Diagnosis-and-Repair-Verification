"""An LLM agent that reviews and repairs one training script through tools.

The agent is given a workspace (train.py and TASK.md, see agent_tasks) and nothing
else. It can read both files, edit train.py and, when the episode allows it, ask the
harness to train the script for a bounded budget. It ends by calling finish with its
own claim: whether training is correct now.

What a training run returns to the agent is decided by the caller, which is how a
study varies the evidence an agent gets; this loop only relays it. The loop makes no
decision about the repair either: a claim is what the agent said, never a verdict.
Every model call, tool call and result is recorded.

The client is an OpenAI-compatible one with tool calling (see llm_client).
"""

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from runsleuth.agent_tasks import SCRIPT_NAME, STATEMENT_NAME

PROMPT_VERSION = 1
MAX_OUTPUT_TOKENS = 4000
# The harness refuses longer scripts (agent_workspace.MAX_SOURCE_CHARACTERS).
MAX_SCRIPT_CHARACTERS = 20_000
MAX_TEXT_REPLIES = 2
# Long enough for a provider to say which quota or which part of a request it refused.
MAX_ERROR_CHARACTERS = 1500


def system_prompt(allow_runs: bool, max_tool_calls: int, max_runs: int) -> str:
    """The instructions every episode starts with; only the budget lines vary."""
    running = (
        f"run_training trains the current {SCRIPT_NAME} for a short, fixed budget and returns "
        f"what the harness measured. You may call it at most {max_runs} times."
        if allow_runs
        else f"You cannot run {SCRIPT_NAME} in this session; judge it by reading it."
    )
    return (
        "You are a machine-learning engineer reviewing a PyTorch training script.\n"
        f"{STATEMENT_NAME} says what {SCRIPT_NAME} is meant to do. Decide whether the script "
        "trains the model correctly. If it does not, fix it with the smallest change that "
        "makes it correct; if it does, leave it alone.\n"
        f"Work only through the tools. {running}\n"
        "When you are done, call finish. Set training_is_correct to true only if the script, "
        "as it stands after your changes, now trains the model correctly. In problem, name "
        "what was wrong, or write none. Keep summary to a few sentences.\n"
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
    tools = [
        tool(
            "read_file",
            "Read a file of the workspace.",
            {"name": {"type": "string", "enum": [SCRIPT_NAME, STATEMENT_NAME]}},
        ),
        tool(
            "edit_file",
            f"Replace one exact occurrence of old_text in {SCRIPT_NAME} with new_text.",
            {"old_text": text, "new_text": text},
        ),
        tool("write_file", f"Replace the whole of {SCRIPT_NAME}.", {"content": text}),
    ]
    if allow_runs:
        tools.append(
            tool("run_training", f"Train the current {SCRIPT_NAME} for a short, fixed budget.")
        )
    tools.append(
        tool(
            "finish",
            "End the session with your conclusion.",
            {"training_is_correct": {"type": "boolean"}, "problem": text, "summary": text},
        )
    )
    return tools


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _with_provider_content(entry: dict[str, Any], source: Any) -> dict[str, Any]:
    """Return what a provider attached to its reply along with the reply.

    Gemini signs its function calls (extra_content.google.thought_signature) and
    rejects a conversation in which a call comes back without its signature.
    """
    extra = getattr(source, "extra_content", None)
    if extra is not None:
        entry["extra_content"] = extra
    return entry


def _rejected(status_code: Any) -> bool:
    """The provider refused the request as it stands, so sending it again cannot help."""
    if not isinstance(status_code, int) or status_code in (408, 429):  # timeout, rate limit
        return False
    return 400 <= status_code < 500


def run_agent(
    client: Any,
    model: str,
    workspace: Path,
    *,
    run_training: Callable[[Path], dict[str, Any]] | None = None,
    max_tool_calls: int = 15,
    max_runs: int = 3,
    pause: float = 0.0,
    error_wait: float = 30.0,
    max_api_errors: int = 3,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Let the model work on a workspace until it finishes or a limit stops it.

    run_training(workspace) trains the script and returns what the agent may see of the
    run; without it the agent can only read and edit. status is finished, or the limit
    that ended the episode: tool_call_limit, no_tool_call (the model kept answering in
    text) or api_error. claim is the agent's own conclusion, None unless it finished.
    pause is waited before every model call and error_wait after a failed one, for
    rate-limited providers; a request the provider rejects is not sent again.
    """
    workspace = Path(workspace)
    script, statement = workspace / SCRIPT_NAME, workspace / STATEMENT_NAME
    allow_runs = run_training is not None
    system = system_prompt(allow_runs, max_tool_calls, max_runs)
    tools = tool_schemas(allow_runs)
    initial = script.read_text(encoding="utf-8")
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                f"{STATEMENT_NAME}:\n\n{statement.read_text(encoding='utf-8')}\n"
                f"{SCRIPT_NAME}:\n\n```python\n{initial}```"
            ),
        },
    ]
    record: dict[str, Any] = {
        "schema_version": 1,
        "task": "agent_loop",
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "system_prompt_sha256": _sha256(system),
        "tools": [tool["function"]["name"] for tool in tools],
        "limits": {"max_tool_calls": max_tool_calls, "max_runs": max_runs},
        "status": "running",
        "claim": None,
        "model_calls": 0,
        "tool_calls": 0,
        "training_runs": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "api_errors": [],
        "initial_script_sha256": _sha256(initial),
        "final_script_sha256": None,
        "script_changed": False,
        "steps": [],
    }

    def edit(arguments: dict[str, Any]) -> dict[str, Any]:
        old, new = arguments.get("old_text"), arguments.get("new_text")
        if not isinstance(old, str) or not isinstance(new, str) or not old:
            return {"error": "old_text and new_text must be strings, and old_text not empty"}
        source = script.read_text(encoding="utf-8")
        if source.count(old) != 1:
            return {"error": f"old_text occurs {source.count(old)} times; it must occur once"}
        return write({"content": source.replace(old, new)})

    def write(arguments: dict[str, Any]) -> dict[str, Any]:
        content = arguments.get("content")
        if not isinstance(content, str) or not content.strip():
            return {"error": "content must be the new text of the script"}
        if len(content) > MAX_SCRIPT_CHARACTERS:
            return {"error": f"the script may have at most {MAX_SCRIPT_CHARACTERS} characters"}
        script.write_text(content, encoding="utf-8", newline="\n")
        return {"ok": True, "characters": len(content)}

    def read(arguments: dict[str, Any]) -> dict[str, Any]:
        if arguments.get("name") not in (SCRIPT_NAME, STATEMENT_NAME):
            return {"error": f"only {SCRIPT_NAME} and {STATEMENT_NAME} can be read"}
        return {"content": (workspace / arguments["name"]).read_text(encoding="utf-8")}

    def train(_arguments: dict[str, Any]) -> dict[str, Any]:
        if not allow_runs:
            return {"error": "the script cannot be run in this session"}
        if record["training_runs"] >= max_runs:
            return {"error": "no training runs are left"}
        record["training_runs"] += 1
        return run_training(workspace)

    def finish(arguments: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(arguments.get("training_is_correct"), bool):
            return {"error": "training_is_correct must be true or false"}
        record["claim"] = {
            "training_is_correct": arguments["training_is_correct"],
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
        usage = getattr(response, "usage", None)
        record["input_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
        record["output_tokens"] += getattr(usage, "completion_tokens", 0) or 0
        message = response.choices[0].message
        calls = list(getattr(message, "tool_calls", None) or [])
        step = {"content": message.content or "", "tool_calls": []}
        record["steps"].append(step)
        if not calls:
            text_replies += 1
            if text_replies > MAX_TEXT_REPLIES:
                record["status"] = "no_tool_call"
                break
            messages.append(
                _with_provider_content(
                    {"role": "assistant", "content": message.content or ""}, message
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
    final = script.read_text(encoding="utf-8")
    record["final_script_sha256"] = _sha256(final)
    record["script_changed"] = final != initial
    return record
