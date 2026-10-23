"""Chat Completions for the agent loops, over Anthropic's own Messages API.

Claude reached through an OpenAI-compatible endpoint loses prompt caching and its
thinking. AnthropicChat gives an anthropic.Anthropic client the one method the loops
call, chat.completions.create, and translates each way:

- the request: the system message becomes a cached system block, the tools Anthropic
  tools with automatic choice, tool results one user message of tool_result blocks per
  turn, and the last message carries the second cache breakpoint, so every turn reads
  the conversation before it from the cache;
- the reply: text, tool calls, a finish reason and the usage the loops record. The
  reply's own content blocks, thinking blocks and their signatures among them, ride
  along as extra_content, which the loops send back with the reply
  (agent_loop._with_provider_content); the next request returns them unchanged, as an
  append-only conversation with thinking requires.

Thinking, effort and sampling stay at the model's defaults, as for every other model;
no server-side fallback is asked for, so a declined request ends with finish reason
refusal instead of being answered by another model.
"""

import json
import types
from typing import Any

CACHE = {"type": "ephemeral"}
FINISH_REASONS = {
    "end_turn": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "refusal": "refusal",
    "pause_turn": "pause",
}


def _tool(schema: dict[str, Any]) -> dict[str, Any]:
    function = schema["function"]
    return {
        "name": function["name"],
        "description": function.get("description", ""),
        "input_schema": function.get("parameters") or {"type": "object", "properties": {}},
    }


def _assistant(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """An assistant turn as Anthropic content blocks: its own, if it came from Claude."""
    if isinstance(entry.get("extra_content"), list):
        return entry["extra_content"]
    blocks: list[dict[str, Any]] = []
    if entry.get("content"):
        blocks.append({"type": "text", "text": entry["content"]})
    for call in entry.get("tool_calls") or []:
        try:
            arguments = json.loads(call["function"]["arguments"] or "{}")
        except ValueError:
            arguments = {}
        blocks.append(
            {
                "type": "tool_use",
                "id": call["id"],
                "name": call["function"]["name"],
                "input": arguments,
            }
        )
    return blocks


def translate(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The system blocks and the messages of a Chat Completions conversation."""
    system: list[dict[str, Any]] = []
    converted: list[dict[str, Any]] = []
    for entry in messages:
        role = entry["role"]
        if role == "system":
            system.append({"type": "text", "text": entry["content"]})
        elif role == "assistant":
            converted.append({"role": "assistant", "content": _assistant(entry)})
        elif role == "tool":
            result = {
                "type": "tool_result",
                "tool_use_id": entry["tool_call_id"],
                "content": entry["content"],
            }
            previous = converted[-1] if converted else None
            if previous and previous["role"] == "user" and previous.get("tool_results"):
                previous["content"].append(result)
            else:
                converted.append({"role": "user", "content": [result], "tool_results": True})
        else:
            converted.append(
                {"role": "user", "content": [{"type": "text", "text": entry["content"]}]}
            )
    for message in converted:
        message.pop("tool_results", None)
    if system:
        system[-1] = {**system[-1], "cache_control": CACHE}
    if converted and converted[-1]["content"]:
        last = converted[-1]["content"]
        converted[-1] = {
            **converted[-1],
            "content": [*last[:-1], {**last[-1], "cache_control": CACHE}],
        }
    return system, converted


def _block(block: Any) -> dict[str, Any]:
    return block.to_dict() if hasattr(block, "to_dict") else dict(block)


def completion(response: Any) -> Any:
    """A Messages API reply in the shape the loops read from a Chat Completions one."""
    blocks = [_block(block) for block in response.content]
    text = "".join(block["text"] for block in blocks if block.get("type") == "text")
    calls = [
        types.SimpleNamespace(
            id=block["id"],
            type="function",
            function=types.SimpleNamespace(
                name=block["name"], arguments=json.dumps(block["input"])
            ),
        )
        for block in blocks
        if block.get("type") == "tool_use"
    ]
    message = types.SimpleNamespace(content=text, tool_calls=calls, extra_content=blocks)
    usage = response.usage
    read = getattr(usage, "cache_read_input_tokens", None) or 0
    written = getattr(usage, "cache_creation_input_tokens", None) or 0
    return types.SimpleNamespace(
        model=response.model,
        choices=[
            types.SimpleNamespace(
                message=message,
                finish_reason=FINISH_REASONS.get(response.stop_reason, response.stop_reason),
            )
        ],
        usage=types.SimpleNamespace(
            prompt_tokens=usage.input_tokens + read + written,
            completion_tokens=usage.output_tokens,
            prompt_tokens_details=types.SimpleNamespace(cached_tokens=read),
            completion_tokens_details=types.SimpleNamespace(reasoning_tokens=None),
            cache_creation_input_tokens=written,
        ),
    )


class AnthropicChat:
    """An anthropic.Anthropic client with chat.completions.create, as the loops call it."""

    def __init__(self, client: Any) -> None:
        self.client = client
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        **_ignored: Any,
    ) -> Any:
        system, converted = translate(messages)
        request: dict[str, Any] = {"model": model, "max_tokens": max_tokens, "messages": converted}
        if system:
            request["system"] = system
        if tools:
            request["tools"] = [_tool(tool) for tool in tools]
            request["tool_choice"] = {"type": tool_choice}
        return completion(self.client.messages.create(**request))
