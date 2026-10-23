"""Tests for Claude through Anthropic's Messages API in the agent loops, with a fake client:
thinking blocks go back unchanged, the cache breakpoints sit where they should."""

import json
import tempfile
import types
import unittest
from pathlib import Path

from runsleuth import anthropic_chat, pipeline_tasks

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

if torch is not None:
    from runsleuth import pipeline_agent


class Block(dict):
    """A content block as the SDK returns it: to_dict gives the wire form."""

    def to_dict(self):
        return dict(self)


def reply(blocks, stop="tool_use", read=0, written=0):
    usage = types.SimpleNamespace(
        input_tokens=100,
        output_tokens=40,
        cache_read_input_tokens=read,
        cache_creation_input_tokens=written,
    )
    return types.SimpleNamespace(
        content=[Block(block) for block in blocks],
        stop_reason=stop,
        model="claude-sonnet-5-5",
        usage=usage,
    )


class FakeAnthropic:
    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        self.messages = types.SimpleNamespace(create=self.create)

    def create(self, **request):
        self.requests.append(json.loads(json.dumps(request)))
        return self.replies.pop(0)


class TranslateTests(unittest.TestCase):
    def test_a_conversation_becomes_cached_system_blocks_and_messages(self):
        system, messages = anthropic_chat.translate(
            [
                {"role": "system", "content": "be careful"},
                {"role": "user", "content": "the task"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "a",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": '{"name": "x"}'},
                        },
                        {
                            "id": "b",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": '{"name": "y"}'},
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "a", "content": "{}"},
                {"role": "tool", "tool_call_id": "b", "content": "{}"},
                {"role": "user", "content": "Use a tool."},
            ]
        )
        self.assertEqual(
            system, [{"type": "text", "text": "be careful", "cache_control": {"type": "ephemeral"}}]
        )
        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "user", "user"])
        self.assertEqual([b["tool_use_id"] for b in messages[2]["content"]], ["a", "b"])
        self.assertEqual(messages[1]["content"][0]["input"], {"name": "x"})
        self.assertEqual(messages[-1]["content"][-1]["cache_control"], {"type": "ephemeral"})
        self.assertNotIn("cache_control", messages[0]["content"][-1])

    def test_a_reply_keeps_its_own_blocks_and_reports_what_it_cost(self):
        response = reply(
            [
                {"type": "thinking", "thinking": "", "signature": "s"},
                {"type": "text", "text": "hi"},
            ],
            stop="max_tokens",
            read=900,
            written=50,
        )
        made = anthropic_chat.completion(response)
        choice = made.choices[0]
        self.assertEqual((choice.finish_reason, choice.message.content), ("length", "hi"))
        self.assertEqual(choice.message.extra_content[0]["signature"], "s")
        self.assertEqual(made.usage.prompt_tokens, 1050)
        self.assertEqual(made.usage.prompt_tokens_details.cached_tokens, 900)
        self.assertEqual(made.model, "claude-sonnet-5-5")


@unittest.skipIf(torch is None, "Install torch and torchvision for the pipeline tests")
class LoopTests(unittest.TestCase):
    def test_the_agent_loop_sends_thinking_back_unchanged_with_cached_turns(self):
        fake = FakeAnthropic(
            [
                reply(
                    [
                        {"type": "thinking", "thinking": "", "signature": "sig-1"},
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "read_file",
                            "input": {"name": "TASK.md"},
                        },
                    ]
                ),
                reply(
                    [
                        {"type": "thinking", "thinking": "", "signature": "sig-2"},
                        {
                            "type": "tool_use",
                            "id": "t2",
                            "name": "finish",
                            "input": {
                                "project_is_correct": True,
                                "problem": "none",
                                "summary": "ok",
                            },
                        },
                    ],
                    read=4000,
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            project = pipeline_tasks.materialize("labels_not_filtered", Path(directory) / "project")
            record = pipeline_agent.run_agent(
                anthropic_chat.AnthropicChat(fake), "claude-sonnet-5-5", project
            )
        self.assertEqual(record["status"], "finished")
        first, second = fake.requests
        for request in fake.requests:
            for setting in ("thinking", "temperature", "fallbacks", "output_config"):
                self.assertNotIn(setting, request)
            self.assertEqual(request["tool_choice"], {"type": "auto"})
            self.assertEqual(request["system"][-1]["cache_control"], {"type": "ephemeral"})
            self.assertEqual(
                request["messages"][-1]["content"][-1]["cache_control"], {"type": "ephemeral"}
            )
        self.assertIn("input_schema", first["tools"][0])
        assistant = second["messages"][1]
        self.assertEqual(assistant["role"], "assistant")
        self.assertEqual(
            assistant["content"][0], {"type": "thinking", "thinking": "", "signature": "sig-1"}
        )
        self.assertEqual(second["messages"][2]["content"][0]["tool_use_id"], "t1")
        self.assertEqual(record["cached_input_tokens"], 4000)
        self.assertEqual(record["served_models"], ["claude-sonnet-5-5"])
        self.assertEqual([step["thinking_blocks"] for step in record["steps"]], [1, 1])
        self.assertEqual(
            [step["finish_reason"] for step in record["steps"]], ["tool_calls", "tool_calls"]
        )


if __name__ == "__main__":
    unittest.main()
