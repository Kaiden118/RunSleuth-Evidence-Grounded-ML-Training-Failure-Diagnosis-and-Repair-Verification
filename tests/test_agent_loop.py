"""Offline tests for the agent loop: a scripted model works on real task workspaces."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from runsleuth import agent_loop, agent_tasks
from runsleuth.agent_tasks import HEALTHY

MISSING_TRAIN = "    for images, labels in batches:"
REPAIR = {"old_text": MISSING_TRAIN, "new_text": "    model.train()\n" + MISSING_TRAIN}


def reply(*calls, content=None, tokens=(100, 20)):
    """One model reply: tool calls as (name, arguments), or text only."""
    tool_calls = [
        SimpleNamespace(
            id=f"call-{index}",
            function=SimpleNamespace(
                name=name,
                arguments=arguments if isinstance(arguments, str) else json.dumps(arguments),
            ),
        )
        for index, (name, arguments) in enumerate(calls)
    ]
    message = SimpleNamespace(content=content, tool_calls=tool_calls or None)
    usage = SimpleNamespace(prompt_tokens=tokens[0], completion_tokens=tokens[1])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def finish(correct=True, problem="none", summary="Done."):
    return reply(
        ("finish", {"training_is_correct": correct, "problem": problem, "summary": summary})
    )


class Scripted:
    """A client that replays prepared replies and keeps the requests it received."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        self.requests.append(copy.deepcopy(request))
        answer = self.replies.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


class RateLimited(Exception):
    status_code = 429

    def __str__(self):
        return "quota exceeded"


class Rejected(Exception):
    status_code = 400

    def __str__(self):
        return "Function call is missing a thought_signature"


class AgentLoopTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.sleeps = []
        self.runs = []

    def workspace(self, task: str) -> Path:
        return agent_tasks.materialize(task, self.root / f"{task}-{len(list(self.root.iterdir()))}")

    def train(self, workspace: Path) -> dict:
        self.runs.append((workspace / "train.py").read_text(encoding="utf-8"))
        return {"status": "completed", "id_validation_accuracy": 0.97}

    def work(self, workspace, *replies, **options):
        client = Scripted(*replies)
        options.setdefault("run_training", self.train)
        record = agent_loop.run_agent(
            client, "scripted-model", workspace, sleep=self.sleeps.append, **options
        )
        return record, client

    def results(self, record) -> list[dict]:
        return [call["result"] for step in record["steps"] for call in step["tool_calls"]]

    def test_the_agent_reads_repairs_trains_and_states_its_claim(self):
        workspace = self.workspace("train_in_eval_mode")
        record, client = self.work(
            workspace,
            reply(("read_file", {"name": "TASK.md"})),
            reply(("edit_file", REPAIR), content="The loop never calls model.train()."),
            reply(("run_training", {})),
            finish(True, "model.train() was missing", "Added it at the start of each epoch."),
            pause=2.0,
        )
        self.assertEqual(record["status"], "finished")
        self.assertEqual(
            record["claim"],
            {
                "training_is_correct": True,
                "problem": "model.train() was missing",
                "summary": "Added it at the start of each epoch.",
            },
        )
        self.assertEqual(
            (record["model_calls"], record["tool_calls"], record["training_runs"]), (4, 4, 1)
        )
        self.assertEqual((record["input_tokens"], record["output_tokens"]), (400, 80))
        # The edit restored the healthy script, and that is what the harness trained.
        healthy = agent_tasks.task_source(HEALTHY)
        self.assertEqual((workspace / "train.py").read_text(encoding="utf-8"), healthy)
        self.assertEqual(self.runs, [healthy])
        self.assertTrue(record["script_changed"])
        self.assertNotEqual(record["final_script_sha256"], record["initial_script_sha256"])
        self.assertEqual(self.sleeps, [2.0] * 4, "a pause before every model call")
        results = self.results(record)
        self.assertIn("Task", results[0]["content"])
        self.assertEqual(
            results[1], {"ok": True, "characters": len(healthy), "tool_calls_left": 13}
        )
        self.assertEqual(results[2]["id_validation_accuracy"], 0.97)
        self.assertEqual(record["steps"][1]["content"], "The loop never calls model.train().")

    def test_the_model_starts_with_the_statement_the_script_and_its_tools(self):
        workspace = self.workspace("stale_head")
        record, client = self.work(workspace, finish(False, "unsure"), max_runs=2)
        first = client.requests[0]
        self.assertEqual(first["model"], "scripted-model")
        system, user = first["messages"]
        self.assertIn("at most 2 times", system["content"])
        self.assertIn("at most 15 tool calls", system["content"])
        self.assertIn(agent_tasks.STATEMENT, user["content"])
        self.assertIn(agent_tasks.task_source("stale_head"), user["content"])
        names = [tool["function"]["name"] for tool in first["tools"]]
        self.assertEqual(names, ["read_file", "edit_file", "write_file", "run_training", "finish"])
        self.assertEqual(record["tools"], names)
        self.assertEqual(
            first["tools"][-1]["function"]["parameters"]["required"],
            ["training_is_correct", "problem", "summary"],
        )
        self.assertNotIn("required", first["tools"][3]["function"]["parameters"])
        self.assertFalse(record["script_changed"])
        self.assertFalse(record["claim"]["training_is_correct"])
        # Nothing in what the model is shown names the fault.
        shown = json.dumps(first).lower()
        for hint in ("stale", "fault", "bug"):
            self.assertNotIn(hint, shown)

    def test_without_a_training_function_the_agent_can_only_read_and_edit(self):
        record, client = self.work(
            self.workspace(HEALTHY),
            reply(("run_training", {})),
            finish(True),
            run_training=None,
        )
        names = [tool["function"]["name"] for tool in client.requests[0]["tools"]]
        self.assertNotIn("run_training", names)
        self.assertIn("cannot run train.py", client.requests[0]["messages"][0]["content"])
        self.assertIn("cannot be run", self.results(record)[0]["error"])
        self.assertEqual((record["training_runs"], self.runs), (0, []))
        self.assertEqual(record["status"], "finished")

    def test_tool_results_reach_the_model_as_tool_messages(self):
        _, client = self.work(
            self.workspace(HEALTHY), reply(("read_file", {"name": "train.py"})), finish()
        )
        assistant, tool = client.requests[1]["messages"][2:]
        self.assertEqual(assistant["role"], "assistant")
        self.assertEqual(assistant["tool_calls"][0]["function"]["name"], "read_file")
        self.assertEqual((tool["role"], tool["tool_call_id"]), ("tool", "call-0"))
        self.assertEqual(json.loads(tool["content"])["tool_calls_left"], 14)

    def test_invalid_tool_use_is_answered_with_an_error_and_the_episode_goes_on(self):
        workspace = self.workspace(HEALTHY)
        before = (workspace / "train.py").read_text(encoding="utf-8")
        record, _ = self.work(
            workspace,
            reply(
                ("edit_file", {"old_text": "not in the script", "new_text": "x"}),
                ("edit_file", {"old_text": "model", "new_text": "net"}),
                ("write_file", {"content": "x = 1\n" * agent_loop.MAX_SCRIPT_CHARACTERS}),
                ("write_file", {"content": "   "}),
                ("read_file", {"name": "../secrets.txt"}),
                ("delete_everything", {}),
                ("read_file", "{not json"),
                ("finish", {"training_is_correct": "yes", "problem": "", "summary": ""}),
            ),
            finish(),
        )
        errors = [result["error"] for result in self.results(record)[:8]]
        for error, expected in zip(
            errors,
            (
                "occurs 0 times",
                "it must occur once",
                "at most 20000 characters",
                "must be the new text",
                "only train.py and TASK.md",
                "unknown tool delete_everything",
                "not a JSON object",
                "must be true or false",
            ),
            strict=True,
        ):
            self.assertIn(expected, error)
        self.assertEqual((workspace / "train.py").read_text(encoding="utf-8"), before)
        self.assertEqual((record["status"], record["tool_calls"]), ("finished", 9))

    def test_tool_call_and_run_limits_stop_the_episode(self):
        read = ("read_file", {"name": "TASK.md"})
        record, client = self.work(
            self.workspace(HEALTHY), reply(read, read, read, read), max_tool_calls=3
        )
        self.assertEqual((record["status"], record["claim"]), ("tool_call_limit", None))
        self.assertEqual(record["tool_calls"], 3)
        self.assertEqual(self.results(record)[2]["tool_calls_left"], 0)
        self.assertEqual(self.results(record)[3], {"error": "the session has ended"})
        self.assertEqual(len(client.requests), 1)
        run = ("run_training", {})
        record, _ = self.work(self.workspace(HEALTHY), reply(run, run), finish(), max_runs=1)
        self.assertEqual(record["training_runs"], 1)
        self.assertIn("no training runs are left", self.results(record)[1]["error"])

    def test_calls_after_finish_in_one_reply_are_not_carried_out(self):
        workspace = self.workspace("train_in_eval_mode")
        record, _ = self.work(
            workspace,
            reply(
                ("finish", {"training_is_correct": True, "problem": "none", "summary": ""}),
                ("edit_file", REPAIR),
            ),
        )
        self.assertEqual((record["status"], record["tool_calls"]), ("finished", 1))
        self.assertFalse(record["script_changed"])

    def test_a_model_that_only_writes_text_is_nudged_then_stopped(self):
        text = reply(content="I think the script is fine.")
        record, client = self.work(self.workspace(HEALTHY), text, text, text)
        self.assertEqual((record["status"], record["claim"]), ("no_tool_call", None))
        self.assertEqual((record["model_calls"], record["tool_calls"]), (3, 0))
        nudge = client.requests[1]["messages"][-1]
        self.assertEqual(
            nudge, {"role": "user", "content": "Use a tool. Call finish when you are done."}
        )
        record, _ = self.work(self.workspace(HEALTHY), text, finish())
        self.assertEqual(record["status"], "finished")

    def test_provider_errors_are_retried_after_a_wait_then_end_the_episode(self):
        record, _ = self.work(
            self.workspace(HEALTHY), RateLimited(), RateLimited(), finish(), error_wait=45.0
        )
        self.assertEqual(record["status"], "finished")
        self.assertEqual(
            record["api_errors"],
            [{"type": "RateLimited", "status_code": 429, "message": "quota exceeded"}] * 2,
        )
        self.assertEqual(self.sleeps.count(45.0), 2)
        self.assertEqual(record["model_calls"], 1)
        record, _ = self.work(self.workspace(HEALTHY), *[RateLimited()] * 3)
        self.assertEqual((record["status"], record["model_calls"]), ("api_error", 0))
        self.assertEqual(len(record["api_errors"]), 3)

    def test_a_request_the_provider_rejects_is_not_sent_again(self):
        record, client = self.work(self.workspace(HEALTHY), Rejected(), finish())
        self.assertEqual((record["status"], record["claim"]), ("api_error", None))
        self.assertEqual([error["status_code"] for error in record["api_errors"]], [400])
        self.assertEqual(len(client.requests), 1)
        self.assertNotIn(30.0, self.sleeps)

    def test_what_the_provider_attached_to_a_tool_call_is_sent_back_with_it(self):
        # Gemini rejects a conversation whose function call lost its thought signature.
        signature = {"google": {"thought_signature": "opaque"}}
        signed = reply(("read_file", {"name": "TASK.md"}), ("read_file", {"name": "train.py"}))
        signed.choices[0].message.tool_calls[0].extra_content = signature
        text = reply(content="Let me think.")
        text.choices[0].message.extra_content = signature
        record, client = self.work(self.workspace(HEALTHY), signed, text, finish())
        self.assertEqual(record["status"], "finished")
        first, second = client.requests[1]["messages"][2]["tool_calls"]
        self.assertEqual(first["extra_content"], signature)
        self.assertNotIn("extra_content", second)
        self.assertEqual(
            client.requests[2]["messages"][5],
            {"role": "assistant", "content": "Let me think.", "extra_content": signature},
        )
        unsigned = client.requests[2]["messages"][2]
        self.assertEqual(sorted(unsigned), ["content", "role", "tool_calls"])

    def test_each_call_records_the_model_that_served_it_and_the_tokens_it_reports(self):
        # OpenAI reports cached tokens in prompt_tokens_details, DeepSeek at the top level.
        first = reply(("read_file", {"name": "TASK.md"}), tokens=(1000, 300))
        first.model = "provider-model-0813"
        first.usage.prompt_tokens_details = SimpleNamespace(cached_tokens=600)
        first.usage.completion_tokens_details = SimpleNamespace(reasoning_tokens=250)
        second = finish()
        second.model = "provider-model-0901"
        second.usage.prompt_cache_hit_tokens = 900
        record, _ = self.work(self.workspace(HEALTHY), first, second)
        self.assertEqual(record["schema_version"], 2)
        self.assertEqual(record["served_models"], ["provider-model-0813", "provider-model-0901"])
        self.assertEqual((record["cached_input_tokens"], record["reasoning_tokens"]), (1500, 250))
        self.assertEqual((record["input_tokens"], record["output_tokens"]), (1100, 320))
        step = record["steps"][0]
        self.assertEqual(
            {
                name: step[name]
                for name in ("served_model", "cached_input_tokens", "reasoning_tokens")
            },
            {
                "served_model": "provider-model-0813",
                "cached_input_tokens": 600,
                "reasoning_tokens": 250,
            },
        )
        self.assertIsNone(record["steps"][1]["reasoning_tokens"])
        # A provider that reports none of it leaves the totals at None.
        silent, _ = self.work(self.workspace(HEALTHY), finish())
        self.assertEqual(
            (silent["cached_input_tokens"], silent["reasoning_tokens"], silent["served_models"]),
            (None, None, []),
        )
        self.assertFalse(silent["echoes_reasoning"])

    def test_reasoning_goes_back_to_the_provider_only_when_asked(self):
        def thinking(*replies):
            for answer in replies:
                answer.choices[0].message.reasoning_content = "I should read the task first."
            return replies

        echoed, client = self.work(
            self.workspace(HEALTHY),
            *thinking(reply(("read_file", {"name": "TASK.md"})), reply(content="Hmm."), finish()),
            echo_reasoning=True,
        )
        self.assertTrue(echoed["echoes_reasoning"])
        sent = client.requests[2]["messages"]
        self.assertEqual(sent[2]["reasoning_content"], "I should read the task first.")
        self.assertEqual(sent[4]["reasoning_content"], "I should read the task first.")
        self.assertNotIn("reasoning_content", sent[2]["tool_calls"][0])
        _, quiet = self.work(
            self.workspace(HEALTHY),
            *thinking(reply(("read_file", {"name": "TASK.md"})), finish()),
        )
        self.assertNotIn("reasoning_content", quiet.requests[1]["messages"][2])

    def test_the_record_is_json_and_names_the_prompt_it_used(self):
        record, client = self.work(self.workspace(HEALTHY), finish())
        saved = json.loads(json.dumps(record))
        self.assertEqual(saved["prompt_version"], agent_loop.PROMPT_VERSION)
        self.assertEqual(len(saved["system_prompt_sha256"]), 64)
        self.assertEqual(saved["limits"], {"max_tool_calls": 15, "max_runs": 3})
        self.assertEqual(client.requests[0]["max_tokens"], agent_loop.MAX_OUTPUT_TOKENS)


if __name__ == "__main__":
    unittest.main()
