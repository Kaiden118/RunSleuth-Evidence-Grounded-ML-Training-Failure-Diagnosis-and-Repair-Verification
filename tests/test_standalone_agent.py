"""Tests for the agent that writes a program and can run it, with a fake client: what it
is told and shown, its limits, and whole episodes that run a tiny program."""

import json
import tempfile
import types
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

if torch is not None:
    from test_standalone_check import TINY

    from runsleuth import (
        pipeline_workspace,
        standalone_agent,
        standalone_check,
        standalone_programs,
        standalone_task,
    )


def call(tool, number, **arguments):
    return types.SimpleNamespace(
        id=f"call-{number}",
        function=types.SimpleNamespace(name=tool, arguments=json.dumps(arguments)),
    )


class FakeClient:
    """Answers with the given replies in turn: a list of tool calls, text, or an exception."""

    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

    def create(self, **request):
        self.requests.append(request)
        answer = self.replies.pop(0)
        if isinstance(answer, Exception):
            raise answer
        calls = answer if isinstance(answer, list) else []
        message = types.SimpleNamespace(
            content=answer if isinstance(answer, str) else "", tool_calls=calls
        )
        usage = types.SimpleNamespace(prompt_tokens=4000, completion_tokens=300)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=message)], usage=usage, model="fake-1"
        )


class ApiError(Exception):
    def __init__(self, status_code):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


def finish(number, meets=True, reason="none"):
    return [call("finish", number, meets_every_requirement=meets, reason=reason)]


@unittest.skipIf(torch is None, "Install torch and torchvision for the standalone tests")
class EpisodeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name)
        self.asked = []

    def shown(self, program, arguments):
        self.asked.append((program.read_text(encoding="utf-8"), arguments))
        return {"status": "completed", "returncode": 0, "output": "ok", "files": []}

    def run_agent(self, replies, **options):
        client = FakeClient(replies)
        record = standalone_agent.run_agent(
            client,
            "fake",
            self.workspace,
            "the statement",
            run_program=self.shown,
            sleep=lambda _seconds: None,
            **options,
        )
        return client, record

    def test_the_agent_is_told_its_tools_and_its_budget(self):
        client, record = self.run_agent([finish(1)])
        request = client.requests[0]
        system, user = request["messages"][:2]
        self.assertEqual(user, {"role": "user", "content": "the statement"})
        for said in (
            "at most 10 times",
            "at most 300 seconds",
            "256 training and 128 validation samples",
            "at most 30 tool calls in total",
            "a run can be stopped and resumed in it",
        ):
            self.assertIn(said, system["content"])
        self.assertNotIn("check", system["content"].lower())
        names = [tool["function"]["name"] for tool in request["tools"]]
        self.assertEqual(names, ["read_file", "write_file", "edit_file", "run_program", "finish"])
        self.assertEqual(record["tools"], names)
        run_tool = request["tools"][3]["function"]["parameters"]
        self.assertEqual(run_tool["required"], ["run", "seed", "epochs"])
        self.assertEqual(
            sorted(run_tool["properties"]), ["epochs", "resume", "run", "seed", "stop_after"]
        )
        self.assertEqual(record["generation"], "agent with a run tool")
        self.assertEqual(record["statement_version"], standalone_task.STATEMENT_VERSION)

    def test_an_episode_writes_runs_and_claims(self):
        client, record = self.run_agent(
            [
                [call("write_file", 1, content="print('one')\n")],
                [call("run_program", 2, run="a", seed=1, epochs=2, stop_after=1)],
                [call("edit_file", 3, old_text="one", new_text="two")],
                [call("run_program", 4, run="a", seed=1, epochs=2, resume=True)],
                [call("read_file", 5)],
                finish(6, meets=False, reason="requirement 5: not tested"),
            ]
        )
        self.assertEqual(record["episode_status"], "finished")
        self.assertEqual(
            record["agent_claim"],
            {"meets_every_requirement": False, "reason": "requirement 5: not tested"},
        )
        self.assertEqual((record["model_calls"], record["tool_calls"]), (6, 6))
        self.assertEqual((record["input_tokens"], record["output_tokens"]), (24000, 1800))
        self.assertEqual(record["served_models"], ["fake-1"])
        self.assertEqual(
            record["runs"],
            [
                {
                    "run": "a",
                    "seed": 1,
                    "epochs": 2,
                    "stop_after": 1,
                    "resume": False,
                    "status": "completed",
                },
                {"run": "a", "seed": 1, "epochs": 2, "resume": True, "status": "completed"},
            ],
        )
        self.assertEqual([source for source, _ in self.asked], ["print('one')\n", "print('two')\n"])
        self.assertEqual(
            (self.workspace / "train.py").read_text(encoding="utf-8"), "print('two')\n"
        )
        results = [step["tool_calls"][0]["result"] for step in record["steps"]]
        self.assertEqual(results[1]["runs_left"], 9)
        self.assertEqual(results[4]["content"], "print('two')\n")
        self.assertEqual(
            [result["tool_calls_left"] for result in results], [29, 28, 27, 26, 25, 24]
        )
        # The conversation the model is sent holds every result as a tool message.
        answers = {
            message["tool_call_id"]: json.loads(message["content"])
            for message in client.requests[-1]["messages"]
            if message["role"] == "tool"
        }
        self.assertEqual(answers["call-2"]["status"], "completed")
        self.assertEqual(answers["call-2"]["runs_left"], 9)

    def test_mistaken_calls_are_answered_with_an_error(self):
        _, record = self.run_agent(
            [
                [call("run_program", 1, run="a", seed=1, epochs=2)],
                [call("edit_file", 2, old_text="x", new_text="y")],
                [call("read_file", 3)],
                [call("write_file", 4, content="  ")],
                [call("write_file", 5, content="x = 1\nx = 1\n")],
                [call("edit_file", 6, old_text="x = 1", new_text="x = 2")],
                [call("run_program", 7, run="../a", seed=1, epochs=2)],
                [call("run_program", 8, run="a", seed="1", epochs=2)],
                [call("run_program", 9, run="a", epochs=2)],
                [call("finish", 10, meets_every_requirement="yes", reason="")],
                [call("delete_file", 11)],
                [
                    types.SimpleNamespace(
                        id="call-12", function=types.SimpleNamespace(name="finish", arguments="{")
                    )
                ],
                finish(13),
            ]
        )
        errors = [step["tool_calls"][0]["result"].get("error") for step in record["steps"]]
        for position, said in enumerate(
            (
                "has not been written yet",
                "has not been written yet",
                "has not been written yet",
                "content must be the text",
                None,
                "occurs 2 times",
                "run must be a name",
                "must be whole numbers",
                "seed and epochs are needed",
                "must be true or false",
                "unknown tool delete_file",
                "not a JSON object",
                None,
            )
        ):
            with self.subTest(position=position):
                if said is None:
                    self.assertIsNone(errors[position])
                else:
                    self.assertIn(said, errors[position])
        self.assertEqual(record["runs"], [])
        self.assertEqual(self.asked, [])
        self.assertEqual(record["episode_status"], "finished")

    def test_the_runs_and_the_tool_calls_are_limited(self):
        replies = [[call("write_file", 1, content="x = 1\n")]] + [
            [call("run_program", number, run="a", seed=1, epochs=1)] for number in range(2, 6)
        ]
        _, record = self.run_agent(replies, max_runs=2, max_tool_calls=5)
        self.assertEqual(record["episode_status"], "tool_call_limit")
        self.assertIsNone(record["agent_claim"])
        self.assertEqual(len(record["runs"]), 2)
        self.assertIn("no runs are left", record["steps"][3]["tool_calls"][0]["result"]["error"])

    def test_text_replies_and_provider_errors_end_an_episode(self):
        client, record = self.run_agent(["I will write it.", "Almost.", "Done."])
        self.assertEqual(record["episode_status"], "no_tool_call")
        self.assertEqual(
            client.requests[1]["messages"][-1]["content"],
            "Use a tool. Call finish when you are done.",
        )
        _, record = self.run_agent([ApiError(401)])
        self.assertEqual(record["episode_status"], "api_error")
        self.assertEqual(record["api_errors"][0]["status_code"], 401)
        _, record = self.run_agent([ApiError(429), finish(1)])
        self.assertEqual((record["episode_status"], len(record["api_errors"])), ("finished", 1))


@unittest.skipIf(torch is None, "Install torch and torchvision for the standalone tests")
class AttemptTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        standalone_task.export(pipeline_workspace.synthetic_pipeline_data(7), self.root / "data")
        self.messages = []

    def write(self, replies, attempts=1):
        return standalone_agent.write_attempts(
            self.root / "programs",
            "camelyon",
            "model-a",
            FakeClient(replies),
            "served-a",
            provider="compatible",
            attempts=attempts,
            data=self.root / "data",
            work=self.root / "work",
            size=16,
            device="cpu",
            progress=self.messages.append,
            sleep=lambda _seconds: None,
        )

    def test_an_agent_runs_its_program_stops_it_and_resumes_it(self):
        (record,) = self.write(
            [
                [call("write_file", 1, content=TINY)],
                [call("run_program", 2, run="whole", seed=3, epochs=2)],
                [call("run_program", 3, run="parts", seed=3, epochs=2, stop_after=1)],
                [call("run_program", 4, run="parts", seed=3, epochs=2, resume=True)],
                finish(5),
            ]
        )
        self.assertEqual((record["status"], record["episode_status"]), ("written", "finished"))
        self.assertEqual(record["claim"], "meets")
        self.assertEqual(record["id"], standalone_agent.attempt_id("model-a", "camelyon", 1))
        self.assertNotEqual(record["id"], standalone_programs.attempt_id("model-a", "camelyon", 1))
        self.assertEqual([run["status"] for run in record["runs"]], ["completed"] * 3)
        whole, stopped, resumed = (step["tool_calls"][0]["result"] for step in record["steps"][1:4])

        def names(shown):
            return [name for name, _size in shown["files"]]

        self.assertIn("delivered.pt", names(whole))
        self.assertEqual(len(whole["log_end"]), 2)
        self.assertNotIn("delivered.pt", names(stopped))
        self.assertNotIn("epoch_2.pt", names(stopped))
        self.assertIn("delivered.pt", names(resumed))
        self.assertEqual(json.loads(resumed["log_end"][-1])["epoch"], 2)
        for shown in (whole, stopped, resumed):
            self.assertEqual(
                sorted(shown),
                [
                    "files",
                    "log_end",
                    "output",
                    "returncode",
                    "runs_left",
                    "seconds",
                    "status",
                    "tool_calls_left",
                ],
            )
        directory = self.root / "programs" / "model-a" / "camelyon" / "attempt-1"
        self.assertEqual((directory / "train.py").read_text(encoding="utf-8"), TINY)
        stored = json.loads((directory / "attempt.json").read_text(encoding="utf-8"))
        self.assertEqual(stored["call"]["input_tokens"], 20000)
        self.assertEqual(stored["program_lines"], len(TINY.splitlines()))
        self.assertFalse((self.root / "work" / "runs" / record["id"]).exists())
        small = json.loads(
            (self.root / "work" / "data-small" / "camelyon" / "data.json").read_text()
        )
        self.assertEqual(small["train"]["of"], 24)
        self.assertIn("3 runs (1 resumed, 0 not completed)", self.messages[0])
        # What standalone_programs and standalone_check read is there.
        (attempt,) = standalone_programs.read_attempts(self.root / "programs")
        self.assertEqual(attempt["generation"], "agent with a run tool")
        self.assertIn("1 attempts, 1 with a program", standalone_agent.list_text([attempt]))
        self.assertEqual(
            standalone_check.screen((directory / "train.py").read_text(encoding="utf-8")), []
        )

    def test_a_refused_program_is_not_run_and_an_agent_may_end_without_one(self):
        records = self.write(
            [
                [call("write_file", 1, content="import subprocess\n")],
                [call("run_program", 2, run="a", seed=1, epochs=1)],
                finish(3, meets=False, reason="could not run it"),
                finish(1, meets=False, reason="nothing written"),
            ],
            attempts=2,
        )
        refused = records[0]["steps"][1]["tool_calls"][0]["result"]
        self.assertEqual(refused["status"], "refused")
        self.assertIn("import of 'subprocess' is not allowed", refused["problems"][0])
        self.assertEqual(records[0]["runs"][0]["status"], "refused")
        self.assertEqual((records[0]["status"], records[0]["claim"]), ("written", "does_not_meet"))
        self.assertEqual((records[1]["status"], records[1]["program_lines"]), ("no_program", None))
        self.assertEqual(self.write([], attempts=2), [])

    def test_a_failed_episode_is_no_attempt_and_is_made_again(self):
        (record,) = self.write([[call("write_file", 1, content="x = 1\n")], ApiError(401)])
        self.assertEqual(record["status"], "api_error")
        self.assertIn("stopped at an API error", self.messages[-1])
        (record,) = self.write([[call("write_file", 1, content="x = 2\n")], finish(2)])
        self.assertEqual(record["status"], "written")
        directory = self.root / "programs" / "model-a" / "camelyon" / "attempt-1"
        self.assertEqual((directory / "train.py").read_text(encoding="utf-8"), "x = 2\n")
        self.assertEqual((directory / "train.py.earlier").read_text(encoding="utf-8"), "x = 1\n")


if __name__ == "__main__":
    unittest.main()
