"""Tests for programs written by a model, with a fake client: what counts as a program
and a claim, one call per attempt, every attempt kept, and resuming."""

import contextlib
import io
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
    from runsleuth import standalone_programs, standalone_task

PROGRAM = "import torch\n\nprint('trained')\n"


def reply(program=PROGRAM, claim="CLAIM: the program meets every requirement"):
    return f"Here is the program.\n\n```python\n{program}```\n\n{claim}\n"


class FakeClient:
    """Answers with the given replies in turn; an exception is raised instead."""

    def __init__(self, replies, finish_reason="stop"):
        self.replies, self.requests, self.finish_reason = list(replies), [], finish_reason
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

    def create(self, **request):
        self.requests.append(request)
        answer = self.replies.pop(0)
        if isinstance(answer, Exception):
            raise answer
        usage = types.SimpleNamespace(
            prompt_tokens=900,
            completion_tokens=2500,
            prompt_tokens_details=types.SimpleNamespace(cached_tokens=0),
            completion_tokens_details=types.SimpleNamespace(reasoning_tokens=1200),
        )
        choice = types.SimpleNamespace(
            message=types.SimpleNamespace(content=answer), finish_reason=self.finish_reason
        )
        return types.SimpleNamespace(choices=[choice], usage=usage, model="fake-1")


@unittest.skipIf(torch is None, "Install torch and torchvision for the standalone tests")
class ReplyTests(unittest.TestCase):
    def test_the_program_is_the_only_python_block(self):
        parsed = standalone_programs.parse_reply(reply())
        self.assertEqual(parsed["program"], PROGRAM)
        self.assertEqual(parsed["python_blocks"], 1)
        shell = reply() + "\n```bash\npython train.py --seed 1\n```\n"
        self.assertEqual(standalone_programs.parse_reply(shell)["program"], PROGRAM)
        two = reply() + "\n```python\nprint('a helper')\n```\n"
        self.assertIsNone(standalone_programs.parse_reply(two)["program"])
        self.assertEqual(standalone_programs.parse_reply(two)["python_blocks"], 2)
        bare = f"```\n{PROGRAM}```\nCLAIM: the program meets every requirement"
        self.assertIsNone(standalone_programs.parse_reply(bare)["program"])
        self.assertEqual(standalone_programs.parse_reply("I cannot write it.")["python_blocks"], 0)

    def test_the_claim_is_one_of_the_two_sentences_asked_for(self):
        parse = standalone_programs.parse_reply
        self.assertEqual(parse(reply())["claim"], "meets")
        self.assertEqual(
            parse(reply(claim="**CLAIM: The program meets every requirement.**"))["claim"], "meets"
        )
        failing = parse(
            reply(
                claim="CLAIM: the program does not meet every requirement\n\n"
                "Requirement 5: the flips of a resumed epoch are drawn again."
            )
        )
        self.assertEqual(failing["claim"], "does_not_meet")
        self.assertEqual(
            failing["claim_reason"], "Requirement 5: the flips of a resumed epoch are drawn again."
        )
        self.assertEqual(parse(reply(claim="CLAIM: it should work"))["claim"], "unparsed")
        self.assertIsNone(parse(reply(claim="It meets every requirement."))["claim"])
        inside = "```python\n# CLAIM: the program meets every requirement\n```\n"
        self.assertIsNone(parse(inside)["claim"])
        self.assertIsNone(parse(reply())["claim_reason"])


@unittest.skipIf(torch is None, "Install torch and torchvision for the standalone tests")
class AttemptTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.messages = []

    def write(self, client, attempts, dataset="camelyon"):
        return standalone_programs.write_attempts(
            self.root,
            dataset,
            "model-a",
            client,
            "served-a",
            provider="compatible",
            attempts=attempts,
            size=16,
            progress=self.messages.append,
        )

    def test_one_call_asks_for_the_statement_and_nothing_else(self):
        client = FakeClient([reply()])
        (record,) = self.write(client, 1)
        (request,) = client.requests
        self.assertEqual(sorted(request), ["max_tokens", "messages", "model"])
        self.assertEqual(request["model"], "served-a")
        self.assertEqual(request["max_tokens"], standalone_programs.MAX_OUTPUT_TOKENS)
        system, user = request["messages"]
        self.assertEqual(system, {"role": "system", "content": standalone_programs.SYSTEM_PROMPT})
        self.assertEqual(
            user, {"role": "user", "content": standalone_task.statement("camelyon", 16)}
        )
        self.assertEqual(record["status"], "written")
        self.assertEqual(record["generation"], "one call, no tools")
        self.assertEqual(record["claim"], "meets")
        self.assertEqual(record["program_lines"], 3)
        self.assertEqual(record["call"]["served_model"], "fake-1")
        self.assertEqual(record["call"]["output_tokens"], 2500)
        self.assertEqual(record["call"]["raw_text"], reply())
        self.assertNotIn("program", record)

    def test_every_attempt_is_kept_with_its_program_beside_it(self):
        client = FakeClient([reply(), "I cannot write this program.", reply() + reply()])
        records = self.write(client, 3)
        self.assertEqual(
            [record["status"] for record in records],
            ["written", "no_program", "several_programs"],
        )
        first = self.root / "model-a" / "camelyon" / "attempt-1"
        self.assertEqual((first / "train.py").read_text(encoding="utf-8"), PROGRAM)
        stored = json.loads((first / "attempt.json").read_text(encoding="utf-8"))
        self.assertEqual(stored["id"], standalone_programs.attempt_id("model-a", "camelyon", 1))
        self.assertNotIn("model-a", stored["id"])
        self.assertEqual(stored["attempt"], 1)
        for number in (2, 3):
            directory = self.root / "model-a" / "camelyon" / f"attempt-{number}"
            self.assertTrue((directory / "attempt.json").is_file())
            self.assertFalse((directory / "train.py").exists())
        self.assertEqual(len(standalone_programs.read_attempts(self.root)), 3)
        self.assertEqual(len({record["id"] for record in records}), 3)

    def test_a_reply_cut_off_leaves_no_program(self):
        (record,) = self.write(FakeClient([reply()], finish_reason="length"), 1)
        self.assertEqual(record["status"], "truncated")
        self.assertIsNone(record["program_sha256"])
        self.assertFalse((self.root / "model-a" / "camelyon" / "attempt-1" / "train.py").exists())

    def test_a_failed_call_is_no_attempt_and_is_made_again(self):
        self.write(FakeClient([reply(), RuntimeError("quota")]), 2)
        statuses = [record["status"] for record in standalone_programs.read_attempts(self.root)]
        self.assertEqual(statuses, ["written", "api_error"])
        client = FakeClient([reply(program="print('second')\n")])
        (record,) = self.write(client, 2)
        self.assertEqual((record["attempt"], record["status"]), (2, "written"))
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(self.write(FakeClient([]), 2), [])

    def test_the_run_stops_after_three_failed_calls_in_a_row(self):
        client = FakeClient([RuntimeError("down")] * 5)
        records = self.write(client, 5)
        self.assertEqual(len(records), 3)
        self.assertIn("stopped after 3 API errors in a row", self.messages[-1])

    def test_a_model_name_must_name_a_directory(self):
        with self.assertRaisesRegex(ValueError, "cannot name a directory"):
            standalone_programs.write_attempts(
                self.root, "camelyon", "../x", FakeClient([]), "m", provider="gemini", attempts=1
            )

    def test_the_list_counts_attempts_programs_and_tokens(self):
        self.assertEqual(standalone_programs.list_text([]), "no attempts")
        self.write(FakeClient([reply(), "no code", RuntimeError("quota")]), 3)
        text = standalone_programs.list_text(standalone_programs.read_attempts(self.root))
        self.assertIn("2 attempts, 1 with a program; 1800 input and 5000 output tokens", text)
        self.assertIn("api_error", text)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            standalone_programs.main(["list", str(self.root)])
        self.assertEqual(output.getvalue().strip(), text)


if __name__ == "__main__":
    unittest.main()
