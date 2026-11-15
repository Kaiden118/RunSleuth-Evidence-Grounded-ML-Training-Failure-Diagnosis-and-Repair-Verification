"""Tests for the matched-evidence screening with a fake client: what each input asks,
the plan of calls, resuming, and the summary."""

import json
import tempfile
import types
import unittest
from pathlib import Path

from runsleuth import llm_reviewer, matched_screening


def reply(statuses=("met",) * 6):
    return json.dumps(
        {
            "requirements": [
                {"requirement": number, "status": status, "evidence": "the records"}
                for number, status in zip(llm_reviewer.REQUIREMENTS, statuses, strict=True)
            ],
            "verdict": llm_reviewer.decision_of(statuses),
            "reason": "as the runs show",
        }
    )


class FakeClient:
    """Answers with the given replies in turn; an exception is raised instead."""

    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

    def create(self, **request):
        self.requests.append(request)
        answer = self.replies.pop(0)
        if isinstance(answer, Exception):
            raise answer
        usage = types.SimpleNamespace(
            prompt_tokens=9000,
            completion_tokens=700,
            prompt_tokens_details=types.SimpleNamespace(cached_tokens=0),
            completion_tokens_details=types.SimpleNamespace(reasoning_tokens=None),
        )
        choice = types.SimpleNamespace(
            message=types.SimpleNamespace(content=answer), finish_reason="stop"
        )
        return types.SimpleNamespace(choices=[choice], usage=usage, model="fake-1")


def fake_items():
    """Items as build_items makes them, for every name the screening asks."""
    items = {}
    for name in matched_screening.WRONG + matched_screening.CONTROLS:
        user = f"files and runs of {name}"
        items[name] = {
            "wrong": name in matched_screening.WRONG,
            "v2": {"decision": "reject" if name in matched_screening.WRONG else "accept"},
            "inputs": {
                "b": (llm_reviewer.SYSTEM_PROMPT, f"old prompt of {name}"),
                "b_prime": (matched_screening.BOUNDED_SYSTEM_PROMPT, f"old prompt of {name}"),
                "c": (matched_screening.BOUNDED_SYSTEM_PROMPT, user),
                "d": (
                    matched_screening.BOUNDED_SYSTEM_PROMPT,
                    user + "\n" + matched_screening.RULES,
                ),
            },
        }
    return items


class InputTests(unittest.TestCase):
    def test_the_bounded_question_differs_from_the_stored_one_only_in_what_it_asks(self):
        bounded, old = matched_screening.BOUNDED_SYSTEM_PROMPT, llm_reviewer.SYSTEM_PROMPT
        self.assertIn("within the runs you are given", bounded)
        self.assertIn("0.01", bounded)
        self.assertIn("the first is delivered", bounded)
        self.assertNotIn("within the runs you are given", old)
        self.assertEqual(bounded.split("Reply with")[1], old.split("Reply with")[1])
        for word in ("v2", "RunSleuth", "verifier", "probe"):
            self.assertNotIn(word, bounded)
            self.assertNotIn(word, matched_screening.RULES)

    def test_the_rules_cover_every_requirement(self):
        for number in llm_reviewer.REQUIREMENTS:
            self.assertIn(f"\n{number}. ", "\n" + matched_screening.RULES.split("\n\n", 1)[1])
        self.assertIn("the earliest on a tie", matched_screening.RULES)

    def test_v2s_parts_become_the_reviewers_requirements(self):
        self.assertEqual(matched_screening.requirement_of("training_labels"), 1)
        self.assertEqual(matched_screening.requirement_of("delivery"), 6)
        self.assertIsNone(matched_screening.requirement_of("runs"))
        parts = {part: "supported" for part in matched_screening.verifier_v2.PARTS}
        self.assertEqual(set(matched_screening.requirements_of(parts).values()), {"met"})
        parts.update(delivery="violated", restart="untested")
        found = matched_screening.requirements_of(parts)
        self.assertEqual((found[5], found[6], found[1]), ("unclear", "violated", "met"))
        failed = {
            **{p: "supported" for p in matched_screening.verifier_v2.PARTS},
            "runs": "violated",
        }
        found = matched_screening.requirements_of(failed)
        self.assertEqual((found[5], found[6], found[4]), ("violated", "violated", "met"))

    def test_items_that_are_one_input_are_asked_once(self):
        items = fake_items()
        twin = "camelyon_pipeline/best_accuracy_not_restored/as_is"
        first = matched_screening.WRONG[0]
        kept = matched_screening.distinct({**items, twin: items[first]}, {twin: first})
        self.assertEqual(list(kept), list(items))
        with self.assertRaisesRegex(ValueError, "does not have the inputs"):
            matched_screening.distinct(
                {**items, twin: items[matched_screening.WRONG[1]]}, {twin: first}
            )
        doubled = {**items, "another": items[first]}
        with self.assertRaisesRegex(ValueError, "are one input"):
            matched_screening.distinct(doubled, {})

    def test_the_passes_ask_44_and_then_24_calls(self):
        items = fake_items()
        first, second = matched_screening.asked(items, 1), matched_screening.asked(items, 2)
        self.assertEqual((len(first), len(second)), (44, 24))
        self.assertEqual({repetition for _, _, repetition in first}, {1})
        self.assertEqual({repetition for _, _, repetition in second}, {2, 3})
        self.assertEqual({name for name, _, _ in second}, set(matched_screening.WRONG))
        self.assertEqual(len(set(first) | set(second)), 68)


class RunTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.items, self.messages = fake_items(), []

    def run_pass(self, replies, pass_=1, limit=None):
        client = FakeClient(replies)
        written = matched_screening.run_pass(
            self.items,
            pass_,
            self.root,
            client,
            "served",
            provider="anthropic",
            limit=limit,
            progress=self.messages.append,
        )
        return client, written

    def test_a_review_asks_the_inputs_system_prompt_and_reads_its_answer(self):
        client, (record,) = self.run_pass([reply(("met",) * 5 + ("violated",))], limit=1)
        (request,) = client.requests
        name = matched_screening.WRONG[0]
        system, prompt = self.items[name]["inputs"]["b"]
        self.assertEqual(request["messages"][0], {"role": "system", "content": system})
        self.assertEqual(request["messages"][1], {"role": "user", "content": prompt})
        self.assertEqual(request["response_format"]["type"], "json_schema")
        self.assertEqual((record["item"], record["input"], record["repetition"]), (name, "b", 1))
        self.assertEqual((record["status"], record["decision"]), ("completed", "reject"))
        self.assertEqual(record["requirements"][6], "violated")
        self.assertEqual(record["calls"][0]["input_tokens"], 9000)
        self.assertIn(
            f"{name} b #1: completed, reject; 9000 input, 700 output tokens", self.messages[0]
        )

    def test_a_pass_resumes_and_asks_failed_calls_again(self):
        self.run_pass([reply(), RuntimeError("quota")], limit=2)
        stored = matched_screening.read_reviews(self.root)
        self.assertEqual([r["status"] for r in stored], ["completed", "api_error"])
        client, written = self.run_pass([reply()], limit=1)
        self.assertEqual((written[0]["input"], written[0]["status"]), ("b_prime", "completed"))
        self.assertEqual(len(matched_screening.read_reviews(self.root)), 3)

    def test_an_unreadable_reply_is_answered_once_and_then_decides_review(self):
        client, (record,) = self.run_pass(["not json", "still not"], limit=1)
        self.assertEqual((record["status"], record["decision"]), ("invalid_output", "review"))
        self.assertEqual(record["model_calls"], 2)
        self.assertIn("could not be used", client.requests[1]["messages"][-1]["content"])

    def test_three_failed_calls_in_a_row_stop_a_pass(self):
        _, written = self.run_pass([RuntimeError("down")] * 3)
        self.assertEqual(len(written), 3)
        self.assertIn("stopped after 3 API errors in a row", self.messages[-1])

    def test_the_summary_sets_each_input_beside_v2(self):
        self.assertEqual(matched_screening.summary_text([], self.items), "no reviews")
        self.run_pass([reply(), reply(("met",) * 5 + ("violated",)), reply(), reply()], limit=4)
        text = matched_screening.summary_text(matched_screening.read_reviews(self.root), self.items)
        self.assertIn("best_accuracy_not_restored/gemini-3.5-flash-lite/r1", text)
        self.assertIn("b: wrong items accepted 1 of 1", text)
        self.assertIn("b_prime: wrong items accepted 0 of 1", text)
        self.assertIn("4 reviews, 4 calls: 36000 input and 2800 output tokens", text)


if __name__ == "__main__":
    unittest.main()
