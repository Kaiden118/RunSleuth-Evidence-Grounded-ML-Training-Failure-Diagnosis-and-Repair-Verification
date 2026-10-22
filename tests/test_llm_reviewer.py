"""Tests for the LLM reviewer with a fake client: prompts, blindness, replies and resuming."""

import ast
import json
import tempfile
import types
import unittest
from pathlib import Path

from runsleuth import llm_reviewer, pipeline_tasks
from runsleuth.pipeline_tasks import HEALTHY

FAMILIES = ("camelyon_pipeline", "ham10000_pipeline")
RUN = {
    "status": "completed",
    "final_metrics": {"id_validation_accuracy": 0.98123456789},
    "workspace": {
        "seed": 7,
        "project_sha256": "a" * 64,
        "step_state_sha256": ["b" * 64] * 32,
        "epoch_state_sha256": ["c" * 64] * 3,
        "data": {"source": "camelyon17", "provenance": {"revision": "x"}},
    },
    "evaluation_holdouts": {"id": {"evaluation_groups": [f"HAM_{n}" for n in range(40)]}},
    "pipeline": {"checkpoints": [{"name": "best", "state_sha256": "d" * 64, "after_epoch": 1}]},
}


def reply(statuses=("met",) * 6, verdict=None):
    return json.dumps(
        {
            "requirements": [
                {"requirement": number, "status": status, "evidence": "train.py, main"}
                for number, status in zip(llm_reviewer.REQUIREMENTS, statuses, strict=True)
            ],
            "verdict": verdict or llm_reviewer.decision_of(statuses),
            "reason": "as the records show",
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
            completion_tokens=600,
            prompt_tokens_details=types.SimpleNamespace(cached_tokens=0),
            completion_tokens_details=types.SimpleNamespace(reasoning_tokens=200),
        )
        choice = types.SimpleNamespace(
            message=types.SimpleNamespace(content=answer), finish_reason="stop"
        )
        return types.SimpleNamespace(choices=[choice], usage=usage, model="fake-1")


class PromptTests(unittest.TestCase):
    def test_the_prompt_names_no_version_task_or_fault(self):
        for family_name in FAMILIES:
            family = pipeline_tasks.family(family_name)
            healthy = pipeline_tasks.version_sources(HEALTHY, family_name)
            for name, item in family.versions.items():
                sources = pipeline_tasks.version_sources(name, family_name)
                for start in (sources, healthy):
                    prompt = llm_reviewer.user_prompt(family.statement, start, sources, RUN, RUN)
                    with self.subTest(family=family_name, version=name):
                        for word in (name, item.summary, HEALTHY, "fault", "development"):
                            self.assertNotIn(word, prompt)
                        for task in family.tasks:
                            if "_" in task:
                                self.assertNotIn(task, prompt)
        self.assertNotIn("healthy", llm_reviewer.SYSTEM_PROMPT)

    def test_the_prompt_holds_the_statement_files_changes_and_both_runs(self):
        family = pipeline_tasks.family("camelyon_pipeline")
        start = pipeline_tasks.version_sources(HEALTHY)
        candidate = pipeline_tasks.version_sources("labels_not_filtered")
        prompt = llm_reviewer.user_prompt(family.statement, start, candidate, RUN, RUN)
        self.assertIn(family.statement, prompt)
        for source in candidate.values():
            self.assertIn(source, prompt)
        self.assertIn("--- start/data.py\n+++ candidate/data.py", prompt)
        self.assertIn("+        self.labels = list(patches.labels)", prompt)
        self.assertEqual(prompt.count("```json"), 2)
        unchanged = llm_reviewer.user_prompt(family.statement, candidate, candidate, RUN, RUN)
        self.assertIn("The agent changed no file.", unchanged)

    def test_run_records_lose_hashes_and_long_lists_but_not_what_they_observed(self):
        summary = llm_reviewer.run_summary(RUN)
        workspace = summary["workspace"]
        self.assertNotIn("step_state_sha256", workspace)
        self.assertNotIn("project_sha256", workspace)
        self.assertNotIn("provenance", workspace["data"])
        self.assertEqual(len(workspace["epoch_state_sha256"]), 3)
        self.assertEqual(summary["evaluation_holdouts"]["id"]["evaluation_groups"], "40 entries")
        self.assertEqual(summary["pipeline"]["checkpoints"], [{"name": "best", "after_epoch": 1}])
        self.assertEqual(summary["final_metrics"]["id_validation_accuracy"], 0.981235)
        self.assertIn("step_state_sha256", RUN["workspace"])

    def test_the_reviewer_reads_neither_the_contract_nor_the_challenge_set(self):
        tree = ast.parse(Path(llm_reviewer.__file__).read_text(encoding="utf-8"))
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module.startswith("runsleuth")
        }
        self.assertEqual(
            imported,
            {
                "runsleuth",
                "runsleuth.diagnose_run",
                "runsleuth.llm_client",
                "runsleuth.pipeline_tasks",
            },
        )
        text = Path(llm_reviewer.__file__).read_text(encoding="utf-8")
        for name in ("pipeline_contract", "challenge_set", "case_provenance", "expected_failures"):
            self.assertNotIn(name, text)


class ReplyTests(unittest.TestCase):
    def test_the_decision_follows_from_the_requirements(self):
        self.assertEqual(llm_reviewer.decision_of(["met"] * 6), "accept")
        self.assertEqual(llm_reviewer.decision_of(["met"] * 5 + ["unclear"]), "review")
        self.assertEqual(llm_reviewer.decision_of(["unclear", "violated"] + ["met"] * 4), "reject")

    def test_a_reply_is_read_through_fences_and_reasoning(self):
        for text in (reply(), f"```json\n{reply()}\n```", f"<think>\nhmm\n</think>\n{reply()}"):
            parsed, issues = llm_reviewer.parse_review(text)
            self.assertEqual(issues, [])
            self.assertEqual(parsed["verdict"], "accept")

    def test_an_unusable_reply_says_what_is_wrong(self):
        missing = json.loads(reply())
        missing["requirements"].pop()
        wrong_status = json.loads(reply())
        wrong_status["requirements"][0]["status"] = "fine"
        for text, expected in (
            ("not json", "not one JSON object"),
            (json.dumps(missing), "one entry for each requirement"),
            (json.dumps(wrong_status), "every status must be one of"),
            (json.dumps({**json.loads(reply()), "verdict": "maybe"}), "verdict must be one of"),
        ):
            parsed, issues = llm_reviewer.parse_review(text)
            self.assertIsNone(parsed)
            self.assertTrue(any(expected in issue for issue in issues), issues)

    def test_a_review_decides_from_the_requirements_and_records_its_cost(self):
        client = FakeClient([reply(("met",) * 5 + ("violated",), verdict="accept")])
        record = llm_reviewer.review(client, "fake", "prompt")
        self.assertEqual((record["status"], record["decision"]), ("completed", "reject"))
        self.assertFalse(record["verdict_consistent"])
        call = record["calls"][0]
        self.assertEqual((call["input_tokens"], call["reasoning_tokens"]), (9000, 200))
        self.assertEqual(call["served_model"], "fake-1")
        request = client.requests[0]
        self.assertEqual(request["response_format"]["type"], "json_schema")
        self.assertEqual(request["messages"][0]["content"], llm_reviewer.SYSTEM_PROMPT)

    def test_an_unusable_reply_is_answered_once_then_decides_review(self):
        client = FakeClient(["not json", reply()])
        record = llm_reviewer.review(client, "fake", "prompt")
        self.assertEqual((record["status"], record["model_calls"]), ("completed", 2))
        self.assertIn("could not be used", client.requests[1]["messages"][-1]["content"])
        record = llm_reviewer.review(FakeClient(["not json", "still not"]), "fake", "prompt")
        self.assertEqual((record["status"], record["decision"]), ("invalid_output", "review"))

    def test_an_api_error_ends_the_review_without_a_decision(self):
        record = llm_reviewer.review(FakeClient([RuntimeError("quota")]), "fake", "prompt")
        self.assertEqual((record["status"], record["decision"]), ("api_error", None))
        self.assertEqual(record["error"]["type"], "RuntimeError")


class ResumeTests(unittest.TestCase):
    ITEMS = [{"id": f"item-{number}", "prompt": f"prompt {number}"} for number in range(4)]

    def test_reviews_resume_where_they_stopped_and_retry_api_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            client = FakeClient([reply(), RuntimeError("quota"), reply(), "x", "y"])
            llm_reviewer.review_all(
                self.ITEMS, output, client, "fake", provider="fake", progress=lambda _: None
            )
            statuses = {r["id"]: r["status"] for r in llm_reviewer.read_reviews(output)}
            self.assertEqual(
                statuses,
                {
                    "item-0": "completed",
                    "item-1": "api_error",
                    "item-2": "completed",
                    "item-3": "invalid_output",
                },
            )
            again = FakeClient([reply()])
            written = llm_reviewer.review_all(
                self.ITEMS, output, again, "fake", provider="fake", progress=lambda _: None
            )
            self.assertEqual([record["id"] for record in written], ["item-1"])
            text = llm_reviewer.summary_text(llm_reviewer.read_reviews(output))
            self.assertIn("4 reviews", text)

    def test_reviewing_stops_after_api_errors_in_a_row(self):
        with tempfile.TemporaryDirectory() as directory:
            client = FakeClient([RuntimeError("down")] * 3)
            written = llm_reviewer.review_all(
                self.ITEMS,
                Path(directory),
                client,
                "fake",
                provider="fake",
                progress=lambda _: None,
            )
            self.assertEqual(len(written), llm_reviewer.MAX_CONSECUTIVE_API_ERRORS)


if __name__ == "__main__":
    unittest.main()
