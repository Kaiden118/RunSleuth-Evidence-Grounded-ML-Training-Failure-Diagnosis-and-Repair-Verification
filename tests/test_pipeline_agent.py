"""CPU tests for the pipeline agent: its tools and limits with a fake client, and what a
short run shows it."""

import json
import tempfile
import types
import unittest
from pathlib import Path

from test_pipeline_workspace import tiny_sources, write_project

from runsleuth import pipeline_tasks

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

if torch is not None:
    from runsleuth import pipeline_agent, pipeline_workspace

FIX = (
    "self.labels = list(patches.labels)",
    "self.labels = [patches.labels[index] for index in self.indices]",
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
        usage = types.SimpleNamespace(prompt_tokens=5000, completion_tokens=100)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=message)], usage=usage, model="fake-1"
        )


class ApiError(Exception):
    def __init__(self, status_code):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


@unittest.skipIf(torch is None, "Install torch and torchvision for the pipeline tests")
class EpisodeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.project = pipeline_tasks.materialize(
            "labels_not_filtered", Path(temporary.name) / "project"
        )

    def run_agent(self, replies, **options):
        client = FakeClient(replies)
        record = pipeline_agent.run_agent(client, "fake", self.project, **options)
        return client, record

    def test_an_agent_reads_edits_and_finishes(self):
        client, record = self.run_agent(
            [
                [call("read_file", 1, name="TASK.md")],
                [call("edit_file", 2, name="data.py", old_text=FIX[0], new_text=FIX[1])],
                [call("finish", 3, project_is_correct=True, problem="labels", summary="fixed")],
            ]
        )
        self.assertEqual(record["status"], "finished")
        self.assertEqual(record["claim"]["project_is_correct"], True)
        self.assertEqual(record["changed_modules"], ["data.py"])
        self.assertIn(FIX[1], (self.project / "data.py").read_text(encoding="utf-8"))
        first = client.requests[0]["messages"][1]["content"]
        for name in pipeline_agent.FILES:
            self.assertIn(f"{name}:\n\n", first)
        self.assertEqual(record["served_models"], ["fake-1"])
        self.assertEqual(record["input_tokens"], 15000)
        tools = [tool["function"]["name"] for tool in client.requests[0]["tools"]]
        self.assertEqual(tools, ["read_file", "edit_file", "write_file", "finish"])

    def test_only_the_modules_can_be_changed_and_edits_must_be_exact(self):
        _, record = self.run_agent(
            [
                [
                    call("write_file", 1, name="TASK.md", content="Do nothing."),
                    call("edit_file", 2, name="train.py", old_text="model", new_text="net"),
                    call("write_file", 3, name="config.py", content=""),
                    call("read_file", 4, name="../secret.txt"),
                ],
                [call("finish", 5, project_is_correct="yes")],
                [call("finish", 6, project_is_correct=False, problem="none", summary="")],
            ]
        )
        results = [tool["result"] for step in record["steps"] for tool in step["tool_calls"]]
        self.assertIn("only config.py, data.py, train.py can be written", results[0]["error"])
        self.assertIn("it must occur once", results[1]["error"])
        self.assertIn("content must be", results[2]["error"])
        self.assertIn("can be read", results[3]["error"])
        self.assertIn("true or false", results[4]["error"])
        self.assertEqual(record["changed_modules"], [])
        self.assertEqual(record["status"], "finished")

    def test_short_runs_are_offered_and_counted(self):
        runs = []

        def run_training(project):
            runs.append(project)
            return {"status": "completed"}

        replies = [[call("run_training", number)] for number in range(1, 5)]
        replies.append([call("finish", 9, project_is_correct=True, problem="none", summary="")])
        client, record = self.run_agent(replies, run_training=run_training, max_runs=3)
        self.assertEqual(len(runs), 3)
        self.assertEqual(record["training_runs"], 3)
        results = [tool["result"] for step in record["steps"] for tool in step["tool_calls"]]
        self.assertEqual(results[3]["error"], "no training runs are left")
        self.assertIn("run_training", [t["function"]["name"] for t in client.requests[0]["tools"]])
        self.assertIn("at most 3 times", client.requests[0]["messages"][0]["content"])

    def test_limits_end_an_episode(self):
        reads = [[call("read_file", number, name="data.py")] for number in range(3)]
        _, record = self.run_agent(reads, max_tool_calls=3)
        self.assertEqual((record["status"], record["claim"]), ("tool_call_limit", None))
        _, record = self.run_agent(["thinking", "still thinking", "done"])
        self.assertEqual(record["status"], "no_tool_call")
        _, record = self.run_agent([ApiError(400)])
        self.assertEqual(record["status"], "api_error")
        self.assertEqual(record["api_errors"][0]["status_code"], 400)


@unittest.skipIf(torch is None, "Install torch and torchvision for the pipeline tests")
class ShortRunTests(unittest.TestCase):
    def test_a_short_run_shows_what_a_developer_sees_and_not_the_harness_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            project = write_project(Path(directory) / "project", tiny_sources())
            report = pipeline_workspace.run_project(
                project,
                Path(directory) / "run",
                pipeline_workspace.data_builder("synthetic"),
                epochs=pipeline_agent.SHORT_RUN["epochs"],
                seed=7,
                device=torch.device("cpu"),
                interrupt_after_epochs=pipeline_agent.SHORT_RUN["interrupt_after_epochs"],
            )
            harness_line = json.dumps({"status": "completed", "report": str(report)})
            outcome = {
                "status": "completed",
                "problems": [],
                "report": str(report),
                "error": None,
                "output": f"epoch 1 done\n{harness_line}\n",
            }
            shown = pipeline_agent.shown_run(outcome)
        self.assertEqual([attempt["epochs_trained"] for attempt in shown["attempts"]], [1, 1])
        self.assertIn("stopped by the harness", shown["attempts"][0]["ended"])
        self.assertEqual([row["epoch"] for row in shown["logged"]], [1, 2])
        self.assertIn("validation_accuracy", shown["logged"][0])
        self.assertTrue(shown["delivered_weights"])
        self.assertEqual(shown["output"], "epoch 1 done")
        text = json.dumps(shown)
        for hidden in ("final_metrics", "id_validation", "ood_validation", "sha256", str(report)):
            self.assertNotIn(hidden, text)

    def test_a_refused_project_shows_why(self):
        shown = pipeline_agent.shown_run(
            {"status": "refused", "problems": ["line 1: import of 'os' is not allowed"]}
        )
        self.assertEqual(shown["status"], "refused")
        self.assertEqual(shown["problems"], ["line 1: import of 'os' is not allowed"])


if __name__ == "__main__":
    unittest.main()
