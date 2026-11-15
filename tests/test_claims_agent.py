"""Tests for the claims-task agent with a fake client: the small copy, what the agent is
told and shown, its limits, and whole episodes that run a tiny program."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

from test_claims_task import camelyon_rows, ham_rows
from test_standalone_agent import FakeClient, call, finish

from runsleuth import claims_agent, claims_task

HAM, CAMELYON = claims_task.SPECS["ham10000"], claims_task.SPECS["camelyon"]


def export(root, rows, chosen):
    def image(position):
        return np.full((3, chosen.size, chosen.size), position % 251, dtype=np.uint8)

    claims_task.export(rows, image, chosen, root, seed=7)
    return root / claims_task.GIVEN


def table(directory):
    with (Path(directory) / claims_task.METADATA).open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


class SmallCopyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_a_small_copy_keeps_whole_lesions_in_the_given_format(self):
        given = export(self.root / "ham", ham_rows(), HAM)
        record = claims_agent.small_copy(given, self.root / "small", HAM, rows=40, seed=1)
        full, small = table(given), table(self.root / "small")
        self.assertLessEqual(len(small), 40)
        self.assertEqual(record["samples"], len(small))
        self.assertEqual([int(line["row"]) for line in small], list(range(len(small))))
        counts = {}
        for line in full:
            counts[line["lesion_id"]] = counts.get(line["lesion_id"], 0) + 1
        for lesion in {line["lesion_id"] for line in small}:
            self.assertEqual(sum(line["lesion_id"] == lesion for line in small), counts[lesion])
        images = np.load(self.root / "small" / claims_task.IMAGES)
        originals = np.load(given / claims_task.IMAGES)
        by_image = {line["image_id"]: int(line["row"]) for line in full}
        for line, pixels in zip(small, images, strict=True):
            self.assertTrue((pixels == originals[by_image[line["image_id"]]]).all())
        self.assertTrue((self.root / "small.json").is_file())
        self.assertEqual(
            sorted(p.name for p in (self.root / "small").iterdir()), ["images.npy", "metadata.csv"]
        )

    def test_a_small_camelyon_copy_takes_a_few_patches_of_every_slide(self):
        given = export(self.root / "camelyon", camelyon_rows(), CAMELYON)
        claims_agent.small_copy(given, self.root / "small", CAMELYON, rows=512, per_group=10)
        small = table(self.root / "small")
        slides = {line["slide"] for line in table(given)}
        self.assertEqual({line["slide"] for line in small}, slides)
        per = {}
        for line in small:
            key = (line["slide"], line["label"])
            per[key] = per.get(key, 0) + 1
        self.assertEqual(max(per.values()), 10)


class PromptTests(unittest.TestCase):
    def test_the_agent_is_told_nothing_of_how_it_is_judged(self):
        text = claims_agent.system_prompt(30, 10, 512, 8137, 300).lower()
        for word in ("leak", "sealed", "lesion", "slide", "group", "audit", "judge", "estimate"):
            self.assertNotIn(word, text)
        self.assertIn("512 of the 8137 samples", text)
        names = [tool["function"]["name"] for tool in claims_agent.tool_schemas()]
        self.assertEqual(
            names, ["read_file", "write_file", "edit_file", "show_data", "run_program", "finish"]
        )


class EpisodeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.given = export(self.root / "ham", ham_rows(), HAM)
        self.asked = []

    def shown(self, program, arguments):
        self.asked.append((program.read_text(encoding="utf-8"), arguments))
        return {"status": "completed", "returncode": 0, "output": "ok", "files": []}

    def run_agent(self, replies, **options):
        client = FakeClient(replies)
        workspace = self.root / "workspace"
        workspace.mkdir(exist_ok=True)
        record = claims_agent.run_agent(
            client,
            "served",
            workspace,
            claims_task.statement("ham10000"),
            given=self.given,
            small_samples=40,
            run_program=self.shown,
            sleep=lambda seconds: None,
            **options,
        )
        return client, record

    def test_an_episode_writes_looks_runs_and_finishes(self):
        client, record = self.run_agent(
            [
                [call("write_file", 1, content="print('hi')\n")],
                [call("show_data", 2)],
                [call("run_program", 3, run="first", seed=4)],
                [call("run_program", 4, run="first", seed=5)],
                finish(5, reason="none"),
            ]
        )
        self.assertEqual(record["episode_status"], "finished")
        self.assertEqual(record["agent_claim"], {"meets_every_requirement": True, "reason": "none"})
        self.assertEqual(record["looked_at_data"], 1)
        self.assertEqual(
            record["runs"], [{"run": "first", "seed": 4, "status": "completed", "prediction": None}]
        )
        self.assertEqual(self.asked, [("print('hi')\n", {"run": "first", "seed": 4})])
        results = [step["tool_calls"][0]["result"] for step in record["steps"]]
        self.assertEqual(results[1]["images.npy"]["shape"][0], len(table(self.given)))
        self.assertEqual(len(results[1]["metadata.csv"]["first_lines"]), 21)
        self.assertIn("has been used", results[3]["error"])
        system = client.requests[0]["messages"][0]["content"]
        self.assertIn(f"40 of the {len(table(self.given))} samples", system)

    def test_runs_are_limited(self):
        replies = [[call("write_file", 0, content="pass\n")]]
        replies += [[call("run_program", n, run=f"r{n}", seed=n)] for n in range(1, 4)]
        _, record = self.run_agent(replies + [finish(9)], max_runs=2)
        self.assertEqual(len(record["runs"]), 2)
        self.assertIn("no runs are left", record["steps"][3]["tool_calls"][0]["result"]["error"])


@unittest.skipIf(torch is None, "Install torch and torchvision for the agent's runs")
class RunnerTests(unittest.TestCase):
    def test_a_run_trains_on_the_small_copy_and_predicts_its_images(self):
        from test_claims_run import TINY

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            given = export(root / "ham", ham_rows(), HAM)
            claims_agent.small_copy(given, root / "small", HAM, rows=40)
            program = root / claims_task.PROGRAM
            program.write_text(TINY, encoding="utf-8")
            run = claims_agent.program_runner(root / "small", root / "runs", device="cpu")
            shown = run(program, {"run": "one", "seed": 3})
            self.assertEqual(shown["status"], "completed", shown["output"])
            self.assertEqual(json.loads(shown["report"])["threshold"], 0.5)
            self.assertIn(
                ["delivered.pt", (root / "runs/one/delivered.pt").stat().st_size], shown["files"]
            )
            written = shown["prediction"]["written"]
            self.assertEqual(
                (shown["prediction"]["status"], written["shape"]),
                ("completed", [written["samples"]]),
            )
            # The images were alone, as in the official run, and their copy is gone.
            self.assertLessEqual(written["max"], 1.0)
            self.assertFalse((root / "runs" / "one-prediction" / claims_task.IMAGES).exists())
            (root / "bad.py").write_text("import subprocess\n", encoding="utf-8")
            self.assertEqual(run(root / "bad.py", {"run": "two", "seed": 3})["status"], "refused")

    def test_attempts_are_kept_with_their_program_and_made_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            export(root / "data" / "ham10000", ham_rows(), HAM)
            replies = [
                [call("write_file", 1, content="pass\n")],
                finish(2, meets=False, reason="x"),
            ]
            lines = []
            (record,) = claims_agent.write_attempts(
                root / "programs",
                "ham10000",
                "pressure",
                "fake",
                FakeClient(replies),
                "served",
                provider="test",
                attempts=1,
                data=root / "data",
                work=root / "work",
                device="cpu",
                progress=lines.append,
                sleep=lambda seconds: None,
            )
            directory = root / "programs" / "fake" / "ham10000" / "pressure" / "attempt-1"
            self.assertEqual((directory / claims_task.PROGRAM).read_text(), "pass\n")
            self.assertEqual(
                (record["status"], record["claim"], record["condition"]),
                ("written", "does_not_meet", "pressure"),
            )
            pressed = claims_task.statement("ham10000", pressure=True)
            self.assertEqual(record["statement_sha256"], claims_agent._sha256(pressed))
            self.assertFalse((root / "work" / "runs" / record["id"]).exists())
            self.assertTrue(
                (root / "work" / "data-small" / "ham10000" / claims_task.METADATA).is_file()
            )
            again = claims_agent.write_attempts(
                root / "programs",
                "ham10000",
                "pressure",
                "fake",
                FakeClient([]),
                "served",
                provider="test",
                attempts=1,
                data=root / "data",
                work=root / "work",
            )
            self.assertEqual(again, [])
            self.assertEqual(len(claims_agent.read_attempts(root / "programs")), 1)


if __name__ == "__main__":
    unittest.main()
