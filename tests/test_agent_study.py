"""CPU tests for the agent study: the draw, what it freezes, and a run on drawn seeds."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_agent_loop import Scripted, finish
from test_agent_workspace import tiny_source

from runsleuth.agent_tasks import HEALTHY

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

if torch is not None:
    from test_agent_episodes import BUDGET, contract, run_in_process

    from runsleuth import agent_episodes, agent_study


@unittest.skipIf(torch is None, "Install torch and torchvision for the study tests")
class AgentStudyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = self.root / "config.json"
        self.config.write_text('{"seed": 7}', encoding="utf-8")
        self.evaluations = self.root / "evaluations"
        self.evaluations.mkdir()
        earlier = {"seeds": [101, 102], "excluded_seeds": [7, 103]}
        (self.evaluations / "earlier_draw.json").write_text(json.dumps(earlier), encoding="utf-8")
        (self.evaluations / "reviews.json").write_text('{"reviews": []}', encoding="utf-8")
        self.draw_path = self.evaluations / "draw.json"
        for target, value in (
            ("runsleuth.agent_tasks.task_source", tiny_source),
            ("runsleuth.agent_study.TASKS", ("stale_head", HEALTHY)),
            (
                "runsleuth.agent_study.CONDITIONS",
                {name: agent_episodes.CONDITIONS[name] for name in ("C0", "C3")},
            ),
        ):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def draw(self, count=2, numbers=(101, 7, 555, 555, 103, 777, 888), status=""):
        numbers = iter(numbers)
        return agent_study.draw_seeds(
            count,
            self.draw_path,
            config=self.config,
            evaluations=self.evaluations,
            randbelow=lambda _limit: next(numbers),
            git=lambda *arguments: status if arguments[0] == "status" else "abc123",
        )

    def read(self):
        return agent_study.read_draw(self.draw_path, config=self.config)

    def test_a_draw_skips_used_seeds_and_records_what_is_frozen(self):
        record = self.draw()
        self.assertEqual(record["seeds"], [555, 777])
        self.assertEqual(record["excluded_seeds"], [7, 11, 42, 101, 102, 103, 2026])
        self.assertEqual((record["code_commit"], record["episodes_per_model"]), ("abc123", 8))
        frozen = record["frozen"]
        self.assertEqual(frozen["tasks"], ["stale_head", HEALTHY])
        self.assertEqual(list(frozen["conditions"]), ["C0", "C3"])
        self.assertEqual(frozen["agent"]["max_tool_calls"], agent_study.MAX_TOOL_CALLS)
        self.assertEqual(frozen["budgets"]["quick"]["epochs"], agent_episodes.QUICK.epochs)
        self.assertEqual(frozen["contract"]["version"], 2)
        self.assertEqual(
            frozen["repair_policy"], {"max_accuracy_drop": 0.01, "max_loss_ratio": 1.1}
        )
        for name in (
            "task_statement_sha256",
            "task_scripts_sha256",
            "failure_signatures_sha256",
            "verification_source_sha256",
        ):
            self.assertRegex(frozen[name], r"^[0-9a-f]{64}$")
        self.assertEqual(json.loads(self.draw_path.read_text(encoding="utf-8")), record)
        self.assertEqual(self.read(), record)
        # A later draw would exclude these seeds too.
        self.assertIn(555, agent_study.used_seeds(self.evaluations))

    def test_a_study_has_one_draw_of_committed_code(self):
        with self.assertRaisesRegex(RuntimeError, "Commit the source changes first"):
            self.draw(status=" M src/runsleuth/agent_tasks.py")
        for count in (1, 11, True):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "two to ten"):
                self.draw(count)
        self.draw()
        with self.assertRaisesRegex(FileExistsError, "one draw"):
            self.draw()

    def test_a_change_to_anything_frozen_stops_the_study(self):
        self.draw()
        changes = {
            "task_statement_sha256": patch("runsleuth.agent_tasks.STATEMENT", "Another task."),
            "task_scripts_sha256, tasks": patch("runsleuth.agent_study.TASKS", (HEALTHY,)),
            "presentation_version": patch("runsleuth.agent_study.PRESENTATION_VERSION", 99),
            "agent": patch("runsleuth.agent_study.MAX_RUNS", 9),
            "budgets": patch("runsleuth.agent_study.FULL", BUDGET),
            "repair_policy": patch.dict(
                "runsleuth.camelyon_optimizer_training.REPAIR_POLICY", {"max_loss_ratio": 2.0}
            ),
        }
        for changed, change in changes.items():
            with self.subTest(changed=changed), change:
                with self.assertRaisesRegex(RuntimeError, f"and changed: {changed}$"):
                    self.read()
        self.config.write_text('{"seed": 8}', encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "and changed: data$"):
            self.read()

    def test_a_model_runs_every_task_and_condition_on_every_drawn_seed(self):
        draw = self.draw()
        store = agent_study.study_store(
            self.root / "study", data="synthetic", config=None, runner=run_in_process
        )
        lines = []
        rows = agent_study.prepare_study(
            draw, store, quick=BUDGET, full=BUDGET, progress=lines.append
        )
        self.assertEqual(lines[0], "seed 555")
        self.assertEqual({row["seed"] for row in rows}, {555, 777})
        self.assertEqual({row["status"] for row in rows}, {"completed"})
        options = {
            "directory": self.root / "study",
            "quick": BUDGET,
            "full": BUDGET,
            "contract": contract,
            "provider": "gemini",
            "sleep": lambda _seconds: None,
        }
        client = Scripted(*[finish(True, "none")] * 8)
        summary = agent_study.run_study("model-a", client, "model-a", draw, store, **options)
        self.assertEqual((summary["episodes"], summary["models"]), (8, ["model-a"]))
        self.assertEqual(len(client.requests), 8)
        episodes = agent_episodes.load_episodes(self.root / "study" / "model-a")
        self.assertEqual({episode["seed"] for episode in episodes}, {555, 777})
        self.assertEqual({episode["condition"] for episode in episodes}, {"C0", "C3"})
        self.assertEqual(
            {episode["agent"]["limits"]["max_runs"] for episode in episodes},
            {agent_study.MAX_RUNS},
        )
        # Unchanged scripts were prepared, so the episodes trained nothing; a second
        # model shares the store, and a repeated command asks nothing.
        self.assertTrue((self.root / "study" / "runs").is_dir())
        for seed in draw["seeds"]:
            cached = store.for_seed(seed).run(tiny_source(HEALTHY), BUDGET, train=False)
            self.assertIsNotNone(cached)
        again = agent_study.run_study("model-a", Scripted(), "model-a", draw, store, **options)
        self.assertEqual(again, summary)
        other = Scripted(*[finish(False, "unsure")] * 8)
        second = agent_study.run_study("model-b", other, "model-b", draw, store, **options)
        self.assertEqual(second["conditions"]["C0"]["false_alarms"], 2)

    def test_command_line_takes_a_name_for_each_model(self):
        parser = agent_study.build_parser()
        run = parser.parse_args(["run", "--name", "luna", "--provider", "openai"])
        self.assertEqual(
            (run.name, run.provider, run.draw), ("luna", "openai", agent_study.DRAW_PATH)
        )
        self.assertEqual(parser.parse_args(["draw"]).count, 5)
        with self.assertRaises(SystemExit):
            parser.parse_args(["run"])


if __name__ == "__main__":
    unittest.main()
