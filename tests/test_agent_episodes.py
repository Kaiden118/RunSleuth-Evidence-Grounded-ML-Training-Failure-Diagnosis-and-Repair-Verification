"""CPU tests for agent episodes: a scripted model, a tiny network and synthetic patches."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_agent_loop import REPAIR, Rejected, Scripted, finish, reply
from test_agent_workspace import tiny_source

from runsleuth.agent_tasks import HEALTHY

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

if torch is not None:
    from runsleuth import agent_contract, agent_episodes, agent_workspace
    from runsleuth.agent_episodes import CONDITIONS, Budget, RunStore

    BUDGET = Budget(epochs=2, max_train_batches=None, max_eval_samples=None)


def run_in_process(workspace, run_dir, *, data, config, epochs, seed, **options):
    """The harness without a child process, for tests."""
    try:
        path = agent_workspace.run_workspace(
            workspace,
            run_dir,
            agent_workspace.data_builder(data, config, options.get("max_eval_samples")),
            epochs=epochs,
            seed=seed,
            device=torch.device("cpu"),
            max_train_batches=options.get("max_train_batches"),
        )
    except agent_workspace.ScriptRefused as refusal:
        return {"status": "refused", "problems": refusal.problems, "report": None, "output": ""}
    status = json.loads(path.read_text(encoding="utf-8"))["status"]
    return {"status": status, "problems": [], "report": str(path), "output": ""}


def contract(script):
    """The contract tests without a child process, on patches the tiny network can take."""
    return agent_contract.run_contract(script, patch_size=16)


@unittest.skipIf(torch is None, "Install torch and torchvision for the episode tests")
class EpisodeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        patcher = patch("runsleuth.agent_tasks.task_source", tiny_source)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.store = RunStore(self.root / "runs", data="synthetic", seed=11, runner=run_in_process)

    def episode(self, task, condition, *replies, name="episode"):
        client = Scripted(*replies)
        result = agent_episodes.run_episode(
            client,
            "scripted-model",
            task,
            condition,
            self.root / name,
            self.store,
            quick=BUDGET,
            full=BUDGET,
            contract=contract,
            sleep=lambda _seconds: None,
        )
        return result, client

    def test_a_repair_that_restores_the_healthy_script_is_correct_and_accepted(self):
        episode, _ = self.episode(
            "train_in_eval_mode",
            "C1",
            reply(("edit_file", REPAIR)),
            reply(("run_training", {})),
            finish(True, "model.train() was missing"),
        )
        self.assertEqual(
            (episode["claimed_correct"], episode["semantically_correct"]), (True, True)
        )
        self.assertEqual(episode["semantic_basis"], "same_program_as_healthy_script")
        self.assertFalse(episode["task_is_correct"])
        self.assertEqual(episode["expected_diagnosis"], "train_in_eval_mode")
        verification = episode["verification"]
        self.assertEqual(
            (verification["decision"], verification["re_diagnosis"]),
            ("accepted", "no_known_fault"),
        )
        self.assertTrue(verification["criteria"]["mechanism_check_and_reference_metrics"])
        self.assertEqual(verification["trajectory"]["outcome"], "identical_full_run")
        shown = episode["agent"]["steps"][1]["tool_calls"][0]["result"]
        self.assertEqual(shown["status"], "completed")
        self.assertEqual(len(shown["epochs"]), 2)
        self.assertNotIn("evidence", shown)
        # The repaired script is the healthy one, so two scripts were trained in all.
        self.assertEqual(self.store.trained, 2)
        # The contract tests label the script, in a record of their own.
        self.assertEqual(
            (episode["contract"]["status"], episode["contract"]["failed"]), ("passed", [])
        )
        self.assertEqual(episode["name"], "episode")
        self.assertEqual(agent_episodes.load_episode(self.root / "episode"), episode)
        saved = json.loads((self.root / "episode" / "episode.json").read_text(encoding="utf-8"))
        self.assertNotIn("contract", saved)
        self.assertEqual(saved["final_script"], tiny_source(HEALTHY))
        self.assertEqual(len(saved["statement_sha256"]), 64)
        scored = json.loads((self.root / "episode" / "contract.json").read_text(encoding="utf-8"))
        self.assertEqual(scored, episode["contract"])

    def test_claiming_a_faulty_script_correct_is_recorded_apart_from_the_verifier(self):
        episode, _ = self.episode(
            "stale_head", "C1", reply(("run_training", {})), finish(True, "none")
        )
        self.assertEqual(
            (episode["claimed_correct"], episode["semantically_correct"]), (True, False)
        )
        self.assertEqual(episode["semantic_basis"], "unchanged_script")
        self.assertFalse(episode["agent"]["script_changed"])
        self.assertEqual(episode["verification"]["decision"], "rejected")
        self.assertEqual(episode["verification"]["re_diagnosis"], "stale_optimizer_binding")
        self.assertFalse(episode["verification"]["criteria"]["mechanism_check"])
        self.assertEqual(episode["contract"]["failed"], ["all_parameters_trained"])
        summary = agent_episodes.summarize([episode])["conditions"]["C1"]
        self.assertEqual((summary["false_claims"], summary["verifier_accepted"]), (1, 0))
        self.assertEqual((summary["contract_failed"], summary["verifier_accepted_failing"]), (1, 0))

    def test_without_runs_the_agent_reads_only_and_a_wrong_doubt_is_a_false_alarm(self):
        episode, client = self.episode(HEALTHY, "C0", finish(False, "the schedule looks wrong"))
        tools = [tool["function"]["name"] for tool in client.requests[0]["tools"]]
        self.assertNotIn("run_training", tools)
        self.assertEqual(
            (episode["claimed_correct"], episode["semantically_correct"]), (False, True)
        )
        self.assertTrue(episode["task_is_correct"])
        self.assertEqual(episode["agent"]["training_runs"], 0)
        self.assertEqual(agent_episodes.summarize([episode])["conditions"]["C0"]["false_alarms"], 1)
        self.assertEqual(self.store.trained, 1, "only the final script is trained, once")

    def test_each_condition_adds_one_thing_to_what_a_run_shows(self):
        library = agent_episodes.load_library()
        healthy = self.store.run(tiny_source(HEALTHY), BUDGET)["report"]
        outcome = self.store.run(tiny_source("frozen_head"), BUDGET)

        def shown(condition):
            return agent_episodes.present(
                outcome,
                CONDITIONS[condition],
                reference=lambda: healthy,
                before=lambda: outcome["report"],
                library=library,
            )

        self.assertEqual(sorted(shown("C1")), ["epochs", "status"])
        self.assertIn("id_validation_accuracy", shown("C1")["epochs"][0])
        self.assertEqual(sorted(shown("C2")), ["epochs", "evidence", "status"])
        evidence = shown("C2")["evidence"]
        self.assertEqual(evidence["head_max_trainable_fraction"], 0.0)
        self.assertFalse([name for name in evidence if name.startswith("reference")])
        self.assertNotIn(None, evidence.values())
        self.assertEqual(sorted(shown("C3")), ["diagnosis", "epochs", "evidence", "status"])
        diagnosis = shown("C3")["diagnosis"]
        self.assertEqual(diagnosis["diagnosis"], "frozen_head")
        self.assertIn("requires_grad", diagnosis["repair"])
        verification = shown("C4")["verification"]
        self.assertEqual(verification["decision"], "rejected")
        self.assertIn("frozen_head", verification["reasons"][0])
        self.assertTrue(verification["trajectory"].startswith("diverged at step"))
        self.assertEqual(shown("C3"), {k: v for k, v in shown("C4").items() if k != "verification"})
        text = json.dumps(shown("C4"))
        self.assertLess(len(text), 6000, "a run's summary must stay small")

    def test_refused_and_failing_scripts_are_shown_as_such(self):
        library = agent_episodes.load_library()

        def shown(source):
            return agent_episodes.present(
                self.store.run(source, BUDGET),
                CONDITIONS["C4"],
                reference=lambda: None,
                before=lambda: None,
                library=library,
            )

        refused = shown("import os\n" + tiny_source(HEALTHY))
        self.assertEqual(
            refused, {"status": "refused", "problems": ["line 1: import of 'os' is not allowed"]}
        )
        failing = shown(
            tiny_source(HEALTHY).replace(
                "        loss.backward()\n", "        raise ValueError('no')\n"
            )
        )
        self.assertEqual(failing["status"], "failed")
        self.assertEqual(failing["error"], {"type": "ValueError", "message": "no"})
        self.assertIn("raise ValueError('no')", failing["traceback"])
        self.assertEqual(failing["verification"]["decision"], "rejected")

    def test_semantic_labels_come_from_the_source_alone(self):
        healthy = tiny_source(HEALTHY)
        stale = tiny_source("stale_head")
        reformatted = healthy.replace(
            "    model.train()\n", "    # put the model in training mode\n    model.train()\n\n"
        )
        self.assertTrue(agent_episodes.same_program(healthy, reformatted))
        self.assertFalse(agent_episodes.same_program(healthy, stale))
        self.assertFalse(agent_episodes.same_program(healthy, "def broken(:"))
        label = agent_episodes.semantic_label
        self.assertEqual(
            label("stale_head", stale, reformatted, healthy),
            (True, "same_program_as_healthy_script"),
        )
        self.assertEqual(label("stale_head", stale, stale, healthy), (False, "unchanged_script"))
        lookalike = tiny_source("head_replaced_after_moving")
        self.assertEqual(
            label("head_replaced_after_moving", lookalike, lookalike, healthy),
            (True, "unchanged_script"),
        )
        other = healthy.replace("LEARNING_RATE = 1e-4", "LEARNING_RATE = 5e-5")
        self.assertEqual(label("stale_head", stale, other, healthy), (None, "needs_review"))
        self.assertEqual(
            label("stale_head", stale, "def broken(:", healthy), (False, "does_not_parse")
        )

    def test_contract_tests_label_scripts_and_reviews_only_audit_them(self):
        slower = {"old_text": "LEARNING_RATE = 1e-4", "new_text": "LEARNING_RATE = 5e-5"}
        apart = {
            "old_text": "hospital not in (VALIDATION_HOSPITAL, TEST_HOSPITAL)",
            "new_text": "hospital != VALIDATION_HOSPITAL and hospital != TEST_HOSPITAL",
        }
        pilot = self.root / "pilot"

        def run(name, task, *replies):
            return self.episode(task, "C0", *replies, name=f"pilot/episodes/{name}")[0]

        episodes = [
            run("a", HEALTHY, reply(("edit_file", slower)), finish(True)),
            run("b", HEALTHY, reply(("edit_file", slower)), finish(True)),
            run("c", "stale_head", finish(True)),
            run("d", HEALTHY, finish(False, "unsure")),
            run("e", HEALTHY, reply(("edit_file", apart)), finish(True)),
        ]
        # The source alone settles neither edited script; the contract tests settle both.
        self.assertEqual(
            [episode["semantic_basis"] for episode in episodes],
            [
                "needs_review",
                "needs_review",
                "unchanged_script",
                "same_program_as_healthy_script",
                "needs_review",
            ],
        )
        self.assertEqual(
            [episode["contract"]["failed"] for episode in episodes],
            [["optimizer"], ["optimizer"], ["all_parameters_trained"], [], []],
        )
        summary = agent_episodes.summarize(episodes)
        row = summary["conditions"]["C0"]
        self.assertEqual(
            (row["contract_passed"], row["contract_failed"], row["unscored"]), (2, 3, 0)
        )
        self.assertEqual((row["false_claims"], row["false_alarms"]), (3, 1))
        self.assertEqual(row["edited_a_correct_script"], 3)
        self.assertEqual(
            summary["criteria"]["agent_claim"],
            {"accepted": 4, "accepted_failing": 3, "rejected_passing": 1},
        )
        self.assertEqual(summary["failed_clauses"], {"optimizer": 2, "all_parameters_trained": 1})
        self.assertEqual(summary["contract_versions"], [agent_contract.CONTRACT_VERSION])
        # The stale script fails the mechanism check and the healthy ones pass it; a script
        # that was not changed cannot have better metrics than before.
        mechanism = summary["criteria"]["mechanism_check"]
        self.assertEqual(mechanism["rejected_passing"], 0)
        slower_accepted = sum(
            e["verification"]["criteria"]["mechanism_check"] for e in episodes[:2]
        )
        self.assertEqual(mechanism["accepted"], slower_accepted + 2)
        self.assertEqual(mechanism["accepted_failing"], slower_accepted)
        self.assertEqual(row["verifier_accepted_failing"], slower_accepted)
        self.assertGreaterEqual(
            summary["criteria"]["before_and_after_metrics"]["rejected_passing"], 1
        )
        # Where the source settles a label, the contract tests agree with it.
        self.assertEqual(summary["checks"]["source"], {"episodes": 2, "disagree": []})
        self.assertEqual(summary["checks"]["reviews"]["episodes"], 0)

        program = agent_episodes.agent_review.program_sha256
        reviews_path = self.root / "reviews.json"
        review = {
            "reason": "Read against the task statement.",
            "reviewer": "a reviewer",
            "blind_to_verifier": True,
        }
        reviews_path.write_text(
            json.dumps(
                {
                    "reviews": [
                        {
                            **review,
                            "program_sha256": program(episodes[0]["final_script"]),
                            "correct": False,
                        },
                        {
                            **review,
                            "program_sha256": program(episodes[4]["final_script"]),
                            "correct": False,
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        reviews = agent_episodes.agent_review.load_reviews(reviews_path)
        audited = agent_episodes.summarize(episodes, reviews)
        # A review that disagrees is reported, and changes no label.
        self.assertEqual(
            audited["checks"]["reviews"],
            {"episodes": 3, "blind_to_verifier": 3, "disagree": ["e"]},
        )
        self.assertEqual(audited["conditions"], summary["conditions"])
        self.assertEqual(audited["criteria"], summary["criteria"])

        def printed(*arguments):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(agent_episodes.main(list(arguments)), 0)
            return output.getvalue()

        empty = self.root / "empty.json"
        empty.write_text('{"reviews": []}', encoding="utf-8")
        listing = printed("review", str(pilot), "--reviews", str(empty))
        self.assertIn(f"program_sha256 {program(episodes[0]['final_script'])}  (ends 2", listing)
        self.assertIn("+LEARNING_RATE = 5e-5", listing)
        self.assertNotIn("optimizer", listing)
        self.assertEqual(
            printed("review", str(pilot), "--reviews", str(reviews_path)).strip(),
            "No script needs a review.",
        )
        table = printed("summarize", str(pilot), "--reviews", str(reviews_path))
        self.assertRegex(table, r"agent claim\s+4\s+3\s+1")
        self.assertIn("failed clauses: optimizer 2, all_parameters_trained 1", table)
        self.assertIn("what the source alone settles: 2 of 2 episodes agree", table)
        self.assertIn(
            "reviews: 2 of 3 episodes agree (3 reviewed blind to the verifier); disagree: e", table
        )

    def test_scoring_is_a_step_that_can_be_run_again(self):
        episode = self.episode("stale_head", "C0", finish(True), name="pilot/episodes/a")[0]
        pilot, scored = (
            self.root / "pilot",
            self.root / "pilot" / "episodes" / "a" / "contract.json",
        )
        calls = []

        def other(script):
            calls.append(script.read_text(encoding="utf-8"))
            return {**contract(script), "contract_version": 99}

        self.assertEqual(agent_episodes.score_episodes(pilot, other), 0, "already scored")
        scored.unlink()
        (unscored,) = agent_episodes.load_episodes(pilot)
        self.assertIsNone(unscored["contract"])
        summary = agent_episodes.summarize([unscored])
        self.assertEqual((summary["unscored"], summary["conditions"]["C0"]["false_claims"]), (1, 0))
        self.assertEqual(summary["criteria"]["agent_claim"]["accepted_failing"], 0)
        lines = []
        self.assertEqual(agent_episodes.score_episodes(pilot, other, progress=lines.append), 1)
        self.assertEqual(lines, ["a: failed all_parameters_trained"])
        self.assertEqual(calls, [episode["final_script"]])
        self.assertEqual(agent_episodes.load_episodes(pilot)[0]["contract"]["contract_version"], 99)
        self.assertEqual(agent_episodes.score_episodes(pilot, contract, redo=True), 1)
        self.assertEqual(agent_episodes.load_episodes(pilot), [episode])
        crashed = {**episode, "contract": {**episode["contract"], "status": "crashed"}}
        self.assertIsNone(agent_episodes.meets_contract(crashed))

    def test_the_store_trains_a_script_once_per_budget_and_recovers_from_an_interruption(self):
        source = tiny_source(HEALTHY)
        first = self.store.run(source, BUDGET)
        self.assertEqual(self.store.run(source, BUDGET), first)
        self.assertEqual(self.store.trained, 1)
        shorter = Budget(epochs=2, max_train_batches=2, max_eval_samples=None)
        self.assertNotEqual(self.store.run(source, shorter)["settings"], first["settings"])
        self.assertEqual(self.store.trained, 2)
        (directory,) = [
            path
            for path in (self.root / "runs").iterdir()
            if json.loads((path / "outcome.json").read_text("utf-8"))["settings"]
            == first["settings"]
        ]
        (directory / "outcome.json").unlink()
        again = self.store.run(source, BUDGET)
        self.assertEqual(self.store.trained, 3)
        self.assertEqual(
            again["report"]["workspace"]["final_state_sha256"],
            first["report"]["workspace"]["final_state_sha256"],
        )

    def test_preparing_trains_what_episodes_need_so_they_train_nothing_more(self):
        lines = []
        rows = agent_episodes.prepare_runs(
            self.store, ("stale_head",), quick=BUDGET, full=BUDGET, progress=lines.append
        )
        self.assertEqual(
            [(row["task"], row["budget"]) for row in rows][::2],
            [(HEALTHY, "quick"), ("stale_head", "quick")],
        )
        self.assertEqual({row["status"] for row in rows}, {"completed"})
        self.assertEqual(len(lines), 4)
        self.assertIn("ood_validation_accuracy", rows[0]["final_metrics"])
        trained = self.store.trained
        self.assertEqual(trained, 2, "equal budgets share one run per script")
        self.episode("stale_head", "C1", reply(("run_training", {})), finish(True, "none"))
        self.assertEqual(self.store.trained, trained)

    def test_a_pilot_runs_every_task_under_every_condition_and_resumes(self):
        replies = (
            finish(True, "none"),  # stale_head, C0: a false claim
            reply(("run_training", {})),  # stale_head, C1
            finish(False, "the head never updates"),
            finish(True, "none"),  # healthy, C0
            finish(True, "none"),  # healthy, C1
        )
        lines = []
        options = {
            "tasks": ("stale_head", HEALTHY),
            "conditions": ("C0", "C1"),
            "repetitions": 1,
            "quick": BUDGET,
            "full": BUDGET,
            "contract": contract,
            "sleep": lambda _seconds: None,
        }
        output = self.root / "pilot"
        summary = agent_episodes.run_pilot(
            output,
            Scripted(*replies),
            "scripted-model",
            self.store,
            progress=lines.append,
            **options,
        )
        self.assertEqual(summary["episodes"], 4)
        zero, one = summary["conditions"]["C0"], summary["conditions"]["C1"]
        self.assertEqual(
            (zero["episodes"], zero["claimed_correct"], zero["false_claims"]), (2, 2, 1)
        )
        self.assertEqual(
            (one["claimed_correct"], one["false_claims"], one["training_runs"]), (1, 0, 1)
        )
        self.assertEqual((zero["contract_failed"], zero["contract_passed"]), (1, 1))
        self.assertIn("contract tests: failed", lines[1])
        self.assertEqual(len(lines), 8)
        self.assertEqual(json.loads((output / "pilot_report.json").read_text("utf-8")), summary)
        # Nothing is asked of the model again, except for an episode that was interrupted.
        self.assertEqual(
            agent_episodes.run_pilot(output, Scripted(), "scripted-model", self.store, **options),
            summary,
        )
        (output / "episodes" / "healthy-C1-1" / "episode.json").unlink()
        resumed = agent_episodes.run_pilot(
            output, Scripted(finish(False, "unsure")), "scripted-model", self.store, **options
        )
        self.assertEqual(resumed["conditions"]["C1"]["false_alarms"], 1)
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            self.assertEqual(agent_episodes.main(["summarize", str(output)]), 0)
        self.assertIn("false claims", printed.getvalue())
        self.assertRegex(printed.getvalue(), r"C0\s+2\s+2\s+2")

    def test_a_pilot_stops_at_a_provider_failure_and_runs_that_episode_again(self):
        options = {
            "tasks": ("stale_head",),
            "conditions": ("C0",),
            "repetitions": 2,
            "quick": BUDGET,
            "full": BUDGET,
            "contract": contract,
            "sleep": lambda _seconds: None,
        }
        output, lines = self.root / "pilot", []
        client = Scripted(Rejected(), finish(True, "none"))
        stopped = agent_episodes.run_pilot(
            output, client, "scripted-model", self.store, progress=lines.append, **options
        )
        # The failed episode is no observation of the agent, and the second was not tried.
        self.assertEqual((stopped["episodes"], stopped["provider_failures"]), (0, 1))
        self.assertEqual(stopped["conditions"], {})
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(len(lines), 2)
        self.assertIn("thought_signature", lines[1])
        resumed = agent_episodes.run_pilot(
            output,
            Scripted(finish(True, "none"), finish(False, "the head never updates")),
            "scripted-model",
            self.store,
            **options,
        )
        self.assertEqual((resumed["episodes"], resumed["provider_failures"]), (2, 0))
        self.assertEqual(resumed["conditions"]["C0"]["false_claims"], 1)


if __name__ == "__main__":
    unittest.main()
