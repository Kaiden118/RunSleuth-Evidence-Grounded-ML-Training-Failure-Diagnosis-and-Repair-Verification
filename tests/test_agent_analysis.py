"""Tests for the study's analysis on made-up episodes; no model and no training."""

import contextlib
import io
import json
import random
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

if torch is not None:
    from runsleuth import agent_analysis

LISTED = ("stale_head", "frozen_head", "train_in_eval_mode", "high_learning_rate")
UNLISTED = ("weight_decay_dropped", "learning_rate_halved")
CRITERIA = (
    "before_and_after_metrics",
    "reference_metrics",
    "mechanism_check",
    "mechanism_check_and_reference_metrics",
)


def episode(
    task,
    condition,
    passes,
    *,
    claims=True,
    triage="accept",
    edited=False,
    status="finished",
    found=None,
    expected=None,
):
    """An episode as load_episode reads it, with only what the analysis uses."""
    accepted = triage != "reject"
    return {
        "repair_task": task,
        "condition": condition,
        "model": "made-up",
        "claimed_correct": claims,
        "expected_diagnosis": expected,
        "agent": {"status": status, "script_changed": edited},
        "contract": None if passes is None else {"status": "passed" if passes else "failed"},
        "verification": {
            "decision": "accepted" if accepted else "rejected",
            "triage": triage,
            "re_diagnosis": found,
            "criteria": dict.fromkeys(CRITERIA, accepted),
        },
    }


@unittest.skipIf(torch is None, "Install torch for the analysis tests")
class AgentAnalysisTests(unittest.TestCase):
    def test_rates_are_per_task_first_so_a_task_counts_once(self):
        episodes = [episode("stale_head", "C0", True) for _ in range(8)]
        episodes += [episode("frozen_head", "C0", False, triage="reject")]
        episodes += [episode("weight_decay_dropped", "C0", False, triage="review")]
        result = agent_analysis.analyze(episodes)
        repaired = result["faulty_tasks"]["all_faults"]["C0"]["repaired"]
        # Eight of ten episodes pass, but one task of three does.
        self.assertEqual(repaired["tasks"], 3)
        self.assertAlmostEqual(repaired["mean"], 1 / 3)
        self.assertLessEqual(repaired["low"], repaired["mean"])
        self.assertGreaterEqual(repaired["high"], repaired["mean"])
        listed = result["faulty_tasks"]["listed_faults"]["C0"]
        self.assertAlmostEqual(listed["repaired"]["mean"], 0.5)
        self.assertAlmostEqual(listed["false_claim"]["mean"], 0.5)
        unlisted = result["faulty_tasks"]["unlisted_faults"]["C0"]
        self.assertEqual((unlisted["repaired"]["tasks"], unlisted["repaired"]["mean"]), (1, 0.0))
        self.assertEqual(
            result["tasks"],
            {"healthy": 0, "lookalike": 0, "listed_fault": 2, "unlisted_fault": 1},
        )
        self.assertEqual(agent_analysis.analyze(episodes), result, "the resampling is seeded")

    def test_planned_comparisons_are_paired_by_task_with_an_exact_sign_flip_test(self):
        episodes = []
        for task in LISTED:  # Every task is repaired under C0, none under C2, all under C3.
            episodes += [episode(task, "C0", True), episode(task, "C3", True)]
            episodes += [episode(task, "C2", False, triage="reject")]
        episodes.append(episode("weight_decay_dropped", "C0", True))  # No C2 episode: unpaired.
        result = agent_analysis.analyze(episodes)
        comparisons = {
            (item["from"], item["to"], item["outcome"]): item
            for item in result["planned_comparisons"]
        }
        self.assertEqual(len(comparisons), 4)
        hurt = comparisons[("C0", "C2", "repaired")]
        self.assertEqual((hurt["difference"]["tasks"], hurt["difference"]["mean"]), (4, -1.0))
        # Two of the sixteen sign patterns are as extreme as the one observed.
        self.assertAlmostEqual(hurt["p_value"], 2 / 16)
        self.assertEqual(comparisons[("C2", "C3", "repaired")]["difference"]["mean"], 1.0)
        self.assertEqual(comparisons[("C0", "C2", "false_claim")]["difference"]["mean"], 1.0)
        rng = random.Random(0)
        self.assertEqual(agent_analysis.sign_flip_p([0.0, 0.0], rng), 1.0)
        self.assertIsNone(agent_analysis.sign_flip_p([], rng))
        many = agent_analysis.sign_flip_p([0.5] * (agent_analysis.EXACT_TASKS + 4), rng)
        self.assertLess(many, 0.001)

    def test_criteria_and_triage_are_counted_on_failing_and_passing_scripts(self):
        episodes = [
            episode("stale_head", "C1", False, triage="reject"),
            episode("weight_decay_dropped", "C1", False, triage="review"),
            episode("learning_rate_halved", "C1", False, triage="accept"),
            episode("healthy", "C1", True),
            episode("cosine_stepped_per_batch_over_the_run", "C1", True, triage="review"),
            episode("two_parameter_groups", "C1", True, claims=False, edited=True),
        ]
        result = agent_analysis.analyze(episodes)
        claim = result["criteria"]["agent_claim"]
        self.assertEqual(claim["accepts_failing"]["all"], [3, 3])
        self.assertEqual(claim["rejects_passing"]["all"], [1, 3])
        both = result["criteria"]["mechanism_check_and_reference_metrics"]
        self.assertEqual(
            both["accepts_failing"],
            {
                "all": [2, 3],
                "listed_fault": [0, 1],
                "unlisted_fault": [2, 2],
            },
        )
        self.assertEqual(both["rejects_passing"]["all"], [0, 3])
        same_run = result["criteria"]["checks_and_same_run_as_reference"]
        self.assertEqual(same_run["accepts_failing"]["unlisted_fault"], [1, 2])
        self.assertEqual(same_run["rejects_passing"]["lookalike"], [1, 2])
        triage = result["triage"]
        self.assertEqual(
            [triage["failing"][name]["all"] for name in ("accept", "review", "reject")],
            [[1, 3], [1, 3], [1, 3]],
        )
        self.assertEqual(triage["passing"]["review"]["lookalike"], [1, 2])
        correct = result["correct_tasks"]["C1"]
        self.assertEqual(correct["left_correct"]["mean"], 1.0)
        self.assertAlmostEqual(correct["edited"]["mean"], 1 / 3)
        self.assertAlmostEqual(correct["false_alarm"]["mean"], 1 / 3)

    def test_secondary_comparisons_are_reported_apart_from_the_planned_ones(self):
        episodes = []
        for task in LISTED:  # Repaired under C1 and C4, not under C2 and C3.
            episodes += [episode(task, "C1", True), episode(task, "C4", True)]
            episodes += [episode(task, name, False, triage="reject") for name in ("C2", "C3")]
        result = agent_analysis.analyze(episodes)
        planned = {(item["from"], item["to"]) for item in result["planned_comparisons"]}
        self.assertEqual(planned, {("C0", "C2"), ("C2", "C3")})
        secondary = result["secondary_comparisons"]
        self.assertEqual(secondary["added"], "2026-10-14")
        found = {
            (item["from"], item["to"], item["outcome"]): item["difference"]["mean"]
            for item in secondary["comparisons"]
        }
        self.assertEqual(found[("C1", "C2", "repaired")], -1.0)
        self.assertEqual(found[("C3", "C4", "repaired")], 1.0)
        self.assertEqual(found[("C3", "C4", "false_claim")], -1.0)

    def test_the_verifier_is_judged_on_c0_to_c3_and_review_is_counted(self):
        episodes = [
            episode("stale_head", "C1", False, triage="reject"),
            episode("weight_decay_dropped", "C1", False, triage="review"),
            episode("learning_rate_halved", "C2", False, triage="accept"),
            episode("healthy", "C2", True),
            episode("cosine_stepped_per_batch_over_the_run", "C3", True, triage="review"),
            # C4 is left out of the pooled figures and only reported on its own.
            episode("learning_rate_halved", "C4", False, triage="accept"),
        ]
        judged = agent_analysis.analyze(episodes)["verifier"]
        self.assertEqual(judged["pooled_conditions"], ["C1", "C2", "C3"])
        triage = judged["pooled"]["checks_and_same_run_as_reference"]
        self.assertEqual(
            triage,
            {
                "false_acceptance": [1, 3],
                "failing_among_accepted": [1, 2],
                "accepted_without_review": [2, 5],
                "sent_to_review": [2, 5],
            },
        )
        two_way = judged["pooled"]["mechanism_check_and_reference_metrics"]
        self.assertEqual(two_way["false_acceptance"], [2, 3])
        self.assertEqual(two_way["accepted_without_review"], [4, 5])
        self.assertEqual(two_way["sent_to_review"], [0, 5], "a two-way decision reviews nothing")
        claim = judged["pooled"]["agent_claim"]
        self.assertEqual(claim["failing_among_accepted"], [3, 5])
        self.assertEqual(
            judged["by_condition"]["C4"]["checks_and_same_run_as_reference"]["false_acceptance"],
            [1, 1],
        )

    def test_the_diagnosis_of_a_failing_script_is_counted_apart_from_the_decision(self):
        stale = "stale_optimizer_binding"
        episodes = [
            episode("stale_head", "C0", False, triage="reject", found=stale, expected=stale),
            # Rejected, but for another fault than the one the task holds.
            episode(
                "stale_head", "C1", False, triage="reject", found="frozen_head", expected=stale
            ),
            episode(
                "weight_decay_dropped",
                "C2",
                False,
                triage="review",
                found="no_known_fault",
                expected="no_known_fault",
            ),
            episode("frozen_head", "C3", False, triage="reject"),  # Never diagnosed.
            episode("stale_head", "C4", False, triage="reject", found=stale, expected=stale),
            episode("healthy", "C0", True, found="no_known_fault", expected="no_known_fault"),
        ]
        table = agent_analysis.analyze(episodes)["verifier"]["failing_scripts_by_diagnosis"]
        self.assertEqual(table["the_task_fault"], {"accept": 0, "review": 0, "reject": 1})
        self.assertEqual(table["another_fault"], {"accept": 0, "review": 0, "reject": 1})
        self.assertEqual(table["no_known_fault"], {"accept": 0, "review": 1, "reject": 0})
        self.assertEqual(table["not_diagnosed"], {"accept": 0, "review": 0, "reject": 1})

    def test_episodes_without_a_label_are_counted_and_left_out(self):
        episodes = [
            episode("stale_head", "C0", True),
            episode("stale_head", "C0", None),
            episode("stale_head", "C0", False, status="api_error"),
            {**episode("stale_head", "C0", True), "contract": {"status": "crashed"}},
        ]
        result = agent_analysis.analyze(episodes)
        self.assertEqual((result["episodes"], result["episodes_without_a_label"]), (1, 3))
        self.assertEqual(agent_analysis.analyze([])["faulty_tasks"]["all_faults"], {})

    def test_command_line_writes_the_analysis_beside_the_episodes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "model"
            for index, item in enumerate(
                [
                    episode("stale_head", "C0", True),
                    episode("stale_head", "C2", False, triage="reject"),
                    episode("healthy", "C0", True),
                ]
            ):
                folder = directory / "episodes" / f"e{index}"
                folder.mkdir(parents=True)
                contract = item.pop("contract")
                (folder / "episode.json").write_text(json.dumps(item), encoding="utf-8")
                (folder / "contract.json").write_text(json.dumps(contract), encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(agent_analysis.main([str(directory)]), 0)
            saved = json.loads((directory / "analysis.json").read_text(encoding="utf-8"))
        text = output.getvalue()
        self.assertEqual(saved["episodes"], 3)
        self.assertIn("model: made-up; 3 episodes with a label, 0 without", text)
        self.assertIn("C2 minus C0, repaired: -1.00 [-1.00, -1.00], sign-flip p = 1.000", text)
        self.assertIn("planned comparisons, over all faults and paired by task:", text)
        self.assertRegex(text, r"agent claim\s+1/1")
        self.assertRegex(text, r"failing\s+0/1 \| 0/1 \| 1/1")
        self.assertEqual(saved["schema_version"], 2)
        self.assertIn("secondary comparisons, added 2026-10-14 after the draw:", text)
        self.assertIn("the verifier on the final scripts of C0, C2 (added 2026-10-14):", text)
        self.assertRegex(text, r"checks and same run as reference\s+0/1\s+0/2\s+2/3\s+0/3")
        self.assertRegex(text, r"not diagnosed\s+0 \| 0 \| 1")


if __name__ == "__main__":
    unittest.main()
