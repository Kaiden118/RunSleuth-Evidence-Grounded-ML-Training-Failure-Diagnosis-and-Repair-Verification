"""Tests for RunSleuth v2's rules on hand-made run records, and its choice of checks with
fake probes."""

import ast
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from runsleuth import verifier_v2

FAMILY = "camelyon_pipeline"
COSINE = [1e-4 * (1 + math.cos(math.pi * (epoch - 1) / 3)) / 2 for epoch in (1, 2, 3)]


def report(rows, delivered, *, final=None, evaluated=False, total=None, interrupt=1, **top):
    """A run record; rows are (epoch, attempt, accuracy, state, rate)."""
    epochs_in_all = total or max(row[0] for row in rows)
    rows = [
        (*row, 1e-4 * (1 + math.cos(math.pi * (row[0] - 1) / epochs_in_all)) / 2)
        if len(row) == 4
        else row
        for row in rows
    ]
    group = {"parameter_tensors": 2, "min_tensors_with_gradient": 2}
    epochs, entries = [], []
    for epoch, attempt, accuracy, _state, rate in rows:
        epochs.append(
            {
                "attempt": attempt,
                "project_epoch": epoch,
                "optimizer_steps": 2,
                "training_forwards": 2,
                "eval_mode_training_forwards": 0,
                "learning_rate": {"first": rate, "last": rate, "changes": 0},
                "project_metrics": {
                    "validation_accuracy": accuracy,
                    "validation_loss": 1 - accuracy,
                },
                "head": dict(group),
                "backbone": dict(group),
            }
        )
        entry = {
            "epoch": epoch,
            "attempt": attempt,
            "reads": 64,
            "distinct": 64,
            "reads_by_group": {"0": 32, "3": 32},
            "loss_targets": 64,
            "loss_targets_matching_read_labels": 64,
        }
        if evaluated:
            entry["evaluation"] = {
                "id_validation_accuracy": accuracy,
                "id_validation_loss": 1 - accuracy,
            }
        entries.append(entry)
    accuracy = {f"epoch {r[0]} (attempt {r[1]})": r[2] for r in rows}[delivered]
    states = [row[3] for row in rows]
    built = {
        "status": "completed",
        "optimizer": "AdamW",
        "head_output_units": 2,
        "trainability": {"head": {"frozen_names": []}, "backbone": {"frozen_names": []}},
        "optimizer_audit": {"missing_trainable_names": []},
        "parameter_group_epochs": epochs,
        "final_metrics": {
            "id_validation_accuracy": accuracy if final is None else final,
            "id_validation_loss": 1 - accuracy,
        },
        "input_statistics": {
            split: {"channel_mean": [0.1, 0.2, 0.3], "channel_std": [1.0, 1.0, 1.0]}
            for split in ("train", "id_eval")
        },
        "workspace": {
            "seed": 7,
            "epochs": epochs_in_all,
            "max_train_batches": None,
            "hashed_steps": 2,
            "interrupt_after_epochs": interrupt,
            "initial_state_sha256": "init",
            "step_state_sha256": ["s1", "s2"],
            "epoch_state_sha256": states,
            "final_state_sha256": states[-1],
        },
        "pipeline": {
            "epochs": entries,
            "training_reads": {"candidates": 72, "candidates_by_group": {"0": 32, "1": 8, "3": 32}},
            "delivery": {"same_as": [delivered]},
        },
    }
    built.update(top)
    return built


REFERENCE = report([(1, 1, 0.90, "a"), (2, 1, 0.92, "b"), (3, 1, 0.93, "c")], "epoch 3 (attempt 1)")
RESTARTED = report([(1, 1, 0.90, "a"), (2, 2, 0.92, "b"), (3, 2, 0.93, "c")], "epoch 3 (attempt 2)")


def statuses(findings):
    found = {}
    for finding in findings:
        found.setdefault(finding["part"], set()).add(finding["status"])
    return found


class RecordTests(unittest.TestCase):
    def test_a_run_identical_to_the_reference_supports_the_parts_training_decides(self):
        found = statuses(verifier_v2.record_checks(RESTARTED, REFERENCE, FAMILY))
        for part in verifier_v2.TRAINING_PARTS:
            self.assertEqual(found.get(part), {"supported"}, part)
        for part in ("validation", "validation_inputs", "delivery"):
            self.assertNotIn("supported", found.get(part, set()), part)
        self.assertNotIn("violated", set().union(*found.values()))

    def test_records_find_clear_violations(self):
        def violated(changed, part):
            found = statuses(verifier_v2.record_checks(changed, REFERENCE, FAMILY))
            self.assertIn("violated", found.get(part, set()), (part, found))

        run = copy.deepcopy(RESTARTED)
        run["trainability"]["head"]["frozen_names"] = ["fc.weight"]
        violated(run, "parameters_trained")
        run = copy.deepcopy(RESTARTED)
        run["optimizer_audit"]["missing_trainable_names"] = ["fc.bias"]
        violated(run, "optimizer")
        violated({**copy.deepcopy(RESTARTED), "optimizer": "SGD"}, "optimizer")
        run = copy.deepcopy(RESTARTED)
        run["parameter_group_epochs"][1]["eval_mode_training_forwards"] = 2
        violated(run, "training_mode")
        run = copy.deepcopy(RESTARTED)
        run["parameter_group_epochs"][0]["optimizer_steps"] = 1
        violated(run, "steps")
        run = copy.deepcopy(RESTARTED)
        run["pipeline"]["epochs"][0]["reads_by_group"] = {"0": 32, "1": 8, "3": 24}
        violated(run, "training_samples")
        run = copy.deepcopy(RESTARTED)
        run["pipeline"]["epochs"][2]["loss_targets_matching_read_labels"] = 40
        violated(run, "training_labels")
        run = copy.deepcopy(RESTARTED)
        run["input_statistics"]["id_eval"]["channel_std"] = [0.2, 0.2, 0.2]
        violated(run, "validation_inputs")
        violated(report(RESTARTED_ROWS, "epoch 3 (attempt 2)", final=0.5), "validation")
        violated(report(RESTARTED_ROWS, "epoch 2 (attempt 2)"), "delivery")

    def test_records_support_nothing_they_cannot_show(self):
        run = copy.deepcopy(RESTARTED)
        run["workspace"]["step_state_sha256"] = ["x1", "x2"]  # not the reference's training
        for entry in run["pipeline"]["epochs"][1:]:
            entry["loss_targets_matching_read_labels"] = None  # a cache: nothing to pair
        found = statuses(verifier_v2.record_checks(run, REFERENCE, FAMILY))
        for part in ("training_samples", "training_labels", "training_inputs", "restart", "loss"):
            self.assertNotIn(part, found, part)
        self.assertEqual(found["steps"], {"supported"})
        self.assertEqual(found["parameters_trained"], {"supported"})
        failed = verifier_v2.record_checks({"status": "failed"}, REFERENCE, FAMILY)
        self.assertEqual([(f["part"], f["status"]) for f in failed], [("runs", "violated")])


RESTARTED_ROWS = [(1, 1, 0.90, "a"), (2, 2, 0.92, "b"), (3, 2, 0.93, "c")]


class ProbeRuleTests(unittest.TestCase):
    UNINTERRUPTED = report(
        [(1, 1, 0.90, "a"), (2, 1, 0.92, "b"), (3, 1, 0.93, "c")],
        "epoch 3 (attempt 1)",
        evaluated=True,
        interrupt=None,
    )

    def test_a_restart_is_judged_against_the_candidate_itself(self):
        supported = verifier_v2.restart_check(RESTARTED, self.UNINTERRUPTED)
        self.assertEqual([f["status"] for f in supported], ["supported"])
        repeated = report(
            RESTARTED_ROWS[:1] + [(1, 2, 0.9, "a"), (2, 2, 0.92, "b"), (3, 2, 0.93, "c")],
            "epoch 3 (attempt 2)",
        )
        self.assertIn(
            "trains epochs [1, 1, 2, 3]",
            verifier_v2.restart_check(repeated, self.UNINTERRUPTED)[0]["evidence"],
        )
        rates = report(
            [(1, 1, 0.90, "a", COSINE[0]), (2, 2, 0.92, "b", COSINE[1]), (3, 2, 0.93, "c", 7e-5)],
            "epoch 3 (attempt 2)",
        )
        self.assertEqual(
            verifier_v2.restart_check(rates, self.UNINTERRUPTED)[0]["status"], "violated"
        )
        weights = report(
            [(1, 1, 0.90, "a"), (2, 2, 0.92, "x"), (3, 2, 0.93, "y")], "epoch 3 (attempt 2)"
        )
        self.assertIsNone(verifier_v2.restart_check(weights, self.UNINTERRUPTED))
        same = verifier_v2.restart_check(weights, self.UNINTERRUPTED, self.UNINTERRUPTED)
        self.assertEqual(same[0]["status"], "violated")
        other = report(
            [(1, 1, 0.90, "a"), (2, 1, 0.92, "z"), (3, 1, 0.93, "w")], "epoch 3 (attempt 1)"
        )
        unsettled = verifier_v2.restart_check(weights, self.UNINTERRUPTED, other)
        self.assertEqual(unsettled[0]["status"], "untested")

    def test_the_schedule_is_a_cosine_with_one_rate_per_epoch(self):
        self.assertEqual(verifier_v2.schedule_check(self.UNINTERRUPTED)[0]["status"], "supported")
        linear = report(
            [(1, 1, 0.9, "a", 1e-4), (2, 1, 0.92, "b", 2e-4 / 3), (3, 1, 0.93, "c", 1e-4 / 3)],
            "epoch 3 (attempt 1)",
        )
        self.assertEqual(verifier_v2.schedule_check(linear)[0]["status"], "violated")
        stepped = copy.deepcopy(self.UNINTERRUPTED)
        stepped["parameter_group_epochs"][0]["learning_rate"]["changes"] = 5
        self.assertEqual(verifier_v2.schedule_check(stepped)[0]["status"], "violated")

    def test_reported_validation_must_be_what_the_weights_score(self):
        found = statuses(verifier_v2.validation_check(self.UNINTERRUPTED, "probe"))
        self.assertEqual(found, {"validation": {"supported"}, "validation_inputs": {"supported"}})
        wrong = copy.deepcopy(self.UNINTERRUPTED)
        wrong["pipeline"]["epochs"][1]["evaluation"]["id_validation_accuracy"] = 0.95
        self.assertEqual(verifier_v2.validation_check(wrong, "probe")[0]["status"], "violated")
        # A sample on the boundary flipping between two evaluations is not a wrong report.
        near = copy.deepcopy(self.UNINTERRUPTED)
        near["pipeline"]["epochs"][1]["evaluation"]["id_validation_accuracy"] += 0.0012
        near["pipeline"]["epochs"][1]["evaluation"]["id_validation_loss"] *= 1.005
        self.assertEqual(verifier_v2.validation_check(near, "probe")[0]["status"], "supported")
        loss = copy.deepcopy(self.UNINTERRUPTED)
        loss["pipeline"]["epochs"][2]["evaluation"]["id_validation_loss"] *= 1.03
        self.assertEqual(verifier_v2.validation_check(loss, "probe")[0]["status"], "violated")
        self.assertEqual(verifier_v2.validation_check(REFERENCE, "probe"), [])

    def test_a_delivery_probe_tries_one_direction_of_the_rule(self):
        kept = report(
            [(1, 1, 0.9, "a"), (2, 1, 0.95, "b"), (3, 2, 0.93, "c"), (4, 2, 0.94, "d")],
            "epoch 2 (attempt 1)",
            evaluated=True,
        )
        self.assertEqual(verifier_v2.delivery_case(kept, "probe"), ("kept", []))
        taken = report(
            [(1, 1, 0.9, "a"), (2, 1, 0.91, "b"), (3, 2, 0.95, "c"), (4, 2, 0.94, "d")],
            "epoch 3 (attempt 2)",
            evaluated=True,
        )
        self.assertEqual(verifier_v2.delivery_case(taken, "probe"), ("taken", []))
        last = report(
            [(1, 1, 0.9, "a"), (2, 1, 0.95, "b"), (3, 2, 0.93, "c"), (4, 2, 0.94, "d")],
            "epoch 4 (attempt 2)",
            evaluated=True,
        )
        case, violated = verifier_v2.delivery_case(last, "probe")
        self.assertIsNone(case)
        self.assertEqual(violated[0]["part"], "delivery")


class FakeProbes(verifier_v2.Probes):
    """Probe runs from hand-made records, written to files as a runner's reports are."""

    def __init__(self, directory, uninterrupted, delivery):
        self.directory, self.calls = Path(directory), []
        self.records = {
            "uninterrupted": uninterrupted,
            **{f"delivery_{s}": r for s, r in delivery.items()},
        }
        settings = {
            "delivery": {
                "seeds": list(delivery),
                "epochs": 4,
                "interrupt_after_epochs": 2,
                "max_eval_samples": 10,
            }
        }
        super().__init__(self.fake, self.fake, settings)

    def fake(self, _sources, _family, seed, tag="", **settings):
        name = "uninterrupted" if "interrupt_after_epochs" not in settings else f"delivery_{seed}"
        self.calls.append(name + (f"/{tag}" if tag else ""))
        path = self.directory / f"{name}-{tag or 'first'}.json"
        path.write_text(json.dumps(self.records[name]), encoding="utf-8")
        return {"status": "completed", "report": str(path), "seconds": 10.0}


KEPT = report(
    [(1, 1, 0.9, "a"), (2, 1, 0.95, "b"), (3, 2, 0.93, "c"), (4, 2, 0.94, "d")],
    "epoch 2 (attempt 1)",
    evaluated=True,
)
TAKEN = report(
    [(1, 1, 0.9, "a"), (2, 1, 0.91, "b"), (3, 2, 0.95, "c"), (4, 2, 0.94, "d")],
    "epoch 3 (attempt 2)",
    evaluated=True,
)
LAST = report(
    [(1, 1, 0.9, "a"), (2, 1, 0.91, "b"), (3, 2, 0.95, "c"), (4, 2, 0.94, "d")],
    "epoch 4 (attempt 2)",
    evaluated=True,
)


class VerifyTests(unittest.TestCase):
    def verify(self, delivery, *, run=RESTARTED, order="gap"):
        with tempfile.TemporaryDirectory() as directory:
            probes = FakeProbes(directory, ProbeRuleTests.UNINTERRUPTED, delivery)
            result = verifier_v2.verify({}, FAMILY, 7, run, REFERENCE, probes, order=order)
        return result, probes.calls

    def test_the_gap_order_skips_what_the_records_settled_and_the_control_does_not(self):
        result, calls = self.verify({101: TAKEN, 102: KEPT})
        self.assertEqual(result["decision"], "accept")
        self.assertEqual(calls, ["delivery_101", "delivery_102"])
        self.assertEqual(result["delivery_cases"], {"taken": 101, "kept": 102})
        self.assertEqual(result["cost"], {"runs": 2, "steps": 16, "seconds": 20.0})
        fixed, calls = self.verify({101: TAKEN, 102: KEPT}, order="fixed")
        self.assertEqual(fixed["decision"], "accept")
        self.assertEqual(calls, ["uninterrupted", "delivery_101", "delivery_102"])

    def test_a_delivery_that_ignores_the_best_epoch_is_rejected_at_the_first_probe(self):
        result, calls = self.verify({101: LAST, 102: KEPT})
        self.assertEqual(result["decision"], "reject")
        self.assertEqual(result["parts"]["delivery"]["status"], "violated")
        self.assertEqual(calls, ["delivery_101"])

    def test_one_direction_alone_leaves_the_delivery_untested(self):
        result, calls = self.verify({101: TAKEN, 102: TAKEN})
        self.assertEqual(result["decision"], "review")
        self.assertEqual(result["parts"]["delivery"]["status"], "untested")
        self.assertEqual(calls, ["delivery_101", "delivery_102"])

    def test_a_violation_in_the_records_needs_no_probe(self):
        run = copy.deepcopy(RESTARTED)
        run["optimizer"] = "SGD"
        result, calls = self.verify({101: TAKEN, 102: KEPT}, run=run)
        self.assertEqual((result["decision"], calls), ("reject", []))

    def test_weights_alone_differing_after_a_restart_call_for_the_determinism_control(self):
        run = report(
            [(1, 1, 0.90, "a"), (2, 2, 0.92, "x"), (3, 2, 0.93, "y")], "epoch 3 (attempt 2)"
        )
        result, calls = self.verify({101: TAKEN, 102: KEPT}, run=run)
        self.assertEqual(calls[:2], ["uninterrupted", "uninterrupted/repeat"])
        self.assertEqual(result["parts"]["restart"]["status"], "violated")
        self.assertEqual(result["decision"], "reject")


class IndependenceTests(unittest.TestCase):
    def test_v2_reads_neither_the_hidden_tests_nor_any_label(self):
        text = Path(verifier_v2.__file__).read_text(encoding="utf-8")
        tree = ast.parse(text)
        top = {
            node.module
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module.startswith("runsleuth")
        }
        self.assertEqual(top, {"runsleuth.agent_verification"})
        for name in (
            "pipeline_contract",
            "verifier_evaluation",
            "labels.json",
            "expected_failures",
        ):
            self.assertNotIn(name, text.replace("imports neither pipeline_contract", ""))


if __name__ == "__main__":
    unittest.main()
