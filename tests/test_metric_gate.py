"""Tests for the metric gate: calibration on correct runs, decisions and the frozen file."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from runsleuth import metric_gate, pipeline_tasks

FAMILY = "camelyon_pipeline"


def check(seed, rows, family=FAMILY, family_hash=None):
    """A pipeline check report with rows (version, id accuracy, ood accuracy)."""
    return {
        "settings": {
            "family": family,
            "family_sha256": family_hash or pipeline_tasks.family_sha256(family),
            "seed": seed,
        },
        "versions": [
            {
                "version": name,
                "status": "completed",
                "metrics": {"id_validation_accuracy": id_, "ood_validation_accuracy": ood},
            }
            for name, id_, ood in rows
        ],
    }


def run(id_, ood, id_loss=0.06, ood_loss=0.33, status="completed"):
    return {
        "status": status,
        "final_metrics": {
            "id_validation_accuracy": id_,
            "ood_validation_accuracy": ood,
            "id_validation_loss": id_loss,
            "ood_validation_loss": ood_loss,
        },
    }


CHECKS = [
    check(
        7,
        [
            ("healthy", 0.982, 0.912),
            # A correct development variant counts; faults and test variants do not.
            ("subset_random_sampler", 0.980, 0.8998),
            ("positions_index_base", 0.7242, 0.4666),
            ("seeded_per_epoch", 0.50, 0.50),
        ],
    ),
    check(11, [("healthy", 0.9805, 0.905)]),
    check(42, [("healthy", 0.981, 0.915)]),
    check(2026, [("healthy", 0.983, 0.910)]),
]


class CalibrationTests(unittest.TestCase):
    def test_thresholds_are_the_largest_spread_of_correct_runs_rounded_up(self):
        gate = metric_gate.calibrate(CHECKS, FAMILY)
        self.assertEqual(
            [(run["version"], run["seed"]) for run in gate["runs"]],
            [("healthy", 7), ("subset_random_sampler", 7), ("healthy", 11), ("healthy", 42)]
            + [("healthy", 2026)],
        )
        # id 0.983 - 0.980 = 0.003; ood 0.915 - 0.8998 = 0.0152, up to 0.016.
        self.assertEqual(gate["thresholds"], {"id": 0.003, "ood": 0.016})

    def test_calibration_takes_every_seed_of_the_current_family(self):
        with self.assertRaises(ValueError):
            metric_gate.calibrate(CHECKS[:3], FAMILY)
        with self.assertRaises(ValueError):
            metric_gate.calibrate([*CHECKS, check(5, [("healthy", 0.98, 0.91)])], FAMILY)
        with self.assertRaises(ValueError):
            metric_gate.calibrate(
                [check(7, [("healthy", 0.98, 0.91)], family_hash="0" * 64)], FAMILY
            )
        broken = check(7, [("healthy", 0.98, 0.91)])
        broken["versions"][0]["status"] = "crashed"
        with self.assertRaises(ValueError):
            metric_gate.calibration_runs(broken, FAMILY)

    def test_a_recorded_gate_is_frozen_and_tied_to_its_family(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.json"
            gate = metric_gate.calibrate(CHECKS, FAMILY)
            metric_gate.write_gate(FAMILY, gate, path)
            self.assertEqual(metric_gate.load_gate(FAMILY, path), gate["thresholds"])
            with self.assertRaises(FileExistsError):
                metric_gate.write_gate(FAMILY, gate, path)
            with patch.object(pipeline_tasks, "family_sha256", return_value="1" * 64):
                with self.assertRaises(ValueError):
                    metric_gate.load_gate(FAMILY, path)


class DecisionTests(unittest.TestCase):
    GATE = {"id": 0.003, "ood": 0.016}
    REFERENCE = run(0.982, 0.912)

    def test_a_candidate_within_the_spread_is_accepted(self):
        result = metric_gate.decide(run(0.980, 0.8998, ood_loss=0.43), self.REFERENCE, self.GATE)
        self.assertEqual(result["decision"], "accept")
        self.assertEqual([check["split"] for check in result["checks"]], ["id", "ood"])
        # Stage one's gate rejects the same run on its loss.
        stage_one = metric_gate.decide_stage_one(run(0.980, 0.8998, ood_loss=0.43), self.REFERENCE)
        self.assertEqual(stage_one, "reject")

    def test_a_drop_beyond_the_threshold_is_rejected(self):
        result = metric_gate.decide(run(0.982, 0.8900), self.REFERENCE, self.GATE)
        self.assertEqual(result["decision"], "reject")
        self.assertIn("ood accuracy 0.8900 is 2.2 points below", result["reasons"][0])

    def test_a_candidate_without_metrics_is_rejected(self):
        for candidate in (None, run(0.98, 0.91, status="crashed"), run(0.98, float("nan"))):
            self.assertEqual(
                metric_gate.decide(candidate, self.REFERENCE, self.GATE)["decision"], "reject"
            )
            self.assertEqual(metric_gate.decide_stage_one(candidate, self.REFERENCE), "reject")
        with self.assertRaises(ValueError):
            metric_gate.decide(run(0.98, 0.91), run(0.98, None), self.GATE)


if __name__ == "__main__":
    unittest.main()
