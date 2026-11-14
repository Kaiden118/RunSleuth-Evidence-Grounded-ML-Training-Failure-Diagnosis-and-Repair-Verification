"""Tests for v2's evidence as an observation packet, on the hand-made records of v2's
own tests: what a packet keeps and leaves out, and the round trip through frozen v2."""

import copy
import json
import tempfile
import unittest

from test_verifier_v2 import KEPT, LAST, REFERENCE, RESTARTED, TAKEN, FakeProbes, ProbeRuleTests

from runsleuth import matched_evidence, verifier_v2

FAMILY = "camelyon_pipeline"


def hashed(report, prefix):
    """A copy of a report with every state hash 64 hexadecimal digits long."""
    report = copy.deepcopy(report)

    def long(value):
        return (prefix + value.encode().hex()).ljust(64, "0")[:64]

    workspace = report["workspace"]
    for field in matched_evidence.STATE_FIELDS:
        value = workspace[field]
        workspace[field] = (
            [long(item) for item in value] if isinstance(value, list) else long(value)
        )
    return report


class Recording(FakeProbes, matched_evidence.Recorded):
    """v2's test probes, keeping every run v2 asked for as the matched evidence does."""

    def __init__(self, directory, uninterrupted, delivery):
        FakeProbes.__init__(self, directory, uninterrupted, delivery)
        self.runs = {}


class PacketTests(unittest.TestCase):
    def test_a_report_keeps_what_v2_reads_and_nothing_else(self):
        report = copy.deepcopy(RESTARTED)
        report["parameter_group_epochs"][0]["gradient_l2_at_epoch_end"] = 0.3
        report["pipeline"]["epochs"][0]["order_sha256"] = "f" * 64
        report["workspace"]["project_sha256"] = "e" * 64
        report["label_classes"] = [0, 1]
        kept = matched_evidence.project(report)
        self.assertNotIn("gradient_l2_at_epoch_end", kept["parameter_group_epochs"][0])
        self.assertNotIn("order_sha256", kept["pipeline"]["epochs"][0])
        self.assertNotIn("project_sha256", kept["workspace"])
        self.assertNotIn("label_classes", kept)
        self.assertEqual(kept["pipeline"]["delivery"], {"same_as": ["epoch 3 (attempt 2)"]})
        self.assertEqual(
            kept["parameter_group_epochs"][0]["learning_rate"],
            report["parameter_group_epochs"][0]["learning_rate"],
        )
        self.assertNotIn("final_trainability", kept)
        self.assertIsNone(matched_evidence.project(None))

    def test_state_hashes_are_named_alike_across_runs(self):
        run, reference = hashed(RESTARTED, "a"), hashed(REFERENCE, "a")
        renamed = matched_evidence.relabel([run, reference, None])
        self.assertIsNone(renamed[2])
        ours, theirs = (report["workspace"] for report in renamed[:2])
        self.assertEqual(ours["initial_state_sha256"], "S1")
        self.assertEqual(ours["epoch_state_sha256"], theirs["epoch_state_sha256"])
        self.assertEqual(len(set(ours["epoch_state_sha256"])), 3)
        self.assertTrue(all(name.startswith("S") for name in ours["step_state_sha256"]))
        self.assertEqual(run["workspace"]["initial_state_sha256"][:1], "a")

    def test_a_packet_holds_observations_and_parses_back(self):
        probes = {"delivery-101": TAKEN, "delivery-102": None}
        settings = {"delivery": {"epochs": 4, "interrupt_after_epochs": 2, "max_eval_samples": 10}}
        text = matched_evidence.packet(RESTARTED, REFERENCE, probes, settings)
        self.assertTrue(text.startswith(matched_evidence.LEGEND))
        self.assertIn("## candidate: the candidate's own run, seed 7, 3 epochs, stopped", text)
        self.assertIn("## reference: the healthy project's run", text)
        self.assertIn("## delivery-101: the candidate on probe seed 101: 4 epochs", text)
        for word in ("supported", "violated", "untested", "decision", "accept", "reject"):
            self.assertNotIn(word, text)
        runs = matched_evidence.parse_packet(text)
        self.assertEqual(list(runs), ["candidate", "reference", "delivery-101", "delivery-102"])
        self.assertIsNone(runs["delivery-102"])
        expected = matched_evidence.relabel(
            [matched_evidence.project(RESTARTED), matched_evidence.project(REFERENCE)]
        )
        self.assertEqual([runs["candidate"], runs["reference"]], expected)
        self.assertEqual(runs["candidate"]["workspace"]["epoch_state_sha256"], ["S4", "S5", "S6"])
        with self.assertRaisesRegex(ValueError, "needs the candidate's run"):
            matched_evidence.parse_packet(text.replace("## reference:", "## delivery-9:"))

    def test_a_replay_answers_by_role_and_refuses_a_run_it_lacks(self):
        settings = {"delivery": {"seeds": [101]}}
        replayed = matched_evidence.Replayed({"delivery-101": TAKEN}, settings)
        report, cost = replayed.run("delivery", {}, FAMILY, 101, 3)
        self.assertEqual((report, cost["runs"]), (TAKEN, 1))
        with self.assertRaisesRegex(KeyError, "the repeat run"):
            replayed.run("uninterrupted", {}, FAMILY, 7, 3, tag="repeat")
        self.assertEqual(matched_evidence.role("uninterrupted", 7), "uninterrupted")
        self.assertEqual(matched_evidence.role("delivery", 102, "x"), "delivery-102")


class RoundTripTests(unittest.TestCase):
    def round_trip(self, delivery, run=RESTARTED, order="gap"):
        with tempfile.TemporaryDirectory() as directory:
            probes = Recording(directory, hashed(ProbeRuleTests.UNINTERRUPTED, "a"), delivery)
            return matched_evidence.round_trip(
                {}, FAMILY, 7, run, hashed(REFERENCE, "a"), probes, order=order
            )

    def test_v2_decides_from_a_packet_as_from_the_stored_runs(self):
        cases = {
            "accepted": ({101: TAKEN, 102: KEPT}, hashed(RESTARTED, "a"), "accept"),
            "delivery rejected": ({101: LAST, 102: KEPT}, hashed(RESTARTED, "a"), "reject"),
            "one direction": ({101: TAKEN, 102: TAKEN}, hashed(RESTARTED, "a"), "review"),
        }
        for name, (delivery, run, decision) in cases.items():
            for order in verifier_v2.ORDERS:
                with self.subTest(case=name, order=order):
                    result = self.round_trip(delivery, run, order)
                    self.assertEqual(result["stored"]["decision"], decision)
                    self.assertEqual(result["packet_result"], result["stored"])

    def test_the_determinism_control_survives_the_packet(self):
        from test_verifier_v2 import report

        run = hashed(
            report(
                [(1, 1, 0.90, "a"), (2, 2, 0.92, "x"), (3, 2, 0.93, "y")], "epoch 3 (attempt 2)"
            ),
            "a",
        )
        result = self.round_trip({101: TAKEN, 102: KEPT}, run)
        self.assertEqual(result["stored"]["checks"][:2], ["records", "self_restart"])
        self.assertEqual(result["stored"]["parts"]["restart"], "violated")
        self.assertEqual(result["packet_result"], result["stored"])
        runs = matched_evidence.parse_packet(result["packet"])
        self.assertEqual(list(runs)[2:], ["uninterrupted", "repeat"])

    def test_a_packet_that_lost_a_field_changes_v2s_result(self):
        result = self.round_trip({101: LAST, 102: KEPT}, hashed(RESTARTED, "a"))
        text = result["packet"]
        # The delivery probe without its delivered epoch: v2 no longer sees the violation.
        runs = matched_evidence.parse_packet(text)
        runs["delivery-101"]["pipeline"]["delivery"]["same_as"] = ["epoch 3 (attempt 2)"]
        runs.setdefault("delivery-102", matched_evidence.project(hashed(KEPT, "a")))
        replayed = matched_evidence.Replayed(
            {k: v for k, v in runs.items() if k not in ("candidate", "reference")},
            {"delivery": {"seeds": [101, 102]}},
        )
        changed = verifier_v2.verify({}, FAMILY, 7, runs["candidate"], runs["reference"], replayed)
        self.assertNotEqual(matched_evidence.summary(changed), result["stored"])
        json.dumps(result)


class CheckTextTests(unittest.TestCase):
    def test_the_check_counts_the_candidates_that_kept_their_result(self):
        kept = {"decision": "accept", "parts": {"runs": "supported"}, "checks": ["records"]}
        stored = {
            "decision": "accept",
            "parts": {"runs": {"status": "supported"}},
            "checks": ["records"],
        }
        results = {
            "a": {"packet": "x" * 10, "stored": kept, "packet_result": kept},
            "b": {
                "packet": "x" * 30,
                "stored": kept,
                "packet_result": {**kept, "decision": "review"},
            },
        }
        text = matched_evidence.check_text(results, {"a": stored, "b": stored})
        self.assertIn("b: replay = stored, packet != replay", text)
        self.assertNotIn("a:", text)
        self.assertIn(
            "2 candidates: the replay gives the stored result for 2, the packet the replay's "
            "for 1; packets of 10 to 30 characters",
            text,
        )


if __name__ == "__main__":
    unittest.main()
