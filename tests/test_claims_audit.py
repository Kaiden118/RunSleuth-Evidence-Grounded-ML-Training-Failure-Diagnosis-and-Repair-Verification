"""Tests for the AI auditors of claims: what they read at each level, how their answers
are read, and the audits kept for every attempt."""

import json
import tempfile
import unittest
from pathlib import Path

from test_standalone_agent import ApiError, FakeClient

from runsleuth import claims_audit, claims_probe, claims_run, claims_task

HAM = claims_task.spec("ham10000-vienna_dias")
SECRET = "SEALED-RESULT-7731"
PROGRAM = "print('the program')\n"


def estimate(value):
    return {
        "balanced_accuracy": {"estimate": value, "interval": [value - 0.05, value + 0.05]},
        "auroc": {"estimate": value + 0.1, "interval": None},
    }


def measured(value):
    found = {"value": value, "low": value - 0.04, "high": value + 0.04}
    return {"balanced_accuracy": found, "auroc": dict(found)}


PROBE = {
    "replication": [
        {
            "seed": 2,
            "training": {"status": "completed", "seconds": 61.0},
            "estimates": {"new_lesions": estimate(0.8), "new_source": {"not_estimable": "x"}},
        },
        {"seed": 3, "training": {"status": "timeout", "seconds": 1800.0}, "estimates": {}},
    ],
    "regime": [
        {
            "site": site,
            "training": {"status": "completed", "seconds": 70.0 + n},
            "estimates": {"new_lesions": estimate(0.78), "new_source": estimate(0.77)},
            "measured": {"new_groups": measured(0.75), "new_site": measured(0.61)},
            "problems": [],
        }
        for n, site in enumerate(("rosendahl", "vidir_modern", "vidir_molemax"))
    ],
}
FACTS = {
    "site_column": "source",
    "group_column": "lesion_id",
    "uses": {
        "new_lesions": {
            "measured_by": [
                {
                    "fit": "delivered",
                    "rows": 900,
                    "groups": 700,
                    "sites": ["rosendahl", "vidir_modern"],
                    "groups_also_trained_on": 12,
                    "groups_also_chosen_on": 0,
                    "sites_also_trained_on": ["rosendahl", "vidir_modern"],
                    "sites_also_chosen_on": [],
                }
            ]
        },
        "new_source": {"measured_by": []},
    },
}


def answer(*, level="code", **changes):
    entries = []
    for use in ("new_lesions", "new_source"):
        entry = {
            "use": use,
            "decision": "not_supported",
            "forecast": {"balanced_accuracy": 0.66, "low": 0.55, "high": 0.75},
            "risk": 0.7,
            "reasons": "the estimate was measured on training sources",
        }
        if level == "requirements":
            entry = {"use": use, "needed_evidence": "a held-out source", **entry}
        entry.update(changes)
        entries.append(entry)
    return json.dumps({"uses": entries})


class Fixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "attempts"
        self.runs = self.root / "runs"
        self.probes = self.root / "probes"
        self.directories = [self.attempt(n) for n in range(6)]

    def attempt(self, number):
        identity = f"id{number:03d}"
        directory = self.output / "writer" / HAM.name / "neutral" / f"attempt-{number}"
        directory.mkdir(parents=True)
        (directory / claims_task.PROGRAM).write_text(PROGRAM, encoding="utf-8")
        record = {
            "id": identity,
            "world": HAM.name,
            "dataset": "ham10000",
            "condition": "neutral",
            "model_name": "writer",
            "attempt": number,
        }
        (directory / "attempt.json").write_text(json.dumps(record), encoding="utf-8")
        (directory / claims_run.RESULT_FILE).write_text(SECRET, encoding="utf-8")
        (directory / claims_probe.PROBE_FILE).write_text(json.dumps(PROBE), encoding="utf-8")
        train = self.runs / identity / "train"
        train.mkdir(parents=True)
        (train / claims_task.REPORT).write_text('{"threshold": 0.4}', encoding="utf-8")
        (train / claims_task.REPORT_MD).write_text(
            "## Intended use\n\nReport text.", encoding="utf-8"
        )
        (self.runs / identity / claims_run.OUTPUT_FILE).write_text(
            "Epoch 5\nDone.", encoding="utf-8"
        )
        (self.runs / identity / claims_run.PROVENANCE_FILE).write_text(
            json.dumps(FACTS), encoding="utf-8"
        )
        for site in ("rosendahl", "vidir_modern", "vidir_molemax"):
            out = self.probes / identity / f"regime-{site}"
            out.mkdir(parents=True)
            (out / "delivered.pt").write_text("x", encoding="utf-8")
            (self.probes / identity / f"regime-{site}-output.txt").write_text(
                "a\nb\nc", encoding="utf-8"
            )
        return directory

    def packet(self, level, number=0):
        directory = self.directories[number]
        attempt = json.loads((directory / "attempt.json").read_text(encoding="utf-8"))
        return claims_audit.packet(attempt, directory, self.runs, level, probes=self.probes)


class PacketTests(Fixture):
    def test_every_level_adds_its_part_and_nothing_sealed(self):
        shown = {level: self.packet(level) for level in claims_audit.LEVELS}
        for level, text in shown.items():
            self.assertIn("## The task the developer was given", text, level)
            self.assertIn("Report text.", text)
            self.assertNotIn(SECRET, text)
            self.assertNotIn("writer", text.split("## The task")[0])
            self.assertIn('"use": "new_source"', text)
            # The form lists the decisions; none is the example's.
            self.assertIn('"decision": "supported | not_supported | cannot_tell"', text)
            self.assertNotIn('"decision": "supported",', text)
        self.assertNotIn("print('the program')", shown["report"])
        for level in claims_audit.LEVELS[1:]:
            self.assertIn("print('the program')", shown[level])
            self.assertIn("Done.", shown[level])
        self.assertIn("## Items to check", shown["checklist"])
        self.assertIn("needed_evidence", shown["requirements"])
        self.assertNotIn("needed_evidence", shown["code"])
        self.assertIn("## Provenance", shown["provenance"])
        self.assertIn("none recorded", shown["provenance"])
        self.assertIn("## Replication", shown["replication"])
        self.assertIn("timeout", shown["replication"])
        self.assertIn("0.610 [0.570, 0.650]", shown["regime"])
        self.assertIn("## Provenance", shown["both"])
        self.assertIn("0.610", shown["both"])
        for level in ("report", "code", "checklist", "requirements", "provenance", "replication"):
            self.assertNotIn("## Held-out sites", shown[level])

    def test_the_placebo_has_the_regime_tables_form_and_none_of_its_numbers(self):
        regime = self.packet("regime").split("## Held-out sites")[1].split("## Your answer")[0]
        placebo = self.packet("placebo").split("## Held-out sites")[1].split("## Your answer")[0]
        regime_rows = [line for line in regime.splitlines() if line.startswith("|")]
        placebo_rows = [line for line in placebo.splitlines() if line.startswith("|")]
        self.assertEqual(len(placebo_rows), len(regime_rows))
        self.assertEqual(
            [row.count("|") for row in placebo_rows], [row.count("|") for row in regime_rows]
        )
        for value in ("0.610", "0.750", "0.770"):
            self.assertNotIn(value, placebo)
        self.assertIn("70.0", placebo)
        with self.assertRaisesRegex(ValueError, "probes"):
            attempt = json.loads((self.directories[0] / "attempt.json").read_text())
            claims_audit.packet(attempt, self.directories[0], self.runs, "placebo")

    def test_a_third_of_attempts_get_the_placebo(self):
        drawn = sum(claims_audit.placebo_drawn(f"attempt-{n}") for n in range(3000))
        self.assertTrue(900 < drawn < 1100, drawn)

    def test_the_order_of_the_levels_is_drawn_per_attempt(self):
        levels = list(claims_audit.EVERY_PROGRAM)
        orders = [claims_audit.call_order(f"attempt-{n}", levels) for n in range(200)]
        self.assertTrue(all(sorted(order) == sorted(levels) for order in orders))
        self.assertEqual(orders[0], claims_audit.call_order("attempt-0", levels))
        first = [order.index("replication") < order.index("regime") for order in orders]
        self.assertTrue(60 < sum(first) < 140, sum(first))


class AnswerTests(unittest.TestCase):
    def test_an_answer_is_read_by_use(self):
        found, issues = claims_audit.parse_answer("```json\n" + answer() + "\n```", HAM, "code")
        self.assertEqual(issues, [])
        self.assertEqual(sorted(found), ["new_lesions", "new_source"])
        self.assertEqual(found["new_source"]["risk"], 0.7)

    def test_a_wrong_answer_says_what_is_wrong(self):
        cases = (
            (answer(decision="maybe"), "decision must be one of"),
            (answer(risk=1.5), "risk must be a probability"),
            (answer(forecast={"balanced_accuracy": 0.9, "low": 0.5, "high": 0.8}), "within low"),
            (answer(forecast={"balanced_accuracy": 0.6}), "needs balanced_accuracy"),
            (json.dumps({"uses": [json.loads(answer())["uses"][0]]}), "no entry for new_source"),
            ("not json", "not one JSON object"),
            (answer(), "needed_evidence must be given"),
        )
        for text, expected in cases:
            level = "requirements" if "needed_evidence" in expected else "code"
            found, issues = claims_audit.parse_answer(text, HAM, level)
            self.assertIsNone(found)
            self.assertTrue(any(expected in issue for issue in issues), (expected, issues))
        found, _ = claims_audit.parse_answer(answer(level="requirements"), HAM, "requirements")
        self.assertEqual(found["new_lesions"]["needed_evidence"], "a held-out source")

    def test_the_schema_asks_for_needed_evidence_only_at_its_level(self):
        code = claims_audit.response_schema(HAM, "code")["properties"]["uses"]["items"]
        asked = claims_audit.response_schema(HAM, "requirements")["properties"]["uses"]["items"]
        self.assertNotIn("needed_evidence", code["required"])
        self.assertEqual(asked["required"][:2], ["use", "needed_evidence"])

    def test_an_unusable_reply_is_asked_again_once(self):
        client = FakeClient(["not json", answer()])
        record = claims_audit.audit(client, "auditor", "packet", HAM, "code")
        self.assertEqual((record["status"], len(record["calls"])), ("completed", 2))
        self.assertIn("could not be used", client.requests[1]["messages"][-1]["content"])
        self.assertEqual(client.requests[0]["response_format"]["type"], "json_schema")
        failed = claims_audit.audit(FakeClient(["no", "no"]), "auditor", "p", HAM, "code")
        self.assertEqual(failed["status"], "invalid_output")
        broken = claims_audit.audit(FakeClient([ApiError(500)]), "auditor", "p", HAM, "code")
        self.assertEqual((broken["status"], broken["error"]["status_code"]), ("api_error", 500))
        plain = FakeClient([answer()])
        claims_audit.audit(plain, "auditor", "p", HAM, "code", response_format="text")
        self.assertNotIn("response_format", plain.requests[0])


class AuditTests(Fixture):
    def test_every_attempt_is_audited_at_every_level_once(self):
        levels = ("report", "regime", "placebo")
        replies = [answer() for _ in range(len(self.directories) * len(levels))]
        client = FakeClient(replies)
        lines = []
        done = claims_audit.audit_attempts(
            self.output,
            self.runs,
            self.root / "audits",
            "auditor",
            client,
            "model",
            levels=levels,
            probes=self.probes,
            progress=lines.append,
        )
        drawn = [
            f"id{n:03d}"
            for n in range(len(self.directories))
            if claims_audit.placebo_drawn(f"id{n:03d}")
        ]
        self.assertEqual(len(done), 2 * len(self.directories) + len(drawn))
        for record in done:
            self.assertEqual(record["status"], "completed")
            self.assertNotIn(SECRET, json.dumps(record))
        placebo = sorted(p.stem for p in (self.root / "audits" / "auditor" / "placebo").iterdir())
        self.assertEqual(placebo, drawn)
        again = claims_audit.audit_attempts(
            self.output,
            self.runs,
            self.root / "audits",
            "auditor",
            FakeClient([]),
            "model",
            levels=levels,
            probes=self.probes,
        )
        self.assertEqual(again, [])

    def test_levels_with_a_table_wait_for_the_probe_and_errors_are_asked_again(self):
        (self.directories[0] / claims_probe.PROBE_FILE).unlink()
        client = FakeClient([ApiError(503)])
        done = claims_audit.audit_attempts(
            self.output,
            self.runs,
            self.root / "audits",
            "auditor",
            client,
            "model",
            levels=("regime",),
            progress=lambda line: None,
        )
        self.assertEqual([record["attempt_id"] for record in done], ["id001"])
        self.assertEqual(done[0]["status"], "api_error")
        retried = claims_audit.audit_attempts(
            self.output,
            self.runs,
            self.root / "audits",
            "auditor",
            FakeClient([answer()]),
            "model",
            levels=("regime",),
            limit=1,
            progress=lambda line: None,
        )
        self.assertEqual(
            [(r["attempt_id"], r["status"]) for r in retried], [("id001", "completed")]
        )


if __name__ == "__main__":
    unittest.main()
