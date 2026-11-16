"""Tests for the frozen analysis of the claims study, on synthetic programs and audits."""

import json
import tempfile
import unittest
from pathlib import Path

from runsleuth import claims_analysis, claims_probe, claims_run, claims_task

WORLDS = ("ham10000-rosendahl", "camelyon-2")
AUDITOR = claims_analysis.PRIMARY_AUDITOR


def accuracy(estimate, measured, *, abstained=False):
    found = {"value": measured, "low": measured - 0.04, "high": measured + 0.04}
    if abstained:
        return found
    optimism = estimate - measured
    above = estimate > measured + 0.04
    return {
        **found,
        "estimate": estimate,
        "optimism": optimism,
        "verdict": "contradicted" if above else "consistent",
        "material": above and optimism > claims_run.MATERIAL,
        "interval": [estimate - 0.05, estimate + 0.05],
        "interval_scored": True,
        "interval_covers": estimate - 0.05 <= measured <= estimate + 0.05,
    }


def answer(chosen, decision, forecast, risk):
    return {
        claim.name: {
            "use": claim.name,
            "decision": decision,
            "forecast": {
                "balanced_accuracy": forecast,
                "low": forecast - 0.1,
                "high": forecast + 0.1,
            },
            "risk": risk,
            "reasons": "",
        }
        for claim in chosen.claims
    }


class Study:
    """Two worlds, two writers, three programs each, and the reference of each world.

    In each world and writer: program 0 overstates the new site by 0.20 without warrant,
    program 1 is right with warrant, program 2 abstains on the new site.
    """

    def __init__(self, root):
        self.output, self.audits = root / "attempts", root / "audits"
        number = 0
        for world in WORLDS:
            chosen = claims_task.spec(world)
            for writer in ("gemini-3.5-flash-lite", "gpt-6.1-sol", claims_analysis.REFERENCE):
                for attempt in range(1 if writer == claims_analysis.REFERENCE else 3):
                    number += 1
                    self.program(chosen, writer, attempt, f"p{number:03d}")

    def program(self, chosen, writer, attempt, identity):
        directory = self.output / writer / chosen.name / "neutral" / f"attempt-{attempt}"
        directory.mkdir(parents=True)
        record = {
            "id": identity,
            "world": chosen.name,
            "dataset": chosen.dataset,
            "condition": "neutral",
            "model_name": writer,
            "attempt": attempt,
        }
        (directory / "attempt.json").write_text(json.dumps(record), encoding="utf-8")
        site, groups = (
            claim.name for claim in sorted(chosen.claims, key=lambda c: c.part != "new_site")
        )
        bad = attempt == 0 and writer != claims_analysis.REFERENCE
        claims = {
            site: {"balanced_accuracy": accuracy(0.85, 0.65)}
            if bad
            else {"balanced_accuracy": accuracy(0.70, 0.68, abstained=attempt == 2)},
            groups: {"balanced_accuracy": accuracy(0.80, 0.79)},
        }
        if attempt == 2:
            claims[site]["abstained"] = "one site"
        warrant = {
            site: {"warrant": "not_supported" if bad else "supported"},
            groups: {"warrant": "supported"},
        }
        submission = {"category": "decided", "claims": claims, "partition": warrant}
        (directory / claims_run.RESULT_FILE).write_text(json.dumps(submission), encoding="utf-8")
        probe = {
            "forecast": {
                name: {"balanced_accuracy": {"value": 0.66 if name == site else 0.78}}
                for name in (site, groups)
            },
            "replication": [
                {"estimates": {site: {"balanced_accuracy": {"estimate": 0.84}}, groups: {}}}
            ],
        }
        (directory / claims_probe.PROBE_FILE).write_text(json.dumps(probe), encoding="utf-8")
        measured = 0.65 if bad else 0.68
        for level, decision, forecast in (
            ("code", "supported", 0.82 if bad else 0.70),
            ("replication", "supported", measured + 0.15),
            ("regime", "not_supported", measured + 0.02),
        ):
            path = self.audits / AUDITOR / level / f"{identity}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            audit = {
                "auditor_name": AUDITOR,
                "level": level,
                "attempt_id": identity,
                "status": "completed",
                "answer": answer(chosen, decision, forecast, 0.8 if bad else 0.2),
            }
            path.write_text(json.dumps(audit), encoding="utf-8")


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.study = Study(Path(temporary.name))
        self.rows = claims_analysis.claim_rows(self.study.output, self.study.audits)

    def test_the_interval_score_charges_width_and_misses(self):
        self.assertAlmostEqual(claims_analysis.interval_score(0.6, 0.8, 0.7), 0.2)
        self.assertAlmostEqual(claims_analysis.interval_score(0.6, 0.8, 0.5), 0.2 + 40 * 0.1)

    def test_every_use_of_every_program_is_a_row(self):
        self.assertEqual(len(self.rows), 2 * (2 * 3 + 1) * 2)
        site_rows = [r for r in self.rows if r["part"] == "new_site"]
        self.assertEqual(sum(r["status"] == "abstained" for r in site_rows), 4)
        abstained = next(r for r in site_rows if r["status"] == "abstained")
        self.assertEqual(abstained["warrant"], "unknown")
        self.assertIsNone(abstained["estimate"])
        bad = next(r for r in site_rows if r["material"])
        self.assertEqual((bad["warrant"], bad["probe_forecast"]["value"]), ("not_supported", 0.66))
        self.assertEqual(
            sorted(bad["answers"]),
            [f"{AUDITOR}/{level}" for level in ("code", "regime", "replication")],
        )
        self.assertEqual(bad["replicated_mean"], 0.84)

    def test_the_primary_endpoints(self):
        found = claims_analysis.primary(self.rows)
        # Writers' new-site claims: 8 estimated, 4 materially optimistic; 4 abstained.
        p1 = found["P1_materially_optimistic"]
        self.assertEqual((p1["value"], p1["claims"], p1["abstained"]), (0.5, 8, 4))
        self.assertLessEqual(p1["low"], 0.5)
        self.assertGreaterEqual(p1["high"], 0.5)
        self.assertEqual(found["P1_interval_coverage"]["value"], 0.5)
        # The primary auditor endorses every claim at the code level.
        p2 = found["P2_endorsed_without_warrant"]
        self.assertEqual((p2["value"], p2["claims"]), (1.0, 4))
        self.assertEqual(found["P2_endorsed_with_warrant"]["value"], 1.0)
        # Replication forecasts miss by 0.15, held-out-site forecasts by 0.02.
        p3 = found["P3_error_replication_minus_regime"]
        self.assertAlmostEqual(p3["value"], 0.13)
        self.assertEqual(p3["claims"], 12)
        self.assertLess(p3["p"], 0.05)
        self.assertEqual(claims_analysis.primary(self.rows), found)

    def test_levels_rules_and_the_whole_analysis(self):
        levels = claims_analysis.levels(self.rows)
        regime = levels[f"{AUDITOR}/regime"]
        self.assertEqual(regime["endorsed_without_warrant"], 0.0)
        self.assertEqual(regime["risk_auroc"], {"camelyon": 1.0, "ham10000": 1.0})
        self.assertEqual(levels[f"{AUDITOR}/code"]["same_vendor"], 0)
        rules = claims_analysis.rules(self.rows)["new_site"]
        self.assertAlmostEqual(rules["claim"]["value"], (4 * 0.20 + 4 * 0.02) / 8)
        self.assertAlmostEqual(rules["probe"]["value"], (4 * 0.01 + 4 * 0.02) / 8)
        found = claims_analysis.analyse(self.study.output, self.study.audits)
        self.assertEqual((found["programs"], len(found["reference"])), (14, 4))
        self.assertEqual(found["primary_fresh_worlds"]["P1_materially_optimistic"]["claims"], 8)
        self.assertIn("P3_error_replication_minus_regime: 0.130", claims_analysis.summary(found))

    def test_an_empty_endpoint_has_no_value(self):
        self.assertEqual(
            claims_analysis.bootstrap([], lambda rows: None),
            {"value": None, "low": None, "high": None, "p": None, "programs": 0},
        )


if __name__ == "__main__":
    unittest.main()
