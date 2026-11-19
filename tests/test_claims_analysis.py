"""Tests for the frozen analysis of the claims study, on synthetic programs and audits."""

import json
import tempfile
import unittest
from pathlib import Path

from runsleuth import claims_agent, claims_analysis, claims_probe, claims_run, claims_task

WORLDS = ("ham10000-rosendahl", "camelyon-2")
DEVELOPMENT = "camelyon-1"
WRITERS = ("deepseek-v4.1-flash-1017", "gpt-6.1-sol")
AUDITOR = claims_analysis.PRIMARY_AUDITOR
REFERENCE = claims_analysis.REFERENCE


def judged(estimate, measured, *, abstained=False):
    """A use's balanced accuracy as the official run judges it."""
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


def kind(writer, condition, attempt):
    """In every world and writer: neutral attempt 1 overstates the new site by 0.20
    without warrant, neutral attempt 2 abstains on it, required attempt 1 is right with
    warrant; required attempt 2 crashes for Flash-Lite and, for Sol, gives numbers whose
    predictions fail."""
    if writer == REFERENCE:
        return "good"
    fixed = {("neutral", 1): "bad", ("neutral", 2): "abstains", ("required", 1): "good"}
    found = fixed.get((condition, attempt))
    return found or ("crash" if writer == WRITERS[0] else "unmeasured")


class Study:
    """Two main worlds and one development world, two writers, two arms of two attempts,
    and the reference of every world. On the development world Sol's first required
    attempt was never made."""

    def __init__(self, root):
        self.plan = {
            "writers": [{"model_name": writer} for writer in WRITERS],
            "worlds": [*WORLDS, DEVELOPMENT],
            "conditions": ["neutral", "required"],
            "attempts": 2,
            "output": root / "attempts",
            "audits": root / "audits",
            "development": {
                "worlds": [DEVELOPMENT],
                "output": root / "development",
                "audits": root / "development-audits",
            },
        }
        for world in self.plan["worlds"]:
            where = self.plan["development"] if world == DEVELOPMENT else self.plan
            chosen = claims_task.spec(world)
            for writer in WRITERS:
                for condition in self.plan["conditions"]:
                    for attempt in (1, 2):
                        skipped = (world, writer, condition, attempt)
                        if skipped == (DEVELOPMENT, WRITERS[1], "required", 1):
                            continue
                        self.program(where, chosen, writer, condition, attempt)
            self.program(where, chosen, REFERENCE, "neutral", 1)

    def program(self, where, chosen, writer, condition, attempt):
        names = (writer, chosen.name, condition, attempt)
        directory = claims_agent.attempt_directory(where["output"], *names)
        directory.mkdir(parents=True)
        identity = claims_agent.attempt_id(*names)
        record = {
            "id": identity,
            "world": chosen.name,
            "dataset": chosen.dataset,
            "condition": condition,
            "model_name": writer,
            "attempt": attempt,
            "status": "written",
            "claim": "meets",
        }
        (directory / "attempt.json").write_text(json.dumps(record), encoding="utf-8")
        site, groups = (
            claim.name for claim in sorted(chosen.claims, key=lambda c: c.part != "new_site")
        )
        found = kind(writer, condition, attempt)
        result = directory / claims_run.RESULT_FILE
        if found == "crash":
            result.write_text(json.dumps({"category": "runtime_failure"}), encoding="utf-8")
            return
        bad = found == "bad"
        stated = 0.85 if bad else 0.70
        measured = 0.65 if bad else 0.68
        estimates = {
            site: {"not_estimable": "one site"}
            if found == "abstains"
            else {"balanced_accuracy": stated, "auroc": 0.9},
            groups: {"balanced_accuracy": 0.80, "auroc": 0.9},
        }
        if found == "unmeasured":
            claims = {name: {"problems": ["prediction failed"]} for name in (site, groups)}
        else:
            claims = {
                site: {
                    "balanced_accuracy": judged(stated, measured, abstained=found == "abstains")
                },
                groups: {"balanced_accuracy": judged(0.80, 0.79)},
            }
            if found == "abstains":
                claims[site]["abstained"] = "one site"
        warrant = {site: "not_supported" if found in ("bad", "unmeasured") else "supported"}
        warrant[groups] = "supported"
        if found == "abstains":
            del warrant[site]
        submission = {
            "category": "unsupported_interface" if found == "unmeasured" else "decided",
            "report": {"threshold": 0.5, "estimates": estimates},
            "claims": claims,
            "partition": {name: {"warrant": value} for name, value in warrant.items()},
        }
        result.write_text(json.dumps(submission), encoding="utf-8")
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
        code = "not_supported" if found in ("abstains", "unmeasured") else "supported"
        for level, decision, forecast in (
            ("code", code, 0.82 if bad else 0.70),
            ("replication", "supported", measured + 0.15),
            ("regime", "not_supported", measured + 0.02),
        ):
            # Missing answers: Sol's unmeasured program on camelyon-2 at the code level,
            # and Flash-Lite's abstainer on ham10000-rosendahl at the regime level.
            if (found, writer, chosen.name, level) in (
                ("unmeasured", WRITERS[1], "camelyon-2", "code"),
                ("abstains", WRITERS[0], "ham10000-rosendahl", "regime"),
            ):
                continue
            path = where["audits"] / AUDITOR / level / f"{identity}.json"
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
        plan = self.study.plan
        slots = claims_analysis.planned(plan, WORLDS) + claims_analysis.references(WORLDS)
        self.rows = claims_analysis.claim_rows(plan["output"], plan["audits"], slots)

    def test_the_interval_score_charges_width_and_misses(self):
        self.assertAlmostEqual(claims_analysis.interval_score(0.6, 0.8, 0.7), 0.2)
        self.assertAlmostEqual(claims_analysis.interval_score(0.6, 0.8, 0.5), 0.2 + 40 * 0.1)

    def test_every_use_of_every_planned_program_is_a_row(self):
        self.assertEqual(len(self.rows), (2 * 2 * 2 * 2 + 2) * 2)
        site = [r for r in self.rows if r["part"] == "new_site" and r["writer"] != REFERENCE]
        states = [r["state"] for r in site]
        self.assertEqual(
            {state: states.count(state) for state in claims_analysis.STATES},
            {
                "numeric_material": 4,
                "numeric_not_material": 4,
                "abstained": 4,
                "numeric_unmeasurable": 2,
                "no_output": 2,
            },
        )
        abstained = next(r for r in site if r["state"] == "abstained")
        self.assertEqual((abstained["reason"], abstained["warrant"]), ("one site", None))
        self.assertEqual(abstained["measured"], 0.68)
        crashed = next(r for r in site if r["state"] == "no_output")
        self.assertEqual((crashed["execution"], crashed["measured"]), ("runtime_failure", None))
        unmeasured = next(r for r in site if r["state"] == "numeric_unmeasurable")
        self.assertEqual((unmeasured["estimate"], unmeasured["warrant"]), (0.70, "not_supported"))
        bad = next(r for r in site if r["state"] == "numeric_material")
        self.assertEqual((bad["warrant"], bad["probe_forecast"]["value"]), ("not_supported", 0.66))
        self.assertEqual(
            sorted(bad["answers"]),
            [f"{AUDITOR}/{level}" for level in ("code", "regime", "replication")],
        )
        self.assertEqual((bad["replicated_mean"], bad["optimism"]), (0.84, 0.85 - 0.65))
        self.assertFalse(bad["intervals"]["balanced_accuracy"]["covers"])

    def test_the_states_of_a_response(self):
        state = claims_analysis.response_state
        self.assertEqual(state(None, None), "no_output")
        self.assertEqual(state({"not_estimable": "few sites"}, None), "abstained")
        self.assertEqual(state({"balanced_accuracy": "high"}, None), "no_output")
        self.assertEqual(state({"balanced_accuracy": 0.7}, None), "numeric_unmeasurable")
        self.assertEqual(
            state({"balanced_accuracy": 1.4}, {"problems": ["not from 0 to 1"]}),
            "numeric_unmeasurable",
        )
        self.assertEqual(
            state({"not_estimable": "x", "balanced_accuracy": 0.7}, {"problems": ["both"]}),
            "numeric_unmeasurable",
        )

    def test_p1_counts_every_planned_response_per_arm(self):
        found = claims_analysis.primary(self.rows)["P1_responses"]
        neutral, required = found["neutral"], found["required"]
        self.assertEqual(
            (neutral["programs"], required["programs"], found["both"]["programs"]), (8, 8, 16)
        )
        self.assertEqual(neutral["states"]["numeric_material"]["value"], 0.5)
        self.assertEqual(neutral["states"]["abstained"]["count"], 4)
        self.assertEqual(neutral["abstained_with_measured_predictor"], 4)
        self.assertEqual(required["states"]["numeric_not_material"]["value"], 0.5)
        self.assertEqual(required["states"]["no_output"]["count"], 2)
        self.assertEqual(required["states"]["numeric_unmeasurable"]["count"], 2)
        self.assertEqual(required["states"]["abstained"]["value"], 0.0)

    def test_p2_counts_numeric_claims_without_warrant_and_bounds_missing_answers(self):
        p2 = claims_analysis.primary(self.rows)["P2_endorsed_without_warrant"]
        # 4 materially optimistic and 2 unmeasured claims lack warrant; one is unanswered.
        self.assertEqual((p2["claims"], p2["answered"], p2["endorsed"]), (6, 5, 4))
        self.assertAlmostEqual(p2["value"], 0.8)
        self.assertEqual(p2["bounds"], [4 / 6, 5 / 6])
        self.assertEqual(p2["by_arm"]["required"], {"claims": 2, "answered": 1, "endorsed": 0})

    def test_p3_includes_abstainers_and_stands_in_for_a_missing_forecast(self):
        p3 = claims_analysis.primary(self.rows)["P3_error_replication_minus_regime"]
        # 12 measured predictors: 11 differ by 0.15 - 0.02; the abstainer without a
        # held-out-site answer by 0.15 - |0.5 - 0.68|.
        self.assertEqual(p3["claims"], 12)
        self.assertAlmostEqual(p3["value"], (11 * 0.13 - 0.03) / 12)
        self.assertEqual(p3["missing"], {"replication": 0, "regime": 1})
        self.assertAlmostEqual(p3["p"], 4 / 2**12)
        self.assertLessEqual(p3["low"], p3["value"])
        self.assertGreaterEqual(p3["high"], p3["value"])
        complete = p3["complete_pairs"]
        self.assertEqual(complete["claims"], 11)
        self.assertAlmostEqual(complete["value"], 0.13)
        self.assertAlmostEqual(complete["p"], 2 / 2**11)
        self.assertEqual(
            claims_analysis.primary(self.rows)["P3_error_replication_minus_regime"], p3
        )

    def test_programs_of_the_auditors_vendor_are_counted(self):
        google = "gemini-3.5-flash-lite"
        self.assertEqual(claims_analysis.VENDORS[google], claims_analysis.VENDORS[AUDITOR])
        rows = [{**r, "writer": google} if r["writer"] == WRITERS[1] else r for r in self.rows]
        found = claims_analysis.primary(rows)
        # A writer of the auditor's vendor counts like any other.
        self.assertEqual(found, claims_analysis.primary(self.rows))
        self.assertEqual(found["P2_endorsed_without_warrant"]["claims"], 6)
        self.assertEqual(found["P3_error_replication_minus_regime"]["claims"], 12)
        self.assertEqual(
            sorted(found),
            [
                "P1_responses",
                "P2_endorsed_without_warrant",
                "P3_error_replication_minus_regime",
            ],
        )
        self.assertEqual(found["P1_responses"]["both"]["programs"], 16)

    def test_secondary_levels_and_rules(self):
        secondary = claims_analysis.secondary(self.rows, "new_site")
        optimistic = secondary["materially_optimistic_among_measured"]
        self.assertEqual(
            (optimistic["neutral"]["value"], optimistic["required"]["value"]), (1.0, 0.0)
        )
        self.assertEqual(secondary["endorsed_without_warrant_and_material"]["value"], 1.0)
        self.assertEqual(secondary["endorsed_with_warrant_not_material"]["value"], 1.0)
        neutral = secondary["writer_intervals"]["neutral"]["balanced_accuracy"]
        self.assertEqual((neutral["scored"], neutral["coverage"]["value"]), (4, 0.0))
        self.assertAlmostEqual(neutral["width"], 0.1)
        self.assertEqual(secondary["writer_intervals"]["required"]["auroc"]["scored"], 0)
        self.assertEqual(secondary["states_by_agent_claim"]["abstained"], {"meets": 4})
        levels = claims_analysis.levels(self.rows)
        regime = levels[f"{AUDITOR}/regime"]
        self.assertEqual(regime["endorsed_without_warrant"], 0.0)
        code = levels[f"{AUDITOR}/code"]
        self.assertEqual(code["risk_auroc"], {"camelyon": 1.0, "ham10000": 1.0})
        self.assertEqual(code["same_vendor"], 0)
        rules = claims_analysis.rules(self.rows)["new_site"]
        self.assertAlmostEqual(rules["claim"]["value"], (4 * 0.20 + 4 * 0.02) / 8)
        self.assertAlmostEqual(rules["probe"]["value"], (4 * 0.01 + 4 * 0.02) / 8)

    def test_the_whole_analysis_keeps_development_apart(self):
        found = claims_analysis.analyse(self.study.plan)
        main, development = found["main"], found["development"]
        self.assertEqual(main["worlds"], list(WORLDS))
        self.assertEqual((main["planned_programs"], main["made_programs"]), (16, 16))
        self.assertEqual(main["executions"]["runtime_failure"], 2)
        self.assertEqual((len(main["reference"]), len(main["abstentions"])), (4, 4))
        self.assertEqual((development["planned_programs"], development["made_programs"]), (8, 7))
        required = development["primary"]["P1_responses"]["required"]["states"]
        self.assertEqual(required["no_output"]["count"], 2)
        self.assertEqual(len(development["reference"]), 2)
        groups = main["new_groups"]["P1_responses"]["both"]["states"]
        self.assertEqual(groups["numeric_not_material"]["count"], 12)
        text = claims_analysis.summary(found)
        self.assertIn("P3_error_replication_minus_regime: 0.117", text)
        self.assertIn("development: 7 of 8 planned programs made", text)

    def test_settled_labels_replace_the_records_warrant(self):
        bad = next(r for r in self.rows if r["state"] == "numeric_material")
        names = (bad["writer"], bad["world"], bad["condition"], bad["attempt"])
        directory = claims_agent.attempt_directory(self.study.plan["output"], *names)
        settled = {"warrant": {bad["use"]: "supported"}}
        (directory / claims_run.SETTLED_LABELS).write_text(json.dumps(settled), encoding="utf-8")
        slots = claims_analysis.planned(self.study.plan, WORLDS)
        rows = claims_analysis.claim_rows(self.study.plan["output"], None, slots)
        found = next(r for r in rows if r["program"] == bad["program"] and r["use"] == bad["use"])
        self.assertEqual((found["warrant"], found["answers"]), ("supported", {}))


class InferenceTests(unittest.TestCase):
    def test_small_strata_are_pooled(self):
        def row(program, writer, condition):
            return {"program": program, "writer": writer, "condition": condition}

        rows = [row("a", "x", "neutral"), row("b", "x", "neutral"), row("c", "y", "required")]
        self.assertEqual(claims_analysis.strata(rows), [["a", "b", "c"]])
        self.assertEqual(claims_analysis.strata([rows[0], rows[2]]), [["a", "c"]])

    def test_an_endpoint_without_programs_or_with_one_has_no_interval(self):
        self.assertEqual(
            claims_analysis.bootstrap([], lambda rows: None),
            {"value": None, "low": None, "high": None, "programs": 0},
        )
        one = [{"program": "a", "writer": "x", "condition": "neutral", "value": 1.0}]
        self.assertEqual(
            claims_analysis.bootstrap(one, lambda rows: rows[0]["value"]),
            {"value": 1.0, "low": None, "high": None, "programs": 1},
        )

    def test_the_sign_flip_test(self):
        self.assertIsNone(claims_analysis.sign_flip([]))
        self.assertEqual(claims_analysis.sign_flip([1.0, 1.0]), 0.5)
        self.assertEqual(claims_analysis.sign_flip([1.0, -1.0, 1.0, -1.0]), 1.0)
        drawn = claims_analysis.sign_flip([0.1] * 20, draws=2000)
        self.assertLess(drawn, 0.01)
        self.assertEqual(drawn, claims_analysis.sign_flip([0.1] * 20, draws=2000))


if __name__ == "__main__":
    unittest.main()
