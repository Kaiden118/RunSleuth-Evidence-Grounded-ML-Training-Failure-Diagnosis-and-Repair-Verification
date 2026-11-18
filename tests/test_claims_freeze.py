"""Tests for the freeze of the claims study: what it covers, what counts as a change and
when the study's commands refuse."""

import json
import tempfile
import unittest
from pathlib import Path

from runsleuth import claims_analysis, claims_freeze, claims_task

FROZEN = {
    "files": {"src/a.py": "1", "src/b.py": "2"},
    "duplicates": {"d.csv": "3"},
    "data": {"x/images.npy": "4"},
    "agent_inputs": {"tools": "5"},
    "auditor_inputs": {"system_prompt": "6"},
    "analysis": {"material": 0.05},
    "plan": {"attempts": 4},
    "rules": {"stop": "x"},
    "statement_version": 2,
    "deviations": [],
}


def now(**changes):
    found = json.loads(json.dumps(FROZEN))
    del found["deviations"]
    for key, value in changes.items():
        found[key] = value
    return found


class CoverageTests(unittest.TestCase):
    def test_the_study_freezes_every_module_it_imports(self):
        names = {path.stem for path in claims_freeze.closure()}
        for name in (
            "claims_agent",
            "claims_run",
            "claims_task",
            "claims_freeze",
            "claims_probe",
            "claims_audit",
            "claims_reference",
            "claims_analysis",
            "claims_labels",
            "claims_partition",
            "standalone_agent",
            "standalone_check",
            "agent_loop",
            "agent_workspace",
            "llm_client",
            "ham10000_data",
            "camelyon_data",
        ):
            self.assertIn(name, names)
        self.assertNotIn("matched_screening", names)
        frozen = {path.as_posix() for path in claims_freeze.frozen_files()}
        for name in ("guard/sitecustomize.py", "reference/train.py"):
            self.assertIn(f"src/runsleuth/{name}", frozen)

    def test_the_writers_step_leaves_the_auditors_readers_and_analysis_to_the_second(self):
        writers = {path.stem for path in claims_freeze.writer_files()}
        for name in ("claims_agent", "claims_run", "claims_probe", "claims_partition"):
            self.assertIn(name, writers)
        for name in claims_freeze.AUDITOR_MODULES:
            self.assertNotIn(name, writers)
        self.assertIn("sitecustomize", writers)
        everything = {path.as_posix() for path in claims_freeze.frozen_files()}
        self.assertLessEqual({p.as_posix() for p in claims_freeze.writer_files()}, everything)
        self.assertEqual(
            set(claims_freeze.RULES),
            set(claims_freeze.WRITER_RULES) | set(claims_freeze.AUDITOR_RULES),
        )
        with self.assertRaisesRegex(ValueError, "Unknown stage"):
            claims_freeze.record("readers")

    def test_imports_are_found_anywhere_in_a_source(self):
        source = (
            "from runsleuth import a, b\nimport runsleuth.c\n"
            "def f():\n    from runsleuth.d import e\n    import os\n"
        )
        self.assertEqual(claims_freeze.imported(source), {"a", "b", "c", "d"})


class DifferenceTests(unittest.TestCase):
    def test_nothing_changed_is_no_difference(self):
        self.assertEqual(claims_freeze.differences(FROZEN, now()), [])

    def test_changed_files_data_inputs_and_rules_are_differences(self):
        found = claims_freeze.differences(
            FROZEN,
            now(
                files={"src/a.py": "9", "src/c.py": "7"},
                data={},
                agent_inputs={"tools": "6"},
                auditor_inputs={"system_prompt": "7"},
                rules={"stop": "y"},
            ),
        )
        self.assertEqual(
            found,
            [
                "src/a.py changed",
                "src/b.py is missing",
                "src/c.py is new in what the study runs",
                "x/images.npy is missing",
                "the agent inputs changed",
                "the auditor inputs changed",
                "the rules changed",
            ],
        )

    def test_a_recorded_deviation_allows_that_text_only(self):
        frozen = {**FROZEN, "deviations": [{"file": "src/a.py", "sha256": "9"}]}
        self.assertEqual(
            claims_freeze.differences(frozen, now(files={"src/a.py": "9", "src/b.py": "2"})), []
        )
        self.assertEqual(
            claims_freeze.differences(frozen, now(files={"src/a.py": "8", "src/b.py": "2"})),
            ["src/a.py changed"],
        )


class GuardTests(unittest.TestCase):
    def test_development_outputs_pass_the_pilot_is_closed_and_the_study_needs_its_freeze(self):
        with tempfile.TemporaryDirectory() as temporary:
            claims_freeze.guard(Path(temporary) / "programs")
            missing = Path(temporary) / "none.json"
            with self.assertRaisesRegex(RuntimeError, "not frozen"):
                claims_freeze.guard(Path(claims_freeze.PLAN["output"]), freeze_file=missing)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            claims_freeze.guard(claims_freeze.PILOT_OUTPUT)

    def test_the_study_runs_only_the_plans_writers_worlds_and_auditors(self):
        study = Path(claims_freeze.PLAN["output"])
        holds = claims_freeze.check
        claims_freeze.check = lambda freeze_file=None: []
        self.addCleanup(setattr, claims_freeze, "check", holds)
        plan = claims_freeze.PLAN
        claims_freeze.guard(
            study,
            writer=plan["writers"][1],
            dataset="camelyon-2",
            condition="required",
            attempts=2,
            auditor=claims_freeze.AUDITORS[0],
            paths={"data": Path(plan["data"]), "work": plan["work"], "probes": plan["probes"]},
            device="cuda",
        )
        with self.assertRaisesRegex(RuntimeError, "not in the plan"):
            claims_freeze.guard(study, writer={"model_name": "other"}, condition="pressure")
        with self.assertRaisesRegex(RuntimeError, "auditor"):
            claims_freeze.guard(study, auditor={"auditor_name": "other"})
        with self.assertRaisesRegex(RuntimeError, "makes 2 attempts"):
            claims_freeze.guard(study, attempts=3)
        with self.assertRaisesRegex(RuntimeError, "development world; its programs go to"):
            claims_freeze.guard(study, dataset="camelyon-1")
        with self.assertRaisesRegex(RuntimeError, "runs is elsewhere, not the plan's"):
            claims_freeze.guard(study, paths={"runs": "elsewhere"})
        with self.assertRaisesRegex(RuntimeError, "probes is None"):
            claims_freeze.guard(study, paths={"probes": None})
        with self.assertRaisesRegex(RuntimeError, "the device is cpu"):
            claims_freeze.guard(study, device="cpu")
        # Outside the study's output nothing is checked.
        claims_freeze.guard(Path(plan["development"]["output"]), dataset="camelyon-1", device="cpu")
        self.assertEqual(set(plan["development"]["worlds"]), {"ham10000-vienna_dias", "camelyon-1"})
        self.assertEqual(sorted(claims_freeze.PLAN["worlds"]), sorted(claims_task.SPECS))
        writers = {writer["model_name"] for writer in claims_freeze.PLAN["writers"]}
        self.assertLessEqual(writers, set(claims_analysis.VENDORS))
        self.assertEqual(len(writers), 5)
        self.assertEqual(claims_analysis.PRIMARY_AUDITOR, claims_freeze.AUDITORS[0]["auditor_name"])
        # The primary auditor never reads a program of its own vendor.
        auditors = {auditor["auditor_name"] for auditor in claims_freeze.AUDITORS}
        self.assertEqual(auditors, {"claude-opus-5.5", "gemini-4-carbon"})
        self.assertLessEqual(auditors, set(claims_analysis.VENDORS))
        primary = claims_analysis.VENDORS[claims_analysis.PRIMARY_AUDITOR]
        self.assertNotIn(primary, {claims_analysis.VENDORS[name] for name in writers})


class StepTests(unittest.TestCase):
    def test_auditors_and_readers_need_the_second_step(self):
        study = Path(claims_freeze.PLAN["output"])
        holds = claims_freeze.check
        claims_freeze.check = lambda freeze_file=None: (
            [] if freeze_file == claims_freeze.FREEZE_FILE else ["the auditors' step is missing"]
        )
        self.addCleanup(setattr, claims_freeze, "check", holds)
        claims_freeze.guard(study, dataset="camelyon-2")
        with self.assertRaisesRegex(RuntimeError, "auditors' step is missing"):
            claims_freeze.guard(study, stage="auditors")
        with self.assertRaisesRegex(RuntimeError, "auditors' step is missing"):
            claims_freeze.guard(study, auditor=claims_freeze.AUDITORS[0])
        with self.assertRaisesRegex(ValueError, "Unknown stage"):
            claims_freeze.guard(study, stage="readers")

    def test_a_deviation_goes_to_every_step_that_holds_the_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "a.py"
            source.write_text("x = 1\n")
            first, second = root / "writers.json", root / "auditors.json"
            name = source.as_posix()
            first.write_text(json.dumps({"files": {name: "0"}, "deviations": []}))
            second.write_text(json.dumps({"files": {"other.py": "0"}, "deviations": []}))
            found = claims_freeze.deviation(source, "fixed a crash", (first, second))
            self.assertEqual([entry["freeze"] for entry in found], [first.as_posix()])
            self.assertEqual(json.loads(second.read_text())["deviations"], [])
            (root / "b.py").write_text("y = 2\n")
            with self.assertRaisesRegex(ValueError, "not a frozen file"):
                claims_freeze.deviation(root / "b.py", "x", (second, root / "none.json"))


class AmendmentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "study"
        self.freeze_file = self.root / "freeze.json"
        self.old = json.loads(json.dumps(claims_freeze.PLAN))
        self.old["writers"][1] = {"model_name": "old", "provider": "p", "model": "old-id"}
        self.new = json.loads(json.dumps(self.old))
        self.new["writers"][1] = {"model_name": "new", "provider": "p", "model": "new-id"}
        self.rules = {"arms": "new text", "stop": "same"}
        frozen = {
            **self.state(self.old, {"arms": "old text", "stop": "same"}),
            "frozen_at": "then",
            "commit": "abc",
            "deviations": [],
        }
        self.freeze_file.write_text(json.dumps(frozen), encoding="utf-8")
        for name, value in (
            ("record", lambda stage="writers", plan=None: self.state(self.new, self.rules)),
            ("_git", lambda *args: "def\n"),
        ):
            self.addCleanup(setattr, claims_freeze, name, getattr(claims_freeze, name))
            setattr(claims_freeze, name, value)

    @staticmethod
    def state(plan, rules):
        return {
            "stage": "writers",
            "plan": plan,
            "schedule": claims_freeze.schedule(plan),
            "rules": rules,
            "files": {"a.py": "1"},
            "duplicates": {},
            "data": {},
            "agent_inputs": {"tools": "1"},
        }

    def amend(self):
        return claims_freeze.amend("newer model", freeze_file=self.freeze_file, output=self.output)

    def test_a_replaced_writer_is_recorded_and_the_others_keep_their_order(self):
        entry = self.amend()
        frozen = json.loads(self.freeze_file.read_text(encoding="utf-8"))
        self.assertEqual(entry["replaced"][0]["from"]["model_name"], "old")
        self.assertEqual(entry["rules_before"], {"arms": "old text"})
        self.assertEqual((entry["commit"], entry["study_programs_then"]), ("def", 0))
        self.assertEqual(frozen["amendments"], [entry])
        self.assertEqual(claims_freeze.differences(frozen, self.state(self.new, self.rules)), [])
        before = [i for i in claims_freeze.schedule(self.old) if i["writer"] != "old"]
        self.assertEqual(before, [i for i in frozen["schedule"] if i["writer"] != "new"])

    def test_no_amendment_once_the_study_holds_a_program_of_either_model(self):
        for name in ("old", "new"):
            attempt = self.output / name / "w" / "neutral" / "attempt-1"
            attempt.mkdir(parents=True)
            (attempt / "attempt.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, f"holds programs of {name}"):
                self.amend()
            (attempt / "attempt.json").unlink()

    def test_only_a_writers_replacement_can_be_amended(self):
        with self.assertRaisesRegex(ValueError, "states its reason"):
            claims_freeze.amend(" ", freeze_file=self.freeze_file, output=self.output)
        self.new["attempts"] = 3
        with self.assertRaisesRegex(ValueError, "replaces writers only; changed: attempts"):
            self.amend()
        self.new["attempts"] = self.old["attempts"]
        self.new["writers"][1] = self.old["writers"][1]
        with self.assertRaisesRegex(ValueError, "no writer is replaced"):
            self.amend()


class ScheduleTests(unittest.TestCase):
    def test_every_planned_attempt_comes_once_with_the_arms_interleaved(self):
        plan = claims_freeze.PLAN
        found = claims_freeze.schedule()
        self.assertEqual(found, claims_freeze.schedule())
        slots = [(i["writer"], i["world"], i["condition"], i["attempt"]) for i in found]
        self.assertEqual(len(slots), len(set(slots)))
        self.assertEqual(
            len(slots),
            len(plan["writers"]) * len(plan["worlds"]) * len(plan["conditions"]) * 2,
        )
        for start in range(0, len(found), 4):
            block = found[start : start + 4]
            self.assertEqual(len({(i["writer"], i["world"]) for i in block}), 1)
            for condition in plan["conditions"]:
                numbers = [i["attempt"] for i in block if i["condition"] == condition]
                self.assertEqual(numbers, [1, 2])
            development = block[0]["world"] in plan["development"]["worlds"]
            self.assertEqual(
                block[0]["output"],
                plan["development"]["output"] if development else plan["output"],
            )
        orders = {tuple(i["condition"] for i in found[s : s + 4]) for s in range(0, len(found), 4)}
        self.assertGreater(len(orders), 1)
        commands = claims_freeze.schedule_commands("gpt-6.2-sol")
        self.assertEqual(len(commands), 36)
        self.assertTrue(all("--provider compatible --model gpt-6.2-sol" in c for c in commands))
        self.assertFalse(any("--echo-reasoning" in c for c in commands))
        deepseek = claims_freeze.schedule_commands("deepseek-v4.1-flash-1017")
        self.assertTrue(all(c.endswith("--echo-reasoning") for c in deepseek))


if __name__ == "__main__":
    unittest.main()
