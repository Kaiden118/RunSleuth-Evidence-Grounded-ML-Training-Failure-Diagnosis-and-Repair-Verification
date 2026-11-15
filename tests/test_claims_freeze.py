"""Tests for the freeze of the claims study: what it covers, what counts as a change and
when the study's commands refuse."""

import json
import tempfile
import unittest
from pathlib import Path

from runsleuth import claims_freeze, claims_run

FROZEN = {
    "files": {"src/a.py": "1", "src/b.py": "2"},
    "duplicates": {"d.csv": "3"},
    "data": {"x/images.npy": "4"},
    "agent_inputs": {"tools": "5"},
    "plan": {"attempts": 2},
    "rules": {"stop": "x"},
    "statement_version": 1,
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
    def test_development_outputs_pass_and_the_study_needs_its_freeze(self):
        with tempfile.TemporaryDirectory() as temporary:
            claims_freeze.guard(Path(temporary) / "programs")
            missing = Path(temporary) / "none.json"
            with self.assertRaisesRegex(RuntimeError, "not frozen"):
                claims_freeze.guard(Path(claims_freeze.PLAN["output"]), freeze_file=missing)

    def test_the_study_runs_only_the_plans_writers_and_labelled_programs(self):
        study = Path(claims_freeze.PLAN["output"])
        holds = claims_freeze.check
        claims_freeze.check = lambda freeze_file=None: []
        self.addCleanup(setattr, claims_freeze, "check", holds)
        claims_freeze.guard(study, writer=claims_freeze.PLAN["writers"][1], attempts=2)
        with self.assertRaisesRegex(RuntimeError, "not in the plan"):
            claims_freeze.guard(study, writer={"model_name": "other"}, condition="calm")
        with self.assertRaisesRegex(RuntimeError, "makes 2 attempts"):
            claims_freeze.guard(study, attempts=3)
        self.assertEqual(claims_run.READERS, ("claude", "gpt-6-astra"))


if __name__ == "__main__":
    unittest.main()
