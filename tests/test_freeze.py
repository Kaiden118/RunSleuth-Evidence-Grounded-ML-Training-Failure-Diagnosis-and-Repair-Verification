"""CPU tests for the freeze: the record of every frozen file, the check every command on
the test split makes first, and deviations. git is replaced, since the Docker image has
no repository."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

if torch is not None:
    from runsleuth import candidates, challenge_set, freeze, pipeline_agent, verifier_evaluation


def fake_git(status: str = ""):
    def git(*args: str) -> str:
        return {"status": status, "rev-parse": "c0ffee\n", "ls-files": ""}[args[0]]

    return git


@unittest.skipIf(torch is None, "Install torch and torchvision for the freeze tests")
class FreezeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        package = self.root / freeze.PACKAGE
        (package / "pipeline_templates" / "camelyon").mkdir(parents=True)
        (package / "__pycache__").mkdir()
        (package / "verifier.py").write_bytes(b"ACCEPT = True\r\n")
        (package / "pipeline_templates" / "camelyon" / "train.py").write_text("x = 1\n")
        (package / "__pycache__" / "verifier.cpython-311.pyc").write_bytes(b"\0")
        for path in freeze.FROZEN_DATA:
            (self.root / path).parent.mkdir(parents=True, exist_ok=True)
            (self.root / path).write_text("{}\n")
        self.record = self.root / "evaluations" / "freeze.json"
        self.seeds = self.root / "evaluations" / "seeds.json"
        for patcher in (
            patch.object(freeze, "_git", fake_git()),
            patch.object(freeze, "settings", lambda: {"models": ["m"]}),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def write(self):
        return freeze.write(self.record, root=self.root, seeds=self.seeds)

    def test_the_record_holds_every_frozen_file_by_its_text(self):
        record = self.write()
        self.assertEqual(
            set(record["files"]),
            {
                "src/runsleuth/verifier.py",
                "src/runsleuth/pipeline_templates/camelyon/train.py",
                *(path.as_posix() for path in freeze.FROZEN_DATA),
            },
        )
        lf = self.root / "lf.py"
        lf.write_bytes(b"ACCEPT = True\n")
        self.assertEqual(record["files"]["src/runsleuth/verifier.py"], freeze._file_sha256(lf))
        self.assertEqual(record["commit"], "c0ffee")
        self.assertEqual(len(record["independence"]), 3)
        self.assertEqual(record["budget"]["cap_usd"], 20)
        self.assertEqual(record["deviations"], [])
        self.assertEqual(json.loads(self.record.read_text(encoding="utf-8")), record)

    def test_a_freeze_is_refused_with_changes_not_committed_after_the_seeds_or_twice(self):
        with patch.object(freeze, "_git", fake_git(" M src/runsleuth/verifier.py\n")):
            with self.assertRaisesRegex(RuntimeError, "verifier.py"):
                self.write()
        self.assertFalse(self.record.exists())
        self.seeds.write_text("{}")
        with self.assertRaises(FileExistsError):
            self.write()
        self.seeds.unlink()
        self.write()
        with self.assertRaises(FileExistsError):
            self.write()

    def test_a_changed_frozen_file_stops_the_test_split_until_its_deviation_is_recorded(self):
        with self.assertRaises(FileNotFoundError):
            freeze.check(self.record, root=self.root)
        self.write()
        freeze.check(self.record, root=self.root)
        verifier = self.root / freeze.PACKAGE / "verifier.py"
        # Line ends alone change nothing.
        verifier.write_bytes(b"ACCEPT = True\n")
        freeze.check(self.record, root=self.root)
        verifier.write_bytes(b"ACCEPT = False\n")
        with self.assertRaisesRegex(RuntimeError, "verifier.py"):
            freeze.check(self.record, root=self.root)
        with self.assertRaises(ValueError):
            freeze.deviation("src/runsleuth/verifier.py", " ", self.record, root=self.root)
        with self.assertRaises(ValueError):
            freeze.deviation("src/runsleuth/new.py", "added", self.record, root=self.root)
        freeze.deviation("src/runsleuth/verifier.py", "fixed a crash", self.record, root=self.root)
        record = freeze.check(self.record, root=self.root)
        self.assertEqual(record["deviations"][0]["reason"], "fixed a crash")
        (self.root / freeze.PACKAGE / "pipeline_templates" / "camelyon" / "train.py").unlink()
        with self.assertRaisesRegex(RuntimeError, "train.py"):
            freeze.check(self.record, root=self.root)

    def test_nothing_runs_on_the_test_split_without_the_freeze(self):
        with patch.object(candidates.freeze, "check", side_effect=FileNotFoundError("freeze")):
            with self.assertRaisesRegex(FileNotFoundError, "freeze"):
                candidates.candidates("test")
        # The development split needs no freeze.
        with patch.object(candidates.freeze, "check", side_effect=AssertionError):
            self.assertEqual(len(candidates.candidates("development")), 48)


@unittest.skipIf(torch is None, "Install torch and torchvision for the freeze tests")
class SettingsTests(unittest.TestCase):
    def test_the_settings_are_those_of_the_frozen_modules(self):
        found = freeze.settings()
        self.assertEqual(found["models"], list(challenge_set.MODELS))
        self.assertEqual(found["reviewers"], list(challenge_set.REVIEWERS))
        self.assertEqual(
            found["plan"],
            {"repair": 180, "version": 13, "partial_repair": 6, "evidence_removed": 26},
        )
        self.assertEqual(found["agent"]["max_output_tokens"], pipeline_agent.MAX_OUTPUT_TOKENS)
        self.assertEqual(found["request_timeout_seconds"], 600.0)
        self.assertEqual(found["bootstrap"], verifier_evaluation.BOOTSTRAP)
        self.assertEqual(found["v2"]["probes"]["delivery"]["epochs"], 5)
        json.dumps(found)


if __name__ == "__main__":
    unittest.main()
