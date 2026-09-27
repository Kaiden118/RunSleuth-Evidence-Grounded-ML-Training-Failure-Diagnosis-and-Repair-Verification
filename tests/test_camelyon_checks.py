"""Check that a failed command prevents training without running external tools."""

import contextlib
import io
import runpy
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_camelyon.py"
RUNNER = runpy.run_path(str(SCRIPT_PATH), run_name="camelyon_check_tests")


class CamelyonCheckRunnerTests(unittest.TestCase):
    def commands(self, train=True):
        return RUNNER["build_commands"](
            train=train, download=train, config="configs/camelyon17_clean.json"
        )

    def test_each_failed_check_prevents_training(self):
        commands = self.commands()
        for failed_index in range(len(commands) - 1):
            error = subprocess.CalledProcessError(1, commands[failed_index])
            with (
                self.subTest(failed_index=failed_index),
                patch("subprocess.run", side_effect=[None] * failed_index + [error]) as run,
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaises(subprocess.CalledProcessError),
            ):
                RUNNER["run_commands"](commands, cwd=SCRIPT_PATH.parents[1])
            actual = [call.args[0] for call in run.call_args_list]
            self.assertEqual(actual, commands[: failed_index + 1])
            self.assertNotIn(commands[-1], actual)

    def test_successful_checks_run_training_last_with_checked_subprocesses(self):
        commands = self.commands()
        with patch("subprocess.run") as run, contextlib.redirect_stdout(io.StringIO()):
            RUNNER["run_commands"](commands, cwd=SCRIPT_PATH.parents[1])
        actual = [call.args[0] for call in run.call_args_list]
        self.assertEqual(actual, commands)
        self.assertIn("runsleuth.camelyon", actual[-1])
        self.assertIn("--download", actual[-1])
        self.assertTrue(all(call.kwargs["check"] for call in run.call_args_list))

    def test_default_checks_do_not_request_training(self):
        commands = self.commands(train=False)
        self.assertFalse(any("runsleuth.camelyon" in command for command in commands))
        self.assertTrue(any("pytest" in command for command in commands))


if __name__ == "__main__":
    unittest.main()
