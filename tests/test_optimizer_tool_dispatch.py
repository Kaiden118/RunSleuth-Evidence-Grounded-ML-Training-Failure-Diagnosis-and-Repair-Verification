"""Opt-in registration, dispatch order, and filesystem boundaries for saved audits."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from runsleuth.diagnostic_tools import DiagnosticTools


class OptimizerToolDispatchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.parent = Path(temporary.name).resolve()
        self.root = self.parent / "project"
        self.root.mkdir()
        self.run = self.root / "artifacts" / "candidate"
        self.run.mkdir(parents=True)

    def test_new_tool_requires_opt_in_without_mutating_default_registry(self):
        enabled = DiagnosticTools(self.root, enable_optimizer_audit=True)
        default = DiagnosticTools(self.root)
        self.assertEqual(
            [item["name"] for item in default.definitions()],
            [
                "inspect_training_source",
                "load_run_config",
                "compare_runs",
            ],
        )
        self.assertEqual(len(enabled.definitions()), 4)
        result = default.execute("inspect_optimizer_run", {"run_directory": "artifacts/candidate"})
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, "unknown_tool")

    def test_optimizer_argument_subclass_reaches_its_reader_before_legacy_loader(self):
        tools = DiagnosticTools(self.root, enable_optimizer_audit=True)
        with (
            patch(
                "runsleuth.diagnostic_tools.inspect_optimizer_run", return_value={"epochs": []}
            ) as reader,
            patch("runsleuth.diagnostic_tools.TrainingConfig.load") as legacy_loader,
        ):
            result = tools.execute(
                "inspect_optimizer_run", {"run_directory": "artifacts/candidate"}
            )
        self.assertTrue(result.ok, result.error_message)
        self.assertEqual(result.data["run_directory"], "artifacts/candidate")
        self.assertEqual(reader.call_args.args, (self.run,))
        self.assertIs(reader.call_args.kwargs["require_file"].__self__, tools)
        legacy_loader.assert_not_called()

    def test_tool_arguments_reject_wrong_types_and_extra_fields(self):
        tools = DiagnosticTools(self.root, enable_optimizer_audit=True)
        for arguments in (
            {},
            {"run_directory": 42},
            {"run_directory": False},
            {"run_directory": "artifacts/candidate", "extra": "ignored?"},
        ):
            with (
                self.subTest(arguments=arguments),
                patch("runsleuth.diagnostic_tools.inspect_optimizer_run") as reader,
            ):
                result = tools.execute("inspect_optimizer_run", arguments)
                self.assertEqual(result.error_code, "invalid_arguments")
                reader.assert_not_called()

    def test_absolute_and_parent_paths_never_reach_the_reader(self):
        tools = DiagnosticTools(self.root, enable_optimizer_audit=True)
        for path in ("../outside", "/outside", "C:/outside", r"C:\outside"):
            with (
                self.subTest(path=path),
                patch("runsleuth.diagnostic_tools.inspect_optimizer_run") as reader,
            ):
                result = tools.execute("inspect_optimizer_run", {"run_directory": path})
                self.assertFalse(result.ok)
                self.assertIn(result.error_code, {"invalid_path", "invalid_arguments"})
                reader.assert_not_called()

    def test_reader_callback_rejects_a_file_symlink_outside_project(self):
        outside = self.parent / "outside.json"
        outside.write_text("{}", encoding="utf-8")
        link = self.run / "run_report.json"
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"Symlink creation unavailable: {error}")
        tools = DiagnosticTools(self.root, enable_optimizer_audit=True)

        def use_callback(directory, *, require_file):
            require_file(directory / "run_report.json")
            raise AssertionError("An escaped file must be rejected before reading")

        with patch("runsleuth.diagnostic_tools.inspect_optimizer_run", side_effect=use_callback):
            result = tools.execute(
                "inspect_optimizer_run", {"run_directory": "artifacts/candidate"}
            )
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, "invalid_path")

    def test_invalid_saved_artifacts_return_error_envelope(self):
        tools = DiagnosticTools(self.root, enable_optimizer_audit=True)
        with patch(
            "runsleuth.diagnostic_tools.inspect_optimizer_run", side_effect=ValueError("bad digest")
        ):
            result = tools.execute(
                "inspect_optimizer_run", {"run_directory": "artifacts/candidate"}
            )
        self.assertFalse(result.ok)
        self.assertIsNone(result.data)
        self.assertEqual(result.error_code, "invalid_artifact")


if __name__ == "__main__":
    unittest.main()
