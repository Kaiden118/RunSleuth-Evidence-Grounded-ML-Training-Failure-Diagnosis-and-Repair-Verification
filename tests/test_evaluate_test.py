"""Offline tests; requires the project's normal CPU torch/torchvision dependencies."""

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import ExitStack, contextmanager, redirect_stdout
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import TensorDataset

from runsleuth import evaluate_test as subject
from runsleuth.config import TrainingConfig


class TemporaryRuns(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cpu = torch.device("cpu")

    def make_run(self, name="run", seed=42, checkpoint=b"selected checkpoint"):
        directory = self.root / name
        directory.mkdir()
        config = TrainingConfig(seed=seed, data_dir=str(self.root / "unused-data"))
        (directory / "config.json").write_text(json.dumps(asdict(config)), encoding="utf-8")
        (directory / "model_state_dict.pt").write_bytes(checkpoint)
        return directory

    @contextmanager
    def fake_inference(self, loss=0.75, accuracy=0.4):
        """Mock every data/checkpoint boundary; never fetch FashionMNIST."""
        with ExitStack() as stack:
            mocks = SimpleNamespace()
            for name in (
                "FashionMNISTCNN",
                "DataLoader",
                "seed_everything",
                "evaluate",
                "transforms",
            ):
                setattr(mocks, name, stack.enter_context(patch.object(subject, name)))
            mocks.dataset = stack.enter_context(patch.object(subject.datasets, "FashionMNIST"))
            mocks.load = stack.enter_context(patch.object(subject.torch, "load"))
            mocks.dataset.return_value = list(range(5))
            mocks.evaluate.return_value = SimpleNamespace(loss=loss, accuracy=accuracy)
            mocks.model = mocks.FashionMNISTCNN.return_value
            yield mocks


class TestOfficialTestEvaluation(TemporaryRuns):
    def test_preparation_snapshots_exact_bytes_and_hashes(self):
        directory = self.make_run(seed=71)
        config_bytes = (directory / "config.json").read_bytes()
        checkpoint_bytes = (directory / "model_state_dict.pt").read_bytes()
        prepared = subject.prepare_runs([directory])[0]
        self.assertEqual(prepared.config.seed, 71)
        self.assertEqual(prepared.config_sha256, hashlib.sha256(config_bytes).hexdigest())
        self.assertEqual(prepared.checkpoint_sha256, hashlib.sha256(checkpoint_bytes).hexdigest())
        self.assertEqual(prepared.checkpoint_bytes, checkpoint_bytes)
        (directory / "config.json").write_text("{}", encoding="utf-8")
        (directory / "model_state_dict.pt").write_bytes(b"replacement")
        self.assertEqual(prepared.config.seed, 71)
        self.assertEqual(prepared.checkpoint_bytes, checkpoint_bytes)
        self.assertEqual(prepared.metadata()["training_config"]["seed"], 71)

    def test_invalid_selections_fail_before_inference(self):
        good = self.make_run("good")
        missing = self.make_run("missing")
        (missing / "model_state_dict.pt").unlink()
        empty = self.make_run("empty", checkpoint=b"")
        cases = [
            ([], ValueError),
            ([good, good / "."], ValueError),
            ([good, missing], FileNotFoundError),
            ([good, empty], ValueError),
        ]
        with (
            patch.object(subject, "resolve_device", return_value=self.cpu),
            patch.object(subject, "evaluate_checkpoint") as evaluate,
        ):
            for selected, error in cases:
                with self.subTest(selected=selected), self.assertRaises(error):
                    subject.run_test_evaluation(selected, output_dir=self.root / "output")
            evaluate.assert_not_called()
        self.assertFalse((self.root / "output").exists())

    def test_checkpoint_loading_and_test_loader_options(self):
        prepared = subject.prepare_runs([self.make_run(seed=19)])[0]
        with self.fake_inference() as mocks:

            def observe_inference(*args):
                self.assertFalse(torch.is_grad_enabled())
                return SimpleNamespace(loss=0.75, accuracy=0.4)

            mocks.evaluate.side_effect = observe_inference
            result = subject.evaluate_checkpoint(prepared, device=self.cpu, batch_size=2)
            args, kwargs = mocks.load.call_args
            self.assertEqual(args[0].getvalue(), prepared.checkpoint_bytes)
            self.assertEqual(kwargs, {"map_location": "cpu", "weights_only": True})
            mocks.model.load_state_dict.assert_called_once_with(
                mocks.load.return_value, strict=True
            )
            mocks.model.to.assert_called_once_with(self.cpu)
            mocks.model.eval.assert_called_once_with()
            mocks.seed_everything.assert_called_once_with(19)
            mocks.dataset.assert_called_once_with(
                root=prepared.config.data_dir,
                train=False,
                download=True,
                transform=mocks.transforms.Compose.return_value,
            )
            mocks.transforms.ToTensor.assert_called_once_with()
            mocks.transforms.Normalize.assert_called_once_with(
                subject.FASHION_MNIST_MEAN,
                subject.FASHION_MNIST_STD,
            )
            mocks.DataLoader.assert_called_once_with(
                mocks.dataset.return_value,
                batch_size=2,
                shuffle=False,
                drop_last=False,
                num_workers=0,
                pin_memory=False,
            )
            call = mocks.evaluate.call_args.args
            self.assertIs(call[0], mocks.model)
            self.assertIs(call[1], mocks.DataLoader.return_value)
            self.assertIsInstance(call[2], nn.CrossEntropyLoss)
            self.assertEqual(call[3], self.cpu)
        self.assertEqual(result, {"test_examples": 5, "test_loss": 0.75, "test_accuracy": 0.4})

    def test_incompatible_state_fails_before_dataset_access(self):
        prepared = subject.prepare_runs([self.make_run()])[0]
        with self.fake_inference() as mocks:
            mocks.model.load_state_dict.side_effect = RuntimeError("incompatible weights")
            with self.assertRaisesRegex(RuntimeError, "incompatible"):
                subject.evaluate_checkpoint(prepared, device=self.cpu, batch_size=2)
            mocks.dataset.assert_not_called()
            mocks.evaluate.assert_not_called()

    def test_invalid_metrics_are_rejected(self):
        prepared = subject.prepare_runs([self.make_run()])[0]
        invalid = [
            (float("nan"), 0.5),
            (float("inf"), 0.5),
            (-0.1, 0.5),
            (0.5, float("nan")),
            (0.5, float("inf")),
            (0.5, -0.1),
            (0.5, 1.1),
        ]
        for loss, accuracy in invalid:
            with self.subTest(loss=loss, accuracy=accuracy), self.fake_inference(loss, accuracy):
                with self.assertRaises(ValueError):
                    subject.evaluate_checkpoint(prepared, device=self.cpu, batch_size=2)

    def test_invalid_batch_sizes_fail_before_loading(self):
        prepared = subject.prepare_runs([self.make_run()])[0]
        with patch.object(subject.torch, "load") as load:
            for batch_size in (0, -1, True, 1.5):
                with self.subTest(batch_size=batch_size), self.assertRaises(ValueError):
                    subject.evaluate_checkpoint(prepared, device=self.cpu, batch_size=batch_size)
            load.assert_not_called()

    def test_empty_test_dataset_fails_before_inference(self):
        prepared = subject.prepare_runs([self.make_run()])[0]
        with self.fake_inference() as mocks:
            mocks.dataset.return_value = []
            with self.assertRaisesRegex(ValueError, "Test dataset is empty"):
                subject.evaluate_checkpoint(prepared, device=self.cpu, batch_size=2)
            mocks.DataLoader.assert_not_called()
            mocks.evaluate.assert_not_called()

    def test_plan_and_all_snapshots_precede_inference_and_errors_have_no_stale_metrics(self):
        first, second = self.make_run("first"), self.make_run("second", seed=73)
        output = self.root / "output"
        second_config = (second / "config.json").read_bytes()
        second_checkpoint = (second / "model_state_dict.pt").read_bytes()
        calls = []

        def evaluate(prepared, **kwargs):
            calls.append(prepared.run_directory)
            plans = list(output.glob("*/test_plan.json"))
            self.assertEqual(len(plans), 1)
            plan = json.loads(plans[0].read_text(encoding="utf-8"))
            self.assertEqual(len(plan["runs"]), 2)
            self.assertEqual(plan["split"], "test")
            self.assertEqual(
                plan["runs"][1]["config_sha256"], hashlib.sha256(second_config).hexdigest()
            )
            self.assertEqual(
                plan["runs"][1]["checkpoint_sha256"], hashlib.sha256(second_checkpoint).hexdigest()
            )
            if prepared.run_directory == first:
                (second / "config.json").write_text("{}", encoding="utf-8")
                (second / "model_state_dict.pt").write_bytes(b"changed after preparation")
                return {"test_examples": 5, "test_loss": 0.4, "test_accuracy": 0.8}
            self.assertEqual(prepared.config.seed, 73)
            self.assertEqual(prepared.checkpoint_bytes, second_checkpoint)
            raise RuntimeError("deliberate second-run failure")

        with (
            patch.object(subject, "resolve_device", return_value=self.cpu),
            patch.object(subject, "evaluate_checkpoint", side_effect=evaluate),
            redirect_stdout(io.StringIO()),
        ):
            path = subject.run_test_evaluation([first, second], batch_size=2, output_dir=output)
        report = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(calls, [first, second])
        self.assertEqual(
            report["summary"], {"selected_runs": 2, "completed_runs": 1, "error_runs": 1}
        )
        completed, failed = report["results"]
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["test_accuracy"], 0.8)
        self.assertEqual(failed["status"], "error")
        self.assertEqual(
            failed["error"], {"type": "RuntimeError", "message": "deliberate second-run failure"}
        )
        for key in ("test_examples", "test_loss", "test_accuracy"):
            self.assertNotIn(key, failed)
        for index, row in enumerate(report["results"], 1):
            stored = json.loads((path.parent / f"run-{index:03d}.json").read_text(encoding="utf-8"))
            self.assertEqual(stored, row)

    def test_existing_report_is_never_overwritten(self):
        directory = self.make_run()
        output = self.root / "output"
        existing = output / "fixed" / "test_report.json"
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"previous report")
        with (
            patch.object(subject, "datetime") as clock,
            patch.object(subject, "resolve_device", return_value=self.cpu),
            patch.object(subject, "evaluate_checkpoint") as evaluate,
        ):
            clock.now.return_value.strftime.return_value = "fixed"
            with self.assertRaises(FileExistsError):
                subject.run_test_evaluation([directory], output_dir=output)
            evaluate.assert_not_called()
        with self.assertRaises(FileExistsError):
            subject._save_json(existing, {"replacement": True})
        self.assertEqual(existing.read_bytes(), b"previous report")

    def test_parser_accepts_multiple_runs_and_cli_exits_one_for_any_error(self):
        args = subject.build_parser().parse_args(
            ["--run", "one", "--run", "two", "--batch-size", "2"]
        )
        self.assertEqual(args.run, [Path("one"), Path("two")])
        self.assertEqual(args.batch_size, 2)
        report = self.root / "report.json"
        report.write_text(json.dumps({"summary": {"error_runs": 1}}), encoding="utf-8")
        with (
            patch("sys.argv", ["evaluate_test", "--run", "one", "--run", "two"]),
            patch.object(subject, "run_test_evaluation", return_value=report) as run,
            redirect_stdout(io.StringIO()),
            self.assertRaises(SystemExit) as exit_result,
        ):
            subject.main()
        self.assertEqual(exit_result.exception.code, 1)
        self.assertEqual(run.call_args.args[0], [Path("one"), Path("two")])


class TestRealCPUCheckpoint(TemporaryRuns):
    def test_real_cpu_checkpoint_inference_counts_partial_batch_and_preserves_files(self):
        class TinyClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 3, bias=False)
                self.observations = []

            def forward(self, inputs):
                self.observations.append((len(inputs), self.training, torch.is_grad_enabled()))
                return self.linear(inputs)

        weights = torch.tensor([[1.0, -1.0], [-1.0, 1.0], [0.25, 0.25]])
        checkpoint = io.BytesIO()
        torch.save({"linear.weight": weights}, checkpoint)
        directory = self.make_run(checkpoint=checkpoint.getvalue())
        files = [directory / "config.json", directory / "model_state_dict.pt"]
        original_bytes = [path.read_bytes() for path in files]
        inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        labels = torch.tensor([0, 1, 0, 1, 2])
        dataset = TensorDataset(inputs, labels)
        expected_loss = nn.functional.cross_entropy(inputs @ weights.T, labels).item()
        expected_accuracy = ((inputs @ weights.T).argmax(1) == labels).float().mean().item()
        model = TinyClassifier()
        with (
            patch.object(subject, "FashionMNISTCNN", return_value=model),
            patch.object(subject.datasets, "FashionMNIST", return_value=dataset) as factory,
        ):
            result = subject.evaluate_checkpoint(
                subject.prepare_runs([directory])[0], device=self.cpu, batch_size=2
            )
        self.assertFalse(factory.call_args.kwargs["train"])
        self.assertEqual(
            model.observations, [(2, False, False), (2, False, False), (1, False, False)]
        )
        self.assertEqual(result["test_examples"], 5)
        self.assertAlmostEqual(result["test_loss"], expected_loss, places=6)
        self.assertAlmostEqual(result["test_accuracy"], expected_accuracy, places=6)
        self.assertEqual([path.read_bytes() for path in files], original_bytes)


if __name__ == "__main__":
    unittest.main()
