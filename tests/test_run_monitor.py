"""RunMonitor on hand-written training loops: each library fault is diagnosed end to end."""

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

try:
    import torch
    from torch import nn
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None
    nn = None

if torch is not None:
    from runsleuth import signature_matching as matching
    from runsleuth.run_monitor import RunMonitor

    class Net(nn.Module):
        def __init__(self, head="fc"):
            super().__init__()
            self.head = head
            self.backbone = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
            setattr(self, head, nn.Linear(16, 2))

        def forward(self, inputs):
            return getattr(self, self.head)(self.backbone(inputs))


@unittest.skipIf(torch is None, "PyTorch is not installed; RunMonitor tests require torch")
class RunMonitorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        generator = torch.Generator().manual_seed(3)
        inputs = torch.randn(32, 8, generator=generator)
        targets = (inputs[:, 0] > 0).long()
        self.batches = list(zip(inputs.split(8), targets.split(8), strict=True))
        self.library = matching.load_library()

    def train(self, scenario, *, head="fc", nan_inputs=False):
        """A user's own loop: zero_grad, forward, backward and (normally) step."""
        torch.manual_seed(5)
        model = Net(head)
        lr = 1e-2 if scenario == "high_learning_rate" else 1e-4
        if scenario == "frozen_head":
            getattr(model, head).requires_grad_(False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        if scenario == "stale_head":
            setattr(model, head, deepcopy(getattr(model, head)))
        monitor = RunMonitor(model, optimizer, self.root / f"{scenario}-{head}", head=head)
        for _ in range(2):
            with monitor.epoch():
                for inputs, targets in self.batches:
                    if nan_inputs:
                        inputs = torch.full_like(inputs, float("nan"))
                    optimizer.zero_grad(set_to_none=True)
                    loss = nn.functional.cross_entropy(model(inputs), targets)
                    loss.backward()
                    if scenario != "missing_optimizer_step":
                        optimizer.step()
            monitor.log(train_loss=loss.item())
        return json.loads(monitor.close().read_text(encoding="utf-8"))

    def diagnose(self, report, reference=None):
        evidence = matching.extract_evidence(report, reference)
        return matching.diagnose(evidence, self.library)

    def test_every_library_fault_is_diagnosed_from_a_hand_written_loop(self):
        clean = self.train("clean")
        self.assertEqual(clean["status"], "completed")
        self.assertEqual(clean["optimizer"], "AdamW")
        self.assertEqual(
            [
                (row["optimizer_steps"], row["training_forwards"])
                for row in clean["parameter_group_epochs"]
            ],
            [(4, 4), (4, 4)],
        )
        expected = {
            "clean": ("no_known_fault", "no_known_fault"),
            "stale_head": ("stale_optimizer_binding", "stale_optimizer_binding"),
            "frozen_head": ("pending_reference", "frozen_head"),
            "high_learning_rate": ("high_learning_rate", "high_learning_rate"),
            "missing_optimizer_step": ("missing_optimizer_step", "missing_optimizer_step"),
        }
        for scenario, (alone, with_reference) in expected.items():
            with self.subTest(scenario=scenario):
                report = clean if scenario == "clean" else self.train(scenario)
                self.assertEqual(self.diagnose(report)["diagnosis"], alone)
                self.assertEqual(self.diagnose(report, clean)["diagnosis"], with_reference)

    def test_missing_step_epochs_record_gradients_without_parameter_change(self):
        report = self.train("missing_optimizer_step")
        for row in report["parameter_group_epochs"]:
            self.assertEqual((row["optimizer_steps"], row["training_forwards"]), (0, 4))
            self.assertEqual(row["epoch_parameter_change_l2"], 0.0)
            self.assertGreater(row["gradient_l2_at_epoch_end"], 0)
            self.assertEqual(row["metrics"].keys(), {"train_loss"})

    def test_custom_head_name_is_grouped_and_diagnosed(self):
        report = self.train("stale_head", head="classifier")
        self.assertEqual(report["head_prefix"], "classifier.")
        self.assertEqual(
            set(report["final_optimizer_audit"]["missing_trainable_names"]),
            {"classifier.weight", "classifier.bias"},
        )
        self.assertEqual(self.diagnose(report)["diagnosis"], "stale_optimizer_binding")

    def test_non_finite_training_is_recorded_as_diverged_and_reraised(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.train("clean", nan_inputs=True)
        report = json.loads((self.root / "clean-fc" / "run_report.json").read_text("utf-8"))
        self.assertEqual(report["status"], "diverged")
        self.assertEqual(report["divergence"]["epoch"], 1)

    def test_misuse_is_rejected(self):
        model = Net()
        optimizer = torch.optim.AdamW(model.parameters())
        monitor = RunMonitor(model, optimizer, self.root / "misuse")
        with self.assertRaisesRegex(RuntimeError, "after a completed"):
            monitor.log(train_loss=1.0)
        with self.assertRaises(FileExistsError):
            RunMonitor(model, optimizer, self.root / "misuse")
        monitor.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            with monitor.epoch():
                pass


if __name__ == "__main__":
    unittest.main()
