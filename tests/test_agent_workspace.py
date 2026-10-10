"""CPU tests for the workspace harness: script checks, monitored runs and subprocess runs."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from runsleuth import agent_tasks
from runsleuth.agent_tasks import FAULTS, HEALTHY, TASKS
from runsleuth.signature_matching import diagnose, extract_evidence, load_library

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

if torch is not None:
    from runsleuth import agent_workspace
    from runsleuth.agent_workspace import ScriptRefused, check_script

# A model small enough for CPU, with the parts the faults act on: BatchNorm and a
# 1000-way fc head that the healthy script replaces.
TINY_MODEL = """

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Conv2d(3, 4, kernel_size=3)
        self.norm = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4, 1000)

    def forward(self, inputs):
        return self.fc(torch.relu(self.norm(self.backbone(inputs))).mean(dim=(2, 3)))
"""


def tiny_source(task: str) -> str:
    """A task's script with the tiny model in place of the pretrained ResNet18."""
    source = agent_tasks.task_source(task)
    resnet = "model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)"
    marker = "\n\ndef use_for_training"
    assert source.count(resnet) == 1 and source.count(marker) == 1
    return source.replace(resnet, "model = Tiny()").replace(marker, TINY_MODEL + marker)


@unittest.skipIf(torch is None, "Install torch and torchvision for the workspace tests")
class ScriptCheckTests(unittest.TestCase):
    def test_every_task_script_may_run(self):
        for task in TASKS:
            with self.subTest(task=task):
                self.assertEqual(check_script(agent_tasks.task_source(task)), [])
                self.assertEqual(check_script(tiny_source(task)), [])

    def test_harmless_idioms_are_allowed(self):
        source = (
            "import math\nfrom torch import nn\n\n"
            "class Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n\n"
            "for _ in range(2):\n    print(math.pi)\n"
            'if __name__ == "__main__":\n    pass\n'
        )
        self.assertEqual(check_script(source), [])

    def test_file_network_and_process_access_is_refused(self):
        refused = {
            "import os": "import of 'os'",
            "import subprocess as sp": "import of 'subprocess'",
            "from pathlib import Path": "import of 'pathlib'",
            "from . import helper": "relative imports",
            "import torch.hub": "import of 'hub'",
            "from torch import hub": "import of 'hub'",
            "from torch import load as read": "import of 'load'",
            "from torch import *": "import of '*'",
            "open('weights.pt')": "'open'",
            "eval('1 + 1')": "'eval'",
            "getattr(torch, 'lo' + 'ad')": "'getattr'",
            "__builtins__": "'__builtins__'",
            "import torch\ntorch.load('weights.pt')": "attribute 'load'",
            "import torch\ntorch.save({}, 'weights.pt')": "attribute 'save'",
            "import torch\ntorch._C": "attribute '_C'",
            "().__class__": "attribute '__class__'",
            "def broken(:": "syntax error",
        }
        for source, reason in refused.items():
            with self.subTest(source=source):
                problems = check_script(source)
                self.assertEqual(len(problems), 1, problems)
                self.assertIn(reason, problems[0])
                self.assertRegex(problems[0], r"^line \d+: ")

    def test_an_overlong_script_is_refused(self):
        problems = check_script("x = 1\n" * agent_workspace.MAX_SOURCE_CHARACTERS)
        self.assertEqual(len(problems), 1)
        self.assertIn("longer than", problems[0])

    def test_scripts_run_with_restricted_builtins(self):
        allowed = agent_workspace.restricted_builtins()
        self.assertNotIn("open", allowed)
        self.assertNotIn("eval", allowed)
        self.assertIs(allowed["__import__"]("math"), __import__("math"))
        for name in ("os", "subprocess", "torch_missing.sub"):
            with self.subTest(name=name), self.assertRaises(ImportError):
                allowed["__import__"](name)
        with self.assertRaises(ImportError):
            allowed["__import__"]("torch", level=1)

    def test_a_script_must_define_the_pipeline_functions(self):
        with tempfile.TemporaryDirectory() as temporary:
            script = Path(temporary) / "train.py"
            script.write_text("def build(device, epochs):\n    return None\n", encoding="utf-8")
            with self.assertRaises(ScriptRefused) as refused:
                agent_workspace.load_script(script)
            self.assertEqual(len(refused.exception.problems), 4)
            self.assertIn("train_one_epoch()", str(refused.exception))
            script.write_text("import os\n", encoding="utf-8")
            with self.assertRaisesRegex(ScriptRefused, "import of 'os'"):
                agent_workspace.load_script(script)

    def test_credentials_are_removed_from_the_child_environment(self):
        cleaned = agent_workspace.clean_environment(
            {"PATH": "bin", "GEMINI_API_KEY": "a", "hf_token": "b", "DB_PASSWORD": "c"}
        )
        self.assertEqual(cleaned, {"PATH": "bin", "PYTHONIOENCODING": "utf-8"})


@unittest.skipIf(torch is None, "Install torch and torchvision for the workspace tests")
class WorkspaceRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.reports = {task: cls.run_task(task, f"run-{task}") for task in TASKS}

    @classmethod
    def workspace(cls, name: str, source: str) -> Path:
        directory = cls.root / "workspaces" / name
        directory.mkdir(parents=True)
        (directory / "train.py").write_text(source, encoding="utf-8", newline="\n")
        return directory

    @classmethod
    def run_task(cls, task: str, name: str) -> dict:
        path = agent_workspace.run_workspace(
            cls.workspace(name, tiny_source(task)),
            cls.root / "runs" / name,
            agent_workspace.synthetic_task_data,
            epochs=2,
            seed=11,
            device=torch.device("cpu"),
        )
        return json.loads(path.read_text(encoding="utf-8"))

    def test_healthy_run_records_metrics_evidence_and_state_hashes(self):
        report = self.reports[HEALTHY]
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["completed_epochs"], 2)
        self.assertEqual(report["head_output_units"], 2)
        for name in ("id_validation_loss", "id_validation_accuracy", "ood_validation_accuracy"):
            self.assertIn(name, report["final_metrics"])
        self.assertEqual(sorted(report["input_statistics"]), ["id_eval", "ood_eval", "train"])
        self.assertEqual(report["label_classes"]["train"], [0, 1])
        record = report["workspace"]
        self.assertEqual(record["data"]["training_hospitals"], [0, 3, 4])
        self.assertEqual(
            set(record["initialization_metrics"]), set(report["final_metrics"]) - {"epoch"}
        )
        # Six batches per epoch for two epochs: every step is hashed and all differ.
        self.assertEqual(len(record["step_state_sha256"]), 12)
        hashes = {record["initial_state_sha256"], *record["step_state_sha256"]}
        self.assertEqual(len(hashes), 13)
        self.assertEqual(record["final_state_sha256"], record["step_state_sha256"][-1])
        self.assertNotIn("traceback", record)

    def test_the_same_script_and_seed_replay_bit_for_bit(self):
        first = self.reports[HEALTHY]["workspace"]
        again = self.run_task(HEALTHY, "run-healthy-again")["workspace"]
        for field in ("initial_state_sha256", "step_state_sha256", "final_state_sha256"):
            self.assertEqual(again[field], first[field])
        self.assertEqual(again["script_sha256"], first["script_sha256"])

    def test_every_fault_is_diagnosed_from_its_run_against_the_healthy_run(self):
        library = load_library()
        healthy = self.reports[HEALTHY]
        self.assertEqual(
            diagnose(extract_evidence(healthy, healthy), library)["diagnosis"], "no_known_fault"
        )
        for task, fault in FAULTS.items():
            with self.subTest(task=task):
                report = self.reports[task]
                self.assertEqual(report["status"], "completed")
                found = diagnose(extract_evidence(report, healthy), library)["diagnosis"]
                # The leakage signature arrives with the Camelyon17 data source.
                missing = fault.signature == "train_evaluation_group_overlap"
                self.assertEqual(found, "no_known_fault" if missing else fault.signature)

    def test_training_faults_leave_the_healthy_trajectory_and_evaluation_faults_do_not(self):
        healthy = self.reports[HEALTHY]["workspace"]
        for task in FAULTS:
            with self.subTest(task=task):
                record = self.reports[task]["workspace"]
                same = record["step_state_sha256"] == healthy["step_state_sha256"]
                # A normalization used only for evaluation does not change training.
                self.assertEqual(same, task == "train_eval_normalization_mismatch")
        mismatch = self.reports["train_eval_normalization_mismatch"]
        self.assertNotEqual(mismatch["final_metrics"], self.reports[HEALTHY]["final_metrics"])
        self.assertEqual(
            self.reports["validation_hospital_in_training"]["workspace"]["data"][
                "training_hospitals"
            ],
            [0, 1, 3, 4],
        )

    def test_budget_and_hash_limits_bound_a_run(self):
        path = agent_workspace.run_workspace(
            self.workspace("bounded", tiny_source(HEALTHY)),
            self.root / "runs" / "bounded",
            agent_workspace.synthetic_task_data,
            epochs=2,
            seed=11,
            device=torch.device("cpu"),
            max_train_batches=2,
            hashed_steps=3,
        )
        report = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            [row["optimizer_steps"] for row in report["parameter_group_epochs"]], [2, 2]
        )
        hashes = report["workspace"]["step_state_sha256"]
        full = self.reports[HEALTHY]["workspace"]["step_state_sha256"]
        self.assertEqual(len(hashes), 3)
        # The first epoch starts as the full run does; the second starts four batches early.
        self.assertEqual(hashes[:2], full[:2])
        self.assertNotEqual(hashes[2], full[2])

    def test_a_failure_during_training_is_recorded_with_its_traceback(self):
        source = tiny_source(HEALTHY).replace(
            "        loss.backward()\n", "        raise RuntimeError('boom')\n"
        )
        path = agent_workspace.run_workspace(
            self.workspace("failing", source),
            self.root / "runs" / "failing",
            agent_workspace.synthetic_task_data,
            epochs=2,
            seed=11,
            device=torch.device("cpu"),
        )
        report = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error"], {"type": "RuntimeError", "message": "boom"})
        self.assertIn("raise RuntimeError('boom')", report["workspace"]["traceback"])
        self.assertIsNone(report["workspace"]["final_state_sha256"])

    def test_a_refused_script_and_a_broken_build_leave_no_report(self):
        cases = {
            "refused": ("import os\n" + tiny_source(HEALTHY), ScriptRefused),
            "unbuilt": (
                tiny_source(HEALTHY).replace("model = Tiny()", "model = Missing()"),
                NameError,
            ),
        }
        for name, (source, error) in cases.items():
            with self.subTest(name=name), self.assertRaises(error):
                agent_workspace.run_workspace(
                    self.workspace(name, source),
                    self.root / "runs" / name,
                    agent_workspace.synthetic_task_data,
                    epochs=1,
                    seed=11,
                    device=torch.device("cpu"),
                )
            self.assertFalse((self.root / "runs" / name).exists())


@unittest.skipIf(torch is None, "Install torch and torchvision for the workspace tests")
class SubprocessRunTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def workspace(self, source: str) -> Path:
        directory = self.root / f"workspace-{len(list(self.root.iterdir()))}"
        directory.mkdir()
        (directory / "train.py").write_text(source, encoding="utf-8", newline="\n")
        return directory

    def run_child(self, source: str, **options) -> dict:
        return agent_workspace.run_in_subprocess(
            self.workspace(source),
            self.root / "run",
            data="synthetic",
            epochs=1,
            seed=11,
            device="cpu",
            **options,
        )

    def test_child_process_trains_and_reports(self):
        outcome = self.run_child(tiny_source(HEALTHY) + "\nprint('script was imported')\n")
        self.assertEqual(outcome["status"], "completed", outcome["output"])
        self.assertEqual(outcome["problems"], [])
        self.assertIn("script was imported", outcome["output"])
        report = json.loads(Path(outcome["report"]).read_text(encoding="utf-8"))
        self.assertEqual(report["completed_epochs"], 1)
        self.assertEqual(len(report["workspace"]["step_state_sha256"]), 6)

    def test_a_refused_script_never_starts_a_process(self):
        with patch("subprocess.run") as run:
            outcome = self.run_child("import socket\n")
        run.assert_not_called()
        self.assertEqual(outcome["status"], "refused")
        self.assertEqual(outcome["problems"], ["line 1: import of 'socket' is not allowed"])
        self.assertIsNone(outcome["report"])

    def test_timeout_and_crash_are_reported(self):
        expired = subprocess.TimeoutExpired(cmd="python", timeout=5)
        with patch("subprocess.run", side_effect=expired) as run:
            outcome = self.run_child(tiny_source(HEALTHY), timeout=5)
        self.assertEqual(outcome["status"], "timeout")
        self.assertIn("5 seconds", outcome["output"])
        environment = run.call_args.kwargs["env"]
        self.assertFalse([name for name in environment if "API_KEY" in name.upper()])
        crashed = subprocess.CompletedProcess([], 1, stdout="", stderr="Fatal Python error")
        with patch("subprocess.run", return_value=crashed):
            outcome = self.run_child(tiny_source(HEALTHY))
        self.assertEqual(outcome["status"], "crashed")
        self.assertIn("Fatal Python error", outcome["output"])

    def test_command_line_reports_a_failed_build_as_json(self):
        source = tiny_source(HEALTHY).replace("model = Tiny()", "model = Missing()")
        arguments = [
            "run",
            "--workspace",
            str(self.workspace(source)),
            "--output",
            str(self.root / "run"),
            "--data",
            "synthetic",
            "--epochs",
            "1",
            "--device",
            "cpu",
        ]
        with patch("builtins.print") as printed, patch("traceback.print_exc"):
            code = agent_workspace.main(arguments)
        self.assertEqual(code, 1)
        outcome = json.loads(printed.call_args.args[0])
        self.assertEqual(outcome["status"], "failed")
        self.assertEqual(outcome["error"]["type"], "NameError")


if __name__ == "__main__":
    unittest.main()
