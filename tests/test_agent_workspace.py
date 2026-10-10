"""CPU tests for the workspace harness: script checks, monitored runs and subprocess runs."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from runsleuth import agent_tasks
from runsleuth.agent_tasks import FAULTS, HEALTHY, LOOKALIKES, TASKS
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


TASK_SOURCE = agent_tasks.task_source  # Bound here, before any test patches it.


def tiny_source(task: str) -> str:
    """A task's script with the tiny model in place of the pretrained ResNet18."""
    source = TASK_SOURCE(task)
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
        self.assertEqual(record["steps_per_epoch"], 6)
        steps = record["step_state_sha256"]
        self.assertEqual(record["epoch_state_sha256"], [steps[5], steps[11]])
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
                self.assertEqual(found, fault.signature)
        for task in LOOKALIKES:
            with self.subTest(task=task):
                found = diagnose(extract_evidence(self.reports[task], healthy), library)
                self.assertEqual(found["diagnosis"], "no_known_fault")
        # Leakage needs no reference: the run itself says which hospital it trained on.
        leaky = self.reports["validation_hospital_in_training"]
        self.assertEqual(
            leaky["evaluation_holdouts"]["ood"],
            {
                "held_out_by": "hospital",
                "evaluation_groups": [1],
                "training_samples_in_evaluation_groups": 8,
            },
        )
        self.assertEqual(
            diagnose(extract_evidence(leaky), library)["diagnosis"],
            "train_evaluation_group_overlap",
        )
        self.assertEqual(
            healthy["evaluation_holdouts"]["ood"]["training_samples_in_evaluation_groups"], 0
        )

    def test_training_faults_leave_the_healthy_trajectory_and_evaluation_faults_do_not(self):
        healthy = self.reports[HEALTHY]["workspace"]
        for task in FAULTS:
            with self.subTest(task=task):
                record = self.reports[task]["workspace"]
                same = record["step_state_sha256"] == healthy["step_state_sha256"]
                # A normalization used only for evaluation does not change training.
                self.assertEqual(same, task == "train_eval_normalization_mismatch")
        # The late fault replays the healthy first epoch and leaves it in the second.
        late = self.reports["schedule_ends_an_epoch_early"]["workspace"]
        self.assertEqual(late["step_state_sha256"][:6], healthy["step_state_sha256"][:6])
        self.assertEqual(late["epoch_state_sha256"][0], healthy["epoch_state_sha256"][0])
        self.assertNotEqual(late["epoch_state_sha256"][1], healthy["epoch_state_sha256"][1])
        # One look-alike trains exactly as the healthy script does; the other is correct
        # on a different trajectory.
        same = {
            task: self.reports[task]["workspace"]["step_state_sha256"]
            == healthy["step_state_sha256"]
            for task in LOOKALIKES
        }
        self.assertEqual(
            same,
            {"head_replaced_after_moving": True, "cosine_stepped_per_batch_over_the_run": False},
        )
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


@unittest.skipIf(torch is None, "Install torch and torchvision for the workspace tests")
class CamelyonTaskDataTests(unittest.TestCase):
    """The Camelyon17 source on a synthetic mirror: 36 training patches from hospitals
    0, 3 and 4, 12 in-distribution validation patches, 12 from validation hospital 1."""

    def setUp(self):
        from test_camelyon_data import SyntheticSubset, SyntheticWILDS

        from runsleuth.camelyon_config import CamelyonConfig

        class Mirror(SyntheticWILDS):
            def get_patches(self, indices, transform=None):
                return SyntheticSubset(self, [int(index) for index in indices], transform)

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.mirror = Mirror(temporary.name)
        self.config = CamelyonConfig(
            device="cpu",
            batch_size=4,
            max_train_samples=18,
            max_id_val_samples=6,
            max_ood_val_samples=6,
        )
        for module in ("agent_workspace", "camelyon_data"):
            patcher = patch(f"runsleuth.{module}.load_camelyon_mirror", return_value=self.mirror)
            patcher.start()
            self.addCleanup(patcher.stop)

    def build(self, use_for_training, seed=3):
        builder = agent_workspace.camelyon_task_data(self.config)
        return builder(use_for_training, torch.device("cpu"), seed)

    @staticmethod
    def indices(loader) -> list[int]:
        return list(loader.dataset.subset.indices)

    def test_a_healthy_script_gets_the_baseline_subsets_as_raw_patches(self):
        from runsleuth.camelyon_data import build_camelyon_data

        data = self.build(lambda hospital: hospital in (0, 3, 4))
        baseline = build_camelyon_data(self.config, device=torch.device("cpu")).manifest["splits"]
        self.assertEqual(self.indices(data.train_loader), baseline["train"]["selected_indices"])
        self.assertEqual(
            self.indices(data.evaluation_loaders["id"]), baseline["id_val"]["selected_indices"]
        )
        self.assertEqual(
            self.indices(data.evaluation_loaders["ood"]), baseline["val"]["selected_indices"]
        )
        images, labels = next(iter(data.train_loader))
        self.assertEqual(tuple(images.shape), (4, 3, 96, 96))
        self.assertEqual(labels.dtype, torch.int64)
        # RGB (255, 0, 128) in [0, 1]: the script, not the harness, normalizes.
        self.assertTrue(torch.allclose(images[0, :, 0, 0], torch.tensor([1.0, 0.0, 128 / 255])))
        self.assertEqual(data.record["training_hospitals"], [0, 3, 4])
        self.assertEqual(data.record["training_samples"], 18)
        self.assertEqual(data.record["evaluation_samples"], {"id": 6, "ood": 6})
        self.assertEqual(
            data.holdouts["ood"],
            {
                "held_out_by": "hospital",
                "evaluation_groups": [1],
                "training_samples_in_evaluation_groups": 0,
                "slides_shared_with_training": 0,
            },
        )
        self.assertIsNone(data.holdouts["id"]["held_out_by"])
        json.dumps({"holdouts": data.holdouts, "record": data.record}, allow_nan=False)

    def test_admitting_the_validation_hospital_leaks_its_slides_but_no_evaluation_patch(self):
        data = self.build(lambda hospital: hospital != 2)
        training = self.indices(data.train_loader)
        evaluation = self.indices(data.evaluation_loaders["ood"])
        self.assertFalse(set(training) & set(evaluation))
        hospitals = self.mirror.metadata_array[:, 0].tolist()
        leaked = [index for index in training if hospitals[index] == 1]
        # 6 spare validation-hospital patches among 42 candidates, 18 drawn by stratum.
        self.assertEqual(len(leaked), 2)
        self.assertEqual(data.record["training_hospitals"], [0, 1, 3, 4])
        self.assertEqual(data.holdouts["ood"]["training_samples_in_evaluation_groups"], 2)
        self.assertGreater(data.holdouts["ood"]["slides_shared_with_training"], 0)
        self.assertEqual(self.indices(data.evaluation_loaders["ood"]), evaluation)

    def test_the_seed_orders_training_and_never_changes_the_subset(self):
        def order(seed):
            loader = self.build(lambda hospital: hospital in (0, 3, 4), seed).train_loader
            return list(loader.sampler)

        self.assertEqual(order(3), order(3))
        self.assertNotEqual(order(3), order(4))
        self.assertEqual(sorted(order(3)), sorted(order(4)))

    def test_a_script_that_admits_no_hospital_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "admits no hospital"):
            self.build(lambda hospital: False)

    def test_data_sources_are_named_on_the_command_line(self):
        self.assertIs(
            agent_workspace.data_builder("synthetic"), agent_workspace.synthetic_task_data
        )
        with self.assertRaisesRegex(ValueError, "--config"):
            agent_workspace.data_builder("camelyon")
        with self.assertRaisesRegex(ValueError, "Unknown data source"):
            agent_workspace.data_builder("imagenet")
        self.config.save(self.root / "config.json")
        builder = agent_workspace.data_builder("camelyon", self.root / "config.json")
        data = builder(lambda hospital: hospital in (0, 3, 4), torch.device("cpu"), 3)
        self.assertEqual(data.record["source"], "camelyon17")


@unittest.skipIf(torch is None, "Install torch and torchvision for the workspace tests")
class TaskCheckTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        patcher = patch("runsleuth.agent_tasks.task_source", tiny_source)
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def run_in_process(workspace, run_dir, *, data, config, epochs, seed, device, **_options):
        path = agent_workspace.run_workspace(
            workspace,
            run_dir,
            agent_workspace.data_builder(data, config),
            epochs=epochs,
            seed=seed,
            device=torch.device(device),
        )
        status = json.loads(path.read_text(encoding="utf-8"))["status"]
        return {"status": status, "problems": [], "report": str(path), "error": None, "output": ""}

    def check(self, **options):
        return agent_workspace.check_tasks(
            self.root / "check", data="synthetic", epochs=2, seed=11, device="cpu", **options
        )

    def test_every_task_is_trained_and_diagnosed_against_the_healthy_run(self):
        lines = []
        result = self.check(run=self.run_in_process, progress=lines.append)
        self.assertEqual(result["tasks_as_expected"], len(TASKS))
        self.assertEqual([row["task"] for row in result["tasks"]], list(TASKS))
        self.assertEqual(len(lines), len(TASKS))
        for row in result["tasks"]:
            with self.subTest(task=row["task"]):
                self.assertEqual(row["status"], "completed")
                self.assertEqual(row["diagnosis"], row["expected"])
                self.assertIsNone(row["output"])
                self.assertIn("ood_validation_accuracy", row["metrics"])
                unchanged = (
                    HEALTHY,
                    "head_replaced_after_moving",
                    "train_eval_normalization_mismatch",
                )
                self.assertEqual(row["same_steps_as_healthy"], row["task"] in unchanged)
                workspace = self.root / "check" / "workspaces" / row["task"]
                self.assertTrue((workspace / "TASK.md").is_file())
        saved = json.loads((self.root / "check" / "check_report.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, result)
        text = agent_workspace.check_text(result)
        self.assertIn(f"{len(TASKS)}/{len(TASKS)} tasks as expected", text)
        self.assertNotIn(" NO ", text)

    def test_a_task_that_does_not_run_is_reported_and_not_as_expected(self):
        def run(workspace, run_dir, **options):
            if workspace.name == "frozen_head":
                return {"status": "crashed", "problems": [], "report": None, "output": "boom"}
            return self.run_in_process(workspace, run_dir, **options)

        result = self.check(run=run, tasks=("frozen_head", HEALTHY, "stale_head"))
        rows = {row["task"]: row for row in result["tasks"]}
        self.assertEqual(list(rows), [HEALTHY, "frozen_head", "stale_head"])
        self.assertEqual(result["tasks_as_expected"], 2)
        self.assertEqual(rows["frozen_head"]["status"], "crashed")
        self.assertEqual(rows["frozen_head"]["output"], "boom")
        self.assertIsNone(rows["frozen_head"]["diagnosis"])
        text = agent_workspace.check_text(result)
        self.assertIn(" NO ", text)
        self.assertIn("frozen_head (crashed) printed:\nboom", text)

    def test_the_check_stops_when_the_healthy_reference_does_not_complete(self):
        calls = []

        def run(workspace, run_dir, **_options):
            calls.append(workspace.name)
            return {"status": "refused", "problems": ["line 1"], "report": None, "output": "no"}

        result = self.check(run=run)
        self.assertEqual(calls, [HEALTHY])
        self.assertEqual([row["task"] for row in result["tasks"]], [HEALTHY])
        self.assertEqual(result["tasks_as_expected"], 0)

    def test_check_needs_the_healthy_task_and_a_new_directory(self):
        with self.assertRaisesRegex(ValueError, "healthy"):
            self.check(run=self.run_in_process, tasks=("stale_head",))
        (self.root / "check").mkdir(exist_ok=True)
        with self.assertRaises(FileExistsError):
            self.check(run=self.run_in_process)

    def test_command_line_prints_the_table_and_fails_unless_every_task_is_as_expected(self):
        row = {
            "task": HEALTHY,
            "status": "completed",
            "diagnosis": "frozen_head",
            "as_expected": False,
            "same_steps_as_healthy": True,
            "metrics": {"id_validation_accuracy": 0.5, "ood_validation_accuracy": 0.25},
        }
        result = {"tasks_as_expected": 0, "tasks": [row]}
        arguments = ["check", "--output", str(self.root / "check"), "--data", "synthetic"]
        with (
            patch("runsleuth.agent_workspace.check_tasks", return_value=result) as check,
            patch("builtins.print") as printed,
        ):
            self.assertEqual(agent_workspace.main(arguments), 1)
            self.assertEqual(check.call_args.kwargs["tasks"], TASKS)
            selected = [*arguments, "--tasks", HEALTHY, "stale_head"]
            agent_workspace.main(selected)
            self.assertEqual(check.call_args.kwargs["tasks"], (HEALTHY, "stale_head"))
        self.assertEqual(check.call_args.kwargs["data"], "synthetic")
        self.assertIn("0.5000  0.2500", printed.call_args.args[0])
        self.assertIn("0/1 tasks as expected", printed.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
