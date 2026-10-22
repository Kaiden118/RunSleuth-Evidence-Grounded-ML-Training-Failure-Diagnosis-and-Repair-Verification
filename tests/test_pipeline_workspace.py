"""CPU tests for the pipeline harness: project checks, recorded runs, restarts, subprocesses."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import runsleuth
from runsleuth.signature_matching import diagnose, extract_evidence, load_library

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

if torch is not None:
    from runsleuth import pipeline_workspace
    from runsleuth.agent_verification import compare_trajectories
    from runsleuth.agent_workspace import ScriptRefused, check_script
    from runsleuth.pipeline_workspace import Samples, check_project, load_project

TEMPLATES = Path(runsleuth.__file__).parent / "pipeline_templates"
# The model of the agent_workspace tests: small enough for CPU, with BatchNorm and a
# 1000-way fc head that the project replaces. Built with weights, its convolution and
# head take the same values every time, as pretrained weights would.
TINY_MODEL = """

class Tiny(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.backbone = nn.Conv2d(3, 4, kernel_size=3)
        self.norm = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4, 1000)
        if weights is not None:
            generator = torch.Generator().manual_seed(0)
            with torch.no_grad():
                for value in (*self.backbone.parameters(), *self.fc.parameters()):
                    value.uniform_(-0.2, 0.2, generator=generator)

    def forward(self, inputs):
        return self.fc(torch.relu(self.norm(self.backbone(inputs))).mean(dim=(2, 3)))
"""
POSITIONS_INDEX_BASE = (
    "data",
    "self.patches.image(self.indices[position])",
    "self.patches.image(position)",
)
LABELS_NOT_FILTERED = (
    "data",
    "self.labels = [patches.labels[index] for index in self.indices]",
    "self.labels = list(patches.labels)",
)
TRAINING = (
    "        with run.epoch(epoch):\n"
    "            train_one_epoch(model, loader, optimizer, loss_function, run.device)\n"
)


def template_sources(template: str = "camelyon") -> dict[str, str]:
    folder = TEMPLATES / template
    return {path.stem: path.read_text(encoding="utf-8") for path in sorted(folder.glob("*.py"))}


def tiny_sources(*edits: tuple[str, str, str], template: str = "camelyon") -> dict[str, str]:
    """A template with the tiny model and four samples a batch, then edits (module, old, new)."""
    sources = template_sources(template)
    replacements = (
        ("train", "model = resnet18(", "model = Tiny("),
        ("train", "\n\ndef build_model", TINY_MODEL + "\n\ndef build_model"),
        ("config", "BATCH_SIZE = 32", "BATCH_SIZE = 4"),
        *edits,
    )
    for module, old, new in replacements:
        assert sources[module].count(old) == 1, (module, old)
        sources[module] = sources[module].replace(old, new)
    return sources


def write_project(directory: Path, sources: dict[str, str]) -> Path:
    directory.mkdir(parents=True)
    for name, source in sources.items():
        (directory / f"{name}.py").write_text(source, encoding="utf-8")
    return directory


@unittest.skipIf(torch is None, "Install torch and torchvision for the pipeline tests")
class ProjectCheckTests(unittest.TestCase):
    def test_the_template_may_run(self):
        self.assertEqual(check_project(template_sources()), [])
        self.assertEqual(check_project(tiny_sources()), [])

    def test_modules_may_import_each_other_and_nothing_else_new(self):
        local = frozenset({"data"})
        self.assertIn("import of 'data'", check_script("import data\n")[0])
        self.assertEqual(check_script("import data\nfrom data import loader\n", local), [])
        self.assertIn("import of 'data.extra'", check_script("import data.extra\n", local)[0])
        self.assertIn("'_hidden'", check_script("from data import _hidden\n", local)[0])
        sources = {**tiny_sources(), "train": "import helpers\n" + tiny_sources()["train"]}
        self.assertEqual(
            check_project(sources), ["train.py: line 1: import of 'helpers' is not allowed"]
        )

    def test_a_refusal_names_the_module(self):
        sources = {**tiny_sources(), "data": "import os\n" + tiny_sources()["data"]}
        self.assertEqual(check_project(sources), ["data.py: line 1: import of 'os' is not allowed"])

    def test_modules_may_not_reuse_library_names(self):
        for name in ("torch", "json", "Data"):
            with self.subTest(name=name):
                problems = check_project({**tiny_sources(), name: "x = 1\n"})
                self.assertTrue(problems and problems[0].startswith(f"{name}.py: "), problems)

    def test_a_project_needs_train_and_its_main(self):
        sources = tiny_sources()
        self.assertEqual(
            check_project({name: sources[name] for name in ("config", "data")}),
            ["the project has no train.py"],
        )
        without_main = {
            **sources,
            "train": sources["train"].replace("def main(run)", "def go(run)"),
        }
        with self.assertRaises(ScriptRefused):
            load_project(without_main)

    def test_the_recording_is_out_of_a_projects_reach(self):
        for line in ("run._recorder = None\n", "run.data.train._reader = None\n"):
            with self.subTest(line=line):
                self.assertIn("is not allowed", check_script(line)[0])

    def test_modules_are_loaded_afresh_and_kept_out_of_sys_modules(self):
        first, second = load_project(tiny_sources()), load_project(tiny_sources())
        self.assertEqual(set(first), {"config", "data", "train"})
        self.assertIsNot(first["data"], second["data"])
        self.assertIs(vars(first["train"])["data"], first["data"])
        for name in first:
            self.assertIsNot(sys.modules.get(name), first[name])


@unittest.skipIf(torch is None, "Install torch and torchvision for the pipeline tests")
class SamplesTests(unittest.TestCase):
    def samples(self, reads: list[int]) -> "Samples":
        images = torch.arange(3, dtype=torch.float32).view(3, 1, 1, 1).expand(3, 3, 2, 2)
        return Samples(
            lambda position: images[position],
            {"labels": [0, 1, 1], "hospitals": [0, 1, 3]},
            labels=[0, 1, 1],
            reader=reads.append,
        )

    def test_reads_are_recorded_and_columns_are_free(self):
        reads = []
        samples = self.samples(reads)
        image, label = samples[1]
        self.assertEqual((float(image[0, 0, 0]), label), (1.0, 1))
        self.assertEqual(float(samples.image(-1)[0, 0, 0]), 2.0)
        self.assertEqual(
            (samples.labels, samples.hospitals, len(samples)), ((0, 1, 1), (0, 1, 3), 3)
        )
        self.assertEqual(reads, [1, 2])
        with self.assertRaises(IndexError):
            samples[3]

    def test_reads_outside_the_main_process_are_refused(self):
        with patch("runsleuth.pipeline_workspace.get_worker_info", return_value=object()):
            with self.assertRaisesRegex(RuntimeError, "num_workers=0"):
                self.samples([])[0]

    def test_columns_need_the_target_and_names_of_their_own(self):
        for columns, labels in (
            ({"hospitals": [0]}, [0]),
            ({"labels": [0], "image": [0]}, [0]),
            ({"labels": [0, 1], "x": [0, 1]}, [0]),
            ({"labels": [0], "x": [0, 1]}, [0]),
        ):
            with self.subTest(columns=columns), self.assertRaises(ValueError):
                Samples(lambda position: torch.zeros(3, 1, 1), columns, labels=labels)

    def test_a_project_may_get_diagnoses_while_the_harness_keeps_the_labels(self):
        pixels = torch.tensor([0, 255], dtype=torch.uint8).view(2, 1, 1, 1).expand(2, 3, 2, 2)
        samples = Samples(
            lambda position: pixels[position],
            {"diagnoses": ["nv", "akiec"], "lesion_ids": ["a", "b"]},
            labels=[0, 1],
            target="diagnoses",
        )
        image, diagnosis = samples[1]
        self.assertEqual((image.dtype, diagnosis), (torch.uint8, "akiec"))
        self.assertFalse(hasattr(samples, "labels"))
        ((images, labels),) = samples._batches(8)
        self.assertEqual((images.dtype, float(images.max())), (torch.float32, 1.0))
        self.assertEqual(labels.tolist(), [0, 1])

    def test_a_state_hash_follows_its_contents(self):
        state = {"epoch": 1, "model": {"w": torch.ones(2)}}
        same = {"model": {"w": torch.ones(2)}, "epoch": 1}
        self.assertEqual(
            pipeline_workspace.state_sha256(state), pipeline_workspace.state_sha256(same)
        )
        for other in (
            {"epoch": 2, "model": {"w": torch.ones(2)}},
            {"epoch": 1, "model": {"w": torch.zeros(2)}},
        ):
            self.assertNotEqual(
                pipeline_workspace.state_sha256(state), pipeline_workspace.state_sha256(other)
            )


@unittest.skipIf(torch is None, "Install torch and torchvision for the pipeline tests")
class RunTests(unittest.TestCase):
    """Three-epoch runs on synthetic patches; the restarted run stops after epoch 1."""

    PROJECTS = {
        "healthy": ((), None),
        "restarted": ((), 1),
        "positions_index_base": ((POSITIONS_INDEX_BASE,), None),
        "labels_not_filtered": ((LABELS_NOT_FILTERED,), None),
        "handler_around_training": (
            (
                (
                    "train",
                    TRAINING,
                    "        try:\n"
                    + "".join("    " + line + "\n" for line in TRAINING.splitlines())
                    + "        except Exception:\n            pass\n",
                ),
            ),
            1,
        ),
        "delivers_nothing": (
            (("train", "    run.deliver(run.checkpoints.read(", "    (run.checkpoints.read("),),
            None,
        ),
        "fails_in_training": (
            (("train", "        loss.backward()\n", "        raise ValueError('broken')\n"),),
            None,
        ),
        "fails_before_tracking": (
            (("train", "run.track(model, optimizer)", "run.track(model, None)"),),
            None,
        ),
    }

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        cls.reports = {}
        for name, (edits, interrupt) in cls.PROJECTS.items():
            project = write_project(root / "projects" / name, tiny_sources(*edits))
            path = pipeline_workspace.run_project(
                project,
                root / "runs" / name,
                pipeline_workspace.synthetic_pipeline_data,
                epochs=3,
                seed=7,
                device=torch.device("cpu"),
                interrupt_after_epochs=interrupt,
            )
            cls.reports[name] = json.loads(path.read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def diagnosis(self, name: str) -> str:
        evidence = extract_evidence(self.reports[name], self.reports["healthy"])
        return diagnose(evidence, load_library())["diagnosis"]

    def test_every_training_sample_is_read_once_an_epoch_with_its_label(self):
        report = self.reports["healthy"]
        self.assertEqual(report["status"], "completed")
        for epoch in report["pipeline"]["epochs"]:
            self.assertEqual((epoch["reads"], epoch["distinct"]), (24, 24))
            self.assertEqual(epoch["reads_by_group"], {"0": 8, "3": 8, "4": 8})
            self.assertEqual(epoch["loss_targets"], 24)
            self.assertEqual(epoch["loss_targets_matching_read_labels"], 24)
        reads = report["pipeline"]["training_reads"]
        self.assertEqual(reads["candidates_by_group"], {"0": 8, "1": 8, "3": 8, "4": 8})
        self.assertEqual(reads["read_while_training"], 24)
        self.assertEqual(
            report["evaluation_holdouts"]["ood"]["training_samples_in_evaluation_groups"], 0
        )
        self.assertEqual(sorted(report["input_statistics"]), ["id_eval", "train"])
        self.assertEqual(self.diagnosis("healthy"), "no_known_fault")

    def test_a_restarted_run_continues_the_uninterrupted_one(self):
        report, healthy = self.reports["restarted"], self.reports["healthy"]
        pipeline = report["pipeline"]
        self.assertEqual(
            [attempt["outcome"] for attempt in pipeline["attempts"]], ["preempted", "completed"]
        )
        self.assertEqual(
            [(epoch["attempt"], epoch["epoch"]) for epoch in pipeline["epochs"]],
            [(1, 1), (2, 2), (2, 3)],
        )
        read = [event for event in pipeline["checkpoints"] if event["action"] == "read"][0]
        self.assertEqual((read["attempt"], read["after_epoch"], read["name"]), (2, 1, "last"))
        first, resumed = pipeline["tracked"]
        self.assertEqual((first["model_same_as"], first["optimizer_steps"]), (["initial state"], 0))
        self.assertEqual(resumed["model_same_as"], ["epoch 1 (attempt 1)"])
        self.assertEqual(resumed["optimizer_steps"], 6)  # 24 patches in batches of 4
        self.assertEqual(compare_trajectories(report, healthy)["outcome"], "identical_full_run")
        self.assertEqual(
            pipeline["delivery"]["state_sha256"], healthy["pipeline"]["delivery"]["state_sha256"]
        )
        self.assertEqual(report["final_metrics"], healthy["final_metrics"])

    def test_checkpoints_and_the_delivery_are_recorded(self):
        pipeline = self.reports["healthy"]["pipeline"]
        writes = [
            (event["after_epoch"], event["name"])
            for event in pipeline["checkpoints"]
            if event["action"] == "write"
        ]
        self.assertEqual(
            [write for write in writes if write[1] == "last"],
            [(1, "last"), (2, "last"), (3, "last")],
        )
        accuracies = [
            row["project_metrics"]["validation_accuracy"]
            for row in self.reports["healthy"]["parameter_group_epochs"]
        ]
        best = accuracies.index(max(accuracies)) + 1
        self.assertIn((best, "best"), writes)
        self.assertEqual(pipeline["delivery"]["same_as"], [f"epoch {best} (attempt 1)"])

    def test_the_delivered_weights_are_evaluated_by_the_harness(self):
        report = self.reports["healthy"]
        rows = report["parameter_group_epochs"]
        accuracies = [row["project_metrics"]["validation_accuracy"] for row in rows]
        delivered = rows[accuracies.index(max(accuracies))]["project_metrics"]
        final = report["final_metrics"]
        self.assertEqual(
            sorted(final),
            [
                "epoch",
                "id_validation_accuracy",
                "id_validation_loss",
                "ood_validation_accuracy",
                "ood_validation_loss",
            ],
        )
        self.assertEqual(final["id_validation_accuracy"], delivered["validation_accuracy"])
        self.assertAlmostEqual(final["id_validation_loss"], delivered["validation_loss"], places=5)

    def test_reading_the_wrong_samples_is_recorded_and_diagnosed(self):
        report = self.reports["positions_index_base"]
        for epoch in report["pipeline"]["epochs"]:
            self.assertEqual(epoch["reads_by_group"], {"0": 8, "1": 8, "3": 8})
            self.assertLess(epoch["loss_targets_matching_read_labels"], epoch["loss_targets"])
        self.assertEqual(
            report["evaluation_holdouts"]["ood"]["training_samples_in_evaluation_groups"], 8
        )
        self.assertEqual(self.diagnosis("positions_index_base"), "train_evaluation_group_overlap")

    def test_labels_that_do_not_follow_their_samples_are_recorded(self):
        report = self.reports["labels_not_filtered"]
        for epoch in report["pipeline"]["epochs"]:
            self.assertEqual(epoch["reads_by_group"], {"0": 8, "3": 8, "4": 8})
            self.assertLess(epoch["loss_targets_matching_read_labels"], epoch["loss_targets"])
        self.assertEqual(
            report["evaluation_holdouts"]["ood"]["training_samples_in_evaluation_groups"], 0
        )

    def test_an_exception_handler_does_not_hold_up_a_restart(self):
        report = self.reports["handler_around_training"]
        self.assertEqual(
            [attempt["outcome"] for attempt in report["pipeline"]["attempts"]],
            ["preempted", "completed"],
        )
        self.assertEqual(
            compare_trajectories(report, self.reports["healthy"])["outcome"], "identical_full_run"
        )

    def test_a_run_must_deliver_weights(self):
        report = self.reports["delivers_nothing"]
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error"]["message"], "the run delivered no weights")
        self.assertIsNone(report["pipeline"]["delivery"])

    def test_a_failing_project_still_reports(self):
        report = self.reports["fails_in_training"]
        self.assertEqual((report["status"], report["error"]["type"]), ("failed", "ValueError"))
        self.assertIn("broken", report["workspace"]["traceback"])
        report = self.reports["fails_before_tracking"]
        self.assertEqual((report["status"], report["error"]["type"]), ("failed", "TypeError"))
        self.assertEqual(report["pipeline"]["epochs"], [])


@unittest.skipIf(torch is None, "Install torch and torchvision for the pipeline tests")
class HamRunTests(unittest.TestCase):
    """The HAM10000 project on synthetic images, three epochs, with and without a restart."""

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        cls.reports = {}
        for name, interrupt in (("healthy", None), ("restarted", 1)):
            project = write_project(root / name, tiny_sources(template="ham10000"))
            path = pipeline_workspace.run_project(
                project,
                root / "runs" / name,
                pipeline_workspace.synthetic_ham_pipeline_data,
                epochs=3,
                seed=7,
                device=torch.device("cpu"),
                interrupt_after_epochs=interrupt,
            )
            cls.reports[name] = json.loads(path.read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_the_template_may_run(self):
        self.assertEqual(check_project(template_sources("ham10000")), [])

    def test_the_cache_reads_every_image_once_and_the_labels_follow_the_diagnoses(self):
        report = self.reports["healthy"]
        self.assertEqual(report["status"], "completed")
        first, *later = report["pipeline"]["epochs"]
        self.assertEqual((first["reads"], first["distinct"]), (24, 24))
        self.assertEqual(first["loss_targets_matching_read_labels"], 24)
        for epoch in later:
            # Cached: nothing is read, so the targets cannot be paired with reads.
            self.assertEqual(epoch["reads"], 0)
            self.assertIsNone(epoch["loss_targets_matching_read_labels"])
        self.assertEqual(report["pipeline"]["training_reads"]["read_while_training"], 24)
        for holdout in report["evaluation_holdouts"].values():
            self.assertEqual(holdout["training_samples_in_evaluation_groups"], 0)

    def test_a_restarted_run_with_flips_and_a_cache_continues_the_uninterrupted_one(self):
        report, healthy = self.reports["restarted"], self.reports["healthy"]
        attempts = [attempt["outcome"] for attempt in report["pipeline"]["attempts"]]
        self.assertEqual(attempts, ["preempted", "completed"])
        self.assertEqual(compare_trajectories(report, healthy)["outcome"], "identical_full_run")
        self.assertEqual(report["final_metrics"], healthy["final_metrics"])

    def test_the_harness_evaluates_with_the_true_labels(self):
        report = self.reports["healthy"]
        rows = report["parameter_group_epochs"]
        accuracies = [row["project_metrics"]["validation_accuracy"] for row in rows]
        delivered = rows[accuracies.index(max(accuracies))]["project_metrics"]
        final = report["final_metrics"]
        self.assertEqual(final["id_validation_accuracy"], delivered["validation_accuracy"])
        self.assertAlmostEqual(final["id_validation_loss"], delivered["validation_loss"], places=5)
        self.assertIn("ood_validation_accuracy", final)


@unittest.skipIf(torch is None, "Install torch and torchvision for the pipeline tests")
class SubprocessTests(unittest.TestCase):
    def test_a_child_process_runs_a_project_with_a_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            project = write_project(Path(directory) / "project", tiny_sources())
            outcome = pipeline_workspace.run_in_subprocess(
                project,
                Path(directory) / "run",
                data="synthetic",
                epochs=2,
                seed=7,
                device="cpu",
                interrupt_after_epochs=1,
                timeout=300,
            )
            self.assertEqual(outcome["status"], "completed", outcome["output"])
            report = json.loads(Path(outcome["report"]).read_text(encoding="utf-8"))
            self.assertEqual(len(report["pipeline"]["attempts"]), 2)

    def test_a_refused_project_never_starts(self):
        with tempfile.TemporaryDirectory() as directory:
            sources = {**tiny_sources(), "data": "import os\n" + tiny_sources()["data"]}
            project = write_project(Path(directory) / "project", sources)
            outcome = pipeline_workspace.run_in_subprocess(
                project, Path(directory) / "run", data="synthetic", epochs=1, seed=7
            )
            self.assertEqual(outcome["status"], "refused")
            self.assertFalse((Path(directory) / "run").exists())


if __name__ == "__main__":
    unittest.main()
