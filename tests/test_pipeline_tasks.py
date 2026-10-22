"""CPU tests for the pipeline tasks, their contract tests and the check that trains them."""

import ast
import json
import tempfile
import unittest
import warnings
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from test_pipeline_workspace import TINY_MODEL, write_project

from runsleuth import pipeline_tasks
from runsleuth.pipeline_tasks import CAMELYON_STATEMENT, CAMELYON_VERSIONS, HEALTHY, TASKS

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

if torch is not None:
    from runsleuth import pipeline_contract, pipeline_workspace
    from runsleuth.agent_verification import compare_trajectories

VERSION_SOURCES = pipeline_tasks.version_sources  # Bound here, before any test patches it.
CASE_SOURCES = None if torch is None else pipeline_contract.case_sources


def tiny(sources: dict[str, str]) -> dict[str, str]:
    """A project with the tiny model in place of the pretrained ResNet18."""
    train = sources["train"]
    resnet = "model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)"
    marker = "\n\ndef build_model"
    assert train.count(resnet) == 1 and train.count(marker) == 1
    return {
        **sources,
        "train": train.replace(resnet, "model = Tiny()").replace(marker, TINY_MODEL + marker),
    }


def tiny_version(name: str, family_name: str = "camelyon_pipeline") -> dict[str, str]:
    """A version with the tiny model and four patches a batch, for runs on synthetic data."""
    sources = tiny(VERSION_SOURCES(name, family_name))
    assert sources["config"].count("BATCH_SIZE = 32") == 1
    return {**sources, "config": sources["config"].replace("BATCH_SIZE = 32", "BATCH_SIZE = 4")}


class TaskTests(unittest.TestCase):
    def test_every_version_is_an_exact_edit_of_the_healthy_project(self):
        healthy = pipeline_tasks.healthy_sources()
        self.assertEqual(set(healthy), {"config", "data", "train"})
        for name in CAMELYON_VERSIONS:
            with self.subTest(version=name):
                sources = pipeline_tasks.version_sources(name)
                self.assertEqual(set(sources), set(healthy))
                self.assertNotEqual(sources, healthy)
                changed = {module for module in healthy if sources[module] != healthy[module]}
                diff = pipeline_tasks.version_diff(name)
                for module in healthy:
                    self.assertEqual(f"+++ {name}/{module}.py" in diff, module in changed)
        self.assertEqual(pipeline_tasks.version_sources(HEALTHY), healthy)
        self.assertEqual(pipeline_tasks.version_diff(HEALTHY), "")

    def test_every_task_has_a_development_and_a_test_fault_and_variant(self):
        for task in TASKS:
            versions = [
                item
                for item in CAMELYON_VERSIONS.values()
                if item.task == task and not item.control
            ]
            for correct in (False, True):
                splits = sorted(item.split for item in versions if item.correct == correct)
                self.assertEqual(splits, ["development", "test"], (task, correct))
        for name, item in CAMELYON_VERSIONS.items():
            self.assertIn(item.task, TASKS)
            self.assertIn(item.split, pipeline_tasks.SPLITS)
            kind = "variant" if item.correct else "control" if item.control else "fault"
            self.assertEqual(pipeline_tasks.kind(name), kind)
            if item.correct:
                self.assertIsNone(item.signature)
                self.assertFalse(item.control)
        self.assertEqual(pipeline_tasks.kind("pretrained_weights_delivered"), "control")
        self.assertEqual(pipeline_tasks.kind(HEALTHY), HEALTHY)
        self.assertEqual(
            pipeline_tasks.expected_diagnosis("positions_index_base"),
            "train_evaluation_group_overlap",
        )
        self.assertEqual(pipeline_tasks.expected_diagnosis("labels_not_filtered"), "no_known_fault")
        with self.assertRaisesRegex(ValueError, "Unknown version"):
            pipeline_tasks.version("missing")

    def test_the_family_hash_changes_with_anything_a_version_is_made_of(self):
        family_hash = pipeline_tasks.family_sha256()
        self.assertEqual(family_hash, pipeline_tasks.family_sha256())
        changed = replace(CAMELYON_VERSIONS["labels_not_filtered"], split="development")
        with patch.dict(CAMELYON_VERSIONS, {"labels_not_filtered": changed}):
            self.assertNotEqual(pipeline_tasks.family_sha256(), family_hash)
        family = pipeline_tasks.FAMILIES["camelyon_pipeline"]
        with patch.dict(
            pipeline_tasks.FAMILIES,
            {"camelyon_pipeline": replace(family, statement=family.statement + "\n")},
        ):
            self.assertNotEqual(pipeline_tasks.family_sha256(), family_hash)
        self.assertEqual(pipeline_tasks.family_sha256(), family_hash)

    def test_a_project_is_its_modules_and_the_statement(self):
        with tempfile.TemporaryDirectory() as directory:
            project = pipeline_tasks.materialize("labels_not_filtered", Path(directory) / "p")
            self.assertEqual(
                sorted(path.name for path in project.iterdir()),
                ["TASK.md", "config.py", "data.py", "train.py"],
            )
            self.assertEqual((project / "TASK.md").read_text(encoding="utf-8"), CAMELYON_STATEMENT)
            self.assertIn(
                "self.labels = list(patches.labels)",
                (project / "data.py").read_text(encoding="utf-8"),
            )
            with self.assertRaises(FileExistsError):
                pipeline_tasks.materialize(HEALTHY, project)

    def test_the_statement_states_the_interface_and_names_no_fault(self):
        for stated in (
            "hospitals 0, 1, 3 and 4",
            "hospitals\n   0, 3 and 4 exactly once, with its own label",
            "Batches hold 32 patches",
            "(0.485, 0.456, 0.406)",
            "(0.229, 0.224, 0.225)",
            "learning rate 1e-4 and weight decay 1e-2",
            "`run.track(model, optimizer)`",
            "`with run.epoch(number):`",
            "`run.log(**metrics)`",
            "`run.deliver(state_dict)`",
            "`write(name, state)`, `read(name)` and `name in run.checkpoints`",
            "`num_workers=0`",
            "`run.log(validation_loss=..., validation_accuracy=...)`",
            "among all the epochs of the run, before and after any restart",
        ):
            self.assertIn(stated, CAMELYON_STATEMENT)
        signatures = {item.signature for item in CAMELYON_VERSIONS.values()} - {None}
        # Tasks name the requirements they test, which the statement states; the
        # versions and signatures would give away a fault.
        for name in (*CAMELYON_VERSIONS, *signatures):
            self.assertNotIn(name, CAMELYON_STATEMENT)


@unittest.skipIf(torch is None, "Install torch and torchvision for the contract tests")
class ContractTests(unittest.TestCase):
    def result(self, sources: dict[str, str]) -> dict:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return pipeline_contract.contract_result(sources)

    def test_every_version_and_case_fails_exactly_the_clauses_it_should(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            rows = pipeline_contract.check_cases(sources=lambda name: tiny(CASE_SOURCES(name)))
        for row in rows:
            with self.subTest(project=row["name"]):
                self.assertTrue(row["as_expected"], row)
        text = pipeline_contract.cases_text(rows)
        self.assertIn(f"{len(rows)}/{len(rows)} projects as expected", text)

    def test_the_cases_cover_correct_and_wrong_projects_and_every_clause(self):
        names = (*pipeline_tasks.family("camelyon_pipeline").names, *pipeline_contract.CASES)
        expected = {name: pipeline_contract.expected_failures(name) for name in names}
        self.assertGreaterEqual(sum(not clauses for clauses in expected.values()), 6)
        failed = {clause for clauses in expected.values() for clause in clauses}
        self.assertEqual(failed, set(pipeline_contract.CLAUSES))
        for clause in pipeline_contract.CLAUSES:
            alone = [name for name, clauses in expected.items() if clauses == (clause,)]
            self.assertGreaterEqual(len(alone), 1, clause)

    def test_codes_survive_normalization_and_nothing_else(self):
        codes = [0, 5, 46, pipeline_contract.VALIDATION_CODES + 3]
        images = pipeline_contract.coded_images(codes, [0.3, 0.5, 0.7, 0.4])
        mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        once = (images - mean) / std
        for inputs in (images, once, (once - mean) / std):
            self.assertEqual(pipeline_contract.read_codes(inputs), codes)
        for changed in (images.flip(-1), images.mean(dim=1, keepdim=True).expand_as(images)):
            self.assertEqual(pipeline_contract.read_codes(changed), [None] * len(codes))
        self.assertEqual(pipeline_contract.read_codes(torch.zeros(2, 5)), [None, None])

    def test_a_project_that_fails_or_is_refused_does_not_pass(self):
        healthy = tiny(CASE_SOURCES(HEALTHY))
        crashing = {
            **healthy,
            "data": healthy["data"].replace("return len(self.indices)", "return 1 / 0"),
        }
        result = self.result(crashing)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failed"], list(pipeline_contract.CLAUSES))
        self.assertIn("ZeroDivisionError", result["clauses"]["training_samples"]["detail"])
        with tempfile.TemporaryDirectory() as directory:
            sources = {**healthy, "data": "import os\n" + healthy["data"]}
            refused = pipeline_contract.run_contract(write_project(Path(directory) / "p", sources))
        self.assertEqual(refused["status"], "refused")
        self.assertEqual(refused["failed"], list(pipeline_contract.CLAUSES))

    def test_a_child_process_gives_the_same_result(self):
        sources = tiny(CASE_SOURCES("labels_not_filtered"))
        with tempfile.TemporaryDirectory() as directory:
            project = write_project(Path(directory) / "p", sources)
            child = pipeline_contract.run_contract_in_subprocess(project, timeout=300)
        self.assertEqual(child, self.result(sources))
        self.assertEqual(child["failed"], ["training_labels"])

    def test_the_checks_share_no_code_with_the_verifier(self):
        tree = ast.parse(Path(pipeline_contract.__file__).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module.startswith("runsleuth"):
                imported.update(f"{node.module}.{alias.name}" for alias in node.names)
        self.assertEqual(
            imported,
            {
                "runsleuth.pipeline_tasks",
                "runsleuth.pipeline_tasks.HEALTHY",
                "runsleuth.agent_workspace.clean_environment",
                "runsleuth.pipeline_workspace.check_project",
                "runsleuth.pipeline_workspace.load_project",
                "runsleuth.pipeline_workspace.project_sources",
            },
        )


@unittest.skipIf(torch is None, "Install torch and torchvision for the pipeline tests")
class CheckTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        patcher = patch("runsleuth.pipeline_tasks.version_sources", tiny_version)
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def run_in_process(project, run_dir, *, data, config, epochs, seed, device, **options):
        path = pipeline_workspace.run_project(
            project,
            run_dir,
            pipeline_workspace.data_builder(data, config),
            epochs=epochs,
            seed=seed,
            device=torch.device(device),
            interrupt_after_epochs=options["interrupt_after_epochs"],
        )
        status = json.loads(path.read_text(encoding="utf-8"))["status"]
        return {"status": status, "problems": [], "report": str(path), "error": None, "output": ""}

    def check(self, **options):
        return pipeline_workspace.check_versions(
            self.root / "check", data="synthetic", epochs=3, seed=11, device="cpu", **options
        )

    def test_every_version_is_trained_and_diagnosed_against_the_healthy_run(self):
        lines = []
        result = self.check(run=self.run_in_process, progress=lines.append)
        names = [HEALTHY, pipeline_workspace.RESTARTED, *CAMELYON_VERSIONS]
        self.assertEqual([row["version"] for row in result["versions"]], names)
        self.assertEqual(len(lines), len(names))
        self.assertEqual(result["versions_as_expected"], len(names))
        rows = {row["version"]: row for row in result["versions"]}
        self.assertEqual(rows[pipeline_workspace.RESTARTED]["kind"], "restart")
        self.assertEqual(
            rows[pipeline_workspace.RESTARTED]["trajectory"]["outcome"], "identical_full_run"
        )
        self.assertEqual(rows["positions_index_base"]["evidence"]["held_out_training_samples"], 8)
        self.assertEqual(rows[HEALTHY]["evidence"]["labels_matching"], 1.0)
        self.assertLess(rows["labels_not_filtered"]["evidence"]["labels_matching"], 1.0)
        variants = ("subset_random_sampler", "seeded_per_epoch")
        for name in variants:
            self.assertEqual(rows[name]["evidence"]["labels_matching"], 1.0)
            self.assertEqual(rows[name]["trajectory"]["outcome"], "diverged")
        # The test variant must not replay the development one.
        reports = {
            name: json.loads(Path(row["report"]).read_text(encoding="utf-8"))
            for name, row in rows.items()
        }
        self.assertEqual(
            compare_trajectories(*(reports[name] for name in variants))["outcome"], "diverged"
        )
        # Every run but the healthy reference is restarted after its first epoch.
        for name in ("scheduler_not_restored", "optimizer_not_restored"):
            self.assertEqual(rows[name]["trajectory"]["outcome"], "diverged")
        for name in (pipeline_workspace.RESTARTED, "checkpoint_in_parts", "numbered_checkpoints"):
            self.assertEqual(rows[name]["trajectory"]["outcome"], "identical_full_run")
            self.assertEqual(len(reports[name]["pipeline"]["attempts"]), 2)
        restored = reports[pipeline_workspace.RESTARTED]["pipeline"]["tracked"][1]
        rebuilt = reports["optimizer_not_restored"]["pipeline"]["tracked"][1]
        for tracked in (restored, rebuilt):
            self.assertEqual(tracked["model_same_as"], ["epoch 1 (attempt 1)"])
        self.assertGreater(restored["optimizer_parameters_with_state"], 0)
        self.assertLess(restored["learning_rates"][0], 1e-4)
        self.assertEqual(rebuilt["optimizer_parameters_with_state"], 0)
        self.assertEqual(rebuilt["learning_rates"], [1e-4])
        # The delivery: the last epoch, the freshly built model, or the healthy run's best.
        delivered = {name: row["evidence"]["delivered"] for name, row in rows.items()}
        self.assertEqual(delivered["last_epoch_delivered"], ["epoch 3 (attempt 2)"])
        self.assertEqual(delivered["pretrained_weights_delivered"], [])
        self.assertEqual(len(delivered[HEALTHY]), 1)
        self.assertEqual(
            result["settings"]["family_sha256"], pipeline_tasks.family_sha256("camelyon_pipeline")
        )
        saved = json.loads((self.root / "check" / "check_report.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, result)
        text = pipeline_workspace.check_text(result)
        self.assertIn(f"{len(names)}/{len(names)} versions as expected", text)
        self.assertNotIn(" NO ", text)

    def test_a_version_that_does_not_run_is_reported_and_not_as_expected(self):
        def run(project, run_dir, **options):
            if project.name == "labels_not_filtered":
                return {"status": "crashed", "problems": [], "report": None, "output": "boom"}
            return self.run_in_process(project, run_dir, **options)

        result = self.check(run=run, versions=("labels_not_filtered", HEALTHY))
        rows = {row["version"]: row for row in result["versions"]}
        self.assertEqual(list(rows), [HEALTHY, pipeline_workspace.RESTARTED, "labels_not_filtered"])
        self.assertEqual(result["versions_as_expected"], 2)
        self.assertEqual(rows["labels_not_filtered"]["output"], "boom")
        self.assertIsNone(rows["labels_not_filtered"]["diagnosis"])
        self.assertIn(" NO ", pipeline_workspace.check_text(result))
        with self.assertRaisesRegex(ValueError, "healthy"):
            pipeline_workspace.check_versions(
                self.root / "other",
                data="synthetic",
                epochs=1,
                seed=1,
                versions=("seeded_per_epoch",),
            )


if __name__ == "__main__":
    unittest.main()
