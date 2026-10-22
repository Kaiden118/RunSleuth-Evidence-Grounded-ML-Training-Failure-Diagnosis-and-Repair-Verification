"""CPU tests for RunSleuth v1 on pipeline runs, on runs of the tiny network."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# The module, not its classes: a TestCase imported here would run here as well.
import test_pipeline_tasks

from runsleuth.pipeline_tasks import HEALTHY

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

if torch is not None:
    from runsleuth import pipeline_workspace, verifier_v1

# Wide enough that the gate never decides: these tests are about v1's other parts.
OPEN_GATE = {"id": 1.0, "ood": 1.0}
VERSIONS = (
    HEALTHY,
    "checkpoint_in_parts",
    "subset_random_sampler",
    "positions_index_base",
)


@unittest.skipIf(torch is None, "Install torch and torchvision for the pipeline tests")
class VerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        with patch("runsleuth.pipeline_tasks.version_sources", test_pipeline_tasks.tiny_version):
            result = pipeline_workspace.check_versions(
                Path(temporary.name) / "check",
                data="synthetic",
                epochs=3,
                seed=11,
                device="cpu",
                versions=VERSIONS,
                run=test_pipeline_tasks.CheckTests.run_in_process,
            )
        cls.reports = {
            row["version"]: json.loads(Path(row["report"]).read_text(encoding="utf-8"))
            for row in result["versions"]
        }

    def verify(self, name, gate=OPEN_GATE):
        return verifier_v1.verify(self.reports[name], self.reports[HEALTHY], gate)

    def test_the_reference_run_restarted_and_a_resume_written_differently_are_accepted(self):
        for name in (pipeline_workspace.RESTARTED, "checkpoint_in_parts"):
            result = self.verify(name)
            self.assertEqual(result["triage"], "accept", result)
            self.assertTrue(result["same_evaluation"])
            self.assertEqual(result["differences"], [])

    def test_a_correct_run_that_differs_goes_to_review(self):
        result = self.verify("subset_random_sampler")
        self.assertEqual(result["triage"], "review")
        self.assertEqual(result["re_diagnosis"], "no_known_fault")
        self.assertTrue(any(line.startswith("model states") for line in result["differences"]))

    def test_a_known_fault_is_rejected(self):
        result = self.verify("positions_index_base")
        self.assertEqual(result["re_diagnosis"], "train_evaluation_group_overlap")
        self.assertEqual(result["triage"], "reject")

    def test_other_reported_metrics_send_an_identical_training_run_to_review(self):
        candidate = copy.deepcopy(self.reports[pipeline_workspace.RESTARTED])
        candidate["parameter_group_epochs"][1]["project_metrics"]["validation_accuracy"] -= 0.01
        result = verifier_v1.verify(candidate, self.reports[HEALTHY], OPEN_GATE)
        self.assertEqual(result["trajectory"]["outcome"], "identical_full_run")
        self.assertEqual(result["triage"], "review")
        self.assertIn(
            "evaluation: the project reported other validation metrics", result["differences"]
        )

    def test_a_candidate_that_did_not_run_or_falls_below_the_gate_is_rejected(self):
        self.assertEqual(
            verifier_v1.verify(None, self.reports[HEALTHY], OPEN_GATE)["triage"], "reject"
        )
        crashed = {**self.reports["checkpoint_in_parts"], "status": "crashed"}
        self.assertEqual(
            verifier_v1.verify(crashed, self.reports[HEALTHY], OPEN_GATE)["triage"], "reject"
        )
        lower = copy.deepcopy(self.reports["checkpoint_in_parts"])
        lower["final_metrics"]["ood_validation_accuracy"] -= 0.5
        result = verifier_v1.verify(lower, self.reports[HEALTHY], {"id": 0.0, "ood": 0.1})
        self.assertEqual(result["triage"], "reject")
        self.assertEqual(result["gate"]["decision"], "reject")


if __name__ == "__main__":
    unittest.main()
