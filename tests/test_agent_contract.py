"""CPU tests for the contract tests of agent scripts, on a tiny network."""

import ast
import tempfile
import unittest
from pathlib import Path

from test_agent_workspace import TINY_MODEL

from runsleuth import agent_tasks
from runsleuth.agent_tasks import ALL_FAULTS, FAULTS, HEALTHY, LOOKALIKES, TASKS

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

if torch is not None:
    from runsleuth import agent_contract
    from runsleuth.agent_contract import CASES, CLAUSES

PATCH_SIZE = 16


def tiny(source: str) -> str:
    """A script with the tiny model in place of the pretrained ResNet18."""
    resnet = "model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)"
    marker = "\n\ndef use_for_training"
    assert source.count(resnet) == 1 and source.count(marker) == 1
    return source.replace(resnet, "model = Tiny()").replace(marker, TINY_MODEL + marker)


@unittest.skipIf(torch is None, "Install torch and torchvision for the contract tests")
class AgentContractTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.reference = self.write("reference", tiny(agent_tasks.task_source(HEALTHY)))

    def write(self, name: str, source: str) -> Path:
        path = self.root / f"{name}.py"
        path.write_text(source, encoding="utf-8", newline="\n")
        return path

    def result(self, source: str) -> dict:
        script = self.write("script", source)
        return agent_contract.run_contract(script, self.reference, patch_size=PATCH_SIZE)

    def test_every_task_and_case_fails_exactly_the_clauses_it_should(self):
        rows = agent_contract.check_cases(
            patch_size=PATCH_SIZE, sources=lambda name: tiny(agent_contract.case_source(name))
        )
        self.assertEqual([row["name"] for row in rows], [*TASKS, *CASES])
        for row in rows:
            with self.subTest(script=row["name"]):
                self.assertEqual(sorted(row["failed"]), sorted(row["expected"]), row["details"])
                self.assertEqual(len(row["details"]), len(row["failed"]))
        self.assertIn(
            f"{len(rows)}/{len(rows)} scripts as expected", agent_contract.cases_text(rows)
        )

    def test_the_cases_cover_correct_and_wrong_scripts_and_every_clause(self):
        for name in (HEALTHY, *LOOKALIKES):
            self.assertEqual(agent_contract.expected_failures(name), ())
        self.assertEqual(set(agent_contract.FAULT_CLAUSES), set(ALL_FAULTS))
        names = (*TASKS, *CASES)
        correct = [name for name in names if not agent_contract.expected_failures(name)]
        wrong = [name for name in names if agent_contract.expected_failures(name)]
        self.assertEqual(set(correct) & set(TASKS), {HEALTHY, *LOOKALIKES})
        self.assertGreaterEqual(len(correct), 9)
        self.assertGreaterEqual(len(wrong), 21)
        failed_somewhere = {clause for case in CASES.values() for clause in case.fails}
        failed_somewhere.update(agent_contract.FAULT_CLAUSES.values())
        self.assertEqual(failed_somewhere, set(CLAUSES))
        self.assertTrue(set(CASES).isdisjoint(TASKS))

    def test_a_result_names_each_clause_and_why_it_failed(self):
        healthy = self.result(tiny(agent_tasks.task_source(HEALTHY)))
        self.assertEqual((healthy["status"], healthy["failed"]), ("passed", []))
        self.assertEqual(tuple(healthy["clauses"]), CLAUSES)
        self.assertEqual(healthy["contract_version"], agent_contract.CONTRACT_VERSION)
        late = self.result(tiny(agent_tasks.task_source("schedule_ends_an_epoch_early")))
        self.assertEqual((late["status"], late["failed"]), ("failed", ["learning_rate_schedule"]))
        detail = late["clauses"]["learning_rate_schedule"]["detail"]
        self.assertIn("1, 0.5, 0 of the first", detail)
        self.assertIn("gives 1, 0.75, 0.25", detail)
        self.assertEqual(late["clauses"]["optimizer"], {"passed": True, "detail": ""})

    def test_a_script_that_fails_or_is_refused_does_not_pass(self):
        crashing = tiny(agent_tasks.task_source(HEALTHY)).replace(
            "    model.train()\n", "    model.train()\n    raise ValueError('no')\n"
        )
        crashed = self.result(crashing)
        self.assertEqual(crashed["status"], "failed")
        self.assertEqual(crashed["failed"], list(CLAUSES[2:]))
        self.assertIn("training failed with ValueError: no", crashed["clauses"]["model"]["detail"])
        self.assertTrue(crashed["clauses"]["training_hospitals"]["passed"])
        refused = self.result("import os\n")
        self.assertEqual((refused["status"], refused["failed"]), ("refused", list(CLAUSES)))
        self.assertIn("import of 'os' is not allowed", refused["problems"][0])
        no_hospital = tiny(agent_tasks.task_source(HEALTHY)).replace(
            "hospital not in (VALIDATION_HOSPITAL, TEST_HOSPITAL)", "False"
        )
        self.assertEqual(self.result(no_hospital)["failed"], ["training_hospitals"])

    def test_a_child_process_gives_the_same_result(self):
        stale = self.write("stale", tiny(agent_tasks.task_source("stale_head")))
        here = agent_contract.run_contract(stale, self.reference, patch_size=PATCH_SIZE)
        child = agent_contract.run_contract_in_subprocess(
            stale, self.reference, patch_size=PATCH_SIZE
        )
        self.assertEqual(child, here)
        self.assertEqual(child["failed"], ["all_parameters_trained"])
        refused = agent_contract.run_contract_in_subprocess(self.write("bad", "import os\n"))
        self.assertEqual(refused["status"], "refused")

    def test_the_statement_states_what_the_checks_require(self):
        statement = agent_tasks.STATEMENT
        for stated in (
            str(agent_contract.MEAN),
            str(agent_contract.STD),
            "learning rate 1e-4",
            "weight decay 1e-2",
            "hospital (0 to 4) except hospitals 1 and 2",
            "once per epoch or once\n   per batch",
            "use_for_training",
            "prepare_training_batch",
            "prepare_evaluation_batch",
            "`build`",
            "train_one_epoch",
        ):
            self.assertIn(stated, statement)
        self.assertEqual((agent_contract.LEARNING_RATE, agent_contract.WEIGHT_DECAY), (1e-4, 1e-2))
        for name in (*ALL_FAULTS, *(fault.signature for fault in FAULTS.values())):
            self.assertNotIn(name, statement)

    def test_the_ham10000_family_has_its_own_clauses_and_cases(self):
        rows = agent_contract.check_cases(
            patch_size=PATCH_SIZE,
            sources=lambda name: tiny(agent_contract.case_source(name, "ham10000")),
            family="ham10000",
        )
        tasks = agent_tasks.family("ham10000").tasks
        self.assertEqual([row["name"] for row in rows], [*tasks, *agent_contract.HAM_CASES])
        for row in rows:
            with self.subTest(script=row["name"]):
                self.assertEqual(sorted(row["failed"]), sorted(row["expected"]), row["details"])
        clauses = agent_contract.SPECS["ham10000"].clauses
        self.assertEqual(clauses[:2], ("training_sources", "held_out_lesions"))
        failed = {clause for row in rows for clause in row["expected"]}
        self.assertEqual(failed, set(clauses))
        script = self.write(
            "image_key", tiny(agent_tasks.task_source("held_out_by_image", "ham10000"))
        )
        reference = self.write("ham", tiny(agent_tasks.task_source(HEALTHY, "ham10000")))
        result = agent_contract.run_contract(
            script, reference, patch_size=PATCH_SIZE, family="ham10000"
        )
        self.assertEqual((result["family"], result["failed"]), ("ham10000", ["held_out_lesions"]))
        self.assertEqual(
            result["clauses"]["held_out_lesions"]["detail"],
            "two images of one lesion get different keys",
        )
        self.assertEqual(tuple(result["clauses"]), clauses)
        refused = agent_contract.run_contract_in_subprocess(
            self.write("bad", "import os\n"), family="ham10000"
        )
        self.assertEqual((refused["status"], refused["failed"]), ("refused", list(clauses)))
        # A camelyon script has no held_out_key, which the ham10000 contract requires.
        camelyon = agent_contract.run_contract(
            self.write("camelyon", tiny(agent_tasks.task_source(HEALTHY))),
            reference,
            patch_size=PATCH_SIZE,
            family="ham10000",
        )
        self.assertIn("held_out_lesions", camelyon["failed"])

    def test_the_structured_family_meets_the_same_clauses_with_cases_of_its_own(self):
        rows = agent_contract.check_cases(
            patch_size=PATCH_SIZE,
            sources=lambda name: tiny(agent_contract.case_source(name, "camelyon_structured")),
            family="camelyon_structured",
        )
        self.assertEqual([row["name"] for row in rows], [*TASKS, *agent_contract.STRUCTURED_CASES])
        for row in rows:
            with self.subTest(script=row["name"]):
                self.assertEqual(sorted(row["failed"]), sorted(row["expected"]), row["details"])
        self.assertEqual(agent_contract.SPECS["camelyon_structured"].clauses, CLAUSES)

    def test_the_checks_share_no_code_with_the_verifier(self):
        tree = ast.parse(Path(agent_contract.__file__).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module.startswith("runsleuth"):
                imported.update(f"{node.module}.{alias.name}" for alias in node.names)
        self.assertEqual(
            imported,
            {
                "runsleuth.agent_tasks",
                "runsleuth.agent_tasks.FAULTS",
                "runsleuth.agent_tasks.HEALTHY",
                "runsleuth.agent_workspace.ScriptRefused",
                "runsleuth.agent_workspace.check_script",
                "runsleuth.agent_workspace.clean_environment",
                "runsleuth.agent_workspace.load_script",
                "runsleuth.agent_workspace.load_source",
                "runsleuth.agent_workspace.synthetic_task_data",
            },
        )


if __name__ == "__main__":
    unittest.main()
