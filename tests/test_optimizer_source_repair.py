"""Source proposals replay saved evidence and never mutate the recorded experiment."""

import ast
import json
import shutil
import unittest
from copy import deepcopy
from pathlib import Path

import test_evaluate_optimizer as comparison_fixtures
import test_optimizer_agent_diagnosis as diagnosis_fixtures
import test_optimizer_evidence as evidence_fixtures

from runsleuth.optimizer_evidence import inspect_optimizer_run
from runsleuth.optimizer_source_repair import prepare_source_proposals, run_source_proposals


class OptimizerSourceRepairTests(unittest.TestCase):
    def setUp(self):
        # Compose fixtures without inheriting and collecting unrelated tests again.
        self.fixture = comparison_fixtures.EvaluateOptimizerTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.manifest = self.fixture.manifest
        self.manifest_path = self.fixture.manifest_path
        self.case, self.agent = self.fixture.case, self.fixture.agent
        self.pair = self.bind_pair(self.case, self.agent, defect=True)
        self.save_manifest()

    def bind_pair(self, case, agent, *, defect):
        pair = (self.root / case["reference_run"]).parent
        reference = pair / "clean"
        candidate = pair / ("stale_head" if defect else "stale_head_repaired")
        (self.root / case["reference_run"]).rename(reference)
        (self.root / case["candidate_run"]).rename(candidate)
        case["reference_run"] = reference.relative_to(self.root).as_posix()
        case["candidate_run"] = candidate.relative_to(self.root).as_posix()
        variants = {}
        for directory in (reference, candidate):
            report = comparison_fixtures.read_json(directory / "run_report.json")
            report.update(variant=directory.name, run_directory=str(directory))
            if directory.name == "stale_head_repaired":
                report["repair"] = {
                    "action": "rebuild_optimizer",
                    "timing": "before_first_optimizer_step",
                    "optimizer_state_entries_before": 0,
                    "model_state_sha256_before": report["initial_state_sha256"],
                    "model_state_sha256_after": report["initial_state_sha256"],
                }
            evidence_fixtures.write_json(directory / "run_report.json", report)
            variants[directory.name] = report
        evidence_fixtures.write_json(
            pair / "optimizer_training_report.json",
            {
                "schema_version": 1,
                "task": "camelyon17_optimizer_rebind_verification",
                "status": "completed",
                "error": None,
                "reference_initial_state_sha256": variants["clean"]["initial_state_sha256"],
                "epochs_per_variant": self.fixture.fixture.config.epochs,
                "variants": variants,
            },
        )
        snapshot = pair / "source_snapshot"
        snapshot.mkdir()
        source_directory = Path(__file__).resolve().parents[1] / "src" / "runsleuth"
        for name in ("optimizer_probe.py", "camelyon_optimizer_training.py"):
            shutil.copyfile(source_directory / name, snapshot / name)
        self.refresh_source_hashes(pair)
        trace = []
        for call_id, name in (
            ("reference-call", case["reference_run"]),
            ("candidate-call", case["candidate_run"]),
        ):
            data = inspect_optimizer_run(
                self.root / name, require_file=self.fixture.fixture.require_file
            )
            data["run_directory"] = name
            entry = diagnosis_fixtures.make_entry(call_id, data)
            entry["result"].update(error_code=None, error_message=None)
            trace.append(entry)
        payload = diagnosis_fixtures.make_payload(trace, defect=defect)
        agent.update(
            tool_trace=trace,
            diagnosis=payload,
            final_text=json.dumps(payload),
            validation_attempts=[
                {
                    "model_call": 2,
                    "raw_final_text": json.dumps(payload),
                    "valid": True,
                    "error": None,
                }
            ],
        )
        self.fixture.save_agent(agent, case)
        return pair

    def refresh_source_hashes(self, pair):
        evidence_fixtures.write_json(
            pair / "environment.json",
            {
                "source_sha256": {
                    path.name: evidence_fixtures.digest(path)
                    for path in (pair / "source_snapshot").glob("*.py")
                }
            },
        )

    def save_manifest(self):
        evidence_fixtures.write_json(self.manifest_path, self.manifest)

    def prepare(self):
        self.save_manifest()
        return prepare_source_proposals(self.manifest_path, workspace_root=self.root)

    def execute(self):
        self.save_manifest()
        return run_source_proposals(self.manifest_path, workspace_root=self.root)

    def input_bytes(self):
        return {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}

    def assert_no_outputs(self):
        self.assertFalse((self.root / "artifacts" / "optimizer_source_proposals").exists())

    def assert_repaired_else_order(self, proposed_source):
        module = ast.parse(proposed_source)
        function = next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "make_probe_optimizer"
        )
        branch = next(
            node
            for node in function.body
            if isinstance(node, ast.If) and ast.unparse(node.test) == "variant == 'clean'"
        )
        self.assertEqual(ast.unparse(branch.orelse[0]), "model.fc = new_head")
        self.assertIsInstance(branch.orelse[1], ast.Assign)
        self.assertEqual(ast.unparse(branch.orelse[1].targets[0]), "optimizer")
        self.assertEqual(ast.unparse(branch.orelse[1].value.func), "torch.optim.AdamW")

    def test_fault_proposes_minimal_order_change_without_writing(self):
        self.save_manifest()
        before = self.input_bytes()
        report = prepare_source_proposals(self.manifest_path, workspace_root=self.root)
        case = report["cases"][0]
        self.assertEqual(case["id"], self.case["id"])
        self.assertEqual(case["decision"], "patch_proposed")
        self.assertTrue(case["source"]["provenance"])
        self.assert_repaired_else_order(case["proposed_patch"]["proposed_source"])
        self.assertEqual(before, self.input_bytes())
        self.assert_no_outputs()

    def test_healthy_repaired_control_has_no_patch(self):
        case, agent = self.fixture.add_case("repaired-control", defect=False)
        self.bind_pair(case, agent, defect=False)
        self.manifest["cases"] = [case]
        row = self.prepare()["cases"][0]
        self.assertEqual(row["decision"], "no_change")
        self.assertIsNone(row["proposed_patch"])

    def test_output_contains_isolated_source_copies_and_preserves_every_input(self):
        self.save_manifest()
        before = self.input_bytes()
        expected_original = (self.pair / "source_snapshot" / "optimizer_probe.py").read_bytes()
        path = run_source_proposals(self.manifest_path, workspace_root=self.root)
        self.assertTrue(path.is_file())
        self.assertTrue(path.is_relative_to(self.root / "artifacts" / "optimizer_source_proposals"))
        self.assertEqual(path.name, "report.json")
        original = path.parent / "cases" / "case-001" / "original" / "optimizer_probe.py"
        self.assertEqual(original.read_bytes(), expected_original)
        proposed = path.parent / "cases" / "case-001" / "proposed" / "optimizer_probe.py"
        self.assert_repaired_else_order(proposed.read_text(encoding="utf-8"))
        self.assertTrue((path.parent / "cases" / "case-001" / "repair.patch").is_file())
        saved = comparison_fixtures.read_json(path)
        self.assertNotIn("proposed_source", saved["cases"][0]["proposed_patch"])
        for original, content in before.items():
            self.assertEqual(original.read_bytes(), content, str(original))

    def test_manifest_ground_truth_does_not_choose_controller_action(self):
        self.case["expected"] = {
            "implementation_status": "no_optimizer_binding_defect",
            "root_cause": None,
            "performance_status": "mixed",
            "recommended_action": "collect_more_evidence",
        }
        row = self.prepare()["cases"][0]
        self.assertEqual(row["decision"], "patch_proposed")

    def test_changed_snapshot_bytes_are_rejected_before_output(self):
        path = self.pair / "source_snapshot" / "optimizer_probe.py"
        path.write_bytes(path.read_bytes() + b"\n# unrecorded edit\n")
        with self.assertRaises(ValueError):
            self.execute()
        self.assert_no_outputs()

    def test_changed_run_artifact_invalidates_saved_agent_evidence(self):
        path = self.root / self.case["candidate_run"] / "config.json"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaises(ValueError):
            self.execute()
        self.assert_no_outputs()

    def test_parent_report_must_contain_the_exact_child_run_report(self):
        path = self.pair / "optimizer_training_report.json"
        parent = comparison_fixtures.read_json(path)
        parent["variants"]["stale_head"]["optimizer_audit"]["foreign_parameter_tensors"] = 3
        evidence_fixtures.write_json(path, parent)
        with self.assertRaises(ValueError):
            self.execute()
        self.assert_no_outputs()

    def test_matching_hash_does_not_accept_unsupported_constructor_source(self):
        path = self.pair / "source_snapshot" / "optimizer_probe.py"
        path.write_text(
            path.read_text(encoding="utf-8").replace("torch.optim.AdamW(", "torch.optim.SGD("),
            encoding="utf-8",
        )
        self.refresh_source_hashes(self.pair)
        with self.assertRaises(ValueError):
            self.execute()
        self.assert_no_outputs()

    def test_matching_hash_does_not_accept_a_different_training_dispatch(self):
        path = self.pair / "source_snapshot" / "camelyon_optimizer_training.py"
        source = path.read_text(encoding="utf-8")
        self.assertIn("variant=variant,", source)
        path.write_text(source.replace("variant=variant,", 'variant="clean",'), encoding="utf-8")
        self.refresh_source_hashes(self.pair)
        with self.assertRaises(ValueError):
            self.execute()
        self.assert_no_outputs()

    def test_recorded_source_is_parsed_without_import_or_execution(self):
        path = self.pair / "source_snapshot" / "optimizer_probe.py"
        path.write_text(
            'raise AssertionError("recorded source must never execute")\n'
            + path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        self.refresh_source_hashes(self.pair)
        row = self.prepare()["cases"][0]
        self.assertEqual(row["decision"], "patch_proposed")
        self.assertIn(
            "recorded source must never execute", row["proposed_patch"]["proposed_source"]
        )

    def test_completed_agent_payload_must_match_validated_response(self):
        self.agent["diagnosis"] = deepcopy(self.agent["diagnosis"])
        self.agent["diagnosis"]["implementation_status"] = "no_optimizer_binding_defect"
        self.fixture.save_agent()
        with self.assertRaises(ValueError):
            self.execute()
        self.assert_no_outputs()

    def test_failed_agent_attempt_cannot_authorize_a_proposal(self):
        self.fixture.save_agent(self.fixture.fail_attempt(self.agent))
        with self.assertRaises(ValueError):
            self.execute()
        self.assert_no_outputs()

    def test_later_case_preflight_failure_leaves_no_partial_proposals(self):
        case, agent = self.fixture.add_case("second-case", defect=True)
        pair = self.bind_pair(case, agent, defect=True)
        path = pair / "source_snapshot" / "camelyon_optimizer_training.py"
        path.write_bytes(path.read_bytes() + b"\n# altered second case\n")
        with self.assertRaises(ValueError):
            self.execute()
        self.assert_no_outputs()


if __name__ == "__main__":
    unittest.main()
