"""Source-only tests for capture order, conservative refusal and portable diffs."""

import ast
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from runsleuth.optimizer_factory_analysis import analyze_optimizer_factory, bind_validated_factory
from runsleuth.optimizer_factory_cases import ORACLE_SOURCE, factory_cases


class FactoryAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.cases = {case["id"]: case for case in factory_cases()}

    def analyze(self, name):
        return analyze_optimizer_factory(self.cases[name]["source"])

    def test_all_versioned_cases_classify_without_executing(self):
        decisions = []
        for case in self.cases.values():
            with self.subTest(case=case["id"]):
                result = analyze_optimizer_factory(case["source"])
                self.assertEqual(result["decision"], case["expected_decision"], result["reason"])
                decisions.append(result["decision"])
        self.assertEqual(decisions.count("patch_proposed"), 4)
        self.assertEqual(decisions.count("no_change"), 4)
        self.assertEqual(decisions.count("unsupported"), 4)

    def test_cache_binding_site_precedes_constructor_even_when_constructor_follows_head(self):
        result = self.analyze("eager_parameter_list_before_head")
        self.assertIn("saved_parameters = list", result["binding_site"]["code"])
        self.assertLess(result["binding_site"]["line"], result["replacement_site"]["line"])
        self.assertLess(result["replacement_site"]["line"], result["constructor_site"]["line"])
        self.assertEqual(
            analyze_optimizer_factory(result["proposed_source"])["decision"], "no_change"
        )

    def test_lazy_iterators_are_controls_not_eager_snapshots(self):
        for name in (
            "healthy_unconsumed_parameter_iterator",
            "healthy_unconsumed_iterator_in_group",
        ):
            with self.subTest(case=name):
                result = self.analyze(name)
                self.assertEqual(result["decision"], "no_change")
                self.assertIsNone(result["proposed_source"])
                self.assertIsNone(result["unified_diff"])

    def test_proposal_changes_only_one_statement_position(self):
        for case in self.cases.values():
            if case["expected_decision"] != "patch_proposed":
                continue
            with self.subTest(case=case["id"]):
                result = analyze_optimizer_factory(case["source"])
                original = ast.parse(case["source"])
                proposed = ast.parse(result["proposed_source"])
                left = next(node for node in original.body if isinstance(node, ast.FunctionDef))
                right = next(node for node in proposed.body if isinstance(node, ast.FunctionDef))
                self.assertCountEqual(
                    [ast.dump(node) for node in left.body], [ast.dump(node) for node in right.body]
                )
                self.assertEqual(ast.dump(original.body[0]), ast.dump(proposed.body[0]))
                self.assertEqual(
                    analyze_optimizer_factory(result["proposed_source"])["decision"], "no_change"
                )

    def test_alias_names_and_spacing_do_not_supply_the_classification(self):
        source = self.cases["eager_parameter_group_before_head"]["source"]
        source = (
            source.replace("model", "network")
            .replace("groups", "bucket")
            .replace("optimizer", "updater")
        )
        source = source.replace("import torch", "import torch as tensorlib").replace(
            "torch.optim", "tensorlib.optim"
        )
        result = analyze_optimizer_factory(source)
        self.assertEqual(result["decision"], "patch_proposed")
        self.assertIn("network.fc", result["proposed_source"])

    def test_materialized_alias_stays_bound_to_old_parameters(self):
        source = self.cases["eager_parameter_list_before_head"]["source"]
        source = source.replace(
            "    model.fc = next_head", "    copied = saved_parameters\n    model.fc = next_head"
        )
        source = source.replace("AdamW(saved_parameters,", "AdamW(copied,")
        result = analyze_optimizer_factory(source)
        self.assertEqual(result["decision"], "patch_proposed")
        self.assertIn("saved_parameters = list", result["binding_site"]["code"])

    def test_consumed_iterators_and_their_aliases_are_refused(self):
        source = self.cases["healthy_unconsumed_parameter_iterator"]["source"]
        for middle in (
            "    used = list(parameters)\n",
            "    alias = parameters\n    used = tuple(alias)\n",
        ):
            with self.subTest(middle=middle):
                result = analyze_optimizer_factory(
                    source.replace("    model.fc", middle + "    model.fc")
                )
                self.assertEqual(result["decision"], "unsupported")
                self.assertIn("consumed", result["reason"])

    def test_side_effects_defaults_decorators_reassignment_and_unknown_calls_refused(self):
        source = self.cases["direct_constructor_before_head"]["source"]
        changes = (
            source + "\nraise RuntimeError('not executed')\n",
            source.replace("def make_optimizer", "@decorate\ndef make_optimizer"),
            source.replace("weight_decay):", "weight_decay=0.01):"),
            source.replace(
                "    return optimizer", "    optimizer = optimizer\n    return optimizer"
            ),
            source.replace("torch.optim.AdamW", "torch.optim.SGD"),
            source.replace("deepcopy(model.fc)", "build_head(model.fc)"),
            source.replace("import torch", "import os"),
            source.replace("    next_head =", "    list = model.parameters()\n    next_head ="),
        )
        for changed in changes:
            with self.subTest(source=changed):
                self.assertEqual(analyze_optimizer_factory(changed)["decision"], "unsupported")

    def test_unknowns_cannot_be_compiled_by_the_execution_adapter(self):
        for case in self.cases.values():
            if case["expected_decision"] == "unsupported":
                with self.subTest(case=case["id"]), self.assertRaises(ValueError):
                    bind_validated_factory(case["source"], object())

    def test_annotations_are_not_executed_and_module_imports_are_trusted_bindings(self):
        class Model:
            def __init__(self):
                self.fc = object()

            def parameters(self):
                yield self.fc

        constructed = []

        def fake_adamw(parameters, *, lr, weight_decay):
            value = (tuple(parameters), lr, weight_decay)
            constructed.append(value)
            return value

        fake_torch = SimpleNamespace(optim=SimpleNamespace(AdamW=fake_adamw))
        source = ORACLE_SOURCE.replace("model,", "model: explode(),").replace(
            "weight_decay):", "weight_decay) -> explode():"
        )
        factory = bind_validated_factory(source, fake_torch)
        model = Model()
        result = factory(model, variant="ignored", learning_rate=0.1, weight_decay=0.01)
        self.assertIs(result[0][0], model.fc)
        self.assertEqual(len(constructed), 1)

    def test_malformed_and_semicolon_edits_are_refused(self):
        for source in (
            None,
            "",
            "not python (",
            "x" * (128 * 1024 + 1),
            ORACLE_SOURCE.replace("\n", "\r"),
        ):
            with self.subTest(source=repr(source)[:30]):
                self.assertEqual(analyze_optimizer_factory(source)["decision"], "unsupported")
        source = self.cases["direct_constructor_before_head"]["source"]
        source = source.replace("\n    model.fc = next_head", "; model.fc = next_head")
        self.assertEqual(analyze_optimizer_factory(source)["decision"], "unsupported")

    @unittest.skipUnless(shutil.which("git"), "Git is required for exact diff application")
    def test_diffs_preserve_lf_crlf_unicode_and_unterminated_final_lines(self):
        for ending in ("\n", "\r\n"):
            for final_newline in (True, False):
                with self.subTest(ending=repr(ending), final_newline=final_newline):
                    source = self.cases["eager_parameter_group_before_head"]["source"].replace(
                        "    groups =",
                        "    # Comment contains Unicode separator \u2028 and 中文.\n    groups =",
                    )
                    source = source.rstrip("\n").replace("\n", ending) + (
                        ending if final_newline else ""
                    )
                    result = analyze_optimizer_factory(source)
                    self.assertEqual(result["decision"], "patch_proposed", result["reason"])
                    with tempfile.TemporaryDirectory() as temporary:
                        root = Path(temporary)
                        target, patch = root / "optimizer_probe.py", root / "repair.patch"
                        target.write_bytes(source.encode("utf-8"))
                        patch.write_bytes(result["unified_diff"].encode("utf-8"))
                        applied = subprocess.run(
                            [
                                "git",
                                "-c",
                                "core.autocrlf=false",
                                "-c",
                                "core.eol=lf",
                                "apply",
                                str(patch),
                            ],
                            cwd=root,
                            capture_output=True,
                            text=True,
                            check=False,
                        )
                        self.assertEqual(applied.returncode, 0, applied.stderr)
                        self.assertEqual(
                            target.read_bytes(), result["proposed_source"].encode("utf-8")
                        )


if __name__ == "__main__":
    unittest.main()
