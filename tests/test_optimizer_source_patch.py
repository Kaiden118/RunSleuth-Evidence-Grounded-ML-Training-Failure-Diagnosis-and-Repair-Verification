"""The controlled source proposal must be precise, inert and fail closed."""

import ast
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from runsleuth.optimizer_source_patch import propose_optimizer_order_patch

SOURCE_PATH = Path(__file__).resolve().parents[1] / "src/runsleuth/optimizer_probe.py"


class OptimizerSourcePatchTests(unittest.TestCase):
    def setUp(self):
        self.source = SOURCE_PATH.read_text(encoding="utf-8")

    def test_current_fixture_is_localized_and_only_stale_order_changes(self):
        result = propose_optimizer_order_patch(self.source)
        self.assertEqual(result["function"], "make_probe_optimizer")
        self.assertEqual(result["branch_condition"], 'variant != "clean"')
        self.assertLess(result["constructor"]["line"], result["head_replacement"]["line"])
        for field in ("constructor", "head_replacement"):
            site = result[field]
            physical_lines = self.source.splitlines()[site["line"] - 1 : site["end_line"]]
            self.assertEqual("\n".join(physical_lines).strip(), site["code"].strip())
        original = ast.parse(self.source)
        proposed = ast.parse(result["proposed_source"])
        target = next(
            node
            for node in original.body
            if isinstance(node, ast.FunctionDef) and node.name == "make_probe_optimizer"
        )
        target.body[-2].orelse.reverse()
        self.assertEqual(ast.dump(original), ast.dump(proposed))
        self.assertIn("--- a/optimizer_probe.py\n", result["unified_diff"])
        self.assertIn("+++ b/optimizer_probe.py\n", result["unified_diff"])
        self.assertEqual(SOURCE_PATH.read_text(encoding="utf-8"), self.source)

    def test_multiline_arguments_and_comments_are_preserved(self):
        constructor = (
            "        optimizer = torch.optim.AdamW(\n"
            "            model.parameters(), lr=learning_rate, weight_decay=weight_decay\n"
            "        )\n"
        )
        reformatted = (
            "        optimizer = torch.optim.AdamW(\n"
            "            model.parameters(),\n"
            "            lr=learning_rate,  # exact learning rate\n"
            "            weight_decay=weight_decay,\n"
            "        )  # constructor comment\n"
        )
        source = self.source.replace(constructor, reformatted)
        source = source.replace(
            "    else:\n" + reformatted + "        model.fc = new_head\n",
            "    else:\n"
            + reformatted
            + "        # Between the two statements.\n"
            + "        model.fc = new_head  # replacement comment\n",
            1,
        )
        result = propose_optimizer_order_patch(source)
        expected = source.replace(
            reformatted
            + "        # Between the two statements.\n"
            + "        model.fc = new_head  # replacement comment\n",
            "        model.fc = new_head  # replacement comment\n"
            + "        # Between the two statements.\n"
            + reformatted,
            1,
        )
        self.assertEqual(result["proposed_source"], expected)

    def test_crlf_and_unterminated_final_line_are_preserved(self):
        for ending in ("\n", "\r\n"):
            for final_newline in (True, False):
                with self.subTest(ending=repr(ending), final_newline=final_newline):
                    source = self.source.rstrip("\n").replace("\n", ending)
                    if final_newline:
                        source += ending
                    proposed = propose_optimizer_order_patch(source)["proposed_source"]
                    self.assertEqual(proposed.count(ending), source.count(ending))
                    self.assertEqual(proposed.endswith(ending), final_newline)
                    if ending == "\r\n":
                        self.assertNotIn("\n", proposed.replace("\r\n", ""))

    def test_changed_constructor_arguments_or_control_flow_are_rejected(self):
        mutations = {
            "different arguments": (
                "            model.parameters(), lr=learning_rate, weight_decay=weight_decay\n",
                "            model.parameters(), lr=0.1, weight_decay=weight_decay\n",
            ),
            "different optimizer": ("torch.optim.AdamW(", "torch.optim.Adam("),
            "different condition": ('if variant == "clean":', 'if variant == "stale_head":'),
            "extra stale statement": (
                "    else:\n        optimizer =",
                "    else:\n        model.eval()\n        optimizer =",
            ),
            "prior rebuild": (
                "    new_head = deepcopy(model.fc)\n",
                "    optimizer = torch.optim.AdamW(model.parameters())\n"
                "    new_head = deepcopy(model.fc)\n",
            ),
            "changed return": ("    return optimizer\n", "    return None\n"),
            "changed copy": ("new_head = deepcopy(model.fc)", "new_head = model.fc"),
            "changed signature": ("    variant: str,\n", '    variant: str = "clean",\n'),
        }
        for label, (old, new) in mutations.items():
            with self.subTest(label=label):
                self.assertIn(old, self.source)
                with self.assertRaises(ValueError):
                    propose_optimizer_order_patch(self.source.replace(old, new, 1))

    def test_semicolons_inline_statements_and_decorators_are_rejected(self):
        inline = (
            "    else: optimizer = torch.optim.AdamW(\n"
            "        model.parameters(), lr=learning_rate, weight_decay=weight_decay\n"
            "    ); model.fc = new_head\n"
        )
        old = (
            "    else:\n"
            "        optimizer = torch.optim.AdamW(\n"
            "            model.parameters(), lr=learning_rate, weight_decay=weight_decay\n"
            "        )\n"
            "        model.fc = new_head\n"
        )
        variants = [
            self.source.replace(old, inline, 1),
            self.source.replace(
                "def make_probe_optimizer(", "@decorator\ndef make_probe_optimizer(", 1
            ),
            self.source.replace("def make_probe_optimizer(", "async def make_probe_optimizer(", 1),
            self.source + "\ndef make_probe_optimizer():\n    pass\n",
        ]
        for source in variants:
            with self.subTest(source_suffix=source[-80:]):
                with self.assertRaises(ValueError):
                    propose_optimizer_order_patch(source)

    def test_source_is_not_executed_and_unrelated_text_is_untouched(self):
        prefix = 'raise RuntimeError("Source must never execute")\n'
        suffix = '\nraise RuntimeError("Neither imports nor calls should execute")\n'
        source = prefix + self.source + suffix
        proposed = propose_optimizer_order_patch(source)["proposed_source"]
        self.assertTrue(proposed.startswith(prefix))
        self.assertTrue(proposed.endswith(suffix))

    @unittest.skipUnless(shutil.which("git"), "Git is required to check the emitted diff")
    def test_emitted_diff_applies_exactly_to_original_bytes(self):
        # Keep the changed function at EOF to exercise a diff hunk containing
        # an unterminated final line as well as Windows CRLF source.
        tree = ast.parse(self.source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "make_probe_optimizer"
        )
        function_source = ast.get_source_segment(self.source, function)
        function_source = function_source.replace(
            "    else:\n", "    else:\n        # Preserve Unicode separator: \u2028 here.\n", 1
        )
        for ending in ("\n", "\r\n"):
            for final_newline in (True, False):
                with self.subTest(ending=repr(ending), final_newline=final_newline):
                    source = function_source.replace("\n", ending)
                    if final_newline:
                        source += ending
                    result = propose_optimizer_order_patch(source)
                    with tempfile.TemporaryDirectory() as temporary:
                        root = Path(temporary)
                        target = root / "optimizer_probe.py"
                        patch = root / "proposal.patch"
                        target.write_bytes(source.encode("utf-8"))
                        patch.write_bytes(result["unified_diff"].encode("utf-8"))
                        command = subprocess.run(
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
                        self.assertEqual(command.returncode, 0, command.stderr)
                        self.assertEqual(
                            target.read_bytes(), result["proposed_source"].encode("utf-8")
                        )


if __name__ == "__main__":
    unittest.main()
