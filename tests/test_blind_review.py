"""CPU tests for the blind review: units, packets that name nothing the annotators must not
know, answer sheets, agreement and the consensus against the labels."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

if torch is not None:
    from runsleuth import blind_review, pipeline_tasks
    from runsleuth.challenge_set import Candidate

FAMILY = "camelyon_pipeline"


def fake_split(root: Path, wrong: int = 6, correct: int = 6):
    """Candidates of a split: per unit two agent repairs with the same code, and an
    evidence-removed run of the first; labels.json beside them."""
    healthy = pipeline_tasks.version_sources(pipeline_tasks.HEALTHY, FAMILY)
    planned, sources, labels = [], {}, {}
    for n in range(wrong + correct):
        label = "wrong" if n < wrong else "correct"
        code = dict(healthy, config=healthy["config"] + f"\nUNIT = {n}\n")
        for repetition in (1, 2):
            candidate = Candidate(
                f"{FAMILY}/start_{n}/model-x/r{repetition}",
                FAMILY,
                "task",
                f"start_{n}",
                "test",
                "agent",
                "repair",
                model="model-x",
                repetition=repetition,
            )
            planned.append(candidate)
            sources[candidate.id] = code
            labels[candidate.id] = {"label": label}
        planned.append(
            Candidate(
                f"{FAMILY}/start_{n}/model-x/r1/no_data_reads",
                FAMILY,
                "task",
                f"start_{n}",
                "test",
                "program_transform",
                "evidence_removed",
                transform="no_data_reads",
                parent=f"{FAMILY}/start_{n}/model-x/r1",
            )
        )
    for split in ("development", "test"):
        path = Path(root) / split / "labels.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"labels": labels}), encoding="utf-8")
    return planned, sources


@unittest.skipIf(torch is None, "Install torch and torchvision for the blind review tests")
class BlindReviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.candidates_root = self.root / "candidates"
        planned, sources = fake_split(self.candidates_root)
        for patcher in (
            patch.object(blind_review.candidate_set, "candidates", lambda _split: planned),
            patch.object(
                blind_review.candidate_set,
                "candidate_sources",
                lambda candidate, _root=None: sources.get(candidate.id),
            ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def packets(self, round_name="test", seed=3):
        return blind_review.write_packets(
            round_name,
            ["author", "second"],
            root=self.root / "review",
            candidates_root=self.candidates_root,
            seed=seed,
        )

    def test_a_unit_is_one_code_with_every_candidate_that_hands_it_in(self):
        found = blind_review.units("test", self.candidates_root)
        self.assertEqual(len(found), 12)
        self.assertTrue(all(len(unit["candidates"]) == 3 for unit in found))
        self.assertEqual(sum(unit["label"] == "wrong" for unit in found), 6)

    def test_packets_name_neither_candidates_nor_labels_and_order_items_apart(self):
        key = self.packets()
        self.assertEqual(sorted(key["items"]), [f"R{n:02d}" for n in range(1, 13)])
        written = json.loads((self.root / "review" / "test" / "key.json").read_text("utf-8"))
        self.assertEqual(written["audit"], blind_review.AUDIT)
        orders = []
        for annotator in ("author", "second"):
            packet = self.root / "review" / "test" / annotator
            text = "\n".join(
                path.read_text(encoding="utf-8-sig")
                for path in sorted(packet.rglob("*"))
                if path.is_file()
            )
            for hidden in ("model-x", "start_", "/r1", "no_data_reads", "## Your job"):
                self.assertNotIn(hidden, text)
            self.assertFalse((packet / "key.json").exists())
            self.assertTrue((packet / "reference" / "camelyon17" / "train.py").exists())
            with (packet / "answers.csv").open(encoding="utf-8-sig") as sheet:
                rows = list(csv.DictReader(sheet))
            self.assertTrue(all(row["verdict"] == "" for row in rows))
            orders.append([row["id"] for row in rows])
        self.assertEqual(sorted(orders[0]), sorted(orders[1]))
        self.assertNotEqual(orders[0], orders[1])
        item = (self.root / "review" / "test" / "author" / "items" / "R01.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("+UNIT = ", item)
        with self.assertRaises(FileExistsError):
            self.packets()

    def test_the_pilot_takes_five_wrong_and_five_correct_units(self):
        key = self.packets("pilot")
        labels = [item["label"] for item in key["items"].values()]
        self.assertEqual((labels.count("wrong"), labels.count("correct")), (5, 5))
        self.assertTrue(all(item_id.startswith("P") for item_id in key["items"]))

    def write_sheet(self, path, verdicts, encoding="utf-8-sig"):
        with Path(path).open("w", encoding=encoding, newline="") as sheet:
            writer = csv.writer(sheet)
            writer.writerow(blind_review.COLUMNS)
            for item_id, verdict in verdicts.items():
                writer.writerow([item_id, "camelyon17", verdict, "", "理由", "6"])

    def test_agreement_comes_before_discussion_and_the_consensus_meets_the_labels(self):
        key = self.packets()
        truth = {item_id: item["label"] for item_id, item in key["items"].items()}
        ids = sorted(truth)
        first = dict(truth)
        second = dict(truth, **{ids[0]: "unsure"})
        consensus = dict(truth, **{ids[1]: "correct" if truth[ids[1]] == "wrong" else "wrong"})
        sheets = self.root / "sheets"
        sheets.mkdir()
        self.write_sheet(sheets / "author.csv", first)
        # Excel on a Chinese Windows saves in GBK.
        self.write_sheet(sheets / "second.csv", second, encoding="gbk")
        self.write_sheet(sheets / "consensus.csv", consensus)
        report = blind_review.score(
            "test",
            [sheets / "author.csv", sheets / "second.csv"],
            sheets / "consensus.csv",
            root=self.root / "review",
        )
        found = report["agreement"]["author / second"]
        self.assertEqual((found["agreed"], found["items"]), (11, 12))
        self.assertEqual(found["disagreements"], [ids[0]])
        self.assertEqual((found["agreed_without_unsure"], found["kappa_without_unsure"]), (11, 1.0))
        self.assertEqual(report["minutes_per_item"], 6.0)
        labels = report["labels"]
        self.assertEqual((labels["agreed_units"], labels["agreed_candidates"]), (11, 33))
        self.assertEqual([item["id"] for item in labels["disputed"]], [ids[1]])
        written = json.loads((self.root / "review" / "test" / "report.json").read_text("utf-8"))
        self.assertEqual(written["audit"], blind_review.AUDIT)

    def test_the_audit_amends_the_freeze_and_keeps_the_main_labels(self):
        audit = blind_review.AUDIT
        self.assertEqual(audit["amends"], "evaluations/freeze.json analysis.labels")
        self.assertIn("never edited", audit["main"])
        self.assertIn("unsure never overturns", audit["confirmed"])
        self.assertIn("nothing trains", audit["sensitivity"])

    def test_a_sheet_must_answer_every_item_once_with_a_known_verdict(self):
        key = self.packets()
        sheet = self.root / "sheet.csv"
        verdicts = {item_id: "correct" for item_id in key["items"]}
        self.write_sheet(sheet, dict(verdicts, R01="maybe"))
        with self.assertRaisesRegex(ValueError, "verdict"):
            blind_review.read_answers(sheet, key)
        del verdicts["R02"]
        self.write_sheet(sheet, verdicts)
        with self.assertRaisesRegex(ValueError, "missing"):
            blind_review.read_answers(sheet, key)


@unittest.skipIf(torch is None, "Install torch and torchvision for the blind review tests")
class KappaTests(unittest.TestCase):
    def test_kappa_corrects_agreement_for_chance(self):
        pairs = [("correct", "correct"), ("correct", "wrong"), ("wrong", "wrong")] + [
            ("wrong", "wrong")
        ]
        self.assertEqual(blind_review.kappa(pairs), 0.5)
        self.assertIsNone(blind_review.kappa([("correct", "correct")] * 3))
        self.assertIsNone(blind_review.kappa([]))

    def test_the_statement_drops_the_agents_instructions(self):
        text = blind_review.statement(FAMILY)
        self.assertIn("## What the project must do", text)
        self.assertNotIn("Your job", text)


if __name__ == "__main__":
    unittest.main()
