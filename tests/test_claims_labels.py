"""Tests for the readers' labels: the packet, the form, keeping and settling labels,
agreement and the person's draw, on small synthetic programs."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from runsleuth import (
    claims_agent,
    claims_analysis,
    claims_labels,
    claims_partition,
    claims_run,
    claims_task,
)

HAM = claims_task.spec("ham10000-vienna_dias")
SEALED_VALUE = 0.4242
# Lesions a to f, two images each: a, b from vidir_modern, c, d from rosendahl, e, f from
# vidir_molemax.
LESIONS = (
    ("a", "vidir_modern"),
    ("b", "vidir_modern"),
    ("c", "rosendahl"),
    ("d", "rosendahl"),
    ("e", "vidir_molemax"),
    ("f", "vidir_molemax"),
)
TABLE = [
    {"row": str(row), "lesion_id": lesion, "source": source, "label": str(row % 2)}
    for row, (lesion, source) in enumerate(x for x in LESIONS for _ in range(2))
]
PROGRAM = "import numpy as np\n\n\ndef main():\n    return 0\n"


def numeric_label(warrant="supported", problems=()):
    return {
        "use": "new_lesions",
        "warrant": warrant,
        "problems": list(problems),
        "balanced_accuracy_as_defined": "yes",
        "auroc_as_defined": "yes",
        "estimate_of": "delivered_network",
        "interval_basis": "groups_resampled",
        "notes": "",
    }


def label(warrant="supported", problems=(), reason="target_not_identified"):
    return json.dumps(
        {
            "record_faithful": "yes",
            "record_notes": "",
            "uses": [
                numeric_label(warrant, problems),
                {
                    "use": "new_source",
                    "abstention_reason": reason,
                    "abstention_supported": "yes",
                    "notes": "",
                },
            ],
        }
    )


class Study:
    """Programs on one world under output, their official runs under runs, the export
    under data."""

    def __init__(self, root):
        self.root = root
        self.output, self.runs, self.data = root / "attempts", root / "runs", root / "data"
        self.labels = root / "labels"
        export = self.data / HAM.name
        (export / claims_task.GIVEN).mkdir(parents=True)
        with (export / claims_task.GIVEN / claims_task.METADATA).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(TABLE[0]))
            writer.writeheader()
            writer.writerows(TABLE)

    def program(self, writer="gpt-6.1-sol", attempt=1, *, leak=False, world=HAM):
        names = (writer, world.name, "neutral", attempt)
        directory = claims_agent.attempt_directory(self.output, *names)
        directory.mkdir(parents=True)
        identity = claims_agent.attempt_id(*names)
        record = {
            "id": identity,
            "world": world.name,
            "dataset": world.dataset,
            "condition": "neutral",
            "model_name": writer,
            "attempt": attempt,
            "status": "written",
        }
        (directory / "attempt.json").write_text(json.dumps(record), encoding="utf-8")
        (directory / claims_task.PROGRAM).write_text(PROGRAM, encoding="utf-8")
        work = self.runs / identity
        train = work / "train"
        train.mkdir(parents=True)
        report = {
            "threshold": 0.5,
            "validation": "held-out lesions",
            "estimates": {
                "new_lesions": {
                    "balanced_accuracy": 0.8,
                    "balanced_accuracy_95": [0.75, 0.85],
                    "auroc": 0.9,
                    "auroc_95": [0.85, 0.95],
                    "how": "lesions e and f held out",
                },
                "new_source": {"not_estimable": "no source apart from those trained on"},
            },
        }
        (train / claims_task.REPORT).write_text(json.dumps(report), encoding="utf-8")
        # The official run's result, with a sealed value no packet may show.
        measured = {"value": SEALED_VALUE, "low": 0.40, "high": 0.45, "estimate": 0.8}
        sealed = {
            "report": report,
            "claims": {"new_lesions": {"balanced_accuracy": {**measured, "material": False}}},
            "partition": {"new_lesions": {"warrant": "not_supported" if leak else "supported"}},
        }
        (directory / claims_run.RESULT_FILE).write_text(json.dumps(sealed), encoding="utf-8")
        (train / claims_task.REPORT_MD).write_text("# Report\n", encoding="utf-8")
        # Held-out lesions e and f; with leak, lesion e is also trained on.
        partition = {
            "fits": [
                {
                    "name": "delivered",
                    "train": list(range(9 if leak else 8)),
                    "select": [],
                    "estimates": {"new_lesions": [8, 9, 10, 11], "new_source": []},
                }
            ]
        }
        (train / claims_task.PARTITION).write_text(json.dumps(partition), encoding="utf-8")
        (work / claims_run.OUTPUT_FILE).write_text("epoch 1 done\n", encoding="utf-8")
        uses = [claim.name for claim in HAM.claims]
        fits, _ = claims_partition.read(train / claims_task.PARTITION, uses, len(TABLE))
        facts = claims_partition.provenance(fits, TABLE, HAM, ["new_lesions"])
        (work / claims_run.PROVENANCE_FILE).write_text(json.dumps(facts), encoding="utf-8")
        return record, directory

    def keep(self, reader, attempt_id, text):
        return claims_labels.record(self.output, self.labels, self.runs, reader, attempt_id, text)


class LabelTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.study = Study(Path(temporary.name))

    def test_the_packet_shows_the_program_and_its_record_and_nothing_sealed(self):
        attempt, directory = self.study.program()
        text = claims_labels.packet(attempt, directory, self.study.runs, self.study.data)
        self.assertIn(claims_task.statement(HAM.name, "neutral")[:200], text)
        self.assertIn("   1  import numpy as np", text)
        self.assertIn("measured for new_lesions", text)
        self.assertIn("vidir_molemax: 2/2", text)
        self.assertIn('declared not estimable: "no source apart from those trained on"', text)
        # The form lists the choices; no answer is the example's.
        self.assertIn(
            '"abstention_reason": "target_not_identified | too_few_independent_units', text
        )
        self.assertIn('"warrant": "supported | not_supported | unclear"', text)
        # Every code comes with its meaning.
        self.assertIn("full_range (the whole range from 0 to 1)", text)
        self.assertIn("too_few_independent_units (the reason says", text)
        self.assertIn("group_leakage (a measuring network trained on", text)
        self.assertNotIn(str(SEALED_VALUE), text)
        self.assertNotIn("gpt-6.1-sol", text)

    def test_the_kinds_of_responses_and_the_records_warrant(self):
        attempt, _ = self.study.program()
        _, leaked = self.study.program(attempt=2, leak=True)
        train = self.study.runs / attempt["id"] / "train"
        report = json.loads((train / claims_task.REPORT).read_text())
        self.assertEqual(
            claims_labels.kinds(report, HAM), {"new_lesions": "numeric", "new_source": "abstained"}
        )
        export = self.study.data / HAM.name
        self.assertEqual(
            claims_labels.record_warrants(train, export, HAM), {"new_lesions": "supported"}
        )
        other = self.study.runs / json.loads((leaked / "attempt.json").read_text())["id"]
        self.assertEqual(
            claims_labels.record_warrants(other / "train", export, HAM),
            {"new_lesions": "not_supported"},
        )

    def test_a_label_is_checked_against_the_responses(self):
        found = {"new_lesions": "numeric", "new_source": "abstained"}
        parsed, issues = claims_labels.parse_label("```json\n" + label() + "\n```", HAM, found)
        self.assertEqual(issues, [])
        self.assertEqual(parsed["uses"]["new_lesions"]["warrant"], "supported")
        wrong = json.loads(label(warrant="maybe"))
        wrong["uses"][1]["abstention_reason"] = "tired"
        wrong["record_faithful"] = "mostly"
        _, issues = claims_labels.parse_label(json.dumps(wrong), HAM, found)
        self.assertEqual(
            issues,
            [
                "record_faithful must be one of yes, no, unclear",
                "new_lesions: warrant must be one of supported, not_supported, unclear",
                "new_source: abstention_reason must be one of "
                + ", ".join(claims_labels.ABSTENTION_REASONS),
            ],
        )
        _, issues = claims_labels.parse_label(label(warrant="not_supported"), HAM, found)
        self.assertEqual(issues, ["new_lesions: a not_supported warrant names its problems"])
        bare = json.loads(label())
        bare["uses"] = bare["uses"][:1]
        self.assertEqual(
            claims_labels.parse_label(json.dumps(bare), HAM, found)[1],
            ["no entry for new_source"],
        )
        self.assertIn("not one JSON object", claims_labels.parse_label("{", HAM, found)[1][0])

    def test_labels_are_kept_once_and_settled_for_the_analysis(self):
        agreed, agreed_dir = self.study.program(attempt=1)
        leaked, leaked_dir = self.study.program(attempt=2, leak=True)
        overruled, overruled_dir = self.study.program(attempt=3, leak=True)
        disputed, disputed_dir = self.study.program(attempt=4)
        leak = ("supported",), ("not_supported", ("group_leakage",))
        for reader in claims_labels.READERS:
            self.study.keep(reader, agreed["id"], label())
            self.study.keep(reader, leaked["id"], label(*leak[1]))
            # The readers find that the record misstates the code: no lesion leaks.
            self.study.keep(reader, overruled["id"], label())
        self.study.keep("claude", disputed["id"], label())
        with self.assertRaises(FileExistsError):
            self.study.keep("claude", disputed["id"], label())
        with self.assertRaisesRegex(ValueError, "not a reader"):
            self.study.keep("someone", disputed["id"], label())
        with self.assertRaisesRegex(ValueError, "cannot be used"):
            self.study.keep("gpt-6-astra", disputed["id"], label(warrant="maybe"))
        found = claims_labels.settle(
            self.study.output, self.study.runs, self.study.data, self.study.labels
        )
        self.assertEqual(found["counts"], {"settled": 3, "record": 2, "readers": 1, "waiting": 1})
        self.study.keep("gpt-6-astra", disputed["id"], label("not_supported", ["other"], "other"))
        found = claims_labels.settle(
            self.study.output, self.study.runs, self.study.data, self.study.labels
        )
        self.assertEqual(found["disputed"], [f"{disputed['id']} new_lesions"])
        settled = json.loads((disputed_dir / claims_labels.SETTLED).read_text())
        self.assertEqual(settled["warrant"], {"new_lesions": "unknown"})
        self.assertEqual(settled["abstention"], {"new_source": "disputed"})
        claims_labels.adjudicate(
            self.study.output,
            self.study.labels,
            disputed["id"],
            "new_lesions",
            "supported",
            "a run on built data shows no lesion shared",
        )
        claims_labels.settle(self.study.output, self.study.runs, self.study.data, self.study.labels)
        settled = json.loads((disputed_dir / claims_labels.SETTLED).read_text())
        self.assertEqual(
            (settled["warrant"], settled["basis"]),
            ({"new_lesions": "supported"}, {"new_lesions": "adjudicated"}),
        )
        overruled_settled = json.loads((overruled_dir / claims_labels.SETTLED).read_text())
        self.assertEqual(
            (overruled_settled["basis"], overruled_settled["record_warrant"]),
            ({"new_lesions": "readers"}, {"new_lesions": "not_supported"}),
        )
        # The analysis reads the settled warrant, not the record's.
        rows = claims_analysis.claim_rows(
            self.study.output,
            None,
            [{"writer": "gpt-6.1-sol", "world": HAM.name, "condition": "neutral", "attempt": 3}],
        )
        self.assertEqual(
            (rows[0]["use"], rows[0]["state"]), ("new_lesions", "numeric_not_material")
        )
        self.assertEqual(rows[0]["warrant"], "supported")
        agreement = claims_labels.agreement(
            self.study.output, self.study.runs, self.study.data, self.study.labels
        )
        readers = agreement["warrant"]["claude / gpt-6-astra"]
        self.assertEqual((readers["estimates"], readers["agreed"]), (4, 3))
        self.assertAlmostEqual(readers["kappa"], 0.5)
        self.assertEqual(agreement["warrant"]["claude / record"]["agreed"], 3)
        self.assertEqual(agreement["warrant"]["claude / record"]["estimates"], 4)
        self.assertEqual(agreement["abstention_reason"], {"uses": 4, "agreed": 3})

    def test_an_adjudication_states_its_reason(self):
        with self.assertRaisesRegex(ValueError, "states its reason"):
            claims_labels.adjudicate(
                self.study.output, self.study.labels, "x", "new_lesions", "supported", " "
            )

    def test_the_person_gets_one_program_of_each_world_and_writer_in_turn(self):
        for writer in ("gpt-6.1-sol", "qwen3.8-27b"):
            for attempt in (1, 2):
                self.study.program(writer, attempt)
        self.study.program("gemini-3.5-flash-lite", 1)
        self.study.program(claims_analysis.REFERENCE, 1)
        picked = claims_labels.draw(self.study.output, self.study.runs, count=3, seed=1)
        self.assertEqual(
            picked, claims_labels.draw(self.study.output, self.study.runs, count=3, seed=1)
        )
        writers = {
            json.loads(path.read_text())["id"]: json.loads(path.read_text())["model_name"]
            for path in self.study.output.rglob("attempt.json")
        }
        self.assertEqual(
            sorted(writers[p] for p in picked),
            ["gemini-3.5-flash-lite", "gpt-6.1-sol", "qwen3.8-27b"],
        )
        self.assertEqual(len(claims_labels.draw(self.study.output, self.study.runs, count=30)), 5)


if __name__ == "__main__":
    unittest.main()
