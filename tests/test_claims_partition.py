"""Tests for a program's partition record: its form, and the errors it shows."""

import json
import tempfile
import unittest
from pathlib import Path

from runsleuth import claims_partition, claims_task

HAM = claims_task.SPECS["ham10000"]
USES = [claim.name for claim in HAM.claims]

# Twelve rows: lesions a to f, two images each; a to d from vidir_modern and rosendahl,
# e and f from vidir_molemax.
TABLE = [
    {"row": str(row), "lesion_id": lesion, "source": source}
    for row, (lesion, source) in enumerate(
        (lesion, source)
        for lesion, source in (
            ("a", "vidir_modern"),
            ("b", "vidir_modern"),
            ("c", "rosendahl"),
            ("d", "rosendahl"),
            ("e", "vidir_molemax"),
            ("f", "vidir_molemax"),
        )
        for _ in range(2)
    )
]


def fit(name, train, select=(), **estimates):
    return {"name": name, "train": list(train), "select": list(select), "estimates": estimates}


class ReadTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / claims_task.PARTITION

    def read(self, record):
        self.path.write_text(record if isinstance(record, str) else json.dumps(record))
        return claims_partition.read(self.path, USES, len(TABLE))

    def test_a_record_as_the_statement_asks_gives_its_fits(self):
        fits, problems = self.read(
            {"fits": [fit("delivered", range(8), [8], new_lesions=[9, 10]), fit("fold", [0])]}
        )
        self.assertEqual(problems, [])
        self.assertEqual([found.name for found in fits], ["delivered", "fold"])
        self.assertEqual(fits[0].select, {8})
        self.assertEqual(fits[0].estimates, {"new_lesions": {9, 10}})
        self.assertEqual(fits[1].estimates, {})

    def test_a_malformed_record_gives_no_fits_and_says_why(self):
        self.assertEqual(claims_partition.read(self.path, USES, 12), (None, ["no partition.json"]))
        self.assertIn("not JSON", self.read("{")[1][0])
        self.assertEqual(self.read({"fits": []}), (None, ["fits is not a list of fits"]))
        fits, problems = self.read(
            {
                "fits": [
                    {"train": [0]},
                    fit("delivered", [0, 12], select=[True], new_lesions=[]),
                    fit("x", [0], other=[1]),
                    fit("x", [0]),
                ]
            }
        )
        self.assertIsNone(fits)
        self.assertEqual(
            problems,
            [
                "fit 0 has no name",
                "fit delivered: train is not a list of rows of the table",
                "fit delivered: select is not a list of rows of the table",
                "fit delivered: the rows of new_lesions are not a list of rows",
                "fit x: 'other' is not a use",
            ],
        )
        self.assertEqual(
            self.read({"fits": [fit("x", [0]), fit("x", [1])]})[1], ["two fits are named x"]
        )
        self.assertEqual(self.read({"fits": [fit("x", [0])]})[1], ["no fit is named delivered"])


class ErrorTests(unittest.TestCase):
    def errors(self, *fits, estimated=tuple(USES)):
        found = [
            claims_partition.Fit(
                item["name"],
                frozenset(item["train"]),
                frozenset(item["select"]),
                {use: frozenset(rows) for use, rows in item["estimates"].items()},
            )
            for item in fits
        ]
        return claims_partition.errors(found, TABLE, HAM, list(estimated))

    def test_held_out_lesions_and_a_held_out_source_show_no_error(self):
        found = self.errors(
            fit("delivered", range(12), [0, 1]),
            fit("lesions", range(8), [0], new_lesions=[8, 9, 10, 11]),
            fit("source", range(8), [2], new_source=[8, 9, 10, 11]),
        )
        self.assertEqual(found["new_lesions"]["fits"], ["lesions"])
        for use in USES:
            self.assertEqual(found[use]["group_leakage"], [])
            self.assertEqual(found[use]["selection_on_estimate_data"], [])
            self.assertTrue(found[use]["estimate_not_of_the_delivered_network"])
        self.assertEqual(found["new_source"]["new_site_from_known_sites"], [])
        self.assertIsNone(found["new_lesions"]["new_site_from_known_sites"])

    def test_the_record_shows_leakage_known_sites_and_choices_on_estimate_rows(self):
        # Trained on one image of lesion e; measured on both images of e and f, and chosen
        # on f's: a split by image, a new source measured on a known one, selection.
        found = self.errors(
            fit("delivered", [0, 1, 2, 3, 8], [10, 11], new_lesions=[9, 10, 11], new_source=[9])
        )
        lesions, source = found["new_lesions"], found["new_source"]
        self.assertEqual(
            lesions["group_leakage"], [{"fit": "delivered", "groups": 1, "rows": 1, "of": 3}]
        )
        self.assertEqual(
            lesions["selection_on_estimate_data"], [{"fit": "delivered", "rows": 2, "of": 3}]
        )
        self.assertFalse(lesions["estimate_not_of_the_delivered_network"])
        self.assertEqual(
            source["new_site_from_known_sites"], [{"fit": "delivered", "sites": ["vidir_molemax"]}]
        )

    def test_an_estimate_the_record_does_not_measure_is_not_recorded(self):
        found = self.errors(fit("delivered", range(8), new_lesions=[8]), estimated=["new_source"])
        self.assertEqual(found, {"new_source": {"recorded": False}})


if __name__ == "__main__":
    unittest.main()
