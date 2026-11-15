"""Tests for the claims task: what is sealed, what the program is given and is told."""

import csv
import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from runsleuth import claims_task

HAM, CAMELYON = claims_task.SPECS["ham10000"], claims_task.SPECS["camelyon"]


def ham_rows():
    """Lesions of four sources, one to three images each; one lesion in two sources."""
    rows = []
    for source, lesions in (
        ("vidir_modern", 30),
        ("rosendahl", 20),
        ("vidir_molemax", 40),
        ("vienna_dias", 10),
    ):
        for number in range(lesions):
            for copy in range(1 + number % 3):
                rows.append(
                    {
                        "image_id": f"ISIC_{len(rows):07d}",
                        "lesion_id": f"{source}_{number}",
                        "dx": "mel" if number % 4 == 0 else "nv",
                        "age": "" if copy else "50.0",
                        "sex": "female",
                        "localization": "back",
                        "source": source,
                        "label": int(number % 4 == 0),
                    }
                )
    twin = dict(rows[0], image_id="ISIC_9999999", source="vidir_molemax")
    return rows + [twin]


def camelyon_rows():
    """Ten slides in each of four hospitals, with more patches of each label than the cap."""
    rows = []
    for hospital in (0, 1, 3, 4):
        for slide in range(10 * hospital, 10 * hospital + 10):
            for label, count in ((0, 160), (1, 40 + slide)):
                rows += [
                    {
                        "patch_id": len(rows) + n,
                        "hospital": hospital,
                        "slide": slide,
                        "label": label,
                    }
                    for n in range(count)
                ]
    return rows


def groups(rows, chosen, positions):
    return {rows[p][chosen.group] for p in positions}


class PartitionTests(unittest.TestCase):
    def check(self, rows, chosen):
        parts = claims_task.partition(rows, chosen, seed=7)
        given, new_groups, new_site = (parts[part] for part in claims_task.PARTS)
        everything = given + new_groups + new_site + parts["dropped"]
        self.assertEqual(len(everything), len(set(everything)))
        for one, other in ((given, new_groups), (given, new_site), (new_groups, new_site)):
            self.assertFalse(groups(rows, chosen, one) & groups(rows, chosen, other))
        sealed = set(chosen.sealed_sites)
        self.assertTrue(all(rows[p][chosen.site] in sealed for p in new_site))
        self.assertFalse({rows[p][chosen.site] for p in given + new_groups} & sealed)
        for site in {row[chosen.site] for row in rows} - sealed:
            self.assertIn(site, {rows[p][chosen.site] for p in new_groups})
        self.assertEqual(parts, claims_task.partition(rows, chosen, seed=7))
        self.assertNotEqual(parts["given"], claims_task.partition(rows, chosen, seed=8)["given"])
        return parts

    def test_ham_seals_a_source_and_whole_lesions_of_every_other_source(self):
        rows = ham_rows()
        parts = self.check(rows, HAM)
        self.assertEqual(parts["dropped"], [])
        self.assertEqual(len(parts["new_site"]), sum(r["source"] == "vienna_dias" for r in rows))
        sealed = groups(rows, HAM, parts["new_groups"])
        self.assertTrue(0.1 < len(sealed) / len(groups(rows, HAM, range(len(rows)))) < 0.25)
        # The lesion with images in two sources is one lesion, on one side.
        twin = [p for p, row in enumerate(rows) if row["lesion_id"] == rows[0]["lesion_id"]]
        self.assertTrue(set(twin) <= set(parts["given"]) or set(twin) <= set(parts["new_groups"]))

    def test_camelyon_seals_a_hospital_and_two_slides_of_every_other_hospital(self):
        rows = camelyon_rows()
        parts = self.check(rows, CAMELYON)
        self.assertEqual({rows[p]["hospital"] for p in parts["new_site"]}, {1})
        by_hospital = Counter(rows[p]["hospital"] for p in parts["new_groups"])
        slides = {
            hospital: {
                rows[p]["slide"] for p in parts["new_groups"] if rows[p]["hospital"] == hospital
            }
            for hospital in by_hospital
        }
        self.assertEqual({h: len(s) for h, s in slides.items()}, {0: 2, 3: 2, 4: 2})
        kept = Counter(
            (rows[p]["slide"], rows[p]["label"]) for part in claims_task.PARTS for p in parts[part]
        )
        self.assertEqual(max(kept.values()), CAMELYON.cap)
        self.assertEqual(kept[(0, 1)], 40)

    def test_a_group_in_a_sealed_site_leaves_the_given_data(self):
        rows = ham_rows()
        intruder = dict(rows[-2], image_id="ISIC_8888888", source="rosendahl")
        rows.append(intruder)
        parts = claims_task.partition(rows, HAM, seed=7)
        self.assertEqual(parts["dropped"], [len(rows) - 1])
        self.assertNotIn(intruder["lesion_id"], groups(rows, HAM, parts["given"]))

    def test_the_given_order_says_nothing_about_the_groups(self):
        rows = ham_rows()
        given = claims_task.partition(rows, HAM, seed=7)["given"]
        self.assertNotEqual(given, sorted(given))

    def test_published_duplicates_keep_their_lesions_on_one_side(self):
        rows = ham_rows()
        apart = claims_task.partition(rows, HAM, seed=7)
        sealed = groups(rows, HAM, apart["new_groups"])
        inside = next(r for r in rows if r["lesion_id"] in sealed and r["source"] != "vienna_dias")
        outside = next(
            r for r in rows if r["lesion_id"] not in sealed and r["source"] == "rosendahl"
        )
        pairs = [
            (inside["image_id"], outside["image_id"]),
            (outside["image_id"], inside["image_id"]),
        ]
        self.assertEqual(claims_task.merge_units(rows, HAM, pairs), 1)
        parts = self.check(rows, HAM)
        merged = {inside["lesion_id"], outside["lesion_id"]}
        sides = [part for part in claims_task.PARTS if groups(rows, HAM, parts[part]) & merged]
        self.assertEqual(len(sides), 1)
        # The given metadata keep the published lesion_id.
        self.assertEqual(
            {r["lesion_id"] for r in rows if r[claims_task.UNIT] == min(merged)}, merged
        )
        with self.assertRaisesRegex(ValueError, "names an image"):
            claims_task.merge_units(rows, HAM, [("ISIC_nowhere", inside["image_id"])])

    def test_the_duplicate_list_is_read_by_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pairs.csv"
            path.write_text(
                "similarity,from_img,to_img\n0.98,ISIC_0033481.jpg,ISIC_0033421.jpg\n",
                encoding="utf-8",
            )
            self.assertEqual(claims_task.read_duplicates(path), [("ISIC_0033481", "ISIC_0033421")])


class StatementTests(unittest.TestCase):
    def test_pressure_adds_one_sentence_and_changes_nothing_else(self):
        for name in claims_task.SPECS:
            neutral = claims_task.statement(name)
            pressed = claims_task.statement(name, pressure=True)
            self.assertIn(claims_task.PRESSURE, pressed)
            self.assertNotIn(claims_task.PRESSURE.strip(), neutral)
            self.assertEqual(pressed.replace(claims_task.PRESSURE, ""), neutral)

    def test_the_statement_asks_every_claim_and_names_every_column(self):
        for name, chosen in claims_task.SPECS.items():
            text = claims_task.statement(name)
            for claim in chosen.claims:
                self.assertIn(f'"{claim.name}"', text)
                self.assertIn(claim.population, text)
            for column, _ in chosen.columns:
                self.assertIn(f"`{column}`", text)
            for word in ("--predict", claims_task.DELIVERED, claims_task.REPORT, "threshold"):
                self.assertIn(word, text)
            self.assertNotRegex(text, r"\{[a-z_]+\}")

    def test_the_statement_gives_no_hint_of_the_traps(self):
        for name, chosen in claims_task.SPECS.items():
            text = claims_task.statement(name, pressure=True).lower()
            for word in ("leak", "sealed", "independ", "confound", "shortcut", "group"):
                self.assertNotIn(word, text)
            for trap in chosen.traps:
                self.assertNotIn(trap.name.replace("_", " "), text)

    def test_every_trap_names_where_it_is_documented(self):
        for chosen in claims_task.SPECS.values():
            names = [trap.name for trap in chosen.traps]
            self.assertEqual(len(names), len(set(names)))
            for trap in chosen.traps:
                self.assertRegex(trap.source, r"(19|20)\d\d")
                self.assertTrue(trap.what)


class ExportTests(unittest.TestCase):
    def test_the_parts_are_written_aligned_and_recorded(self):
        rows = ham_rows()

        def image(position):
            return np.full((3, HAM.size, HAM.size), position % 251, dtype=np.uint8)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = claims_task.export(rows, image, HAM, root, seed=7, source={"doi": "x"})
            parts = claims_task.partition(rows, HAM, seed=7)
            for part in claims_task.PARTS:
                directory = root / ("given" if part == "given" else f"sealed/{part}")
                images = np.load(directory / claims_task.IMAGES)
                with (directory / claims_task.METADATA).open(encoding="utf-8") as stream:
                    table = list(csv.DictReader(stream))
                self.assertEqual(len(images), len(table), part)
                self.assertEqual([int(r["row"]) for r in table], list(range(len(table))))
                for line, position, pixels in zip(table, parts[part], images, strict=True):
                    self.assertEqual(line["image_id"], rows[position]["image_id"])
                    self.assertEqual(int(line["label"]), rows[position]["label"])
                    self.assertEqual(int(pixels[0, 0, 0]), position % 251)
                found = record["parts"][part]
                self.assertEqual(found["samples"], len(table))
                for name, digest in found["sha256"].items():
                    self.assertEqual(
                        hashlib.sha256((directory / name).read_bytes()).hexdigest(), digest
                    )
            self.assertEqual(
                sorted(p.name for p in (root / "given").iterdir()), ["images.npy", "metadata.csv"]
            )
            stored = json.loads((root / claims_task.RECORD).read_text(encoding="utf-8"))
            self.assertEqual(
                stored["claims"], {"new_lesions": "new_groups", "new_source": "new_site"}
            )
            self.assertEqual(len(stored["traps"]), len(HAM.traps))
            self.assertEqual(stored["source"], {"doi": "x"})

    def test_images_of_the_wrong_size_are_refused(self):
        rows = camelyon_rows()
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "shape"):
                claims_task.export(
                    rows, lambda p: np.zeros((3, 8, 8), dtype=np.uint8), CAMELYON, Path(temporary)
                )


if __name__ == "__main__":
    unittest.main()
