"""HAM10000 dermatoscopic images for workspace scripts.

HAM10000 (Tschandl, Rosendahl and Kittler, Scientific Data 2018) holds 10,015
dermatoscopic images of pigmented skin lesions from four sources, each image with the
lesion it shows. It is published on Harvard Dataverse (doi:10.7910/DVN/DBW86T) under
CC BY-NC 4.0. prepare() checks the two image archives and the metadata against the
published MD5 sums, reads every image straight out of its archive without extracting
anything, crops it to its central square, resizes it, and stores all images once as
one uint8 array beside a table of their metadata.

The task is binary: malignant (melanoma, basal cell carcinoma, actinic keratosis and
intraepithelial carcinoma) against benign. Sources stand where Camelyon17's hospitals
stand: one is held out for out-of-distribution validation, one is sealed and never
loaded for training, the others train. In-distribution validation holds out a share
of the training sources' lesions and evaluates one image of each, so its images are
not near-duplicates of one another. Their other images, of the same lesions, must stay
out of training too. Which images a script trains on depends on two of its functions:
use_for_training(source) admits sources, and a held-out key decides which images are
held out with the validation images. Keyed by lesion, every image of a validation
lesion stays out of training; keyed by image, only the validation images themselves
do, and the other images of their lesions are trained on. The validation sets never
depend on the script.

    python -m runsleuth.ham10000_data prepare --config configs/ham10000.json
    python -m runsleuth.ham10000_data summary --config configs/ham10000.json
    python -m runsleuth.ham10000_data audit --config configs/ham10000.json \
        --output evaluations/ham10000_audit.json
"""

import argparse
import csv
import hashlib
import io
import json
import re
import zipfile
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

DOI = "doi:10.7910/DVN/DBW86T"
LICENSE = "CC BY-NC 4.0"
CITATION = (
    "Tschandl P, Rosendahl C, Kittler H. The HAM10000 dataset, a large collection of "
    "multi-source dermatoscopic images of common pigmented skin lesions. Sci Data 5, "
    "180161 (2018)."
)
# The published files and their MD5 sums, as Harvard Dataverse lists them.
PUBLISHED = {
    "HAM10000_images_part_1.zip": "4639bfa73ab251610530a97c898e6e46",
    "HAM10000_images_part_2.zip": "da43d6cc50f6613013be07e8986b384b",
    "HAM10000_metadata.csv": "8f85fb1aa29d80a2797247e434deb79d",
}
MALIGNANT = frozenset({"mel", "bcc", "akiec"})
# The pairs of splits an audit checks for shared images and lesions.
AUDIT_PAIRS = (
    ("training", "id validation"),
    ("training", "ood validation"),
    ("training", "sealed"),
    ("id validation", "ood validation"),
)
IMAGE_NAME = re.compile(r"(?:[^/]+/)?(ISIC_\d{7})\.jpg")
SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class HamConfig:
    """Where HAM10000 lies, how it is prepared and how a script's data are drawn from it."""

    data_dir: str
    sources: dict[str, int]
    ood_source: str
    sealed_source: str
    image_size: int = 128
    subset_seed: int = 2026
    id_val_fraction: float = 0.15
    max_ood_val_samples: int | None = 1000
    batch_size: int = 32
    num_workers: int = 0

    def __post_init__(self) -> None:
        if sorted(self.sources.values()) != list(range(len(self.sources))):
            raise ValueError("sources must number the sources 0, 1, 2, ...")
        for name in (self.ood_source, self.sealed_source):
            if name not in self.sources:
                raise ValueError(f"{name!r} is not one of the sources")
        if self.ood_source == self.sealed_source:
            raise ValueError("the out-of-distribution and the sealed source must differ")
        if not 0 < self.id_val_fraction < 1:
            raise ValueError("id_val_fraction must lie between 0 and 1")
        for name in ("image_size", "batch_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8-sig")))

    @property
    def download_dir(self) -> Path:
        return Path(self.data_dir) / "download"

    @property
    def cache_dir(self) -> Path:
        return Path(self.data_dir) / f"prepared_{self.image_size}"


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_metadata(path: Path) -> list[dict[str, Any]]:
    """The published table, one row per image, sorted by image."""
    with Path(path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return sorted(
        (
            {
                "image_id": row["image_id"],
                "lesion_id": row["lesion_id"],
                "dx": row["dx"],
                "label": int(row["dx"] in MALIGNANT),
                "source": row["dataset"],
            }
            for row in rows
        ),
        key=lambda row: row["image_id"],
    )


def _square(image: Any, size: int) -> Any:
    """The image's central square, resized to size x size."""
    from PIL import Image

    width, height = image.size
    side = min(width, height)
    left, top = (width - side) // 2, (height - side) // 2
    square = image.crop((left, top, left + side, top + side))
    return square.resize((size, size), Image.Resampling.LANCZOS)


def prepare(config: HamConfig, *, published: dict[str, str] = PUBLISHED) -> Path:
    """Check the published files, read and shrink every image, and store them once."""
    import numpy as np
    from PIL import Image

    download = config.download_dir
    checksums = {name: md5(download / name) for name in published}
    wrong = sorted(name for name, digest in checksums.items() if digest != published[name])
    if wrong:
        raise ValueError(f"MD5 differs from the published one for {', '.join(wrong)}")
    rows = read_metadata(download / "HAM10000_metadata.csv")
    unknown = sorted({row["source"] for row in rows} - set(config.sources))
    if unknown:
        raise ValueError(f"the metadata names sources the config does not: {unknown}")
    members: dict[str, tuple[str, str]] = {}
    for archive in sorted(name for name in published if name.endswith(".zip")):
        with zipfile.ZipFile(download / archive) as opened:
            for name in opened.namelist():
                if match := IMAGE_NAME.fullmatch(name):
                    members[match.group(1)] = (archive, name)
    missing = [row["image_id"] for row in rows if row["image_id"] not in members]
    if missing:
        raise ValueError(f"{len(missing)} images of the metadata are in no archive")
    size = config.image_size
    images = np.zeros((len(rows), size, size, 3), dtype=np.uint8)
    opened_archives: dict[str, zipfile.ZipFile] = {}
    try:
        for index, row in enumerate(rows):
            archive, name = members[row["image_id"]]
            if archive not in opened_archives:
                opened_archives[archive] = zipfile.ZipFile(download / archive)
            data = opened_archives[archive].read(name)
            with Image.open(io.BytesIO(data)) as image:
                images[index] = np.asarray(_square(image.convert("RGB"), size))
    finally:
        for opened in opened_archives.values():
            opened.close()
    cache = config.cache_dir
    cache.mkdir(parents=True, exist_ok=True)
    np.save(cache / "images.npy", images)
    (cache / "metadata.json").write_text(json.dumps(rows) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task": "ham10000_prepared",
        "dataset": "HAM10000",
        "doi": DOI,
        "license": LICENSE,
        "citation": CITATION,
        "published_md5": checksums,
        "prepared_at": datetime.now(UTC).isoformat(),
        "image_size": size,
        "preprocessing": "RGB, central square crop, Lanczos resize, uint8",
        "images": len(rows),
        "images_sha256": hashlib.sha256(images.tobytes()).hexdigest(),
        "metadata_sha256": hashlib.sha256((cache / "metadata.json").read_bytes()).hexdigest(),
        "label": "malignant (mel, bcc, akiec) = 1, benign = 0",
    }
    (cache / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return cache


@dataclass(frozen=True)
class HamData:
    images: Any  # uint8, images x height x width x 3
    rows: list[dict[str, Any]]
    manifest: dict[str, Any]


def load(config: HamConfig) -> HamData:
    """The prepared images, read into memory (about 0.5 GB at 128x128), with their metadata."""
    import numpy as np

    cache = config.cache_dir
    if not (cache / "manifest.json").is_file():
        raise FileNotFoundError(f"{cache} holds no prepared HAM10000; run prepare first")
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    rows = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
    images = np.load(cache / "images.npy")
    if len(rows) != len(images) or manifest["image_size"] != config.image_size:
        raise ValueError(f"{cache} does not match the config; prepare it again")
    return HamData(images=images, rows=rows, manifest=manifest)


def _fraction(key: str, seed: int) -> float:
    """A fixed number in [0, 1) for a key and seed, the same on every machine."""
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def by_lesion(_image_id: str, lesion_id: str) -> str:
    """The held-out key of a correct script: every image of a held-out lesion."""
    return lesion_id


def evaluation_split(config: HamConfig, rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    """The validation images, fixed by the data alone.

    in-distribution: one image of each lesion in a fixed share of the training sources'
    lesions; out-of-distribution: a label-stratified sample of that source's images.
    """
    training_sources = set(config.sources) - {config.ood_source, config.sealed_source}
    first: dict[str, int] = {}
    for index, row in enumerate(rows):
        lesion = row["lesion_id"]
        if (
            row["source"] in training_sources
            and _fraction(lesion, config.subset_seed) < config.id_val_fraction
        ):
            rank = _fraction(row["image_id"], config.subset_seed)
            if lesion not in first or rank < _fraction(
                rows[first[lesion]]["image_id"], config.subset_seed
            ):
                first[lesion] = index
    id_val = list(first.values())
    pool = [index for index, row in enumerate(rows) if row["source"] == config.ood_source]
    cap = config.max_ood_val_samples
    if cap is not None and cap < len(pool):
        strata: dict[int, list[int]] = {}
        for index in pool:
            strata.setdefault(rows[index]["label"], []).append(index)
        chosen: list[int] = []
        for _label, members in sorted(strata.items()):
            members.sort(key=lambda index: _fraction(rows[index]["image_id"], config.subset_seed))
            chosen += members[: round(cap * len(members) / len(pool))]
        pool = chosen
    return {"id": sorted(id_val), "ood": sorted(pool)}


def training_split(
    config: HamConfig,
    rows: list[dict[str, Any]],
    use_for_training: Callable[[int], bool],
    held_out_key: Callable[[str, str], str] = by_lesion,
) -> list[int]:
    """The images a script trains on: admitted sources, minus whatever it holds out."""
    evaluation = evaluation_split(config, rows)
    evaluated = set(evaluation["id"]) | set(evaluation["ood"])
    held_out = {held_out_key(rows[i]["image_id"], rows[i]["lesion_id"]) for i in evaluated}
    admitted = {name for name, number in config.sources.items() if use_for_training(number)}
    return [
        index
        for index, row in enumerate(rows)
        if row["source"] in admitted
        and row["source"] != config.sealed_source
        and index not in evaluated
        and held_out_key(row["image_id"], row["lesion_id"]) not in held_out
    ]


def holdouts(
    config: HamConfig, rows: list[dict[str, Any]], training: list[int]
) -> dict[str, dict[str, Any]]:
    """Per validation split, the groups it holds out and the training images among them."""
    evaluation = evaluation_split(config, rows)
    id_lesions = {rows[index]["lesion_id"] for index in evaluation["id"]}
    ood_lesions = {rows[index]["lesion_id"] for index in evaluation["ood"]}
    ood_number = config.sources[config.ood_source]
    return {
        "id": {
            "held_out_by": "lesion",
            "evaluation_groups": len(id_lesions),
            "training_samples_in_evaluation_groups": sum(
                rows[index]["lesion_id"] in id_lesions for index in training
            ),
        },
        "ood": {
            "held_out_by": "source",
            "evaluation_groups": [ood_number],
            "training_samples_in_evaluation_groups": sum(
                rows[index]["source"] == config.ood_source for index in training
            ),
            "lesions_shared_with_training": len(
                ood_lesions & {rows[index]["lesion_id"] for index in training}
            ),
        },
    }


def summary(config: HamConfig, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts by source and label, and the sizes of the healthy script's splits."""
    by_source = Counter((row["source"], row["label"]) for row in rows)
    evaluation = evaluation_split(config, rows)
    healthy = training_split(
        config,
        rows,
        lambda number: (
            number not in (config.sources[config.ood_source], config.sources[config.sealed_source])
        ),
    )
    leaky = training_split(
        config,
        rows,
        lambda number: (
            number not in (config.sources[config.ood_source], config.sources[config.sealed_source])
        ),
        lambda image_id, _lesion_id: image_id,
    )
    return {
        "images": len(rows),
        "lesions": len({row["lesion_id"] for row in rows}),
        "sources": {
            name: {
                "number": number,
                "benign": by_source[(name, 0)],
                "malignant": by_source[(name, 1)],
            }
            for name, number in sorted(config.sources.items(), key=lambda item: item[1])
        },
        "splits": {
            "training (keyed by lesion)": len(healthy),
            "training (keyed by image)": len(leaky),
            "id validation": len(evaluation["id"]),
            "ood validation": len(evaluation["ood"]),
        },
        "malignant_share": {
            name: round(sum(rows[i]["label"] for i in indices) / len(indices), 3)
            for name, indices in (
                ("training", healthy),
                ("id validation", evaluation["id"]),
                ("ood validation", evaluation["ood"]),
            )
        },
        "keyed_by_image": holdouts(config, rows, leaky)["id"],
    }


def _describe(rows: list[dict[str, Any]], indices: list[int]) -> dict[str, Any]:
    """Images, lesions, malignant share and sources of a set of images."""
    return {
        "images": len(indices),
        "lesions": len({rows[index]["lesion_id"] for index in indices}),
        "malignant_share": (
            round(sum(rows[index]["label"] for index in indices) / len(indices), 3)
            if indices
            else None
        ),
        "by_source": dict(sorted(Counter(rows[index]["source"] for index in indices).items())),
    }


def audit(
    config: HamConfig, rows: list[dict[str, Any]], manifest: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Every split of the data and what any two of them share, under either held-out key.

    A correct script, keyed by lesion, must share no image and no lesion between
    training and any validation split; keyed by image, only the lesions of the
    in-distribution validation images are shared, never the images themselves.
    """
    evaluation = evaluation_split(config, rows)
    sealed = [index for index, row in enumerate(rows) if row["source"] == config.sealed_source]
    held_out = (config.sources[config.ood_source], config.sources[config.sealed_source])
    keys = {"keyed_by_lesion": by_lesion, "keyed_by_image": lambda image_id, _lesion: image_id}
    training = {}
    for name, key in keys.items():
        splits = {
            "training": training_split(config, rows, lambda number: number not in held_out, key),
            "id validation": evaluation["id"],
            "ood validation": evaluation["ood"],
            "sealed": sealed,
        }
        lesions = {
            split: {rows[index]["lesion_id"] for index in indices}
            for split, indices in splits.items()
        }
        training[name] = {
            "training": _describe(rows, splits["training"]),
            "shared": {
                f"{first} / {second}": {
                    "images": len(set(splits[first]) & set(splits[second])),
                    "lesions": len(lesions[first] & lesions[second]),
                }
                for first, second in AUDIT_PAIRS
            },
        }
    sources = {}
    for name in sorted(config.sources, key=config.sources.get):
        of_source = [index for index, row in enumerate(rows) if row["source"] == name]
        sources[name] = {"number": config.sources[name], **_describe(rows, of_source)}
    return {
        "schema_version": SCHEMA_VERSION,
        "task": "ham10000_audit",
        "dataset": {
            "doi": DOI,
            "license": LICENSE,
            "images": len(rows),
            "lesions": len({row["lesion_id"] for row in rows}),
            "published_md5": (manifest or {}).get("published_md5"),
        },
        "config": {
            "sources": dict(config.sources),
            "ood_source": config.ood_source,
            "sealed_source": config.sealed_source,
            "id_val_fraction": config.id_val_fraction,
            "subset_seed": config.subset_seed,
            "max_ood_val_samples": config.max_ood_val_samples,
            "image_size": config.image_size,
        },
        "sources": sources,
        "validation": {
            "id validation": _describe(rows, evaluation["id"]),
            "ood validation": _describe(rows, evaluation["ood"]),
            "sealed": _describe(rows, sealed),
        },
        "training": training,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("prepare", "summary", "audit"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Where audit writes its JSON.")
    args = parser.parse_args(argv)
    config = HamConfig.load(args.config)
    if args.command == "prepare":
        print(f"prepared {prepare(config)}")
    data = load(config)
    if args.command == "audit":
        result = audit(config, data.rows, data.manifest)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0
    print(json.dumps(summary(config, data.rows), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
