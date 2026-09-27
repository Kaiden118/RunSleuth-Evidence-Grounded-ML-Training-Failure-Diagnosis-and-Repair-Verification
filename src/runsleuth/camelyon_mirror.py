"""Pinned community Parquet mirror; its identity with the WILDS archive is unverified.

Only train and validation shards are read. Importing this module needs no ML stack.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from array import array
from collections import Counter
from numbers import Integral
from pathlib import Path
from typing import Any

MIRROR_REPO = "wltjr1007/Camelyon17-WILDS"
MIRROR_REVISION = "d784d5344ba6c967f83f9f3d9b2f1e2a4d6eb78f"
EXPECTED_COUNTS = {"train": 302436, "id_val": 33560, "val": 34904}
SPLIT_IDS = {"train": 0, "id_val": 1, "test": 2, "val": 3}
CENTER_NAMES = {
    "train-center1": 0,
    "validation-center": 1,
    "test-center": 2,
    "train-center2": 3,
    "train-center3": 4,
}
LABEL_NAMES = {"non-tumor": 0, "tumor": 1}
SHARDS = {
    split: tuple(f"data/{split}-{i:05d}-of-{count:05d}.parquet" for i in range(count))
    for split, count in (("train", 14), ("validation", 3))
}


def canonical_value(value: Any, feature: Any, *, field: str) -> int:
    """Decode ClassLabel names before interpreting an integer as a hospital/label.

    Some mirrors encode a split's subset of classes using consecutive local IDs.
    Integer Value features instead must already use the documented canonical IDs.
    """
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field} must contain integer values, got {value!r}.")
    value = int(value)
    mapping = CENTER_NAMES if field == "center" else LABEL_NAMES if field == "label" else None
    if mapping is None:
        raise ValueError(f"Unsupported categorical field: {field}.")
    if callable(getattr(feature, "int2str", None)):
        try:
            name = feature.int2str(value)
        except (ValueError, IndexError, KeyError) as exc:
            raise ValueError(f"Invalid encoded {field} value: {value}.") from exc
        if name not in mapping:
            raise ValueError(f"Unrecognized {field} class name: {name!r}.")
        return mapping[name]
    if value not in mapping.values():
        raise ValueError(f"Invalid canonical {field} value: {value}.")
    return value


def _metadata_rows(raw: Any, expected: dict[str, int]) -> list[tuple[int, int, int, int, int]]:
    """Return (source image ID, hospital, slide, label, split) without decoding images."""
    rows, seen_ids, slide_hospitals = [], set(), {}
    counts, labels, hospitals = Counter(), {}, {}
    for source in ("train", "validation"):
        dataset = raw[source]
        required = ["image_id", "center", "slide", "label"]
        if not set(required + ["image"]).issubset(dataset.column_names):
            raise ValueError(f"Mirror {source} lacks required image/metadata columns.")
        wanted = expected["train"] if source == "train" else expected["id_val"] + expected["val"]
        if len(dataset) != wanted:
            raise ValueError(f"Mirror {source} has {len(dataset)} rows; expected {wanted}.")
        columns = dataset.select_columns(required).to_dict()
        for image_id, center, slide, label in zip(*(columns[key] for key in required), strict=True):
            hospital = canonical_value(center, dataset.features["center"], field="center")
            target = canonical_value(label, dataset.features["label"], field="label")
            if hospital == 2 or (source == "train" and hospital not in (0, 3, 4)):
                raise ValueError(f"Unexpected hospital {hospital} in mirror {source}.")
            if isinstance(image_id, bool) or not isinstance(image_id, Integral) or image_id < 0:
                raise ValueError(f"Invalid source image ID: {image_id!r}.")
            if int(image_id) in seen_ids:
                raise ValueError(f"Duplicate source image ID: {image_id}.")
            if isinstance(slide, bool) or not isinstance(slide, Integral) or not 0 <= slide < 50:
                raise ValueError(f"Invalid slide ID: {slide!r}.")
            if slide_hospitals.setdefault(int(slide), hospital) != hospital:
                raise ValueError(f"Slide {slide} maps to multiple hospitals.")
            split = "train" if source == "train" else "val" if hospital == 1 else "id_val"
            counts[split] += 1
            labels.setdefault(split, set()).add(target)
            hospitals.setdefault(split, set()).add(hospital)
            seen_ids.add(int(image_id))
            rows.append((int(image_id), hospital, int(slide), target, SPLIT_IDS[split]))
    if dict(counts) != expected:
        raise ValueError(f"Mirror split counts {dict(counts)} differ from expected {expected}.")
    for split in expected:
        if labels[split] != {0, 1}:
            raise ValueError(f"Mirror {split} must contain both binary labels.")
        if hospitals[split] != ({1} if split == "val" else {0, 3, 4}):
            raise ValueError(f"Unexpected hospital set for {split}: {hospitals[split]}.")
    return rows


class MirrorSubset:
    """Top-level, pickleable view with indices into combined mirror row order."""

    def __init__(self, dataset: CamelyonMirror, split: str, transform: Any) -> None:
        self.dataset, self.transform = dataset, transform
        self.indices = [
            i for i, value in enumerate(dataset.split_array) if value == SPLIT_IDS[split]
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[Any, Any, Any]:
        global_index = self.indices[index]
        source, offset = (
            ("train", global_index)
            if global_index < self.dataset.train_size
            else ("validation", global_index - self.dataset.train_size)
        )
        image = self.dataset.raw[source][offset]["image"]
        if self.transform is not None:
            image = self.transform(image)
        return image, self.dataset.y_array[global_index], self.dataset.metadata_array[global_index]


class CamelyonMirror:
    version = "1.0"
    split_scheme = "official"
    split_dict = SPLIT_IDS
    metadata_fields = ["hospital", "slide", "y"]

    def __init__(self, raw: Any, rows: list, provenance_path: Path, provenance: dict) -> None:
        import torch

        self.raw, self.train_size = raw, len(raw["train"])
        self.sample_ids = [row[0] for row in rows]
        self.y_array = torch.tensor([row[3] for row in rows], dtype=torch.long)
        self.metadata_array = torch.tensor([row[1:4] for row in rows], dtype=torch.long)
        self.split_array = array("b", (row[4] for row in rows))
        self.source_metadata_path, self.provenance = provenance_path, provenance

    def get_subset(self, split: str, transform: Any = None) -> MirrorSubset:
        if split not in ("train", "id_val", "val"):
            raise ValueError(
                "This adapter only exposes train, id_val, and val; test remains sealed."
            )
        return MirrorSubset(self, split, transform)


def load_camelyon_mirror(config: Any, download: bool = False) -> CamelyonMirror:
    """Load 17 explicitly named pinned shards; never request the test shards."""
    if config.dataset_version != "1.0" or config.hf_revision != MIRROR_REVISION:
        raise ValueError(
            "This adapter supports only Camelyon17 v1.0 at the pinned mirror revision."
        )
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    data_root = Path(config.data_dir).expanduser().resolve()
    root = data_root / "community_mirror" / MIRROR_REVISION
    cache = root / "cache"
    # Datasets embeds its cache path again in lock filenames. Keep this root short
    # while preserving the existing Hub cache so downloaded shards are reused.
    arrow_cache = data_root / "arrow"
    data_files = {}
    for split, names in SHARDS.items():
        data_files[split] = [
            hf_hub_download(
                repo_id=MIRROR_REPO,
                repo_type="dataset",
                filename=name,
                revision=MIRROR_REVISION,
                cache_dir=str(cache / "hub"),
                local_files_only=not download,
            )
            for name in names
        ]
    # The built-in Parquet reader receives only local files: no repository code is executed.
    # Separate builders preserve each split's own ClassLabel encoding metadata.
    raw = {
        split: load_dataset(
            "parquet", data_files={split: files}, split=split, cache_dir=str(arrow_cache)
        )
        for split, files in data_files.items()
    }
    rows = _metadata_rows(raw, EXPECTED_COUNTS)
    row_hash = hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()
    provenance = {
        "source": "community_mirror",
        "repo": MIRROR_REPO,
        "revision": MIRROR_REVISION,
        "equivalence_not_verified": True,
        "dataset_version": "1.0",
        "caveat": "Community mirror: image and metadata identity with the official WILDS archive has not been verified.",
        "split_scheme": "official_reconstructed_from_mirror",
        "split_counts": EXPECTED_COUNTS,
        "shards": {key: list(value) for key, value in SHARDS.items()},
        "test_shards_loaded": False,
        "metadata_columns": ["image_id", "hospital", "slide", "y", "split"],
        "metadata_sha256": row_hash,
        "metadata_hash_encoding": "SHA-256 of combined train then validation rows as compact UTF-8 JSON",
    }
    root.mkdir(parents=True, exist_ok=True)
    path = root / "metadata_provenance.json"
    payload = json.dumps(provenance, sort_keys=True, indent=2) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root, delete=False) as stream:
        temporary_path = Path(stream.name)
        stream.write(payload)
    temporary_path.replace(path)
    return CamelyonMirror(raw, rows, path, provenance)
