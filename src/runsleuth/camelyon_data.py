"""Camelyon17-WILDS mirror splits with auditable, fixed sample selection.

Subset membership depends on ``subset_seed``; training order depends on ``seed``.
The pinned mirror reconstructs official split semantics from its metadata.
Importing this module does not require Hugging Face datasets, torchvision, or torch.
"""

from __future__ import annotations

import hashlib
import json
import random
import tempfile
from array import array
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader

    from runsleuth.camelyon_config import CamelyonConfig

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class CamelyonData:
    train_loader: DataLoader
    id_val_loader: DataLoader
    ood_val_loader: DataLoader
    manifest: dict[str, Any]


def select_stratified_indices(
    indices: Sequence[int],
    labels: Sequence[int],
    hospitals: Sequence[int],
    max_samples: int | None,
    *,
    seed: int,
) -> list[int]:
    """Sample aligned global IDs by (label, hospital), using largest remainders.

    The selected count is min(cap, population), or the full population for None.
    Integer arithmetic and sorted stratum ties make quota allocation reproducible.
    Returned IDs are sorted; selection never mutates the process-wide random state.
    """
    if len(indices) != len(labels) or len(indices) != len(hospitals):
        raise ValueError("Indices, labels, and hospitals must have equal lengths.")
    if max_samples is not None and (type(max_samples) is not int or max_samples <= 0):
        raise ValueError("Sample caps must be positive integers or None.")
    if len(set(indices)) != len(indices):
        raise ValueError("Global sample indices must be unique.")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("Camelyon17 labels must be 0 or 1.")
    size = len(indices)
    count = size if max_samples is None else min(max_samples, size)
    if count == size:
        return sorted(int(index) for index in indices)
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, label, hospital in zip(indices, labels, hospitals, strict=True):
        groups[(int(label), int(hospital))].append(int(index))
    keys = sorted(groups)
    quotas = {key: count * len(groups[key]) // size for key in keys}
    remainder_order = sorted(keys, key=lambda key: (-(count * len(groups[key]) % size), key))
    for key in remainder_order[: count - sum(quotas.values())]:
        quotas[key] += 1
    rng = random.Random(seed)
    return sorted(index for key in keys for index in rng.sample(sorted(groups[key]), quotas[key]))


class NativeRGBPatch:
    """Pickleable input check; the central tumor label assumes native geometry."""

    def __call__(self, image: Any) -> Any:
        if image.size != (96, 96):
            raise ValueError(f"Expected a native 96x96 Camelyon17 patch, got {image.size}.")
        return image.convert("RGB")


class PairDataset:
    """Expose a WILDS triple as (input, target), preserving original labels."""

    def __init__(self, subset: Any, positions: Sequence[int]) -> None:
        self.subset = subset
        self.positions = list(positions)

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        image, target, _metadata = self.subset[self.positions[index]]
        return image, target


def seed_worker(_worker_id: int) -> None:
    """Seed Python and NumPy from the worker seed assigned by the loader."""
    import numpy as np
    import torch

    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _counts(values: Sequence[int]) -> dict[str, int]:
    return {str(key): value for key, value in sorted(Counter(values).items())}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_camelyon_data(
    config: CamelyonConfig, *, device: torch.device, download: bool = False
) -> CamelyonData:
    """Build train, ID-validation, and OOD-validation loaders from the pinned mirror."""
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms

    if config.dataset_version != "1.0":
        raise ValueError("This baseline is pinned to Camelyon17-WILDS version 1.0.")
    dataset = load_camelyon_mirror(config, download=download)
    if str(dataset.version) != config.dataset_version or dataset.split_scheme != "official":
        raise ValueError("The loaded dataset does not match version 1.0 and official splits.")
    transform = transforms.Compose(
        [
            NativeRGBPatch(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    hospital_column = dataset.metadata_fields.index("hospital")
    labels = [int(value) for value in dataset.y_array.tolist()]
    hospitals = [int(value) for value in dataset.metadata_array[:, hospital_column].tolist()]
    split_ids = [int(value) for value in dataset.split_array.tolist()]
    split_caps = {
        "train": config.max_train_samples,
        "id_val": config.max_id_val_samples,
        "val": config.max_ood_val_samples,
    }
    loaders, records, official_hospitals = {}, {}, {}
    seen_ids: set[int] = set()
    for split, cap in split_caps.items():
        subset = dataset.get_subset(split, transform=transform)
        official_ids = [int(value) for value in subset.indices]
        if not official_ids:
            raise ValueError(f"Official {split} split is empty.")
        if any(index < 0 or index >= len(labels) for index in official_ids):
            raise ValueError(f"Official {split} contains invalid global sample indices.")
        if any(split_ids[index] != dataset.split_dict[split] for index in official_ids):
            raise ValueError(f"Official {split} contains samples assigned to another split.")
        official_hospitals[split] = {hospitals[index] for index in official_ids}
        selected = select_stratified_indices(
            official_ids,
            [labels[index] for index in official_ids],
            [hospitals[index] for index in official_ids],
            cap,
            seed=config.subset_seed,
        )
        if seen_ids.intersection(selected):
            raise ValueError("Selected official splits overlap.")
        if any(split_ids[index] == dataset.split_dict["test"] for index in selected):
            raise ValueError("Test samples must never be selected for this baseline.")
        seen_ids.update(selected)
        local_positions = {global_id: position for position, global_id in enumerate(official_ids)}
        pairs = PairDataset(subset, [local_positions[index] for index in selected])
        # Each loader owns its RNG so evaluating validation cannot alter training order.
        generator = torch.Generator().manual_seed(config.seed)
        loaders[split] = DataLoader(
            pairs,
            batch_size=config.batch_size,
            shuffle=split == "train",
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
            generator=generator,
            worker_init_fn=seed_worker if config.num_workers else None,
        )
        records[split] = {
            "official_split": split,
            "available_samples": len(official_ids),
            "selected_samples": len(selected),
            "requested_cap": cap,
            "selected_indices": selected,
            "selected_indices_sha256": hashlib.sha256(
                json.dumps(selected, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "selected_sample_ids": [dataset.sample_ids[index] for index in selected],
            "selected_sample_ids_sha256": hashlib.sha256(
                json.dumps(
                    [dataset.sample_ids[index] for index in selected], separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest(),
            "label_counts": _counts([labels[index] for index in selected]),
            "hospital_counts": _counts([hospitals[index] for index in selected]),
        }
    if official_hospitals["train"].intersection(official_hospitals["val"]):
        raise ValueError("Official OOD validation hospitals overlap the training hospitals.")
    if set(records["train"]["label_counts"]) != {"0", "1"}:
        raise ValueError("The selected training subset must contain both labels; increase its cap.")
    majority_label = max(
        (0, 1), key=lambda label: records["train"]["label_counts"].get(str(label), 0)
    )
    manifest = {
        "dataset": "camelyon17",
        "dataset_version": config.dataset_version,
        "split_scheme": "official_reconstructed_from_mirror",
        "upstream_split_scheme": "official",
        "subset_seed": config.subset_seed,
        "selection_method": "proportional_label_hospital_stratification_largest_remainder",
        "indices_hash_encoding": "SHA-256 of sorted global IDs as compact UTF-8 JSON",
        "indices_meaning": "Combined mirror row offsets; selected_sample_ids are stable mirror image_id values",
        "sample_ids_hash_encoding": "SHA-256 of image_id values in selected_indices order as compact UTF-8 JSON",
        "source_metadata_sha256": _file_sha256(Path(dataset.source_metadata_path)),
        "source_provenance": dataset.provenance,
        "transform": {
            "resolution": [96, 96],
            "color_mode": "RGB",
            "to_tensor": True,
            "normalize_mean": list(IMAGENET_MEAN),
            "normalize_std": list(IMAGENET_STD),
            "resize": False,
            "augmentation": False,
        },
        "splits": records,
        "majority_baseline": {
            "label_from_train": majority_label,
            "accuracy_by_split": {
                split: record["label_counts"].get(str(majority_label), 0)
                / record["selected_samples"]
                for split, record in records.items()
            },
        },
        "validation_checks": {
            "selected_splits_disjoint": True,
            "test_selected": False,
            "ood_val_hospitals_disjoint_from_train": True,
        },
    }
    return CamelyonData(loaders["train"], loaders["id_val"], loaders["val"], manifest)


# Pinned community Parquet mirror; its identity with the WILDS archive is unverified.
# Only train and validation shards are read.

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

    def __init__(
        self,
        dataset: CamelyonMirror,
        split: str,
        transform: Any,
        indices: Sequence[int] | None = None,
    ) -> None:
        self.dataset, self.transform = dataset, transform
        self.indices = (
            [i for i, value in enumerate(dataset.split_array) if value == SPLIT_IDS[split]]
            if indices is None
            else [int(index) for index in indices]
        )

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

    def get_patches(self, indices: Sequence[int], transform: Any = None) -> MirrorSubset:
        """A view over chosen train and validation patches, by combined row index."""
        return MirrorSubset(self, "train", transform, indices)


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
