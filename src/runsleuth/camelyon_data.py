"""Camelyon17-WILDS mirror splits with auditable, fixed sample selection.

Subset membership depends on ``subset_seed``; training order depends on ``seed``.
The pinned mirror reconstructs official split semantics from its metadata.
Importing this module does not require Hugging Face datasets, torchvision, or torch.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
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

    from runsleuth.camelyon_mirror import load_camelyon_mirror

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
