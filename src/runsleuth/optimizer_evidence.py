"""Validate recorded optimizer evidence without loading models or running training."""

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import asdict
from math import isfinite
from pathlib import Path

from runsleuth.camelyon_config import CamelyonConfig

FILES = (
    "run_report.json",
    "config.json",
    "data_manifest.json",
    "metrics.jsonl",
    "domain_metrics.jsonl",
    "parameter_group_metrics.jsonl",
)
DOMAINS = ("id", "ood")
MAX_FILE_BYTES = 16 * 1024 * 1024


def _object(value: object, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: object, name: str, *, accuracy: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    try:
        finite = isfinite(value)
    except OverflowError as error:
        raise ValueError(f"{name} is outside the finite numeric range") from error
    if not finite or value < 0 or (accuracy and value > 1):
        raise ValueError(f"{name} is outside its finite range")
    return float(value)


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant: {value}")


def _parse(text: str) -> dict:
    return _object(
        json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        ),
        "JSON record",
    )


def _same(left: object, right: object, name: str) -> None:
    # JSON comparison distinguishes False from 0, unlike Python equality.
    if json.dumps(left, sort_keys=True, allow_nan=False) != json.dumps(
        right, sort_keys=True, allow_nan=False
    ):
        raise ValueError(f"{name} does not match its recorded counterpart")


def _epoch_rows(text: str, count: int, name: str) -> list[dict]:
    rows = [_parse(line) for line in text.splitlines() if line.strip()]
    if len(rows) != count:
        raise ValueError(f"{name} must contain exactly {count} epochs")
    for epoch, row in enumerate(rows, start=1):
        if _integer(row.get("epoch"), "epoch", 1) != epoch:
            raise ValueError(f"{name} epochs must be ordered and contiguous")
    return rows


def _audit(value: object) -> dict:
    source = _object(value, "optimizer_audit")
    names = source.get("missing_trainable_names")
    if not isinstance(names, list) or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Missing parameter names must be nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError("Missing parameter names must be unique")
    audit = {
        key: _integer(source.get(key), key)
        for key in (
            "model_parameter_tensors",
            "trainable_parameter_tensors",
            "optimizer_unique_parameter_tensors",
            "foreign_parameter_tensors",
            "duplicate_parameter_occurrences",
        )
    }
    model, trainable = audit["model_parameter_tensors"], audit["trainable_parameter_tensors"]
    present = audit["optimizer_unique_parameter_tensors"] - audit["foreign_parameter_tensors"]
    if not 0 < trainable <= model or not len(names) <= trainable:
        raise ValueError("Model and missing-parameter counts are inconsistent")
    if not trainable - len(names) <= present <= model - len(names):
        raise ValueError("Optimizer membership counts are inconsistent")
    audit["missing_trainable_names"] = names
    audit["missing_trainable_parameter_tensors"] = len(names)
    return audit


def _data_identity(manifest: dict, config: CamelyonConfig) -> tuple[dict, int]:
    for field in ("dataset", "dataset_version", "subset_seed"):
        _same(manifest.get(field), getattr(config, field), field)
    provenance = _object(manifest.get("source_provenance"), "source_provenance")
    if provenance.get("repo") != "wltjr1007/Camelyon17-WILDS":
        raise ValueError("Unexpected Camelyon17 mirror")
    if provenance.get("revision") != config.hf_revision:
        raise ValueError("Manifest revision does not match the configuration")
    metadata_hash = _hash(provenance.get("metadata_sha256"), "metadata_sha256")
    transform = {
        "resolution": [96, 96],
        "color_mode": "RGB",
        "to_tensor": True,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
        "resize": False,
        "augmentation": False,
    }
    _same(manifest.get("transform"), transform, "Input transform")
    splits = _object(manifest.get("splits"), "splits")
    counts, hashes, seen = {}, {}, set()
    for split, cap in (
        ("train", config.max_train_samples),
        ("id_val", config.max_id_val_samples),
        ("val", config.max_ood_val_samples),
    ):
        record = _object(splits.get(split), split)
        ids = record.get("selected_sample_ids")
        if not isinstance(ids, list) or not ids:
            raise ValueError("Selected sample IDs must be nonempty")
        for value in ids:
            _integer(value, "sample ID")
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError("Selected sample IDs must be unique and splits disjoint")
        seen.update(ids)
        available = _integer(record.get("available_samples"), "available_samples", 1)
        selected = _integer(record.get("selected_samples"), "selected_samples", 1)
        expected = available if cap is None else min(cap, available)
        if selected != len(ids) or selected != expected:
            raise ValueError("Selected sample counts do not match configuration and IDs")
        digest = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
        if _hash(record.get("selected_sample_ids_sha256"), "sample IDs digest") != digest:
            raise ValueError("Selected sample IDs do not match their recorded digest")
        counts[split], hashes[split] = selected, digest
    return {
        "dataset": config.dataset,
        "dataset_version": config.dataset_version,
        "subset_seed": config.subset_seed,
        "repo": provenance["repo"],
        "revision": provenance["revision"],
        "metadata_sha256": metadata_hash,
        "selected_sample_ids_sha256": hashes,
        "transform": transform,
    }, counts["train"]


def inspect_optimizer_run(
    directory: Path,
    *,
    require_file: Callable[[Path], Path],
) -> dict:
    """Read one completed variant; the caller supplies path-boundary enforcement.

    Digests and duplicate records detect inconsistent saved evidence, not a forged
    experiment. The result is recorded telemetry, never a live optimizer inspection.
    """
    texts, hashes = {}, {}
    for name in FILES:
        path = require_file(directory / name)
        with path.open("rb") as stream:
            payload = stream.read(MAX_FILE_BYTES + 1)
        if len(payload) > MAX_FILE_BYTES:
            raise ValueError("Optimizer artifact exceeds the input size limit")
        texts[name] = payload.decode("utf-8-sig")
        hashes[name] = hashlib.sha256(payload).hexdigest()
    report = _parse(texts["run_report.json"])
    try:
        config = CamelyonConfig(**_parse(texts["config.json"]))
    except (TypeError, OverflowError) as error:
        raise ValueError("Invalid Camelyon17 configuration fields") from error
    if not 1 <= config.epochs <= 3:
        raise ValueError("Optimizer training evidence must use the supported 1-3 epoch budget")
    if report.get("task") != "camelyon17_controlled_optimizer_training":
        raise ValueError("Unsupported optimizer artifact task")
    if report.get("status") != "completed" or report.get("error") is not None:
        raise ValueError("Optimizer run must have completed successfully")
    if _integer(report.get("completed_epochs"), "completed_epochs", 1) != config.epochs:
        raise ValueError("Completed epochs do not match the configured budget")
    if (
        _hash(report.get("data_manifest_sha256"), "data manifest digest")
        != hashes["data_manifest.json"]
    ):
        raise ValueError("Data manifest does not match the run report digest")
    identity, train_count = _data_identity(_parse(texts["data_manifest.json"]), config)
    audit = _audit(report.get("optimizer_audit"))
    initial_hash = _hash(report.get("initial_state_sha256"), "initial_state_sha256")
    metrics = _epoch_rows(texts["metrics.jsonl"], config.epochs, "metrics")
    domains = _epoch_rows(texts["domain_metrics.jsonl"], config.epochs, "domain metrics")
    groups = _epoch_rows(texts["parameter_group_metrics.jsonl"], config.epochs, "group metrics")
    _same(report.get("parameter_group_epochs"), groups, "Parameter-group epochs")
    expected_steps = (train_count + config.batch_size - 1) // config.batch_size
    output = []
    for metric, domain, group in zip(metrics, domains, groups, strict=True):
        steps = _integer(group.get("optimizer_steps"), "optimizer_steps", 1)
        if steps != expected_steps:
            raise ValueError("Optimizer steps do not match the saved sample and batch counts")
        row = {"epoch": metric["epoch"], "optimizer_steps": steps}
        for name in ("head", "backbone"):
            component = _object(group.get(name), name)
            row[name] = {
                key: _number(component.get(key), key)
                for key in ("mean_gradient_l2_norm", "mean_parameter_update_l2_norm")
            }
        for key in (
            "train_loss",
            "train_accuracy",
            "validation_loss",
            "validation_accuracy",
            "mean_gradient_norm",
            "mean_parameter_update_norm",
        ):
            _number(metric.get(key), key, accuracy=key.endswith("accuracy"))
        for domain_name in DOMAINS:
            for measure in ("loss", "accuracy"):
                key = f"{domain_name}_validation_{measure}"
                row[key] = _number(domain.get(key), key, accuracy=measure == "accuracy")
        for measure in ("loss", "accuracy"):
            _same(metric[f"validation_{measure}"], domain[f"id_validation_{measure}"], measure)
        _number(domain.get("epoch_seconds"), "epoch_seconds")
        if domain.get("peak_cuda_memory_mb") is not None:
            _number(domain["peak_cuda_memory_mb"], "peak_cuda_memory_mb")
        output.append(row)
    _same(report.get("final_metrics"), {**metrics[-1], **domains[-1]}, "Final metrics")
    scientific_config = {
        key: value
        for key, value in asdict(config).items()
        if key not in ("run_name", "data_dir", "output_dir")
    }
    return {
        "run_directory": directory.name,
        "config": scientific_config,
        "optimizer_audit": audit,
        "initial_state_sha256": initial_hash,
        "data_identity": identity,
        "epochs": output,
        "provenance": {
            "kind": "recorded_training_artifacts",
            "live_optimizer_inspected": False,
            "files_sha256": hashes,
        },
    }
