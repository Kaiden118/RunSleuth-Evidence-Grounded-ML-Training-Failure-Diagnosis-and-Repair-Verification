"""Compare correct and stale optimizer bindings on one identical Camelyon batch."""

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict
from datetime import UTC, datetime
from math import isclose
from pathlib import Path

from runsleuth.camelyon_config import CamelyonConfig


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def load_reference(reference_run: Path) -> tuple[CamelyonConfig, dict, Path]:
    """Check saved artifact hashes before reusing a completed baseline's initialization."""
    config = CamelyonConfig.load(reference_run / "config.json")
    report = read_json(reference_run / "run_report.json")
    if (
        report.get("status") != "completed"
        or report.get("completed_epochs") != config.epochs
        or report.get("task") != "camelyon17_resnet18_development_baseline"
    ):
        raise ValueError("Reference must be a completed Camelyon17 baseline")
    checkpoint = reference_run / "initial_state_dict.pt"
    manifest_path = reference_run / "data_manifest.json"
    for path, key in (
        (checkpoint, "initial_checkpoint_sha256"),
        (manifest_path, "data_manifest_sha256"),
    ):
        if file_sha256(path) != report.get(key):
            raise ValueError(f"Reference artifact hash mismatch: {path.name}")
    return config, read_json(manifest_path), checkpoint


def check_matching_data(reference: dict, current: dict) -> None:
    """Require the same pinned metadata and selected IDs, not just matching counts."""
    for name in ("dataset", "dataset_version", "subset_seed"):
        if reference[name] != current[name]:
            raise ValueError(f"Data identity mismatch: {name}")
    for name in ("repo", "revision", "metadata_sha256"):
        if reference["source_provenance"][name] != current["source_provenance"][name]:
            raise ValueError(f"Data provenance mismatch: {name}")
    for split in ("train", "id_val", "val"):
        left = reference["splits"][split]["selected_sample_ids"]
        right = current["splits"][split]["selected_sample_ids"]
        if not left or left != right:
            raise ValueError(f"Selected sample IDs differ: {split}")


def assess_probe(variants: dict, expected_state_hash: str) -> dict[str, bool]:
    """Check a reproduced mechanism, not end-to-end performance recovery."""
    clean, stale = variants["clean"], variants["stale_head"]
    clean_audit, stale_audit = clean["optimizer_audit"], stale["optimizer_audit"]
    return {
        "same_reference_initial_state": (
            clean["initial_state_sha256"] == stale["initial_state_sha256"] == expected_state_hash
        ),
        "same_pre_update_loss": isclose(
            clean["pre_update_loss"], stale["pre_update_loss"], rel_tol=1e-6, abs_tol=1e-6
        ),
        "clean_optimizer_covers_current_parameters": (
            not clean_audit["missing_trainable_names"]
            and clean_audit["foreign_parameter_tensors"] == 0
            and clean_audit["duplicate_parameter_occurrences"] == 0
        ),
        "stale_optimizer_misses_exactly_current_head": (
            set(stale_audit["missing_trainable_names"]) == {"fc.weight", "fc.bias"}
            and stale_audit["foreign_parameter_tensors"] == 2
            and stale_audit["duplicate_parameter_occurrences"] == 0
        ),
        "clean_head_has_gradients_and_updates": (
            clean["head"]["gradient_l2_norm"] > 0 and clean["head"]["parameter_update_l2_norm"] > 0
        ),
        "stale_head_has_gradients_without_updates": (
            stale["head"]["gradient_l2_norm"] > 0 and stale["head"]["parameter_update_l2_norm"] == 0
        ),
        "both_backbones_update": (
            clean["backbone"]["parameter_update_l2_norm"] > 0
            and stale["backbone"]["parameter_update_l2_norm"] > 0
        ),
    }


def run_probe(reference_run: Path, output_dir: Path, requested_device: str | None = None) -> Path:
    """Use cached data and the reference initial checkpoint; run one step per variant."""
    import torch
    from torch.utils.data import default_collate

    from runsleuth.camelyon import build_camelyon_model
    from runsleuth.camelyon_data import build_camelyon_data
    from runsleuth.optimizer_probe import run_probe_step, state_dict_sha256
    from runsleuth.train import resolve_device, seed_everything

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    directory = output_dir / f"probe-{timestamp}"
    directory.mkdir(parents=True)
    report_path = directory / "optimizer_probe_report.json"
    reference_run = reference_run.expanduser().resolve()
    report = {
        "schema_version": 1,
        "status": "running",
        "task": "camelyon17_optimizer_binding_probe",
        "reference_run": str(reference_run),
        "optimizer_steps_per_variant": 1,
        "performance_recovery_evaluated": False,
        "test_evaluated": False,
        "error": None,
    }
    print(f"probe_directory={directory}", flush=True)
    try:
        config, reference_manifest, checkpoint = load_reference(reference_run)
        device = resolve_device(requested_device or config.device)
        report["config"] = asdict(config)
        report["device"] = str(device)
        report["torch_version"] = str(torch.__version__)
        report["reference_config_sha256"] = file_sha256(reference_run / "config.json")
        report["reference_initial_checkpoint_sha256"] = file_sha256(checkpoint)
        report["reference_manifest_sha256"] = file_sha256(reference_run / "data_manifest.json")
        source_directory = directory / "source_snapshot"
        source_directory.mkdir()
        report["source_sha256"] = {}
        for name in (
            "optimizer_audit.py",
            "optimizer_probe.py",
            "camelyon_optimizer_probe.py",
            "camelyon.py",
            "camelyon_config.py",
            "camelyon_data.py",
            "camelyon_mirror.py",
            "train.py",
        ):
            destination = source_directory / name
            shutil.copyfile(Path(__file__).parent / name, destination)
            report["source_sha256"][name] = file_sha256(destination)
        print("phase=load_cached_data", flush=True)
        data = build_camelyon_data(config, device=device, download=False)
        check_matching_data(reference_manifest, data.manifest)
        positions = list(next(iter(data.train_loader.batch_sampler)))
        inputs, targets = default_collate([data.train_loader.dataset[i] for i in positions])
        if len(positions) != len(targets) or not positions:
            raise ValueError("The selected batch is empty or inconsistent")
        sample_ids = data.manifest["splits"]["train"]["selected_sample_ids"]
        report["batch"] = {
            "sample_ids": [sample_ids[i] for i in positions],
            "input_shape": list(inputs.shape),
            "targets": targets.tolist(),
            "tensor_sha256": state_dict_sha256({"inputs": inputs, "targets": targets}),
        }
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        expected_state_hash = state_dict_sha256(state)
        report["reference_initial_state_sha256"] = expected_state_hash
        inputs, targets = inputs.to(device), targets.to(device)
        variants = {}
        report["variants"] = variants
        for variant in ("clean", "stale_head"):
            print(f"phase=single_step variant={variant}", flush=True)
            seed_everything(config.seed)
            model = build_camelyon_model(pretrained=False)
            model.load_state_dict(state, strict=True)
            model.to(device)
            variants[variant] = run_probe_step(
                model,
                inputs,
                targets,
                variant=variant,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            print(json.dumps(variants[variant], allow_nan=False), flush=True)
            del model
        checks = assess_probe(variants, expected_state_hash)
        report["checks"] = checks
        report["mechanism_reproduced"] = all(checks.values())
        report["status"] = "completed" if report["mechanism_reproduced"] else "inconclusive"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        with report_path.open("x", encoding="utf-8") as output:
            output.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"optimizer_probe_report={report_path}", flush=True)
        raise
    with report_path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"optimizer_probe_report={report_path}", flush=True)
    print(f"status={report['status']}", flush=True)
    print("checks=" + json.dumps(report["checks"]), flush=True)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/optimizer_probes"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    path = run_probe(args.reference_run, args.output_dir, args.device)
    if read_json(path)["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
