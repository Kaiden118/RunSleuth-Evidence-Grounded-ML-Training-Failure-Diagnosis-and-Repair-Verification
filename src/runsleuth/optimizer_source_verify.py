"""Verify exported source edits with a bounded, paired optimizer-step experiment."""

import argparse
import json
import platform
from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from runsleuth.camelyon_optimizer_probe import file_sha256, load_reference
from runsleuth.evaluate_optimizer import _read, _scoped
from runsleuth.optimizer_evidence import _data_identity, _same, inspect_optimizer_run
from runsleuth.optimizer_source_repair import (
    MAX_SOURCE_BYTES,
    _sha,
    prepare_source_proposals,
)

VARIANTS = ("clean", "stale_head", "source_patched")


def _file_bytes(root: Path, path: Path, expected: bytes) -> None:
    resolved = _scoped(root, path, file=True)
    if resolved != path:
        raise ValueError("Exported proposal files must not be redirected through symlinks")
    with resolved.open("rb") as stream:
        value = stream.read(MAX_SOURCE_BYTES * 2 + 1)
    if value != expected:
        raise ValueError(f"Exported proposal bytes differ from the verified proposal: {path.name}")


def _baseline_job(root: Path, case: dict) -> dict:
    pair_record = _scoped(root, case["source"]["provenance"]["training_report"]["path"], file=True)
    parent, _ = _read(pair_record)
    baseline = _scoped(root, parent["reference_run"])
    names = ("config.json", "data_manifest.json", "initial_state_dict.pt", "run_report.json")
    paths = {name: _scoped(root, baseline / name, file=True) for name in names}
    hashes = {name: file_sha256(path) for name, path in paths.items()}
    for name, field in (
        ("config.json", "reference_config_sha256"),
        ("data_manifest.json", "reference_manifest_sha256"),
        ("initial_state_dict.pt", "reference_initial_checkpoint_sha256"),
    ):
        _same(hashes[name], parent.get(field), f"Pair baseline {name}")
    config, manifest, checkpoint = load_reference(baseline)
    baseline_report, _ = _read(paths["run_report.json"])
    if baseline_report.get("error") is not None:
        raise ValueError("Recorded baseline contains an execution error")
    reference = inspect_optimizer_run(
        _scoped(root, case["reference_run"]),
        require_file=lambda value: _scoped(root, value, file=True),
    )
    scientific = {
        key: value
        for key, value in asdict(config).items()
        if key not in {"run_name", "data_dir", "output_dir"}
    }
    _same(scientific, reference["config"], "Baseline scientific configuration")
    identity, _ = _data_identity(manifest, config)
    _same(identity, reference["data_identity"], "Baseline data identity")
    return {
        "id": case["id"],
        "seed": case["seed"],
        "config": config,
        "manifest": manifest,
        "checkpoint": checkpoint,
        "initial_state_sha256": reference["initial_state_sha256"],
        "original_source": case["source"]["original_source"],
        "proposed_source": case["proposed_patch"]["proposed_source"],
        "baseline": {"path": baseline.relative_to(root).as_posix(), "files_sha256": hashes},
    }


def prepare_source_verification(proposal_report: Path, *, workspace_root: Path) -> dict:
    """Replay all provenance and byte checks before importing Torch or creating outputs."""
    root = workspace_root.resolve()
    path = _scoped(root, proposal_report, file=True)
    saved, report_hash = _read(path)
    fresh = prepare_source_proposals(Path(saved["manifest"]["path"]), workspace_root=root)
    expected = deepcopy(fresh)
    jobs = []
    for index, (case, export) in enumerate(zip(fresh["cases"], expected["cases"], strict=True), 1):
        original = export["source"].pop("original_source")
        patch = export["proposed_patch"]
        if patch is None:
            continue
        proposed = patch.pop("proposed_source")
        folder = path.parent / "cases" / f"case-{index:03d}"
        for label, text in (("original", original), ("proposed", proposed)):
            source_path = folder / label / "optimizer_probe.py"
            payload = text.encode("utf-8")
            _file_bytes(root, source_path, payload)
            patch[f"{label}_file"] = source_path.relative_to(root).as_posix()
            patch[f"{label}_sha256"] = _sha(payload)
        diff_path = folder / "repair.patch"
        _file_bytes(root, diff_path, patch["unified_diff"].encode("utf-8"))
        patch["patch_file"] = diff_path.relative_to(root).as_posix()
        jobs.append(_baseline_job(root, case))
    _same(saved, expected, "Replayed source proposal report")
    if len(jobs) > 2:
        raise ValueError(
            "This verification batch permits at most two patches (six optimizer steps)"
        )
    report = {
        "schema_version": 1,
        "task": "optimizer_source_step_verification",
        "status": "prepared",
        "source_proposal": {"path": path.relative_to(root).as_posix(), "sha256": report_hash},
        "budget": {
            "patch_cases": len(jobs),
            "variants_per_patch": list(VARIANTS),
            "optimizer_steps_per_variant": 1,
            "optimizer_step_upper_bound": 3 * len(jobs),
            "api_calls": 0,
            "training_epochs": 0,
        },
        "execution_scope": "extracted_make_probe_optimizer_only",
        "performance_recovery_evaluated": False,
        "repair_accepted": None,
        "test_evaluated": False,
        "error": None,
        "summary": {
            "total_cases": len(fresh["cases"]),
            "mechanism_verified": 0,
            "no_change": fresh["summary"]["no_change"],
            "failed_cases": 0,
            "optimizer_steps_recorded": 0,
            "step_accounting_complete": True,
        },
        "cases": [
            {
                "id": case["id"],
                "seed": case["seed"],
                "status": "pending" if case["proposed_patch"] else "no_change",
                "execution_started": False,
                "result": None,
                "error": None,
                "source": case["source"]["provenance"],
                "optimizer_steps_recorded": 0,
                "step_accounting_complete": True,
            }
            for case in fresh["cases"]
        ],
    }
    return {"report": report, "jobs": jobs}


def _execute_case(job: dict, requested_device: str | None = None) -> dict:
    """Load pinned cached data and reuse one new controlled batch in all three variants."""
    import torch
    from torch.utils.data import default_collate

    from runsleuth.camelyon_data import build_camelyon_data
    from runsleuth.optimizer_probe import state_dict_sha256
    from runsleuth.optimizer_source_runtime import run_source_step_probe
    from runsleuth.train import resolve_device, seed_everything

    config = job["config"]
    device = resolve_device(requested_device or config.device)
    _same(
        file_sha256(job["checkpoint"]),
        job["baseline"]["files_sha256"]["initial_state_dict.pt"],
        "Initial checkpoint before load",
    )
    state = torch.load(job["checkpoint"], map_location="cpu", weights_only=True)
    _same(state_dict_sha256(state), job["initial_state_sha256"], "Loaded initial model state")
    seed_everything(config.seed)
    print(f"phase=load_cached_data case={job['id']}", flush=True)
    data = build_camelyon_data(config, device=device, download=False)
    identity, _ = _data_identity(data.manifest, config)
    recorded_identity, _ = _data_identity(job["manifest"], config)
    _same(identity, recorded_identity, "Current selected data and transform")
    positions = list(next(iter(data.train_loader.batch_sampler)))
    inputs, targets = default_collate([data.train_loader.dataset[i] for i in positions])
    if not positions or len(positions) != len(targets):
        raise ValueError("Probe batch is empty or inconsistent")
    selected_ids = data.manifest["splits"]["train"]["selected_sample_ids"]
    batch = {
        "selection": "new controlled batch; shared unchanged by all three variants",
        "sample_ids": [selected_ids[i] for i in positions],
        "input_shape": list(inputs.shape),
        "targets": targets.tolist(),
        "tensor_sha256": state_dict_sha256({"inputs": inputs, "targets": targets}),
    }
    print(f"phase=source_step_probe case={job['id']}", flush=True)
    result = run_source_step_probe(
        job["original_source"],
        job["proposed_source"],
        state=state,
        inputs=inputs.to(device),
        targets=targets.to(device),
        config=config,
        device=device,
    )
    result["batch"] = batch
    result["environment"] = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "device": str(device),
        "cuda_runtime": torch.version.cuda,
    }
    return result


def _write_report(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_source_verification(
    proposal_report: Path,
    *,
    workspace_root: Path,
    requested_device: str | None = None,
) -> Path:
    """Run at most six optimizer steps, retaining failed attempts and their budget."""
    if requested_device not in (None, "auto", "cpu", "cuda"):
        raise ValueError("Device must be auto, cpu, or cuda")
    root = workspace_root.resolve()
    plan = prepare_source_verification(proposal_report, workspace_root=root)
    report = plan["report"]
    directory = _scoped(root, "artifacts/optimizer_source_verifications") / datetime.now(
        UTC
    ).strftime("%Y%m%dT%H%M%S%fZ")
    directory.mkdir(parents=True, exist_ok=False)
    report_path = directory / "report.json"
    report["runtime_source_sha256"] = {
        name: file_sha256(Path(__file__).parent / name)
        for name in (
            "optimizer_source_verify.py",
            "optimizer_source_runtime.py",
            "optimizer_probe.py",
            "optimizer_audit.py",
            "camelyon.py",
            "camelyon_data.py",
            "camelyon_config.py",
            "camelyon_mirror.py",
            "train.py",
        )
    }
    print(json.dumps(report["budget"]), flush=True)
    report["status"] = "running"
    _write_report(report_path, report)
    by_id = {case["id"]: case for case in report["cases"]}
    for job in plan["jobs"]:
        case = by_id[job["id"]]
        case.update(
            status="running",
            execution_started=True,
            baseline=job["baseline"],
            step_accounting_complete=False,
        )
        report["summary"]["step_accounting_complete"] = False
        _write_report(report_path, report)
        try:
            result = _execute_case(job, requested_device)
            checks = result.get("checks")
            if (
                not isinstance(checks, dict)
                or not checks
                or any(type(value) is not bool for value in checks.values())
                or type(result.get("mechanism_verified")) is not bool
                or result["mechanism_verified"] != all(checks.values())
                or type(result.get("optimizer_steps")) is not int
                or result["optimizer_steps"] != 3
                or result.get("execution_scope") != "extracted_make_probe_optimizer_only"
                or result.get("source_patch_executed") is not True
                or result.get("performance_recovery_evaluated") is not False
                or set(result.get("variants", {})) != set(VARIANTS)
            ):
                raise ValueError("Incomplete source-step probe result")
            case.update(
                result=result,
                optimizer_steps_recorded=3,
                step_accounting_complete=True,
                status="mechanism_verified" if result["mechanism_verified"] else "not_verified",
            )
            report["summary"]["optimizer_steps_recorded"] += 3
            report["summary"]["step_accounting_complete"] = True
            if result["mechanism_verified"]:
                report["summary"]["mechanism_verified"] += 1
            else:
                report["summary"]["failed_cases"] += 1
        except (Exception, KeyboardInterrupt) as error:
            case.update(
                status="failed",
                error={"type": type(error).__name__, "message": str(error)},
                optimizer_steps_recorded=None,
                step_accounting_complete=False,
            )
            report["error"] = case["error"]
            report["summary"]["failed_cases"] += 1
            for remaining in report["cases"]:
                if remaining["status"] == "pending":
                    remaining["status"] = "not_run"
            break
        finally:
            _write_report(report_path, report)
    report["status"] = (
        "completed" if not report["summary"]["failed_cases"] else "completed_with_failures"
    )
    _write_report(report_path, report)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal-report", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    try:
        path = run_source_verification(
            args.proposal_report, workspace_root=Path.cwd(), requested_device=args.device
        )
    except (OSError, ValueError, TypeError, KeyError, SyntaxError) as error:
        parser.error(str(error))
    report, _ = _read(path)
    print(f"source_verification_report={path}")
    print(f"status={report['status']}")
    print(json.dumps(report["summary"]))
    for case in report["cases"]:
        print(
            json.dumps(
                {
                    "case": case["id"],
                    "status": case["status"],
                    "checks": case["result"]["checks"] if case["result"] else None,
                    "error": case["error"],
                }
            )
        )
    if report["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
