"""Train verified source edits for three epochs and apply a fixed development policy."""

import argparse
import ast
import json
from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from runsleuth.camelyon_optimizer_probe import file_sha256
from runsleuth.camelyon_optimizer_training import REPAIR_POLICY, write_json
from runsleuth.evaluate_optimizer import _read, _scoped
from runsleuth.optimizer_evidence import (
    _audit,
    _data_identity,
    _epoch_rows,
    _hash,
    _integer,
    _number,
    _same,
)
from runsleuth.optimizer_source_repair import _sha
from runsleuth.optimizer_source_runtime import _checks
from runsleuth.optimizer_source_training_policy import _report_valid, assess_source_training
from runsleuth.optimizer_source_training_runtime import train_source_patch
from runsleuth.optimizer_source_verify import VARIANTS, _write_report, prepare_source_verification

EPOCHS = 3
HELPERS = (
    "train.py",
    "telemetry.py",
    "parameter_group_monitor.py",
    "camelyon.py",
    "camelyon_data.py",
    "camelyon_config.py",
    "camelyon_mirror.py",
    "optimizer_audit.py",
    "optimizer_probe.py",
)


def _check_step_receipt(saved: dict, planned: dict, jobs: list[dict]) -> None:
    """Recompute the prior mechanism checks instead of trusting its success label."""
    for field in (
        "schema_version",
        "task",
        "source_proposal",
        "budget",
        "execution_scope",
        "performance_recovery_evaluated",
        "repair_accepted",
        "test_evaluated",
        "error",
    ):
        _same(saved.get(field), planned[field], f"Step receipt {field}")
    _same(saved.get("status"), "completed", "Step receipt status")
    expected_summary = dict(
        planned["summary"], mechanism_verified=len(jobs), optimizer_steps_recorded=3 * len(jobs)
    )
    _same(saved.get("summary"), expected_summary, "Step receipt summary")
    cases = saved.get("cases")
    if not isinstance(cases, list) or len(cases) != len(planned["cases"]):
        raise ValueError("Step receipt cases differ from the proposal")
    by_id = {job["id"]: job for job in jobs}
    for case, expected in zip(cases, planned["cases"], strict=True):
        for field in ("id", "seed", "source", "error", "step_accounting_complete"):
            _same(case.get(field), expected[field], f"Step case {field}")
        if expected["status"] == "no_change":
            _same(case, expected, "No-change step case")
            continue
        job = by_id[case["id"]]
        for field, value in (
            ("status", "mechanism_verified"),
            ("execution_started", True),
            ("optimizer_steps_recorded", 3),
            ("baseline", job["baseline"]),
        ):
            _same(case.get(field), value, f"Verified step case {field}")
        result = case.get("result")
        if not isinstance(result, dict):
            raise ValueError("Missing source-step result")
        for field, value in (
            ("execution_scope", "extracted_make_probe_optimizer_only"),
            ("source_patch_executed", True),
            ("training_mode", "from_reference_initial_state"),
            ("optimizer_steps", 3),
            ("mechanism_verified", True),
            ("performance_recovery_evaluated", False),
            ("reference_initial_state_sha256", job["initial_state_sha256"]),
        ):
            _same(result.get(field), value, f"Source-step result {field}")
        variants = result.get("variants")
        if not isinstance(variants, dict) or set(variants) != set(VARIANTS):
            raise ValueError("Source-step variants are incomplete")
        for name, row in variants.items():
            _same(
                row.get("constructor_variant"),
                "clean" if name == "clean" else "stale_head",
                "Source-step constructor",
            )
            _same(
                _integer(row.get("optimizer_state_entries_before"), "Optimizer state"),
                0,
                "Fresh source-step optimizer",
            )
            for field in ("initial_state_sha256", "post_step_state_sha256"):
                _hash(row.get(field), field)
            _audit(row.get("optimizer_audit"))
            _number(row.get("pre_update_loss"), "Pre-update loss")
            _number(row.get("pre_update_accuracy"), "Pre-update accuracy", accuracy=True)
            for group in ("head", "backbone"):
                for metric in ("gradient_l2_norm", "parameter_update_l2_norm"):
                    _number(row[group].get(metric), f"Source-step {group} {metric}")
        checks = _checks(variants, job["initial_state_sha256"])
        _same(result.get("checks"), checks, "Recomputed source-step checks")
        if not all(checks.values()):
            raise ValueError("Source-step mechanism was not verified")


def _historical_controls(root: Path, case: dict, job: dict) -> dict:
    """Require matching recorded helper semantics before reusing historical controls."""
    pair_path = _scoped(root, case["source"]["training_report"]["path"], file=True)
    parent, _ = _read(pair_path)
    environment, _ = _read(_scoped(root, case["source"]["environment"]["path"], file=True))
    _same(parent.get("epochs_per_variant"), EPOCHS, "Historical epoch budget")
    if not isinstance(environment.get("torch"), str) or not environment["torch"]:
        raise ValueError("Historical PyTorch version is missing")
    if parent.get("device") not in ("cpu", "cuda"):
        raise ValueError("Historical device must be cpu or cuda")
    compatibility = {}
    for name in HELPERS:
        snapshot = _scoped(root, pair_path.parent / "source_snapshot" / name, file=True)
        recorded_hash = file_sha256(snapshot)
        _same(
            recorded_hash,
            environment.get("source_sha256", {}).get(name),
            f"Historical helper {name}",
        )
        live = Path(__file__).parent / name
        old_tree = ast.dump(ast.parse(snapshot.read_text(encoding="utf-8-sig")))
        new_tree = ast.dump(ast.parse(live.read_text(encoding="utf-8-sig")))
        _same(old_tree, new_tree, f"Training helper semantics: {name}")
        compatibility[name] = {
            "recorded_sha256": recorded_hash,
            "current_sha256": file_sha256(live),
        }
    controls, provenance = {}, {}
    for name in ("clean", "stale_head"):
        path = _scoped(root, pair_path.parent / name / "run_report.json", file=True)
        value, digest = _read(path)
        _same(value, parent["variants"][name], f"Historical {name} report")
        if not _report_valid(value, EPOCHS):
            raise ValueError(f"Historical {name} lacks complete three-epoch evidence")
        controls[name] = value
        provenance[name] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": digest,
            "execution": "reused_historical_run",
        }
    job.update(
        epochs=EPOCHS,
        comparators=controls,
        expected_environment={"torch": environment["torch"], "device": parent["device"]},
    )
    return {
        "runs": provenance,
        "helper_compatibility": compatibility,
        "expected_environment": job["expected_environment"],
    }


def prepare_source_training(verification_report: Path, *, workspace_root: Path) -> dict:
    """Validate every input before allocating an output directory or importing Torch."""
    root = workspace_root.resolve()
    path = _scoped(root, verification_report, file=True)
    saved, digest = _read(path)
    proposal = _scoped(root, saved["source_proposal"]["path"], file=True)
    plan = prepare_source_verification(proposal, workspace_root=root)
    _check_step_receipt(saved, plan["report"], plan["jobs"])
    cases = []
    by_id = {job["id"]: job for job in plan["jobs"]}
    for prior in saved["cases"]:
        case = {
            "id": prior["id"],
            "seed": prior["seed"],
            "status": "no_change" if prior["status"] == "no_change" else "pending",
            "execution_started": False,
            "decision": None,
            "verification": None,
            "training_report": None,
            "completed_epochs": 0,
            "optimizer_steps_recorded": 0,
            "step_accounting_complete": True,
            "partial_epoch_possible": False,
            "error": None,
        }
        if case["status"] == "pending":
            job = by_id[case["id"]]
            if type(job["config"].epochs) is not int or job["config"].epochs != EPOCHS:
                raise ValueError("Source training requires three-epoch historical baselines")
            case["historical_controls"] = _historical_controls(root, prior, job)
            case["baseline"] = deepcopy(job["baseline"])
        cases.append(case)
    policy = {
        "thresholds": dict(REPAIR_POLICY),
        "epochs": EPOCHS,
        "comparators": ["clean", "stale_head"],
        "comparators_reused": True,
        "domains": ["id", "ood"],
        "checkpoint_selection": "fixed_final_epoch",
        "ood_used_for_repair_acceptance": True,
        "ood_used_for_checkpoint_selection": False,
        "test_evaluated": False,
        "scope": "development verification; no held-out test or statistical recovery claim",
    }
    report = {
        "schema_version": 1,
        "task": "optimizer_source_training_verification",
        "status": "prepared",
        "error": None,
        "source_verification": {"path": path.relative_to(root).as_posix(), "sha256": digest},
        "source_proposal": deepcopy(saved["source_proposal"]),
        "policy": policy,
        "action_owner": "controller_ast",
        "test_evaluated": False,
        "budget": {
            "new_training_runs": len(by_id),
            "epochs_per_run": EPOCHS,
            "new_training_epoch_upper_bound": EPOCHS * len(by_id),
            "api_calls": 0,
            "historical_control_runs_reused": 2 * len(by_id),
        },
        "cases": cases,
    }
    _summarize(report)
    return {"report": report, "jobs": plan["jobs"]}


def _validate_training_output(job: dict, directory: Path, returned: dict) -> tuple[dict, dict]:
    """Cross-check the saved telemetry/checkpoint with the runtime receipt."""
    report, digest = _read(directory / "run_report.json")
    _same(report, returned, "Returned training receipt")
    for field, expected in (
        ("task", "camelyon17_source_patch_training"),
        ("status", "completed"),
        ("variant", "source_patched"),
        ("error", None),
        ("completed_epochs", EPOCHS),
        ("source_patch_executed", True),
        ("training_started", True),
        ("execution_scope", "extracted_make_probe_optimizer_only"),
        ("training_mode", "from_reference_initial_state"),
        ("test_evaluated", False),
        ("checkpoint_selection", "fixed_final_epoch"),
        ("step_accounting_complete", True),
        ("partial_epoch_possible", False),
    ):
        _same(report.get(field), expected, f"Training receipt {field}")
    _same(
        report["source_execution"],
        {
            "original_source_sha256": _sha(job["original_source"].encode("utf-8")),
            "proposed_source_sha256": _sha(job["proposed_source"].encode("utf-8")),
            "constructor_variant": "stale_head",
        },
        "Executed source identity",
    )
    config, _ = _read(directory / "config.json")
    _same(config, asdict(job["config"]), "Training configuration")
    manifest, manifest_hash = _read(directory / "data_manifest.json")
    _same(manifest_hash, report["data_manifest_sha256"], "Training data manifest bytes")
    identity, train_count = _data_identity(manifest, job["config"])
    _same(identity, _data_identity(job["manifest"], job["config"])[0], "Training data identity")
    rows = {}
    for name in ("metrics", "domain_metrics", "parameter_group_metrics"):
        rows[name] = _epoch_rows(
            (directory / f"{name}.jsonl").read_text(encoding="utf-8"), EPOCHS, name
        )
    _same(
        rows["parameter_group_metrics"], report["parameter_group_epochs"], "Saved group telemetry"
    )
    _same(
        {**rows["metrics"][-1], **rows["domain_metrics"][-1]},
        report["final_metrics"],
        "Saved final metrics",
    )
    expected_steps = (train_count + job["config"].batch_size - 1) // job["config"].batch_size
    for metrics, domains, groups in zip(
        rows["metrics"], rows["domain_metrics"], rows["parameter_group_metrics"], strict=True
    ):
        _same(groups["optimizer_steps"], expected_steps, "Steps per training epoch")
        for key in (
            "train_loss",
            "train_accuracy",
            "validation_loss",
            "validation_accuracy",
            "mean_gradient_norm",
            "mean_parameter_update_norm",
        ):
            _number(metrics.get(key), key, accuracy=key.endswith("accuracy"))
        for domain in ("id", "ood"):
            for metric in ("accuracy", "loss"):
                _number(
                    domains.get(f"{domain}_validation_{metric}"),
                    f"{domain} {metric}",
                    accuracy=metric == "accuracy",
                )
        _same(metrics["validation_accuracy"], domains["id_validation_accuracy"], "ID accuracy")
        _same(metrics["validation_loss"], domains["id_validation_loss"], "ID loss")
    _same(report["optimizer_steps_recorded"], expected_steps * EPOCHS, "Total optimizer steps")
    for key, value in job["expected_environment"].items():
        _same(report["environment"].get(key), value, f"Training environment {key}")
    baseline, _ = _read(directory / "baseline_metrics.json")
    _same(baseline, {"epoch": 0, **report["initialization_metrics"]}, "Initialization metrics")
    _same(
        file_sha256(directory / "model_state_dict.pt"),
        report["checkpoint_sha256"],
        "Final checkpoint bytes",
    )
    _hash(report.get("final_state_sha256"), "Final state")
    return report, {"path": str(directory / "run_report.json"), "sha256": digest}


def _account(case: dict, runtime: dict) -> None:
    for field in ("completed_epochs", "optimizer_steps_recorded"):
        case[field] = _integer(runtime.get(field), field)
    if case["completed_epochs"] > EPOCHS:
        raise ValueError("Recorded training exceeded the epoch budget")
    for field in ("step_accounting_complete", "partial_epoch_possible"):
        if type(runtime.get(field)) is not bool:
            raise ValueError(f"Invalid training accounting: {field}")
        case[field] = runtime[field]


def _summarize(report: dict) -> None:
    cases = report["cases"]
    report["summary"] = {
        "total_cases": len(cases),
        "accepted": sum(case["status"] == "accepted" for case in cases),
        "rejected": sum(case["status"] == "rejected" for case in cases),
        "no_change": sum(case["status"] == "no_change" for case in cases),
        "error_cases": sum(case["status"] == "error" for case in cases),
        "not_run": sum(case["status"] == "not_run" for case in cases),
        "completed_new_epochs": sum(case["completed_epochs"] for case in cases),
        "optimizer_steps_recorded": sum(case["optimizer_steps_recorded"] for case in cases),
        "step_accounting_complete": all(case["step_accounting_complete"] for case in cases),
        "partial_epoch_possible": any(case["partial_epoch_possible"] for case in cases),
    }


def run_source_training(
    verification_report: Path,
    *,
    workspace_root: Path,
    requested_device: str | None = None,
) -> Path:
    """Train only proposed patches; persist rejections and partial failures without retries."""
    if requested_device not in (None, "auto", "cpu", "cuda"):
        raise ValueError("Device must be auto, cpu, or cuda")
    root = workspace_root.resolve()
    plan = prepare_source_training(verification_report, workspace_root=root)
    for job in plan["jobs"]:
        if requested_device not in (None, "auto", job["expected_environment"]["device"]):
            raise ValueError("Use the historical controls' device for this matched comparison")
    report = plan["report"]
    directory = _scoped(root, "artifacts/optimizer_source_training") / datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "report.json"
    write_json(directory / "policy.json", report["policy"])
    report["policy_sha256"] = file_sha256(directory / "policy.json")
    snapshot = directory / "source_snapshot"
    snapshot.mkdir()
    report["runtime_source_sha256"] = {}
    for name in (
        *HELPERS,
        "optimizer_source_train.py",
        "optimizer_source_training_policy.py",
        "optimizer_source_training_runtime.py",
        "optimizer_source_runtime.py",
        "optimizer_source_patch.py",
    ):
        (snapshot / name).write_bytes((Path(__file__).parent / name).read_bytes())
        report["runtime_source_sha256"][name] = file_sha256(snapshot / name)
    report["status"] = "running"
    _write_report(path, report)
    print(json.dumps(report["budget"]), flush=True)
    print(f"source_training_directory={directory}", flush=True)
    by_id = {job["id"]: job for job in plan["jobs"]}
    for index, case in enumerate(report["cases"], 1):
        if case["status"] == "no_change":
            continue
        job = by_id[case["id"]]
        run_directory = directory / "cases" / f"case-{index:03d}"
        case.update(status="running", execution_started=True, step_accounting_complete=False)
        _summarize(report)
        _write_report(path, report)
        try:
            runtime = train_source_patch(job, run_directory, requested_device)
            runtime, receipt = _validate_training_output(job, run_directory, runtime)
            _account(case, runtime)
            receipt["path"] = Path(receipt["path"]).relative_to(root).as_posix()
            case["training_report"] = receipt
            result = assess_source_training(
                job["comparators"]["clean"],
                job["comparators"]["stale_head"],
                runtime,
                expected_hash=job["initial_state_sha256"],
                epochs=EPOCHS,
            )
            case.update(status=result["decision"], decision=result["decision"], verification=result)
        except (Exception, KeyboardInterrupt) as error:
            case.update(status="error", error={"type": type(error).__name__, "message": str(error)})
            report["error"] = case["error"]
            partial_path = run_directory / "run_report.json"
            if partial_path.is_file():
                try:
                    partial, digest = _read(partial_path)
                    _account(case, partial)
                    case["training_report"] = {
                        "path": partial_path.relative_to(root).as_posix(),
                        "sha256": digest,
                    }
                except (OSError, ValueError, TypeError, KeyError):
                    case.update(step_accounting_complete=False, partial_epoch_possible=True)
            else:
                case.update(step_accounting_complete=False, partial_epoch_possible=True)
            for remaining in report["cases"]:
                if remaining["status"] == "pending":
                    remaining["status"] = "not_run"
            break
        finally:
            _summarize(report)
            _write_report(path, report)
        print(json.dumps({"case": case["id"], "decision": case["decision"]}), flush=True)
    report["status"] = (
        "completed_with_errors"
        if report["summary"]["error_cases"]
        else "completed_with_rejections"
        if report["summary"]["rejected"]
        else "completed"
    )
    _write_report(path, report)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verification-report", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    try:
        path = run_source_training(
            args.verification_report, workspace_root=Path.cwd(), requested_device=args.device
        )
    except (OSError, ValueError, KeyError, TypeError, SyntaxError) as error:
        parser.error(str(error))
    report, _ = _read(path)
    print(f"source_training_report={path}")
    print(f"status={report['status']}")
    print(json.dumps(report["summary"]))
    for case in report["cases"]:
        result = case["verification"] or {}
        print(
            json.dumps(
                {
                    "case": case["id"],
                    "decision": case["decision"],
                    "status": case["status"],
                    "structure_verified": result.get("structure_verified"),
                    "performance_nonregression": result.get("performance_nonregression"),
                    "failed_performance_checks": [
                        check
                        for check in result.get("performance_checks", [])
                        if not check["passed"]
                    ],
                    "error": case["error"],
                }
            )
        )
    if report["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
