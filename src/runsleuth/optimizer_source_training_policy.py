"""Fixed development acceptance policy for training a verified source proposal."""

from math import isclose, isfinite

from runsleuth.camelyon_optimizer_training import REPAIR_POLICY, assess_training

_DOMAINS = ("id", "ood")
_NORMS = ("mean_gradient_l2_norm", "mean_parameter_update_l2_norm")
_GROUPS = ("head", "backbone")
_AUDIT_COUNTS = (
    "model_parameter_tensors",
    "trainable_parameter_tensors",
    "optimizer_unique_parameter_tensors",
    "foreign_parameter_tensors",
    "duplicate_parameter_occurrences",
)


def _number(value) -> bool:
    try:
        return type(value) in (int, float) and isfinite(value)
    except OverflowError:
        return False


def _metrics_valid(value) -> bool:
    return isinstance(value, dict) and all(
        _number(value.get(f"{domain}_validation_{metric}"))
        and value[f"{domain}_validation_{metric}"] >= 0
        and (metric != "accuracy" or value[f"{domain}_validation_{metric}"] <= 1)
        for domain in _DOMAINS
        for metric in ("accuracy", "loss")
    )


def _audit_valid(audit) -> bool:
    if not isinstance(audit, dict) or not all(
        type(audit.get(key)) is int and audit[key] >= 0 for key in _AUDIT_COUNTS
    ):
        return False
    missing = audit.get("missing_trainable_names")
    if not isinstance(missing, (list, tuple)) or not all(
        isinstance(name, str) and bool(name) for name in missing
    ):
        return False
    if len(set(missing)) != len(missing):
        return False
    model = audit["model_parameter_tensors"]
    trainable = audit["trainable_parameter_tensors"]
    unique = audit["optimizer_unique_parameter_tensors"]
    foreign = audit["foreign_parameter_tensors"]
    return (
        model == trainable > 0
        and unique > 0
        and len(missing) <= trainable
        and unique - foreign == trainable - len(missing)
    )


def _coverage(audit) -> bool:
    return (
        _audit_valid(audit)
        and not audit["missing_trainable_names"]
        and audit["foreign_parameter_tensors"] == 0
        and audit["duplicate_parameter_occurrences"] == 0
    )


def _stale_coverage(audit) -> bool:
    return (
        _audit_valid(audit)
        and set(audit["missing_trainable_names"]) == {"fc.bias", "fc.weight"}
        and audit["foreign_parameter_tensors"] == 2
        and audit["duplicate_parameter_occurrences"] == 0
    )


def _rows_valid(rows, epochs) -> bool:
    return (
        isinstance(rows, list)
        and len(rows) == epochs
        and all(
            isinstance(row, dict)
            and type(row.get("epoch")) is int
            and row["epoch"] == index
            and type(row.get("optimizer_steps")) is int
            and row["optimizer_steps"] > 0
            and all(
                isinstance(row.get(group), dict)
                and all(
                    _number(row[group].get(metric)) and row[group][metric] >= 0 for metric in _NORMS
                )
                for group in _GROUPS
            )
            for index, row in enumerate(rows, 1)
        )
    )


def _report_valid(report, epochs) -> bool:
    return (
        isinstance(report, dict)
        and report.get("status") == "completed"
        and "error" in report
        and report["error"] is None
        and type(report.get("completed_epochs")) is int
        and report["completed_epochs"] == epochs
        and isinstance(report.get("initial_state_sha256"), str)
        and bool(report["initial_state_sha256"])
        and _metrics_valid(report.get("initialization_metrics"))
        and _metrics_valid(report.get("final_metrics"))
        and type(report["final_metrics"].get("epoch")) is int
        and report["final_metrics"]["epoch"] == epochs
        and _rows_valid(report.get("parameter_group_epochs"), epochs)
        and _audit_valid(report.get("optimizer_audit"))
        and _audit_valid(report.get("final_optimizer_audit"))
    )


def _all_positive(rows, groups) -> bool:
    return bool(rows) and all(
        _number(row.get(group, {}).get(metric)) and row[group][metric] > 0
        for row in rows
        for group in groups
        for metric in _NORMS
    )


def _performance_checks(clean, stale, patched) -> list[dict]:
    checks = []
    actual_metrics = patched.get("final_metrics", {})
    if not isinstance(actual_metrics, dict):
        actual_metrics = {}
    for comparator, report in (("clean", clean), ("stale_head", stale)):
        reference_metrics = report.get("final_metrics", {})
        if not isinstance(reference_metrics, dict):
            reference_metrics = {}
        valid = _metrics_valid(actual_metrics) and _metrics_valid(reference_metrics)
        for domain in _DOMAINS:
            for metric in ("accuracy", "loss"):
                actual = actual_metrics.get(f"{domain}_validation_{metric}")
                reference = reference_metrics.get(f"{domain}_validation_{metric}")
                accuracy = metric == "accuracy"
                threshold = REPAIR_POLICY["max_accuracy_drop" if accuracy else "max_loss_ratio"]
                observed = None
                passed = False
                if valid:
                    if accuracy:
                        observed = reference - actual
                        passed = actual >= reference - threshold
                    else:
                        # A zero reference loss permits only a zero patched loss.
                        # The ratio itself is undefined, so record JSON null.
                        observed = actual / reference if reference > 0 else None
                        passed = actual <= reference * threshold
                checks.append(
                    {
                        "comparator": comparator,
                        "domain": domain,
                        "metric": "accuracy_drop" if accuracy else "loss_ratio",
                        "repaired_value": actual if _number(actual) else None,
                        "reference_value": reference if _number(reference) else None,
                        "observed_value": observed if _number(observed) else None,
                        "operator": "<=",
                        "threshold": threshold,
                        "passed": passed,
                    }
                )
    return checks


def assess_source_training(
    clean: dict,
    stale: dict,
    patched: dict,
    *,
    expected_hash: str,
    epochs: int = 3,
) -> dict:
    """Require source repair structure and nonregression against both controls.

    Inputs are recorded run reports. The caller must separately verify their
    provenance and matching data/configuration. No runtime work occurs here.
    Invalid or incomplete evidence is rejected rather than coerced into success.
    """
    clean = clean if isinstance(clean, dict) else {}
    stale = stale if isinstance(stale, dict) else {}
    patched = patched if isinstance(patched, dict) else {}
    budget_valid = type(epochs) is int and 1 <= epochs <= 3
    reports = (clean, stale, patched)
    valid = budget_valid and all(_report_valid(report, epochs) for report in reports)
    controls_valid = budget_valid and all(
        _report_valid(report, epochs) for report in (clean, stale)
    )
    structural = {"historical_controls_complete_and_valid": controls_valid}
    if controls_valid:
        structural.update(
            assess_training({"clean": clean, "stale_head": stale}, expected_hash, epochs)
        )
    structural.update(
        {
            "all_variants_completed_fixed_budget_with_valid_evidence": valid,
            "all_use_reference_initial_state": (
                isinstance(expected_hash, str)
                and bool(expected_hash)
                and all(report.get("initial_state_sha256") == expected_hash for report in reports)
            ),
            "source_patch_executed": patched.get("source_patch_executed") is True,
            "fresh_optimizer_before_first_step": (
                type(patched.get("optimizer_state_entries_before")) is int
                and patched["optimizer_state_entries_before"] == 0
            ),
            "source_constructor_preserves_initial_model_state": (
                isinstance(expected_hash, str)
                and bool(expected_hash)
                and patched.get("model_state_sha256_before_constructor")
                == patched.get("model_state_sha256_after_constructor")
                == expected_hash
            ),
            "patched_optimizer_covers_current_parameters_initially_and_finally": all(
                _coverage(patched.get(key)) for key in ("optimizer_audit", "final_optimizer_audit")
            ),
            "historical_clean_coverage_persists": all(
                _coverage(clean.get(key)) for key in ("optimizer_audit", "final_optimizer_audit")
            ),
            "historical_stale_head_mismatch_persists": all(
                _stale_coverage(stale.get(key))
                for key in ("optimizer_audit", "final_optimizer_audit")
            ),
            "same_parameter_tensor_counts": valid
            and all(
                report[key]["model_parameter_tensors"]
                == clean["optimizer_audit"]["model_parameter_tensors"]
                for report in reports
                for key in ("optimizer_audit", "final_optimizer_audit")
            ),
            "same_initial_validation_metrics_after_source_patch": valid
            and all(
                isclose(
                    clean["initialization_metrics"][f"{domain}_validation_{metric}"],
                    report["initialization_metrics"][f"{domain}_validation_{metric}"],
                    rel_tol=1e-6,
                    abs_tol=1e-6,
                )
                for report in reports
                for domain in _DOMAINS
                for metric in ("accuracy", "loss")
            ),
            "same_optimizer_step_counts_after_source_patch": valid
            and all(
                clean["parameter_group_epochs"][index]["optimizer_steps"]
                == stale["parameter_group_epochs"][index]["optimizer_steps"]
                == patched["parameter_group_epochs"][index]["optimizer_steps"]
                for index in range(epochs)
            ),
            "patched_head_and_backbone_update_every_epoch": valid
            and _all_positive(patched["parameter_group_epochs"], _GROUPS),
            "all_backbones_have_gradients_and_updates_every_epoch": valid
            and all(
                _all_positive(report["parameter_group_epochs"], ("backbone",)) for report in reports
            ),
        }
    )
    performance = _performance_checks(clean, stale, patched)
    structure_verified = all(structural.values())
    performance_nonregression = all(check["passed"] for check in performance)
    return {
        "decision": "accepted" if structure_verified and performance_nonregression else "rejected",
        "structure_verified": structure_verified,
        "performance_nonregression": performance_nonregression,
        "structural_checks": structural,
        "performance_checks": performance,
        "policy": dict(REPAIR_POLICY),
        "scope": (
            "development verification against matched historical controls; "
            "not a held-out test or a statistical performance-recovery claim"
        ),
    }
