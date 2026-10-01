"""Deterministic optimizer-audit baseline over the same evidence as the Agent.

These rules mirror the Agent validator's constrained audit classifications.
Agreement on this task does not establish an advantage for LLM diagnosis.
"""

from runsleuth.optimizer_agent_diagnosis import (
    AUDIT_FIELDS,
    METRIC_FIELDS,
    _comparable,
    _validate_run_data,
)

RULE_VERSION = "optimizer_audit_v1"


def diagnose_optimizer_rules(reference: dict, candidate: dict) -> dict:
    """Classify recorded binding defects and observed final-epoch performance.

    Inputs are normalized ``inspect_optimizer_run`` results. Classification
    does not use directory names, variant labels, or benchmark expectations.
    The caller is responsible for loading and verifying the source artifacts.
    """
    for run in (reference, candidate):
        if not isinstance(run, dict):
            raise ValueError("Optimizer evidence must be a JSON object")
        _validate_run_data(run)

    audit = {field: candidate["optimizer_audit"][field] for field in AUDIT_FIELDS}
    mismatch = any(count > 0 for count in audit.values())
    final = candidate["epochs"][-1]
    comparable = _comparable(reference, candidate)
    comparisons = []
    performance_status = "inconclusive"
    if comparable:
        reference_final = reference["epochs"][-1]
        for metric in METRIC_FIELDS:
            higher_is_better = metric.endswith("accuracy")
            difference = final[metric] - reference_final[metric]
            improvement = difference if higher_is_better else -difference
            comparisons.append(
                {
                    "metric": metric,
                    "reference_value": reference_final[metric],
                    "candidate_value": final[metric],
                    "better_when": "higher" if higher_is_better else "lower",
                    "signed_improvement": improvement,
                }
            )
        improvements = [item["signed_improvement"] for item in comparisons]
        if all(value >= 0 for value in improvements):
            performance_status = "no_observed_degradation"
        elif all(value <= 0 for value in improvements):
            performance_status = "degraded"
        else:
            performance_status = "mixed"

    if mismatch:
        action = "review_optimizer_binding"
    elif performance_status == "no_observed_degradation":
        action = "no_change"
    else:
        action = "collect_more_evidence"

    return {
        "rule_version": RULE_VERSION,
        "implementation_status": (
            "optimizer_binding_defect" if mismatch else "no_optimizer_binding_defect"
        ),
        "root_cause": "optimizer_parameter_mismatch" if mismatch else None,
        "performance_status": performance_status,
        "recommended_action": action,
        "evidence": {
            "candidate_audit_counts": audit,
            "candidate_final_epoch": final["epoch"],
            "candidate_head_gradient_l2_norm": final["head"]["mean_gradient_l2_norm"],
            "candidate_head_parameter_update_l2_norm": (
                final["head"]["mean_parameter_update_l2_norm"]
            ),
            "comparable": comparable,
            "final_metric_comparisons": comparisons,
        },
    }
