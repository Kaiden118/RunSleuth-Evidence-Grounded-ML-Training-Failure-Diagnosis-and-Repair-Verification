"""Match run evidence against the declarative failure signature library.

Evidence comes from a variant run report written by the Camelyon17 runners and,
optionally, a healthy reference run of the same setup. Every signature receives
one verdict with the observed value behind each condition:

- supported: every required condition holds and no contradiction holds
- supported_pending_reference: only pending reference-dependent conditions are unevaluated
- not_supported: a required condition fails
- contradicted: a contradicting condition holds
- insufficient_evidence: a required value is missing

No thresholds are learned here; they live in the library with their sources.
"""

import argparse
import json
import math
import operator
from pathlib import Path

LIBRARY_PATH = Path(__file__).with_name("failure_signatures.json")
GROUPS = ("head", "backbone")
OPERATORS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}
GROUND_TRUTH = {
    "clean": "no_known_fault",
    "stale_head": "stale_optimizer_binding",
    "frozen_head": "frozen_head",
    "frozen_head_repaired": "no_known_fault",
    "high_learning_rate": "high_learning_rate",
    "high_learning_rate_repaired": "no_known_fault",
    "missing_optimizer_step": "missing_optimizer_step",
    "missing_optimizer_step_repaired": "no_known_fault",
}


def _conditions(signature: dict):
    for role in ("requires", "contradicts"):
        for condition in signature.get(role, []):
            yield from condition.get("any_of", [condition])


def load_library(path: Path = LIBRARY_PATH) -> dict:
    """Load and validate the library: unique ids, known operators, known thresholds."""
    library = json.loads(path.read_text(encoding="utf-8"))
    if library.get("schema_version") != 1:
        raise ValueError("Unsupported signature library schema version")
    ids = [signature["id"] for signature in library["signatures"]]
    if len(ids) != len(set(ids)) or "no_known_fault" in ids:
        raise ValueError("Signature ids must be unique and must not be no_known_fault")
    for signature in library["signatures"]:
        if not signature.get("requires"):
            raise ValueError(f"Signature {signature['id']} has no required conditions")
        for condition in _conditions(signature):
            if condition["op"] not in OPERATORS:
                raise ValueError(f"Unknown operator {condition['op']!r} in {signature['id']}")
            if condition.get("missing_reference", "pending") not in ("pending", "ignore"):
                raise ValueError(f"Unknown missing_reference value in {signature['id']}")
            value = condition["value"]
            if isinstance(value, str) and value.startswith("$"):
                if value[1:] not in library["thresholds"]:
                    raise ValueError(f"Unknown threshold {value!r} in {signature['id']}")
    return library


def _group_stats(run: dict, group: str) -> list[dict]:
    return [row[group] for row in run.get("parameter_group_epochs", []) if row.get(group)]


def _ratios(stats: list[dict], numerator: str) -> list[float] | None:
    if not stats or any(numerator not in item for item in stats):
        return None
    return [item[numerator] / item["parameter_tensors"] for item in stats]


def _inferred_first_step_lr(run: dict, group: str) -> float | None:
    rows = run.get("parameter_group_epochs", [])
    first = rows[0].get(group) if rows else None
    if not first or "first_step_parameter_update_l2_norm" not in first:
        return None
    return first["first_step_parameter_update_l2_norm"] / math.sqrt(first["parameter_elements"])


def _optimizer_family(optimizer_name: str | None) -> str | None:
    if optimizer_name is None:
        return None
    name = optimizer_name.lower()
    return "adam" if name.startswith("adam") else name


def extract_evidence(
    run: dict, reference: dict | None = None, optimizer_name: str | None = None
) -> dict:
    """Normalize one run report into named evidence values; None means unavailable.

    The optimizer name defaults to the one RunMonitor records. Gradient and update
    evidence fall back to RunMonitor's epoch-level measurements when the report has
    no per-batch training metrics.
    """
    rows = run.get("parameter_group_epochs", [])
    final = run.get("final_metrics", {})
    # The final audit also covers a head replaced after the monitor was created.
    audit = run.get("final_optimizer_audit") or run.get("optimizer_audit") or {}
    last = rows[-1] if rows else {}
    head_prefix = run.get("head_prefix", "fc.")
    optimizer_name = optimizer_name or run.get("optimizer")
    # Rows written before training_forwards existed come from the strict monitor,
    # which rejects any forward without a step, so they imply one step per forward.
    steps_per_forward = [
        row["optimizer_steps"] / forwards
        for row in rows
        if (forwards := row.get("training_forwards", row["optimizer_steps"])) > 0
    ]
    missing = audit.get("missing_trainable_names")
    evidence = {
        "optimizer_family": _optimizer_family(optimizer_name),
        "status_diverged": run.get("status") == "diverged",
        "max_optimizer_steps_per_forward": max(steps_per_forward) if steps_per_forward else None,
        "final_mean_gradient_norm": final.get(
            "mean_gradient_norm", last.get("gradient_l2_at_epoch_end")
        ),
        "final_mean_update_norm": final.get(
            "mean_parameter_update_norm", last.get("epoch_parameter_change_l2")
        ),
        "audit_missing_trainable_tensors": None if missing is None else len(missing),
        "audit_missing_head_tensors": None
        if missing is None
        else sum(name.startswith(head_prefix) for name in missing),
        "audit_foreign_tensors": audit.get("foreign_parameter_tensors"),
        "reference_available": reference is not None,
    }
    for group in GROUPS:
        stats = _group_stats(run, group)
        for bound, reduce in (("min", min), ("max", max)):
            for name, field in (
                ("trainable", "trainable_tensors"),
                ("gradient", "tensors_with_gradient"),
            ):
                values = _ratios(stats, f"{bound}_{field}")
                evidence[f"{group}_{bound}_{name}_fraction"] = (
                    None if values is None else reduce(values)
                )
        updates = [item["mean_parameter_update_l2_norm"] for item in stats]
        evidence[f"{group}_min_update_norm"] = min(updates) if updates else None
        evidence[f"{group}_max_update_norm"] = max(updates) if updates else None
        evidence[f"{group}_inferred_first_step_lr"] = _inferred_first_step_lr(run, group)
    evidence["reference_head_min_trainable_fraction"] = None
    evidence["backbone_inferred_lr_ratio_to_reference"] = None
    if reference is not None:
        base = extract_evidence(reference, None, optimizer_name)
        evidence["reference_head_min_trainable_fraction"] = base["head_min_trainable_fraction"]
        current, known = (
            evidence["backbone_inferred_first_step_lr"],
            base["backbone_inferred_first_step_lr"],
        )
        if current is not None and known:
            evidence["backbone_inferred_lr_ratio_to_reference"] = current / known
    return evidence


def _evaluate_condition(condition: dict, evidence: dict, thresholds: dict) -> dict:
    if "any_of" in condition:
        parts = [_evaluate_condition(part, evidence, thresholds) for part in condition["any_of"]]
        results = [part["result"] for part in parts]
        result = True if True in results else (None if None in results else False)
        return {"any_of": parts, "result": result}
    expected = condition["value"]
    if isinstance(expected, str) and expected.startswith("$"):
        expected = thresholds[expected[1:]]["value"]
    observed = evidence.get(condition["evidence"])
    needs_reference = bool(condition.get("needs_reference"))
    status = "evaluated"
    if needs_reference and not evidence.get("reference_available"):
        ignored = condition.get("missing_reference", "pending") == "ignore"
        status, result = ("skipped_without_reference" if ignored else "needs_reference"), None
    elif observed is None:
        status, result = "missing", None
    else:
        result = bool(OPERATORS[condition["op"]](observed, expected))
    return {
        "evidence": condition["evidence"],
        "op": condition["op"],
        "expected": expected,
        "observed": observed,
        "result": result,
        "status": status,
        "needs_reference": needs_reference,
        "meaning": condition.get("meaning"),
    }


def evaluate_signature(signature: dict, evidence: dict, thresholds: dict) -> dict:
    requires = [_evaluate_condition(item, evidence, thresholds) for item in signature["requires"]]
    active = [item for item in requires if item.get("status") != "skipped_without_reference"]
    contradicts = [
        _evaluate_condition(item, evidence, thresholds) for item in signature.get("contradicts", [])
    ]
    if any(item["result"] is True for item in contradicts):
        verdict = "contradicted"
    elif all(item["result"] is True for item in active):
        verdict = "supported"
    elif any(item["result"] is False for item in active):
        verdict = "not_supported"
    elif all(item["result"] is True or item.get("status") == "needs_reference" for item in active):
        verdict = "supported_pending_reference"
    else:
        verdict = "insufficient_evidence"
    return {
        "id": signature["id"],
        "subsystem": signature["subsystem"],
        "scope": signature["scope"],
        "verdict": verdict,
        "requires": requires,
        "contradicts": contradicts,
        "repair": signature["repair"],
    }


def diagnose(evidence: dict, library: dict) -> dict:
    """Evaluate every signature and summarize; no ranking beyond the verdicts yet."""
    verdicts = [
        evaluate_signature(signature, evidence, library["thresholds"])
        for signature in library["signatures"]
    ]
    supported = [item["id"] for item in verdicts if item["verdict"] == "supported"]
    pending = [item["id"] for item in verdicts if item["verdict"] == "supported_pending_reference"]
    if len(supported) == 1:
        diagnosis = supported[0]
    elif supported:
        diagnosis = "ambiguous"
    elif pending:
        diagnosis = "pending_reference"
    else:
        diagnosis = "no_known_fault"
    return {
        "diagnosis": diagnosis,
        "supported": supported,
        "pending_reference": pending,
        "reference_available": evidence["reference_available"],
        "evidence": evidence,
        "verdicts": verdicts,
    }


def _experiment_optimizer(report_path: Path) -> str | None:
    environment = report_path.parent / "environment.json"
    if not environment.is_file():
        return None
    return json.loads(environment.read_text(encoding="utf-8")).get("optimizer")


def evaluate_reports(report_paths: list[Path], library: dict) -> dict:
    """Diagnose every labeled variant with and without its experiment's clean run."""
    cases = []
    for path in report_paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") != "completed":
            raise ValueError(f"Experiment report is not completed: {path}")
        optimizer_name = _experiment_optimizer(path)
        reference = report["variants"]["clean"]
        for name, run in report["variants"].items():
            if name not in GROUND_TRUTH:
                raise ValueError(f"No ground-truth label for variant {name!r} in {path}")
            outcome = {"report": str(path), "seed": report.get("seed"), "variant": name}
            outcome["truth"] = GROUND_TRUTH[name]
            for mode, base in (("reference_free", None), ("with_reference", reference)):
                result = diagnose(extract_evidence(run, base, optimizer_name), library)
                outcome[mode] = {
                    "diagnosis": result["diagnosis"],
                    "supported": result["supported"],
                    "pending_reference": result["pending_reference"],
                }
            cases.append(outcome)
    summary = {}
    for mode in ("reference_free", "with_reference"):
        confusion: dict[str, dict[str, int]] = {}
        for case in cases:
            row = confusion.setdefault(case["truth"], {})
            predicted = case[mode]["diagnosis"]
            row[predicted] = row.get(predicted, 0) + 1
        healthy = [case for case in cases if case["truth"] == "no_known_fault"]
        summary[mode] = {
            "exact_matches": sum(case[mode]["diagnosis"] == case["truth"] for case in cases),
            "pending_reference_with_true_fault": sum(
                case[mode]["diagnosis"] == "pending_reference"
                and case["truth"] in case[mode]["pending_reference"]
                for case in cases
            ),
            "cases": len(cases),
            "healthy_false_positives": sum(
                case[mode]["diagnosis"] != "no_known_fault" for case in healthy
            ),
            "healthy_cases": len(healthy),
            "confusion": confusion,
        }
    return {
        "scope": "development evaluation on author-injected faults; thresholds were not "
        "calibrated on these runs",
        "summary": summary,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    single = commands.add_parser("diagnose", help="Diagnose one variant run report")
    single.add_argument("--run", type=Path, required=True)
    single.add_argument("--reference", type=Path)
    single.add_argument(
        "--optimizer", help="Optimizer class name, for example AdamW (default: from the report)"
    )
    batch = commands.add_parser("evaluate", help="Score labeled experiment reports")
    batch.add_argument("reports", type=Path, nargs="+")
    batch.add_argument("--output", type=Path)
    args = parser.parse_args()
    library = load_library()
    if args.command == "diagnose":
        run = json.loads(args.run.read_text(encoding="utf-8"))
        reference = (
            json.loads(args.reference.read_text(encoding="utf-8")) if args.reference else None
        )
        result = diagnose(extract_evidence(run, reference, args.optimizer), library)
        print(json.dumps(result, indent=2, allow_nan=False))
        return
    result = evaluate_reports(args.reports, library)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", "utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
