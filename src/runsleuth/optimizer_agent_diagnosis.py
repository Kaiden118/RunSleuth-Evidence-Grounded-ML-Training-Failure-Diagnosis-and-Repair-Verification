"""Structured, read-only optimizer diagnosis with checked recorded evidence."""

import json
import math
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    StringConstraints,
    model_validator,
)

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
Scalar = StrictBool | StrictInt | StrictFloat | Annotated[StrictStr, Field(max_length=1000)] | None
PathPart = Annotated[StrictStr, Field(min_length=1, max_length=120)] | StrictInt

AUDIT_FIELDS = (
    "missing_trainable_parameter_tensors",
    "foreign_parameter_tensors",
    "duplicate_parameter_occurrences",
)
METRIC_FIELDS = (
    "id_validation_accuracy",
    "id_validation_loss",
    "ood_validation_accuracy",
    "ood_validation_loss",
)
COMPARABLE_CONFIG_FIELDS = (
    "dataset",
    "dataset_version",
    "model",
    "pretrained_weights",
    "hf_revision",
    "seed",
    "subset_seed",
    "epochs",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "max_train_samples",
    "max_id_val_samples",
    "max_ood_val_samples",
    "num_workers",
    "device",
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class OptimizerEvidence(_StrictModel):
    call_id: Annotated[str, Field(min_length=1, max_length=200)]
    path: Annotated[list[PathPart], Field(min_length=1, max_length=12)]
    expected_value: Scalar


class OptimizerCause(_StrictModel):
    root_cause: Literal["optimizer_parameter_mismatch"]
    confidence: Literal["high", "medium", "low"]
    explanation: Text
    evidence: Annotated[list[OptimizerEvidence], Field(min_length=1, max_length=24)]


class OptimizerPerformance(_StrictModel):
    status: Literal["no_observed_degradation", "degraded", "mixed", "inconclusive"]
    explanation: Text
    evidence: Annotated[list[OptimizerEvidence], Field(max_length=24)]


class OptimizerDiagnosis(_StrictModel):
    """A structural finding and a separate performance observation; never a repair."""

    profile: Literal["optimizer_binding"]
    implementation_status: Literal[
        "optimizer_binding_defect", "no_optimizer_binding_defect", "inconclusive"
    ]
    summary: Text
    implementation_evidence: Annotated[list[OptimizerEvidence], Field(min_length=1, max_length=24)]
    ranked_causes: Annotated[list[OptimizerCause], Field(max_length=1)]
    performance: OptimizerPerformance
    uncertainties: Annotated[list[Text], Field(min_length=1, max_length=5)]
    recommended_action: Literal["review_optimizer_binding", "no_change", "collect_more_evidence"]
    proposed_patch: None

    @model_validator(mode="after")
    def consistent_status(self) -> "OptimizerDiagnosis":
        if self.implementation_status == "optimizer_binding_defect":
            if (
                len(self.ranked_causes) != 1
                or self.recommended_action != "review_optimizer_binding"
            ):
                raise ValueError(
                    "An optimizer defect requires one cause and review_optimizer_binding"
                )
        elif self.ranked_causes:
            raise ValueError("Only an optimizer defect may include a ranked cause")
        elif self.implementation_status == "inconclusive":
            if self.recommended_action != "collect_more_evidence":
                raise ValueError("An inconclusive implementation finding requires more evidence")
        elif self.recommended_action == "review_optimizer_binding":
            raise ValueError("Reviewing a supported binding defect requires a defect finding")
        if self.performance.status in {"degraded", "mixed", "inconclusive"}:
            if self.recommended_action == "no_change":
                raise ValueError("Unresolved performance requires review or more evidence")
        return self


def optimizer_api_schema() -> dict:
    """Adapt the wire schema without relaxing local diagnosis validation."""
    schema = OptimizerDiagnosis.model_json_schema()
    fixed_fields = (
        schema["properties"]["profile"],
        schema["$defs"]["OptimizerCause"]["properties"]["root_cause"],
    )
    for field in fixed_fields:
        field["enum"] = [field.pop("const")]

    schema["properties"]["proposed_patch"] = {
        "type": ["string", "null"],
        "description": "Must be null. This read-only profile does not propose repairs.",
    }
    return schema


def _same_value(left: object, right: object) -> bool:
    """Preserve JSON types: false, zero, and 0.0 are different evidence values."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _same_value(v, right[k]) for k, v in left.items()
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _same_value(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, float) and not math.isfinite(left):
        return False
    return left == right


def _unique_object(items: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate key in tool arguments")
        result[key] = value
    return result


def _lookup(data: dict, path: list[str | int]) -> object:
    current = data
    for component in path:
        if type(component) is str and isinstance(current, dict) and component in current:
            current = current[component]
        elif type(component) is int and isinstance(current, list) and 0 <= component < len(current):
            current = current[component]
        else:
            raise ValueError(f"Evidence path does not exist: {path}")
    if type(current) not in {str, int, float, bool, type(None)}:
        raise ValueError("Evidence must reference a scalar value")
    if isinstance(current, float) and not math.isfinite(current):
        raise ValueError("Evidence must be finite")
    return current


def _number(value: object, name: str, *, accuracy: bool = False) -> float:
    try:
        valid = type(value) in {int, float} and math.isfinite(value) and value >= 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"Invalid nonnegative metric: {name}")
    if accuracy and value > 1:
        raise ValueError(f"Invalid accuracy: {name}")
    return float(value)


def _validate_run_data(data: dict) -> None:
    audit = data.get("optimizer_audit")
    if not isinstance(audit, dict):
        raise ValueError("Missing optimizer audit")
    for key in AUDIT_FIELDS:
        if type(audit.get(key)) is not int or audit[key] < 0:
            raise ValueError(f"Invalid audit count: {key}")
    names = audit.get("missing_trainable_names")
    if (
        not isinstance(names, list)
        or any(type(name) is not str or not name for name in names)
        or len(set(names)) != len(names)
        or len(names) != audit["missing_trainable_parameter_tensors"]
    ):
        raise ValueError("Missing-parameter names and count are inconsistent")
    rows = data.get("epochs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("No recorded optimizer epochs")
    for expected_epoch, row in enumerate(rows, 1):
        if (
            not isinstance(row, dict)
            or type(row.get("epoch")) is not int
            or row["epoch"] != expected_epoch
        ):
            raise ValueError("Epoch records must be complete and ordered")
        if type(row.get("optimizer_steps")) is not int or row["optimizer_steps"] <= 0:
            raise ValueError("Each epoch must record positive optimizer steps")
        for key in METRIC_FIELDS:
            _number(row.get(key), key, accuracy=key.endswith("accuracy"))
        for group in ("head", "backbone"):
            values = row.get(group)
            if not isinstance(values, dict):
                raise ValueError("Missing parameter-group telemetry")
            for key in ("mean_gradient_l2_norm", "mean_parameter_update_l2_norm"):
                _number(values.get(key), f"{group}.{key}")
    config = data.get("config")
    if (
        not isinstance(config, dict)
        or type(config.get("epochs")) is not int
        or config["epochs"] != len(rows)
    ):
        raise ValueError("Recorded epochs must match the run configuration")


def _collect_runs(trace: list[dict], allowed_runs: set[str]) -> tuple[dict, dict, dict]:
    all_calls = {}
    calls = {}
    runs = {}
    run_call_ids = {path: set() for path in allowed_runs}
    for entry in trace:
        if not isinstance(entry, dict):
            raise ValueError("Malformed tool trace entry")
        call_id = entry.get("call_id")
        if type(call_id) is not str or not call_id or call_id in all_calls:
            raise ValueError("Tool call IDs must be present and unique")
        all_calls[call_id] = entry
        result = entry.get("result")
        if not isinstance(result, dict) or result.get("ok") is not True:
            continue
        if entry.get("tool_name") != "inspect_optimizer_run":
            continue
        if result.get("tool_name") != "inspect_optimizer_run":
            raise ValueError("Tool result name does not match the request")
        try:
            arguments = json.loads(entry["raw_arguments"], object_pairs_hook=_unique_object)
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("Invalid recorded tool arguments") from error
        if not isinstance(arguments, dict) or set(arguments) != {"run_directory"}:
            raise ValueError("Unexpected optimizer inspection arguments")
        path = arguments["run_directory"]
        if type(path) is not str or path not in allowed_runs:
            raise ValueError("Optimizer inspection is outside the fixed task scope")
        data = result.get("data")
        if not isinstance(data, dict) or data.get("run_directory") != path:
            raise ValueError("Tool data does not belong to the requested run")
        _validate_run_data(data)
        if path in runs and not _same_value(runs[path], data):
            raise ValueError("Repeated inspections returned inconsistent run evidence")
        runs[path] = data
        calls[call_id] = data
        run_call_ids[path].add(call_id)
    if set(runs) != allowed_runs:
        raise ValueError("Both reference and candidate require successful optimizer inspection")
    return calls, runs, run_call_ids


def _require_paths(
    evidence: list[OptimizerEvidence], call_ids: set[str], paths: list[list]
) -> None:
    for path in paths:
        if not any(item.call_id in call_ids and item.path == path for item in evidence):
            raise ValueError(f"A relevant evidence citation is required for {path}")


def _comparable(reference: dict, candidate: dict) -> bool:
    if not reference.get("initial_state_sha256") or not reference.get("data_identity"):
        return False
    for key in ("initial_state_sha256", "data_identity"):
        if not _same_value(reference.get(key), candidate.get(key)):
            return False
    for key in COMPARABLE_CONFIG_FIELDS:
        if key not in reference["config"] or key not in candidate["config"]:
            return False
        if not _same_value(reference["config"][key], candidate["config"][key]):
            return False
    return True


def _performance_improvements(reference: dict, candidate: dict) -> dict[str, float]:
    left, right = reference["epochs"][-1], candidate["epochs"][-1]
    return {
        key: (right[key] - left[key]) * (1 if key.endswith("accuracy") else -1)
        for key in METRIC_FIELDS
    }


def _performance_status(reference: dict, candidate: dict) -> str:
    if not _comparable(reference, candidate):
        return "inconclusive"
    improvements = _performance_improvements(reference, candidate).values()
    if all(value >= 0 for value in improvements):
        return "no_observed_degradation"
    if all(value <= 0 for value in improvements):
        return "degraded"
    return "mixed"


def build_optimizer_validation_feedback(
    error: str,
    trace: list[dict],
    *,
    reference_run: str,
    candidate_run: str,
) -> dict[str, object]:
    """Build correction guidance from recorded tools, independently of rejected model JSON."""
    feedback: dict[str, object] = {
        "error": error,
        "citation_rule": (
            "Use scalar evidence values, never an entire list or object. For a ranked "
            "optimizer mismatch, cite a nonzero candidate audit count. Individual names "
            "may be extra citations using an integer list index."
        ),
    }
    try:
        _, runs, run_call_ids = _collect_runs(trace, {reference_run, candidate_run})
    except ValueError as evidence_error:
        feedback["performance_check"] = {"available": False, "error": str(evidence_error)}
        return feedback

    reference, candidate = runs[reference_run], runs[candidate_run]
    expected_status = _performance_status(reference, candidate)
    comparisons = []
    if expected_status != "inconclusive":
        left, right = reference["epochs"][-1], candidate["epochs"][-1]
        for metric, improvement in _performance_improvements(reference, candidate).items():
            comparisons.append(
                {
                    "metric": metric,
                    "reference_value": left[metric],
                    "candidate_value": right[metric],
                    "better_when": "higher" if metric.endswith("accuracy") else "lower",
                    "signed_improvement": improvement,
                    "observation": (
                        "improved"
                        if improvement > 0
                        else "worsened"
                        if improvement < 0
                        else "unchanged"
                    ),
                }
            )
    feedback["performance_check"] = {
        "available": True,
        "comparable": expected_status != "inconclusive",
        "expected_status": expected_status,
        "reference_run": reference_run,
        "candidate_run": candidate_run,
        "reference_call_ids": sorted(run_call_ids[reference_run]),
        "candidate_call_ids": sorted(run_call_ids[candidate_run]),
        "final_metric_comparisons": comparisons,
    }
    return feedback


def validate_optimizer_diagnosis_evidence(
    diagnosis: OptimizerDiagnosis,
    trace: list[dict],
    *,
    reference_run: str,
    candidate_run: str,
) -> None:
    """Check citations and classification against the scoped recorded tool data.

    This verifies machine-readable observations, not the semantic correctness of
    every sentence in the explanation or a causal claim about model performance.
    """
    calls, runs, run_call_ids = _collect_runs(trace, {reference_run, candidate_run})
    evidence = list(diagnosis.implementation_evidence)
    evidence.extend(item for cause in diagnosis.ranked_causes for item in cause.evidence)
    evidence.extend(diagnosis.performance.evidence)
    for item in evidence:
        if item.call_id not in calls:
            raise ValueError("Evidence must cite a successful, scoped inspect_optimizer_run call")
        actual = _lookup(calls[item.call_id], item.path)
        if not _same_value(actual, item.expected_value):
            raise ValueError("Evidence value or JSON type does not match the tool result")

    reference, candidate = runs[reference_run], runs[candidate_run]
    mismatch = any(candidate["optimizer_audit"][key] > 0 for key in AUDIT_FIELDS)
    if diagnosis.implementation_status == "optimizer_binding_defect" and not mismatch:
        raise ValueError("Optimizer mismatch is not supported by the candidate audit")
    if diagnosis.implementation_status == "no_optimizer_binding_defect" and mismatch:
        raise ValueError("The candidate audit contains an optimizer mismatch")
    if diagnosis.implementation_status != "inconclusive":
        required = [["optimizer_audit", key] for key in AUDIT_FIELDS]
        if mismatch:
            last = len(candidate["epochs"]) - 1
            required.extend(
                [
                    ["epochs", last, "head", "mean_gradient_l2_norm"],
                    ["epochs", last, "head", "mean_parameter_update_l2_norm"],
                ]
            )
        _require_paths(diagnosis.implementation_evidence, run_call_ids[candidate_run], required)
        if mismatch:
            cause_evidence = diagnosis.ranked_causes[0].evidence
            mismatch_paths = [
                ["optimizer_audit", key]
                for key in AUDIT_FIELDS
                if candidate["optimizer_audit"][key] > 0
            ]
            if not any(
                item.call_id in run_call_ids[candidate_run] and item.path in mismatch_paths
                for item in cause_evidence
            ):
                raise ValueError("The ranked cause must cite a nonzero candidate mismatch count")

    expected_status = _performance_status(reference, candidate)
    if diagnosis.performance.status != expected_status:
        raise ValueError(f"Performance status must be {expected_status} for these recorded metrics")
    if expected_status != "inconclusive":
        for path in {reference_run, candidate_run}:
            last = len(runs[path]["epochs"]) - 1
            required = [["epochs", last, key] for key in METRIC_FIELDS]
            _require_paths(diagnosis.performance.evidence, run_call_ids[path], required)
