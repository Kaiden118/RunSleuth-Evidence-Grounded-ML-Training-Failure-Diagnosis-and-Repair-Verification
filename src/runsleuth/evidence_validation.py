"""Check citation values and proposed-patch provenance against recorded tool results."""

from math import isfinite

from runsleuth.agent_diagnosis import AgentDiagnosis
from runsleuth.tool_contracts import LoadRunConfigArgs


def _values_match(actual: object, expected: object) -> bool:
    """Allow equal JSON numbers, but never confuse booleans with numbers."""
    if type(actual) is bool or type(expected) is bool:
        return actual is expected
    if type(actual) in (int, float) and type(expected) in (int, float):
        return (
            (type(actual) is not float or isfinite(actual))
            and (type(expected) is not float or isfinite(expected))
            and actual == expected
        )
    return type(actual) is type(expected) and actual == expected


def _read_path(data: dict[str, object], path: list[str | int]) -> object:
    """Resolve dictionary keys and list indices without evaluating code."""
    current: object = data
    for part in path:
        if type(current) is dict and type(part) is str and part in current:
            current = current[part]
        elif type(current) is list and type(part) is int and 0 <= part < len(current):
            current = current[part]
        else:
            raise ValueError(f"Evidence path does not exist: {path!r}")

    if current is not None and type(current) not in (str, int, float, bool):
        raise ValueError(f"Evidence path must identify a scalar value: {path!r}")
    return current


def _successful_call(
    calls: dict[str, dict[str, object]],
    call_id: str,
) -> tuple[dict[str, object], dict[str, object]]:
    entry = calls.get(call_id)
    if entry is None:
        raise ValueError(f"Unknown evidence call ID: {call_id}")

    result = entry.get("result")
    if not isinstance(result, dict) or result.get("ok") is not True:
        raise ValueError(f"Evidence call did not succeed: {call_id}")
    if result.get("tool_name") != entry.get("tool_name"):
        raise ValueError(f"Inconsistent tool identity: {call_id}")

    data = result.get("data")
    if not isinstance(data, dict):
        raise ValueError(f"Evidence call has no data object: {call_id}")
    return entry, data


def validate_diagnosis_evidence(
    diagnosis: AgentDiagnosis,
    tool_trace: list[dict[str, object]],
    *,
    candidate_run: str,
) -> None:
    """Reject invalid citations and patches; do not certify causal correctness."""
    calls: dict[str, dict[str, object]] = {}
    for entry in tool_trace:
        call_id = entry.get("call_id")
        if not isinstance(call_id, str) or not call_id.strip() or call_id in calls:
            raise ValueError("Tool trace contains an invalid or duplicate call ID")
        calls[call_id] = entry

    for cause in diagnosis.ranked_causes:
        for citation in cause.evidence:
            _, data = _successful_call(calls, citation.call_id)
            try:
                actual = _read_path(data, citation.path)
            except ValueError as error:
                raise ValueError(f"Call {citation.call_id}: {error}") from error
            if not _values_match(actual, citation.expected_value):
                raise ValueError(
                    f"Evidence value mismatch: {citation.call_id} at {citation.path!r}"
                )

    patch = diagnosis.proposed_patch
    if patch is None:
        return

    entry, data = _successful_call(calls, patch.config_call_id)
    if entry.get("tool_name") != "load_run_config":
        raise ValueError("Patch must cite a load_run_config call")

    raw_arguments = entry.get("raw_arguments")
    if not isinstance(raw_arguments, str):
        raise ValueError("Configuration call has no recorded JSON arguments")
    arguments = LoadRunConfigArgs.model_validate_json(raw_arguments)
    if arguments.run_directory != candidate_run.strip():
        raise ValueError("Patch must target the candidate run configuration")

    config_path: list[str | int] = ["config", patch.field]
    recorded_value = _read_path(data, config_path)
    if not _values_match(recorded_value, patch.old_value):
        raise ValueError("Patch old_value does not match the recorded configuration")

    if not diagnosis.ranked_causes:
        raise ValueError("Patch requires a supported root cause")
    top_cause = diagnosis.ranked_causes[0]
    if not any(
        citation.call_id == patch.config_call_id
        and citation.path == config_path
        and _values_match(citation.expected_value, patch.old_value)
        for citation in top_cause.evidence
    ):
        raise ValueError("First-ranked cause must cite the configuration value being changed")
