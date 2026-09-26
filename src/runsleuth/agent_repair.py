"""Prepare controlled repairs from validated agent reports."""

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from runsleuth.agent_diagnosis import AgentDiagnosis
from runsleuth.config import TrainingConfig
from runsleuth.diagnosis import RootCause
from runsleuth.diagnostic_tools import DiagnosticTools
from runsleuth.evidence_validation import validate_diagnosis_evidence
from runsleuth.experiment import run_experiment
from runsleuth.repair import apply_config_patch
from runsleuth.verification import (
    VerificationPolicy,
    VerificationReport,
    verify_repair,
)


@dataclass(frozen=True)
class PreparedAgentRepair:
    """An agent proposal checked against current local evidence."""

    agent_report: str
    model: str
    diagnosis: AgentDiagnosis
    failed_config: TrainingConfig
    repaired_config: TrainingConfig


def _result_json(value: object) -> str:
    """Compare JSON values without treating False as equal to zero."""
    return json.dumps(value, sort_keys=True, allow_nan=False)


def prepare_agent_repair(
    agent_report_path: Path,
    *,
    workspace_root: Path,
    source_path: Path,
    reference_run: Path,
    failed_run: Path,
) -> PreparedAgentRepair:
    """Recheck saved evidence and prepare an allowlisted config change."""
    report_path = workspace_root / agent_report_path
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise ValueError("Agent report must be a JSON object")

    if payload.get("status") != "completed" or payload.get("pending_evidence") != []:
        raise ValueError("Repair requires a completed agent report")

    model = payload.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Agent report must identify the model")

    diagnosis = AgentDiagnosis.model_validate(payload.get("diagnosis"))
    if diagnosis.status != "failure_detected" or diagnosis.proposed_patch is None:
        raise ValueError("Agent diagnosis does not contain a supported repair")

    trace = payload.get("tool_trace")
    if not isinstance(trace, list) or not all(isinstance(entry, dict) for entry in trace):
        raise ValueError("Agent report must contain a valid tool trace")

    reference_name = reference_run.as_posix()
    failed_name = failed_run.as_posix()

    validate_diagnosis_evidence(
        diagnosis,
        trace,
        candidate_run=failed_name,
    )

    expected_requests = [
        (
            "inspect_training_source",
            {"source_path": source_path.as_posix()},
        ),
        (
            "load_run_config",
            {"run_directory": reference_name},
        ),
        (
            "load_run_config",
            {"run_directory": failed_name},
        ),
        (
            "compare_runs",
            {
                "reference_run": reference_name,
                "candidate_run": failed_name,
            },
        ),
    ]
    remaining_requests = expected_requests.copy()
    tools = DiagnosticTools(workspace_root)
    failed_config = None

    for entry in trace:
        saved_result = entry.get("result")
        if not isinstance(saved_result, dict) or type(saved_result.get("ok")) is not bool:
            raise ValueError("Tool trace contains an invalid result")

        if saved_result["ok"] is False:
            continue

        raw_arguments = entry.get("raw_arguments")
        if not isinstance(raw_arguments, str):
            raise ValueError("Tool call must contain recorded JSON arguments")

        arguments = json.loads(raw_arguments)
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be a JSON object")

        name = entry.get("tool_name")
        request = (name, arguments)
        if request not in expected_requests:
            raise ValueError(f"Unexpected successful evidence request: {name}")

        fresh_result = tools.execute(name, arguments)
        if not fresh_result.ok or fresh_result.data is None:
            raise ValueError(f"Cannot refresh evidence from tool: {name}")

        if _result_json(asdict(fresh_result)) != _result_json(saved_result):
            raise ValueError(
                f"Saved evidence differs from current tool results: {entry['call_id']}"
            )

        if request in remaining_requests:
            remaining_requests.remove(request)

        if request == (
            "load_run_config",
            {"run_directory": failed_name},
        ):
            failed_config = TrainingConfig(**fresh_result.data["config"])

    if remaining_requests or failed_config is None:
        raise ValueError("Agent report is missing required successful evidence")

    patch = diagnosis.proposed_patch
    repaired_config = apply_config_patch(
        failed_config,
        root_cause=RootCause(diagnosis.ranked_causes[0].root_cause),
        field=patch.field,
        old_value=patch.old_value,
        new_value=patch.new_value,
    )

    return PreparedAgentRepair(
        agent_report=str(agent_report_path),
        model=model,
        diagnosis=diagnosis,
        failed_config=failed_config,
        repaired_config=repaired_config,
    )


@dataclass(frozen=True)
class AgentRepairAttemptReport:
    """Record the agent proposal, training budget, and verification result."""

    agent_report: str
    model: str
    diagnosis: AgentDiagnosis
    max_epochs: int
    repaired_config: TrainingConfig
    repaired_run: str
    verification: VerificationReport

    def to_json(self) -> str:
        payload = {
            "agent_report": self.agent_report,
            "model": self.model,
            "diagnosis": self.diagnosis.model_dump(mode="json"),
            "training_mode": "from_scratch",
            "max_epochs": self.max_epochs,
            "repaired_config": asdict(self.repaired_config),
            "repaired_run": self.repaired_run,
            "verification": asdict(self.verification),
        }
        return json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)


def run_bounded_agent_repair(
    agent_report_path: Path,
    *,
    source_path: Path,
    reference_run: Path,
    failed_run: Path,
    max_epochs: int = 2,
    policy: VerificationPolicy | None = None,
) -> AgentRepairAttemptReport:
    """Execute one agent-proposed repair from the current project directory."""
    if type(max_epochs) is not int or max_epochs < 1:
        raise ValueError("max_epochs must be a positive integer")

    prepared = prepare_agent_repair(
        agent_report_path,
        workspace_root=Path.cwd(),
        source_path=source_path,
        reference_run=reference_run,
        failed_run=failed_run,
    )

    original_epochs = prepared.failed_config.epochs
    if type(original_epochs) is not int or original_epochs < 1:
        raise ValueError("Failed configuration must contain positive integer epochs")

    repaired_config = replace(
        prepared.repaired_config,
        run_name=f"{prepared.failed_config.run_name}-agent-repair",
        epochs=min(original_epochs, max_epochs),
    )

    repaired_run = run_experiment(repaired_config)

    verification = verify_repair(
        reference_run=reference_run,
        failed_run=failed_run,
        repaired_run=repaired_run,
        policy=policy,
    )

    return AgentRepairAttemptReport(
        agent_report=prepared.agent_report,
        model=prepared.model,
        diagnosis=prepared.diagnosis,
        max_epochs=max_epochs,
        repaired_config=repaired_config,
        repaired_run=str(repaired_run),
        verification=verification,
    )
