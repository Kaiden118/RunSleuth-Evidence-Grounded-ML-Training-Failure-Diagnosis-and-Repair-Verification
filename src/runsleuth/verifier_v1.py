"""RunSleuth v1 on pipeline runs: stage one's verifier, applied as it was built.

v1 decides from a candidate's run and the healthy project's run on the same seed:

- reject: re-diagnosis against the reference finds a known fault signature, the metric
  gate rejects, or the candidate did not complete a run;
- accept: none of these, and the run is the reference's run: every hashed model state
  agrees through the final state of a full run, and so does every validation metric;
- review: none of these, but the run is not the reference's, or that cannot be shown.
  differences says where the runs part; whether that is harmless is left to a reviewer.

Adapted to the pipeline harness, not changed. Stage one's harness evaluated the network
after every epoch; the pipeline harness leaves that to the project and evaluates the
delivered weights instead, so "every validation metric" is read from what the project
reported after every epoch and from the harness's evaluation of the delivered weights.
The gate is the calibrated metric gate, so that a comparison with v2 is not one of
thresholds; stage one's own gate is recorded beside it. agent_verification itself is
left as it is: the stage-one study's frozen sources stay unchanged.

    python -m runsleuth.verifier_v1 --family camelyon_pipeline \\
        --candidate runs/fixed/run_report.json --reference runs/healthy/run_report.json
"""

import argparse
import json
from pathlib import Path
from typing import Any

from runsleuth import metric_gate, pipeline_tasks
from runsleuth.agent_verification import NO_FAULT, compare_trajectories, differences
from runsleuth.signature_matching import diagnose, extract_evidence, load_library


def _reported(run: dict[str, Any]) -> list[Any]:
    return [row.get("project_metrics") for row in run.get("parameter_group_epochs", [])]


def same_evaluation(candidate: dict[str, Any], reference: dict[str, Any]) -> bool:
    """Whether two runs reported the same validation metrics after every epoch and
    delivered weights the harness evaluated the same."""
    ours = _reported(candidate)
    return (
        bool(ours)
        and None not in ours
        and ours == _reported(reference)
        and candidate.get("final_metrics") == reference.get("final_metrics")
    )


def _triage(diagnosis: str, gate: str, same_run: bool) -> str:
    if diagnosis != NO_FAULT or gate != "accept":
        return "reject"
    return "accept" if same_run else "review"


def verify(
    candidate: dict[str, Any] | None,
    reference: dict[str, Any],
    gate: dict[str, float],
    library: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """v1's triage of a candidate's run report against the reference's on its seed."""
    gated = metric_gate.decide(candidate, reference, gate)
    stage_one = metric_gate.decide_stage_one(candidate, reference)
    if candidate is None or candidate.get("status") != "completed":
        return {
            "triage": "reject",
            "triage_with_stage_one_gate": "reject",
            "re_diagnosis": None,
            "gate": gated,
            "trajectory": None,
            "same_evaluation": False,
            "differences": gated["reasons"],
        }
    library = library if library is not None else load_library()
    found = diagnose(extract_evidence(candidate, reference), library)["diagnosis"]
    trajectory = compare_trajectories(candidate, reference)
    evaluated = same_evaluation(candidate, reference)
    same_run = trajectory["outcome"] == "identical_full_run" and evaluated
    found_differences = []
    if not same_run:
        # Stage one's differences, but for its evaluation line, which reads metrics the
        # pipeline harness does not record.
        found_differences = [
            line
            for line in differences(candidate, reference, trajectory)
            if not line.startswith("evaluation:")
        ]
        if _reported(candidate) != _reported(reference):
            found_differences.append("evaluation: the project reported other validation metrics")
        if candidate.get("final_metrics") != reference.get("final_metrics"):
            found_differences.append("evaluation: the delivered weights score otherwise")
    return {
        "triage": _triage(found, gated["decision"], same_run),
        "triage_with_stage_one_gate": _triage(found, stage_one, same_run),
        "re_diagnosis": found,
        "gate": gated,
        "trajectory": trajectory,
        "same_evaluation": evaluated,
        "differences": found_differences,
    }


def verification_text(result: dict[str, Any]) -> str:
    lines = [
        f"Triage: {result['triage']} (with stage one's gate: "
        f"{result['triage_with_stage_one_gate']})",
        f"Re-diagnosis: {result['re_diagnosis']}",
        f"Metric gate: {result['gate']['decision']}",
    ]
    lines += [f"  {reason}" for reason in result["gate"]["reasons"]]
    lines += [f"Differs: {line}" for line in result["differences"]]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RunSleuth v1 on a pipeline run.")
    parser.add_argument("--family", choices=tuple(pipeline_tasks.FAMILIES), required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    candidate, reference = (
        json.loads(path.read_text(encoding="utf-8")) for path in (args.candidate, args.reference)
    )
    result = verify(candidate, reference, metric_gate.load_gate(args.family))
    print(json.dumps(result, indent=2) if args.json else verification_text(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
