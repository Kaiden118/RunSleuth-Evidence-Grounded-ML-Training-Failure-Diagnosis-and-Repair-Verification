"""Verify a repaired workspace run, and say what each part of verification decides.

verify_repair_run() decides as runsleuth verify does: a repair is accepted when
re-diagnosis against the healthy reference finds no known fault (the mechanism check)
and no validation metric regresses against that reference. It adds two records for
studying how agents repair training code:

- criteria: what four acceptance criteria would each decide on the same run, so the
  parts of verification can be compared. Metrics alone accept an unrepaired fault that
  is silent in metrics; the mechanism check alone accepts a repair that removes the
  fault and harms performance; before-and-after metrics reject a correct repair whose
  metrics fall, as they do when a leak is removed.
- trajectory: whether the model states the two runs hashed are the same. A run can be
  stopped early (stop_after_epochs) to make this a cheap, bounded check; agreement
  then covers only the optimizer steps and epoch ends both runs hashed. A fault that
  triggers later, or one that acts only on evaluation, agrees too, so the comparison
  never decides by itself, and a correct repair need not agree at all.

    python -m runsleuth.agent_verification --candidate runs/fixed/run_report.json \\
        --reference runs/healthy/run_report.json --faulty runs/faulty/run_report.json
"""

import argparse
import json
from pathlib import Path

from runsleuth.cli import validation_splits, verification_text, verify_runs

NO_FAULT = "no_known_fault"
# What the harness must have set alike for two runs' hashed states to be comparable. The
# steps per epoch are left out: they follow from the data a script admits, so a script
# that trains on other data simply diverges at its first step.
SETUP = ("seed", "epochs", "max_train_batches", "hashed_steps")


def compare_trajectories(candidate: dict, reference: dict) -> dict:
    """Compare the model states two workspace runs hashed, under one seed and setup.

    outcome is one of:
    - identical_full_run: every hashed state agrees, through the final state of a
      run that trained all its epochs;
    - identical_over_checked_steps: every state both runs hashed agrees, but at least
      one run stopped early, so later steps are unknown;
    - diverged: a state differs, and first_difference names the first one;
    - not_comparable: reason says why.
    checked counts what agreement covers: optimizer steps hashed one by one, and
    epoch ends out of the epochs a full run trains.
    """
    ours, theirs = candidate.get("workspace"), reference.get("workspace")
    result = {"outcome": "not_comparable", "first_difference": None, "checked": None}
    if not ours or not theirs:
        return {**result, "reason": "a run recorded no model states"}
    different = [field for field in SETUP if ours.get(field) != theirs.get(field)]
    if different:
        return {**result, "reason": f"the runs differ in {', '.join(different)}"}
    for name, run in (("candidate", candidate), ("reference", reference)):
        if run.get("status") != "completed":
            return {**result, "reason": f"the {name} run did not complete"}
    steps, epochs = (
        list(zip(ours.get(field, []), theirs.get(field, []), strict=False))
        for field in ("step_state_sha256", "epoch_state_sha256")
    )
    result["checked"] = {
        "steps_hashed": len(steps),
        "epoch_ends": len(epochs),
        "epochs": ours["epochs"],
    }

    def diverged(at: str, index: int | None = None) -> dict:
        return {**result, "outcome": "diverged", "first_difference": {"at": at, "index": index}}

    if ours["initial_state_sha256"] != theirs["initial_state_sha256"]:
        return diverged("initial_state")
    for index, (mine, yours) in enumerate(steps, start=1):
        if mine != yours:
            return diverged("step", index)
    for index, (mine, yours) in enumerate(epochs, start=1):
        if mine != yours:
            return diverged("epoch", index)
    finals = (ours["final_state_sha256"], theirs["final_state_sha256"])
    if None in finals:
        return {**result, "outcome": "identical_over_checked_steps"}
    if finals[0] != finals[1]:
        return diverged("final_state")
    return {**result, "outcome": "identical_full_run"}


def trajectory_text(trajectory: dict) -> str:
    """One line for a trajectory comparison."""
    outcome = trajectory["outcome"]
    if outcome == "not_comparable":
        return f"not comparable: {trajectory['reason']}"
    if outcome == "diverged":
        where = trajectory["first_difference"]
        place = where["at"].replace("_", " ")
        return (
            f"diverged at {place}"
            if where["index"] is None
            else f"diverged at {place} {where['index']}"
        )
    checked = trajectory["checked"]
    scope = (
        f"{checked['steps_hashed']} hashed steps and "
        f"{checked['epoch_ends']} of {checked['epochs']} epoch ends"
    )
    if outcome == "identical_full_run":
        return f"identical over the full run ({scope}, final state)"
    return f"identical over the checked steps only ({scope}); later steps are unknown"


def _accuracy(run: dict, splits: tuple[str, ...]) -> float:
    return sum(run["final_metrics"][f"{split}_validation_accuracy"] for split in splits) / len(
        splits
    )


def verify_repair_run(
    candidate: dict, reference: dict, faulty: dict | None = None, *, optimizer_name=None
) -> dict:
    """Verify a repaired run against a healthy reference; see the module docstring.

    criteria.before_and_after_metrics is None without a faulty run to compare with;
    otherwise it says whether mean validation accuracy rose from the faulty run.
    """
    result = verify_runs(candidate, reference, faulty, optimizer_name=optimizer_name)
    mechanism = result["re_diagnosis"] == NO_FAULT
    metrics = bool(result["validation_splits"]) and result["performance_nonregression"]
    before_and_after = None
    if faulty is not None and (splits := validation_splits(candidate, faulty)):
        before_and_after = _accuracy(candidate, splits) > _accuracy(faulty, splits)
    return {
        **result,
        "criteria": {
            "before_and_after_metrics": before_and_after,
            "reference_metrics": metrics,
            "mechanism_check": mechanism,
            "mechanism_check_and_reference_metrics": mechanism and metrics,
        },
        "trajectory": compare_trajectories(candidate, reference),
    }


def repair_text(result: dict) -> str:
    lines = [verification_text(result), "Acceptance criteria on this run:"]
    for name, accepts in result["criteria"].items():
        verdict = "not available" if accepts is None else "accepts" if accepts else "rejects"
        lines.append(f"  {name.replace('_', ' ')}: {verdict}")
    lines.append(f"Trajectory against the reference: {trajectory_text(result['trajectory'])}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify a repaired workspace run.")
    parser.add_argument("--candidate", type=Path, required=True, help="Run report after the repair")
    parser.add_argument("--reference", type=Path, required=True, help="Run report of a healthy run")
    parser.add_argument("--faulty", type=Path, help="Run report before the repair")
    args = parser.parse_args(argv)

    def read(path: Path | None) -> dict | None:
        return None if path is None else json.loads(path.read_text(encoding="utf-8"))

    result = verify_repair_run(read(args.candidate), read(args.reference), read(args.faulty))
    print(repair_text(result))
    return 0 if result["decision"] == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
