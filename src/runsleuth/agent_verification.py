"""Verify a repaired workspace run, and say what each part of verification decides.

verify_repair_run() decides as runsleuth verify does: a repair is accepted when
re-diagnosis against the healthy reference finds no known fault (the mechanism check)
and no validation metric regresses against that reference. It adds three records for
studying how agents repair training code: criteria, trajectory and triage.

- criteria: what four acceptance criteria would each decide on the same run, so the
  parts of verification can be compared. Metrics alone accept an unrepaired fault that
  is silent in metrics; the mechanism check alone accepts a repair that removes the
  fault and harms performance; before-and-after metrics reject a correct repair whose
  metrics fall, as they do when a leak is removed.
- trajectory: whether the model states the two runs hashed are the same. A run can be
  stopped early (stop_after_epochs) to make this a cheap, bounded check; agreement
  then covers only the optimizer steps and epoch ends both runs hashed. A fault that
  triggers later, or one that acts only on evaluation, agrees too, so the comparison
  never accepts or rejects by itself, and a correct repair need not agree at all.

"No known fault" is not "correct": a fault outside the signature library with metrics
inside the gate is accepted. triage therefore says how far an acceptance reaches:

- accept: accepted, and the run is the reference's run. Every hashed model state
  agrees through the final state of a full run, and so does every validation metric.
- review: accepted, but the run is not the reference's, or that cannot be shown.
  differences lists where the two runs part; whether the difference is harmless is
  left to a reviewer. A correct repair written differently lands here too.
- reject: rejected.

    python -m runsleuth.agent_verification --candidate runs/fixed/run_report.json \\
        --reference runs/healthy/run_report.json --faulty runs/faulty/run_report.json
"""

import argparse
import json
import math
from pathlib import Path

from runsleuth.cli import validation_splits, verification_text, verify_runs

NO_FAULT = "no_known_fault"
TRIAGE = ("accept", "review", "reject")
TRIAGE_MEANING = {
    "accept": "No known fault, metrics as good as the reference's, and the same training "
    "run as the reference.",
    "review": "No known fault and metrics as good as the reference's, but the run is not "
    "the reference's run. The differences are not verified.",
    "reject": "A known fault was found, or a metric is worse than the reference's.",
}
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


def same_evaluation(candidate: dict, reference: dict) -> bool:
    """Whether two runs recorded exactly the same validation metrics, epoch by epoch."""

    def metrics(run: dict) -> list:
        before = (run.get("workspace") or {}).get("initialization_metrics")
        return [before, *(row.get("metrics") for row in run.get("parameter_group_epochs", []))]

    ours, theirs = metrics(candidate), metrics(reference)
    return len(ours) > 1 and None not in ours[1:] and ours == theirs


def differences(candidate: dict, reference: dict, trajectory: dict) -> list[str]:
    """Where a run is not the reference's run, from what both run reports record."""
    found = []
    if trajectory["outcome"] != "identical_full_run":
        found.append(f"model states: {trajectory_text(trajectory)}")
    ours, theirs = candidate.get("workspace") or {}, reference.get("workspace") or {}
    data, expected = ours.get("data") or {}, theirs.get("data") or {}
    if any(
        data.get(key) != expected.get(key)
        for key in ("training_hospitals", "training_samples", "training_indices_sha256")
    ):
        found.append(
            f"training data: {data.get('training_samples')} samples from hospitals "
            f"{data.get('training_hospitals')}, where the reference has "
            f"{expected.get('training_samples')} from {expected.get('training_hospitals')}"
        )
    if ours.get("steps_per_epoch") != theirs.get("steps_per_epoch"):
        found.append(
            f"steps per epoch: {ours.get('steps_per_epoch')}, where the reference has "
            f"{theirs.get('steps_per_epoch')}"
        )
    rows = zip(
        candidate.get("parameter_group_epochs", []),
        reference.get("parameter_group_epochs", []),
        strict=False,
    )
    for mine, yours in rows:
        rate, expected_rate = mine.get("learning_rate") or {}, yours.get("learning_rate") or {}
        ends = [(rate.get(end), expected_rate.get(end)) for end in ("first", "last")]
        if any(
            None not in pair and not math.isclose(*pair, rel_tol=1e-9, abs_tol=1e-15)
            for pair in ends
        ):
            (first, first_expected), (last, last_expected) = (
                tuple("unknown" if rate is None else f"{rate:.6g}" for rate in pair)
                for pair in ends
            )
            found.append(
                f"learning rate: epoch {mine.get('epoch')} runs from {first} to {last}, "
                f"where the reference runs from {first_expected} to {last_expected}"
            )
            break
    if trajectory["outcome"] == "identical_full_run" and not same_evaluation(candidate, reference):
        found.append("evaluation: validation metrics differ although the model states agree")
    return found


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
    trajectory = compare_trajectories(candidate, reference)
    same_run = trajectory["outcome"] == "identical_full_run" and same_evaluation(
        candidate, reference
    )
    accepted = result["decision"] == "accepted"
    return {
        **result,
        "criteria": {
            "before_and_after_metrics": before_and_after,
            "reference_metrics": metrics,
            "mechanism_check": mechanism,
            "mechanism_check_and_reference_metrics": mechanism and metrics,
        },
        "trajectory": trajectory,
        "triage": "reject" if not accepted else "accept" if same_run else "review",
        "differences": [] if same_run else differences(candidate, reference, trajectory),
    }


def repair_text(result: dict) -> str:
    lines = [verification_text(result), "Acceptance criteria on this run:"]
    for name, accepts in result["criteria"].items():
        verdict = "not available" if accepts is None else "accepts" if accepts else "rejects"
        lines.append(f"  {name.replace('_', ' ')}: {verdict}")
    lines.append(f"Trajectory against the reference: {trajectory_text(result['trajectory'])}")
    lines.append(f"Triage: {result['triage']}")
    lines += [f"  differs in {difference}" for difference in result["differences"]]
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
