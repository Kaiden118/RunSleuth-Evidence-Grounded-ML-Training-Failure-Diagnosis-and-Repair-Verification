"""The metric gate: a baseline verifier that accepts a candidate whose delivered weights
score as well as the healthy project's, within the spread of correct runs.

The harness evaluates the weights a run delivers on the in-distribution validation
split (id) and on a held-out one (ood). The gate compares the candidate's accuracy on
each with that of the healthy project's run on the same seed, the reference, and
accepts when neither is lower by more than the family's threshold. A candidate that did
not run, or has no finite accuracy, is rejected. It decides accept or reject only. The
losses are recorded but do not decide: on the held-out split, batch order alone changed
the loss of a correct development variant by 30%, so a loss threshold that passes
correct runs would be too loose to decide anything.

The thresholds are fixed per family before any test candidate is run, by one rule
applied to correct runs only: the healthy project on the development seeds 7, 11, 42
and 2026 (all used in stage one, so never drawn as test seeds), and the correct
development variants on seed 7. On each split, the threshold is the largest difference
in accuracy between two of these runs, rounded up to a tenth of a point. No fault's
metrics enter it.

The gate of stage one (an accuracy drop of at most one point and a loss ratio of at
most 1.10 on each split, not calibrated to the spread) is kept beside it, to show what
an uncalibrated gate decides on the same runs.

    python -m runsleuth.metric_gate calibrate --family camelyon_pipeline \\
        --checks artifacts/pipeline_tasks/check-6 artifacts/pipeline_tasks/noise-camelyon-11
    python -m runsleuth.metric_gate decide --family camelyon_pipeline \\
        --candidate runs/fixed/run_report.json --reference runs/healthy/run_report.json
"""

import argparse
import json
import math
from datetime import date
from pathlib import Path
from typing import Any

from runsleuth import pipeline_tasks
from runsleuth.camelyon_optimizer_training import REPAIR_POLICY
from runsleuth.pipeline_tasks import HEALTHY

SPLITS = ("id", "ood")
CALIBRATION_SEEDS = (7, 11, 42, 2026)
GATE_FILE = Path("evaluations/metric_gate.json")
STAGE_ONE = dict(REPAIR_POLICY)


def _metric(metrics: dict[str, Any] | None, split: str, name: str) -> float | None:
    value = (metrics or {}).get(f"{split}_validation_{name}")
    return float(value) if type(value) in (int, float) and math.isfinite(value) else None


def calibration_runs(check: dict[str, Any], family: str) -> list[dict[str, Any]]:
    """The correct runs of a pipeline check that calibrate a family's gate: the healthy
    project and the correct development variants, on a calibration seed."""
    settings = check["settings"]
    if settings["family"] != family or settings["family_sha256"] != pipeline_tasks.family_sha256(
        family
    ):
        raise ValueError(f"The check is not of {family} as it is now")
    if settings["seed"] not in CALIBRATION_SEEDS:
        raise ValueError(f"Seed {settings['seed']} is not a calibration seed")
    versions = pipeline_tasks.family(family).versions
    runs = []
    for row in check["versions"]:
        name = row["version"]
        correct = name == HEALTHY or (
            name in versions and versions[name].correct and versions[name].split == "development"
        )
        if not correct:
            continue
        accuracies = {split: _metric(row["metrics"], split, "accuracy") for split in SPLITS}
        if row["status"] != "completed" or None in accuracies.values():
            raise ValueError(f"{name} on seed {settings['seed']} has no accuracy to calibrate on")
        runs.append({"version": name, "seed": settings["seed"], **accuracies})
    return runs


def thresholds(runs: list[dict[str, Any]]) -> dict[str, float]:
    """On each split, the largest difference in accuracy between two of the runs, rounded
    up to a tenth of a point."""
    if len(runs) < 2:
        raise ValueError("Calibrating takes at least two correct runs")
    found = {}
    for split in SPLITS:
        values = [run[split] for run in runs]
        found[split] = math.ceil(round((max(values) - min(values)) * 1000, 9)) / 1000
    return found


def calibrate(checks: list[dict[str, Any]], family: str) -> dict[str, Any]:
    """A family's gate from pipeline checks: its runs, and thresholds every seed shares."""
    runs = [run for check in checks for run in calibration_runs(check, family)]
    seeds = sorted({run["seed"] for run in runs})
    if seeds != sorted(CALIBRATION_SEEDS):
        raise ValueError(f"The checks cover seeds {seeds}, not {list(CALIBRATION_SEEDS)}")
    return {
        "family_sha256": pipeline_tasks.family_sha256(family),
        "runs": runs,
        "thresholds": thresholds(runs),
    }


def decide(
    candidate: dict[str, Any] | None, reference: dict[str, Any], gate: dict[str, float]
) -> dict[str, Any]:
    """Accept or reject a candidate's run report against the reference's on its seed."""
    expected = {
        split: _metric(reference.get("final_metrics"), split, "accuracy") for split in SPLITS
    }
    if None in expected.values():
        raise ValueError("The reference has no accuracy on every split")
    metrics = (candidate or {}).get("final_metrics")
    if candidate is None or candidate.get("status") != "completed":
        status = "did not run" if candidate is None else f"ended {candidate.get('status')}"
        return {"decision": "reject", "reasons": [f"the candidate {status}"], "checks": []}
    checks, reasons = [], []
    for split in SPLITS:
        accuracy = _metric(metrics, split, "accuracy")
        drop = None if accuracy is None else expected[split] - accuracy
        passed = drop is not None and drop <= gate[split] + 1e-12
        checks.append(
            {
                "split": split,
                "accuracy": accuracy,
                "reference": expected[split],
                "drop": drop,
                "threshold": gate[split],
                "loss": _metric(metrics, split, "loss"),
                "reference_loss": _metric(reference.get("final_metrics"), split, "loss"),
                "passed": passed,
            }
        )
        if accuracy is None:
            reasons.append(f"no finite {split} accuracy")
        elif not passed:
            reasons.append(
                f"{split} accuracy {accuracy:.4f} is {drop * 100:.1f} points below the "
                f"reference's {expected[split]:.4f}, beyond {gate[split] * 100:.1f}"
            )
    return {"decision": "reject" if reasons else "accept", "reasons": reasons, "checks": checks}


def decide_stage_one(candidate: dict[str, Any] | None, reference: dict[str, Any]) -> str:
    """What stage one's uncalibrated gate decides: accept or reject."""
    if candidate is None or candidate.get("status") != "completed":
        return "reject"
    for split in SPLITS:
        values = [
            _metric(run.get("final_metrics"), split, name)
            for run in (candidate, reference)
            for name in ("accuracy", "loss")
        ]
        if None in values:
            return "reject"
        accuracy, loss, expected_accuracy, expected_loss = values
        if accuracy < expected_accuracy - STAGE_ONE["max_accuracy_drop"]:
            return "reject"
        if expected_loss <= 0 or loss > expected_loss * STAGE_ONE["max_loss_ratio"]:
            return "reject"
    return "accept"


def load_gate(family: str, path: Path = GATE_FILE) -> dict[str, float]:
    """A family's frozen thresholds; refused if the family has changed since."""
    record = json.loads(Path(path).read_text(encoding="utf-8"))["families"][family]
    if record["family_sha256"] != pipeline_tasks.family_sha256(family):
        raise ValueError(f"The gate of {family} was calibrated on another state of the family")
    return record["thresholds"]


def write_gate(family: str, gate: dict[str, Any], path: Path = GATE_FILE) -> None:
    """Record a family's gate; a family's gate, once recorded, is not recalibrated."""
    path = Path(path)
    record = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    record = record or {
        "schema_version": 1,
        "task": "metric_gate",
        "rule": "accept when neither accuracy is below the same seed's healthy run by more "
        "than the threshold; thresholds are the largest accuracy difference between two "
        "correct development runs, rounded up to a tenth of a point",
        "calibration_seeds": list(CALIBRATION_SEEDS),
        "stage_one_gate": STAGE_ONE,
        "families": {},
    }
    if family in record["families"]:
        raise FileExistsError(f"{path} already holds the gate of {family}")
    record["families"][family] = {**gate, "calibrated_on": date.today().isoformat()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="The metric gate, a baseline verifier.")
    commands = parser.add_subparsers(dest="command", required=True)
    calibrated = commands.add_parser("calibrate", help=f"Fix a family's thresholds in {GATE_FILE}.")
    calibrated.add_argument("--checks", type=Path, nargs="+", required=True)
    decided = commands.add_parser("decide", help="Accept or reject one candidate's run.")
    decided.add_argument("--candidate", type=Path, required=True)
    decided.add_argument("--reference", type=Path, required=True)
    for command in (calibrated, decided):
        command.add_argument("--family", choices=tuple(pipeline_tasks.FAMILIES), required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "calibrate":
        checks = [_read(directory / "check_report.json") for directory in args.checks]
        gate = calibrate(checks, args.family)
        write_gate(args.family, gate)
        print(json.dumps(gate, indent=2))
        return 0
    result = decide(_read(args.candidate), _read(args.reference), load_gate(args.family))
    result["stage_one_gate"] = decide_stage_one(_read(args.candidate), _read(args.reference))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
