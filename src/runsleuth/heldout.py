"""Held-out seeds: draw them after freezing the evaluated components, then run everything.

    python -m runsleuth.heldout draw --count 2
    python -m runsleuth.heldout run --reference-run <source baseline run>

draw picks random training seeds and records them with the commit and the hashes of
everything the evaluation must not change afterwards: the signature library, the LLM
prompt and the repair gate. Commit the seeds file before training, so the history
shows the draw came first. run trains a new baseline and the stale-binding pair per
seed through the sweep, then the frozen-head, configuration and DeiT experiments,
and writes the held-out diagnosis and repair records. It refuses to start if a
frozen component changed since the draw, and resumes after an interruption by
skipping finished steps.
"""

import argparse
import hashlib
import json
import secrets
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from runsleuth.camelyon_optimizer_probe import file_sha256
from runsleuth.rescore_repairs import rescore_report

# The development seeds and the source baseline's seed.
USED_SEEDS = (7, 42, 2026)
SEED_LIMIT = 2**31
EXPERIMENTS = ("frozen_head", "config_faults", "vit")
DEFAULT_SEEDS_FILE = Path("evaluations/heldout_seeds.json")


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], capture_output=True, text=True, check=True
    ).stdout.strip()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def frozen_components() -> dict:
    """What decides the held-out results; none of it may change after the draw."""
    from runsleuth.camelyon_optimizer_training import REPAIR_POLICY
    from runsleuth.camelyon_variant_runner import REPAIR_GATE
    from runsleuth.diagnose_run import PROMPT_VERSION, system_prompt
    from runsleuth.signature_matching import LIBRARY_PATH

    return {
        "failure_signatures_sha256": file_sha256(LIBRARY_PATH),
        "llm_prompt_version": PROMPT_VERSION,
        "llm_system_prompt_sha256": hashlib.sha256(system_prompt().encode()).hexdigest(),
        "repair_gate": dict(REPAIR_GATE),
        "repair_policy": dict(REPAIR_POLICY),
    }


def draw_seeds(count: int, path: Path, *, randbelow=secrets.randbelow, git=_git) -> dict:
    """Draw new training seeds once; an existing draw is never replaced."""
    if isinstance(count, bool) or count not in (1, 2, 3):
        raise ValueError("Draw one to three seeds; the sweep runs at most three")
    if path.exists():
        raise FileExistsError(f"{path} already holds a draw; keep it or choose another --output")
    if git("status", "--porcelain", "--", "src"):
        raise RuntimeError("Commit the source changes first, so the draw records the exact code")
    seeds: list[int] = []
    while len(seeds) < count:
        seed = randbelow(SEED_LIMIT)
        if seed not in USED_SEEDS and seed not in seeds:
            seeds.append(seed)
    record = {
        "schema_version": 1,
        "task": "heldout_seed_draw",
        "drawn_at": datetime.now(UTC).isoformat(),
        "method": f"secrets.randbelow({SEED_LIMIT}), skipping seeds already used",
        "excluded_seeds": list(USED_SEEDS),
        "seeds": seeds,
        "code_commit": git("rev-parse", "HEAD"),
        "frozen": frozen_components(),
        "scope": "training seeds never used in development; data subset and budgets unchanged",
    }
    _write(path, record)
    return record


def _runners() -> dict:
    from runsleuth.camelyon_config_faults import run_config_fault_training
    from runsleuth.camelyon_frozen_head_training import run_frozen_head_training
    from runsleuth.camelyon_vit import run_vit_faults

    return {
        "frozen_head": run_frozen_head_training,
        "config_faults": run_config_fault_training,
        "vit": run_vit_faults,
    }


def _sweep(reference_run: Path, seeds: list[int]) -> Path:
    from runsleuth.camelyon_optimizer_sweep import run_optimizer_sweep

    return run_optimizer_sweep(
        reference_run, workspace_root=Path.cwd(), seeds=tuple(seeds), epochs=3, execute=True
    )


def _relative(path) -> str:
    """Paths relative to the workspace (the working directory), as the records store them."""
    return Path(path).resolve().relative_to(Path.cwd().resolve()).as_posix()


def repair_outcomes(stale_reports: list, experiment_reports: list) -> dict:
    """Gate v2 decisions per held-out repair; the stale-binding fixture still records v1."""
    cases = []
    for seed, path in stale_reports:
        report = _read(Path(path))
        for case in rescore_report(report):
            cases.append(
                {
                    "seed": seed,
                    "report": path,
                    "fault": case["fault"],
                    "decision": case["v2_decision"],
                    "structure_verified": case["structure_verified"],
                    "faulty_passes_gate_vs_reference": case["faulty_passes_gate_vs_reference"],
                }
            )
    for seed, path in experiment_reports:
        recorded = _read(Path(path))["repair_verification"]
        if "decision" in recorded:  # one repair, named by its informational faulty run
            recorded = {recorded["informational_comparators"][0]: recorded}
        for fault, verification in recorded.items():
            cases.append(
                {
                    "seed": seed,
                    "report": path,
                    "fault": fault,
                    "decision": verification["decision"],
                    "structure_verified": verification["structure_verified"],
                    "faulty_passes_gate_vs_reference": verification[
                        "faulty_passes_gate_vs_reference"
                    ],
                }
            )
    faults = {}
    for case in cases:
        counts = faults.setdefault(
            case["fault"], {"cases": 0, "accepted": 0, "faulty_passes_gate_vs_reference": 0}
        )
        counts["cases"] += 1
        counts["accepted"] += case["decision"] == "accepted"
        counts["faulty_passes_gate_vs_reference"] += case["faulty_passes_gate_vs_reference"] is True
    return {"faults": faults, "cases": cases}


def run_heldout(
    seeds_path: Path,
    reference_run: Path,
    *,
    device: str | None = None,
    sweep=_sweep,
    runners=None,
    git=_git,
) -> Path:
    """Train and evaluate every drawn seed; returns the batch state path."""
    draw = _read(seeds_path)
    if frozen_components() != draw["frozen"]:
        raise RuntimeError(
            "A frozen component changed since the draw (signatures, prompt or gate); "
            "held-out results would no longer test what was frozen"
        )
    runners = runners or _runners()
    seeds = draw["seeds"]
    name = "seeds-" + "-".join(map(str, seeds))
    directory = Path("artifacts/heldout") / name
    state_path = directory / "batch.json"
    state = (
        _read(state_path)
        if state_path.is_file()
        else {
            "schema_version": 1,
            "task": "heldout_batch",
            "seeds_file": {"path": seeds_path.as_posix(), "sha256": file_sha256(seeds_path)},
            "status": "running",
            "sweep": None,
            "seeds": {str(seed): {} for seed in seeds},
        }
    )
    state["run_commit"] = git("rev-parse", "HEAD")
    state["source_changed_since_draw"] = bool(
        git("diff", "--name-only", draw["code_commit"], "--", "src")
    )
    _write(state_path, state)

    if state["sweep"] is None or state["sweep"]["status"] not in (
        "completed",
        "completed_with_errors",
    ):
        summary_path = sweep(reference_run, seeds)
        summary = _read(Path(summary_path))
        state["sweep"] = {"summary": _relative(summary_path), "status": summary["status"]}
        for case in summary["cases"]:
            entry = state["seeds"][str(case["seed"])]
            entry["baseline_run"] = case.get("baseline_run")
            entry["stale_binding"] = {
                "status": case["status"],
                "report": case.get("training_report", {}).get("path"),
            }
        _write(state_path, state)

    for seed in seeds:
        entry = state["seeds"][str(seed)]
        if not entry.get("baseline_run"):
            continue
        for experiment in EXPERIMENTS:
            if entry.get(experiment, {}).get("status") in ("completed", "inconclusive"):
                continue
            print(f"phase=seed-{seed}-{experiment}", flush=True)
            try:
                path = runners[experiment](
                    Path(entry["baseline_run"]), directory / f"seed-{seed}" / experiment, 3, device
                )
                entry[experiment] = {
                    "status": _read(Path(path))["status"],
                    "report": _relative(path),
                }
            except KeyboardInterrupt:
                entry[experiment] = {"status": "interrupted"}
                _write(state_path, state)
                raise
            except Exception as error:  # record it and keep the other experiments going
                entry[experiment] = {
                    "status": "failed",
                    "error": {"type": type(error).__name__, "message": str(error)},
                }
            _write(state_path, state)

    from runsleuth.signature_matching import evaluate_reports, load_library

    completed = [
        (int(seed), entry[experiment]["report"])
        for seed, entry in state["seeds"].items()
        for experiment in EXPERIMENTS
        if entry.get(experiment, {}).get("status") == "completed"
    ]
    stale = [
        (int(seed), entry["stale_binding"]["report"])
        for seed, entry in state["seeds"].items()
        if entry.get("stale_binding", {}).get("status") == "completed"
    ]
    scope = (
        f"held-out seeds {seeds}, drawn after the signatures, LLM prompt and repair gate were "
        "frozen; author-injected faults, development thresholds, not a statistical benchmark"
    )
    results = Path("evaluations/results")
    diagnosis = evaluate_reports([Path(path) for _, path in completed], load_library())
    diagnosis["scope"] = scope
    repairs = {"scope": scope, **repair_outcomes(stale, completed)}
    state["results"] = {
        "signature_matching": (results / f"signature-matching-heldout-{name}.json").as_posix(),
        "repairs": (results / f"repairs-heldout-{name}.json").as_posix(),
    }
    _write(Path(state["results"]["signature_matching"]), diagnosis)
    _write(Path(state["results"]["repairs"]), repairs)
    finished = state["sweep"]["status"] == "completed" and len(completed) == len(seeds) * len(
        EXPERIMENTS
    )
    state["status"] = "completed" if finished else "completed_with_errors"
    _write(state_path, state)
    return state_path


def summary_text(state: dict) -> str:
    repairs = _read(Path(state["results"]["repairs"]))
    diagnosis = _read(Path(state["results"]["signature_matching"]))["summary"]
    lines = [f"{'fault':<36}{'accepted':>10}{'silent in metrics':>19}"]
    for fault, counts in repairs["faults"].items():
        lines.append(
            f"{fault:<36}{counts['accepted']:>7}/{counts['cases']:<2}"
            f"{counts['faulty_passes_gate_vs_reference']:>16}/{counts['cases']}"
        )
    for mode in ("with_reference", "reference_free"):
        summary = diagnosis[mode]
        lines.append(
            f"diagnosis {mode}: {summary['exact_matches']}/{summary['cases']} exact, "
            f"{summary['pending_reference_with_true_fault']} deferred to a reference, "
            f"{summary['healthy_false_positives']}/{summary['healthy_cases']} healthy flagged"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    draw = commands.add_parser("draw", help="Draw and record held-out seeds")
    draw.add_argument("--count", type=int, default=2)
    draw.add_argument("--output", type=Path, default=DEFAULT_SEEDS_FILE)
    run = commands.add_parser("run", help="Train and evaluate the drawn seeds")
    run.add_argument("--reference-run", type=Path, required=True, help="Source baseline run")
    run.add_argument("--seeds-file", type=Path, default=DEFAULT_SEEDS_FILE)
    run.add_argument("--device", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    if args.command == "draw":
        record = draw_seeds(args.count, args.output)
        print(f"seeds={record['seeds']} commit={record['code_commit']}")
        print(f"seeds_file={args.output}")
        return
    state_path = run_heldout(args.seeds_file, args.reference_run, device=args.device)
    state = _read(state_path)
    print(summary_text(state))
    print(f"status={state['status']}")
    print(f"batch={state_path}")
    print("results=" + json.dumps(state["results"]))
    if state["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
