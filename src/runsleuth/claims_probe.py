"""Evidence about a claims-task program that its official run does not give, made without
the world's sealed data.

Replication trains the program again on the same given data with other seeds and keeps
the estimates each run reports. Regime matching applies the world's own sealing rule to
the given data: each development site in turn is sealed whole, with some lesions or
patients of every other site, as a fold. The program trains on a fold's given part through
its own interface, and its delivered predictor predicts the fold's sealed parts; each fold
keeps the run's own estimates and threshold, the size of what it trained on, and what its
predictor measures there. A fold's network is another network than the delivered one,
trained on fewer sites and groups.

A world with K development sites gets K folds and K replication seeds (2 to K + 1), so
that both tables cost the same number of training calls; their time differs, since fold
runs train on less data.

From the folds comes a forecast for each use of the world: the mean of what the folds
measured, with the interval mean ± t(K−1) · s · sqrt(1 + 1/K) over K folds, clipped to
[0, 1], and the mean amount by which the folds' own estimates exceeded what they measured.
The rule is a heuristic fixed before the study; it promises no 95% coverage for a new
site.

Both kinds of run use the guard, the given copy and the prediction call of the official
run (claims_run).

    python -m runsleuth.claims_probe folds --world ham10000-vienna_dias \\
        --data C:/Users/ADMIN/rs-data/claims --out C:/Users/ADMIN/rs-data/claims-folds
    python -m runsleuth.claims_probe run --out artifacts/claims/dev-v2 \\
        --data C:/Users/ADMIN/rs-data/claims --folds C:/Users/ADMIN/rs-data/claims-folds \\
        --work C:/Users/ADMIN/rs-data/claims-probe-runs
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from runsleuth import claims_freeze, claims_run, claims_task

# The first replication seed; a world with K development sites gets seeds 2 to K + 1.
FIRST_REPLICATION_SEED = 2
PROBE_FILE = "probe.json"
FOLD_RECORD = "fold.json"
# Two-sided 95% quantiles of Student's t by degrees of freedom.
T_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}


def development_rows(
    export: Path, chosen: claims_task.Spec
) -> tuple[list[dict[str, Any]], Callable[[int], np.ndarray]]:
    """The rows of a world's given part with their sealing units, and its images."""
    given = Path(export) / claims_task.GIVEN
    with (given / claims_task.METADATA).open(encoding="utf-8", newline="") as stream:
        table = list(csv.DictReader(stream))
    units = json.loads((Path(export) / claims_task.UNITS).read_text(encoding="utf-8"))["given"]
    if len(units) != len(table):
        raise ValueError(f"{len(units)} units for {len(table)} given rows")
    rows = []
    for line, unit in zip(table, units, strict=True):
        row = {key: value for key, value in line.items() if key != "row"}
        row["label"] = int(row["label"])
        row[claims_task.UNIT] = unit
        rows.append(row)
    images = np.load(given / claims_task.IMAGES, mmap_mode="r")
    return rows, lambda position: np.asarray(images[position])


def fold_spec(chosen: claims_task.Spec, site: str) -> claims_task.Spec:
    """The world's sealing rule with one development site sealed."""
    return replace(chosen, name=f"{chosen.name}.fold-{site}", sealed_sites=(str(site),))


def build_folds(export: Path, chosen: claims_task.Spec, root: Path) -> list[tuple[str, Path]]:
    """One fold per site of the world's given part, under root/<world>/fold-<site>, made
    once and made again when the given part changed."""
    export = Path(export)
    record = json.loads((export / claims_task.RECORD).read_text(encoding="utf-8"))
    given_sha = record["parts"]["given"]["sha256"]
    sites = sorted(record["parts"]["given"]["sites"])
    rows, image = None, None
    folds = []
    for site in sites:
        directory = Path(root) / chosen.name / f"fold-{site}"
        marker = directory / FOLD_RECORD
        if marker.is_file() and json.loads(marker.read_text(encoding="utf-8")) == given_sha:
            folds.append((site, directory))
            continue
        if rows is None:
            rows, image = development_rows(export, chosen)
        claims_task.export(
            rows,
            image,
            fold_spec(chosen, site),
            directory,
            seed=claims_task.EXPORT_SEED,
            source={"world": chosen.name, "given_sha256": given_sha},
        )
        marker.write_text(json.dumps(given_sha) + "\n", encoding="utf-8")
        folds.append((site, directory))
    return folds


def estimates_of(report: dict[str, Any] | None, chosen: claims_task.Spec) -> dict[str, Any]:
    """What a report says of each use: its numbers and intervals, why it is not
    estimable, or why it cannot be read."""
    if report is None or not isinstance(report.get("estimates"), dict):
        return {claim.name: {"problems": ["no readable report"]} for claim in chosen.claims}
    wrong = claims_run.estimate_problems(report, chosen)
    found: dict[str, Any] = {}
    for claim in chosen.claims:
        estimate = report["estimates"].get(claim.name)
        if wrong[claim.name]:
            found[claim.name] = {"problems": wrong[claim.name]}
        elif claims_run.abstains(estimate):
            found[claim.name] = {"not_estimable": str(estimate[claims_task.NOT_ESTIMABLE])}
        else:
            found[claim.name] = {
                metric: {
                    "estimate": float(estimate[metric]),
                    "interval": claims_run.interval_of(estimate, metric),
                }
                for metric in claims_task.METRICS
            }
    return found


def _train(
    program: Path,
    data: Path,
    out: Path,
    seed: int,
    chosen: claims_task.Spec,
    *,
    device: str,
    timeout: float,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """A training call on data into out, and the report it wrote when it completed and
    the report can be read."""
    out.mkdir(parents=True)
    call = claims_run.call_program(
        program,
        claims_run.training_arguments(data, out, seed, device),
        out,
        timeout=timeout,
        read=[data],
        write=[out],
    )
    (out.parent / f"{out.name}-output.txt").write_text(call.pop("output"), encoding="utf-8")
    if call["status"] != "completed":
        return call, None
    report, _problems = claims_run.read_report(out / claims_task.REPORT, chosen)
    return call, report


def replicate(
    program: Path,
    given: Path,
    chosen: claims_task.Spec,
    work: Path,
    seed: int,
    *,
    device: str,
    timeout: float,
) -> dict[str, Any]:
    """The program trained again on the given data with another seed: what it reports."""
    out = Path(work) / f"replicate-{seed}"
    call, report = _train(program, given, out, seed, chosen, device=device, timeout=timeout)
    return {"seed": seed, "training": call, "estimates": estimates_of(report, chosen)}


def regime_run(
    program: Path,
    fold: Path,
    site: str,
    chosen: claims_task.Spec,
    work: Path,
    *,
    seed: int,
    device: str,
    train_timeout: float,
    predict_timeout: float,
) -> dict[str, Any]:
    """The program trained on a fold's given part, and its predictor measured on the
    fold's sealed parts at the run's own threshold."""
    out = Path(work) / f"regime-{site}"
    call, report = _train(
        program, fold / claims_task.GIVEN, out, seed, chosen, device=device, timeout=train_timeout
    )
    record = json.loads((fold / claims_task.RECORD).read_text(encoding="utf-8"))
    trained_on = record["parts"]["given"]
    found: dict[str, Any] = {
        "site": site,
        "training": call,
        "trained_on": {
            "samples": trained_on["samples"],
            "groups": trained_on["groups"],
            "sites": len(trained_on["sites"]),
        },
    }
    found["estimates"] = estimates_of(report, chosen)
    threshold = report.get("threshold") if isinstance(report, dict) else None
    found["threshold"] = threshold if isinstance(threshold, int | float) else None
    problems = [] if call["status"] == "completed" else [f"training {call['status']}"]
    if not problems:
        if not (isinstance(threshold, int | float) and 0 <= threshold <= 1):
            problems.append("no threshold")
        problems += claims_run.check_weights(out / claims_task.DELIVERED)
    measured: dict[str, Any] = {}
    if not problems:
        units = json.loads((fold / claims_task.UNITS).read_text(encoding="utf-8"))
        for part in sorted({claim.part for claim in chosen.claims}):
            sealed = fold / claims_task.SEALED / part
            place = Path(work) / f"regime-{site}-predict-{part}"
            predicted = claims_run.predict(
                program,
                out / claims_task.DELIVERED,
                sealed / claims_task.IMAGES,
                place,
                device=device,
                timeout=predict_timeout,
            )
            with (sealed / claims_task.METADATA).open(encoding="utf-8", newline="") as stream:
                table = list(csv.DictReader(stream))
            scores, wrong = claims_run.read_predictions(place / claims_run.PREDICTIONS, len(table))
            if predicted["status"] != "completed" or wrong:
                measured[part] = {"problems": [f"prediction {predicted['status']}", *wrong]}
                continue
            measured[part] = claims_run.measure(
                np.asarray([int(line["label"]) for line in table]),
                scores,
                units[part],
                float(threshold),
            )
    found["problems"] = problems
    found["measured"] = measured
    return found


def replication_seeds(folds: int) -> tuple[int, ...]:
    """As many replication seeds as a world has folds."""
    return tuple(range(FIRST_REPLICATION_SEED, FIRST_REPLICATION_SEED + folds))


def forecast(values: list[float]) -> dict[str, Any]:
    """The mean of values with the interval mean ± t(K−1) · s · sqrt(1 + 1/K), clipped to
    [0, 1]; no interval from fewer than two values."""
    found: dict[str, Any] = {"folds": len(values)}
    if not values:
        return {**found, "value": None, "low": None, "high": None}
    mean = statistics.fmean(values)
    found["value"] = mean
    if len(values) < 2:
        return {**found, "low": None, "high": None}
    spread = statistics.stdev(values)
    half = T_975[min(len(values) - 1, max(T_975))] * spread * math.sqrt(1 + 1 / len(values))
    return {**found, "sd": spread, "low": max(0.0, mean - half), "high": min(1.0, mean + half)}


def forecasts(chosen: claims_task.Spec, regime: list[dict[str, Any]]) -> dict[str, Any]:
    """For each use and metric, the forecast from the folds that measured its part, and
    the mean excess of the folds' own estimates over what they measured."""
    found: dict[str, Any] = {}
    for claim in chosen.claims:
        found[claim.name] = {}
        for metric in claims_task.METRICS:
            values, excess = [], []
            for fold in regime:
                measured = fold["measured"].get(claim.part)
                if not measured or "problems" in measured:
                    continue
                values.append(measured[metric]["value"])
                estimate = fold["estimates"].get(claim.name, {})
                if metric in estimate:
                    excess.append(estimate[metric]["estimate"] - measured[metric]["value"])
            found[claim.name][metric] = {
                **forecast(values),
                "own_estimate_excess": statistics.fmean(excess) if excess else None,
                "estimated_folds": len(excess),
            }
    return found


def probe(
    program: Path,
    export: Path,
    chosen: claims_task.Spec,
    folds_root: Path,
    work: Path,
    *,
    device: str = "cuda",
    seeds: tuple[int, ...] | None = None,
    regime_seed: int = claims_run.SEED,
    train_timeout: float = claims_run.TRAIN_TIMEOUT,
    predict_timeout: float = claims_run.PREDICT_TIMEOUT,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Replication and regime matching of a program on a world, kept under work. seeds
    default to as many as the world has development sites."""
    work = Path(work)
    if work.exists():
        raise FileExistsError(f"{work} exists; remove it to probe the program again")
    work.mkdir(parents=True)
    source = Path(program).read_text(encoding="utf-8")
    problems = claims_run.screen(source)
    result: dict[str, Any] = {"world": chosen.name, "problems": problems}
    if problems:
        return {**result, "category": "refused"}
    (work / "program").mkdir()
    program = work / "program" / claims_task.PROGRAM
    program.write_text(source, encoding="utf-8", newline="\n")
    given = claims_run.given_copy(export, work.parent, chosen.name)
    folds = build_folds(export, chosen, folds_root)
    if seeds is None:
        seeds = replication_seeds(len(folds))
    result["replication"] = []
    for seed in seeds:
        result["replication"].append(
            replicate(program, given, chosen, work, seed, device=device, timeout=train_timeout)
        )
        progress(f"  replication seed {seed}: {result['replication'][-1]['training']['status']}")
    result["regime"] = []
    for site, fold in folds:
        found = regime_run(
            program,
            fold,
            site,
            chosen,
            work,
            seed=regime_seed,
            device=device,
            train_timeout=train_timeout,
            predict_timeout=predict_timeout,
        )
        result["regime"].append(found)
        progress(f"  fold {site}: {found['training']['status']} {found['problems'][:1]}")
    result["forecast"] = forecasts(chosen, result["regime"])
    result["category"] = "probed"
    (work / PROBE_FILE).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run_attempts(
    output: Path,
    data: Path,
    folds_root: Path,
    work: Path,
    *,
    device: str = "cuda",
    progress: Callable[[str], None] = print,
    **options,
) -> list[dict[str, Any]]:
    """Probe every attempt with a program and no probe yet, one at a time, and keep the
    result beside the program. In the study's output the freeze must hold, with the
    plan's paths and device."""
    claims_freeze.guard(
        output, paths={"data": data, "folds": folds_root, "probes": work}, device=device
    )
    done = []
    for directory in claims_run.attempt_directories(output):
        attempt = json.loads((directory / "attempt.json").read_text(encoding="utf-8"))
        program = directory / claims_task.PROGRAM
        if (directory / PROBE_FILE).is_file() or not program.is_file():
            continue
        chosen = claims_task.spec(attempt.get("world", attempt["dataset"]))
        progress(
            f"{attempt['model_name']} {chosen.name} {attempt['condition']} {attempt['attempt']}"
        )
        result = probe(
            program,
            Path(data) / chosen.name,
            chosen,
            folds_root,
            Path(work) / attempt["id"],
            device=device,
            progress=progress,
            **options,
        )
        result = {"schema_version": 1, "task": "claims_probe", "id": attempt["id"], **result}
        (directory / PROBE_FILE).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        done.append(result)
    return done


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replication and regime-matched probes.")
    commands = parser.add_subparsers(dest="command", required=True)
    folds = commands.add_parser("folds", help="Write a world's regime-matched folds.")
    folds.add_argument("--world", required=True, choices=sorted(claims_task.SPECS))
    folds.add_argument("--data", type=Path, required=True, help="The exports, one per world.")
    folds.add_argument("--out", type=Path, required=True, help="Where the folds are kept.")
    run = commands.add_parser("run", help="Probe every attempt that has no probe yet.")
    run.add_argument("--out", type=Path, required=True, help="The attempts' directory.")
    run.add_argument("--data", type=Path, required=True, help="The exports, one per world.")
    run.add_argument("--folds", type=Path, required=True, help="Where the folds are kept.")
    run.add_argument("--work", type=Path, required=True, help="Where the probes keep files.")
    run.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "folds":
        chosen = claims_task.spec(args.world)
        for site, directory in build_folds(args.data / chosen.name, chosen, args.out):
            record = json.loads((directory / claims_task.RECORD).read_text(encoding="utf-8"))
            parts = record["parts"]
            print(
                f"fold {site}: given {parts['given']['samples']}, new groups "
                f"{parts['new_groups']['samples']}, held-out site {parts['new_site']['samples']} "
                f"({parts['new_site']['positive']} positive)"
            )
        return 0
    for result in run_attempts(args.out, args.data, args.folds, args.work, device=args.device):
        print(json.dumps(result["forecast"])[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
