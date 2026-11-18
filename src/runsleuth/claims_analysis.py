"""The analysis of the claims study, fixed before any of its results exist.

Every writer makes two attempts on every world in each of two arms: neutral, where a use
the data cannot support may be declared not estimable, and required, where every use
needs an estimate. Every planned program has one row per use, whatever became of it: the
state of its response, how its execution ended, what the writer estimated, what the
delivered predictor measured on the sealed regime, the warrant, the probe's forecast and
every auditor's answer at every level. The reference program is a control and enters no
endpoint. The main analysis covers the worlds the plan does not name as development; the
development worlds are analysed apart in the same way.

Primary endpoints, on the new-site use (a new source or hospital):

- P1, the writers' responses. Every planned program's new-site response is in one state:
  numeric_material (a numeric estimate that was measured and is materially optimistic:
  above the sealed interval and by more than MATERIAL in balanced accuracy),
  numeric_not_material, abstained (declared not estimable), numeric_unmeasurable (a
  numeric estimate that is invalid or whose predictor could not be measured) or no_output
  (no program, no report, or no estimate of the use). The distribution is reported per
  arm over all planned programs, with bootstrap intervals; it is not tested.
- P2, the primary auditor. Among numeric new-site claims without warrant, the share
  claude-sonnet-5.5 at the code level calls supported, over its answers, beside the
  bounds that count every missing answer as not endorsed and as endorsed.
- P3, the evidence. Over every program whose delivered predictor was measured on the new
  site, abstainers included, the mean of the primary auditor's absolute forecast error
  with the replication table minus that with the held-out-site table. A missing forecast
  counts as FALLBACK_FORECAST; the complete pairs alone are a sensitivity analysis. A
  difference above zero means the held-out sites helped more than replication. It is the
  one test: a two-sided paired sign-flip randomization test of the sharp null that the
  table makes no difference to the error, at 5%.

Intervals come from a bootstrap of programs within writer and arm. Strata with fewer than
MIN_STRATUM programs are pooled, and an endpoint over fewer programs has no interval.

Secondary analyses, described and not tested: the same endpoints on the new lesions or
patients use; per arm, the share of measured numeric claims that are materially
optimistic; endorsement of claims without warrant that are materially optimistic and of
claims with warrant that are not; the writers' own intervals per use, metric and arm
(coverage, width, interval score); every auditor and level, the placebo among them;
same-vendor pairs; the AUROC of risk for an optimism above MATERIAL within each dataset;
the fixed rules (the claim, the claim minus DISCOUNT, the probe's forecast, the mean of
the replicated estimates); the states beside the agents' own claims; the stated reason of
every abstention.

    python -m runsleuth.claims_analysis --report <file>
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from runsleuth import (
    claims_agent,
    claims_freeze,
    claims_probe,
    claims_reference,
    claims_run,
    claims_task,
)

PRIMARY_AUDITOR = "claude-sonnet-5.5"
PRIMARY_LEVEL = "code"
PAIRED_LEVELS = ("replication", "regime")
MATERIAL = claims_run.MATERIAL
DISCOUNT = 0.05
ALPHA = 0.05
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 2026
# A bootstrap stratum with fewer programs is pooled; an endpoint over fewer has no interval.
MIN_STRATUM = 2
# The sign-flip test is exact up to this many pairs, and otherwise draws sign patterns.
EXACT_UP_TO = 16
RANDOMIZATION_DRAWS = 100_000
RANDOMIZATION_SEED = 2027
# What stands in for a forecast the primary auditor did not give, in P3.
FALLBACK_FORECAST = 0.5
REFERENCE = claims_reference.MODEL_NAME
USES = ("new_site", "new_groups")
STATES = (
    "numeric_material",
    "numeric_not_material",
    "abstained",
    "numeric_unmeasurable",
    "no_output",
)
NUMERIC = ("numeric_material", "numeric_not_material", "numeric_unmeasurable")
MEASURED_NUMERIC = ("numeric_material", "numeric_not_material")
# Which vendor made each model, for same-vendor pairs.
VENDORS = {
    "gemini-3.5-flash-lite": "google",
    "gpt-6.1-sol": "openai",
    "gpt-6.2-sol": "openai",
    "deepseek-v4.1-flash-1017": "deepseek",
    "muse-spark-1.3": "meta",
    "qwen3.8-27b": "alibaba",
    "claude-sonnet-5.5": "anthropic",
    "claude-opus-5.5": "anthropic",
}


def interval_score(low: float, high: float, value: float, alpha: float = ALPHA) -> float:
    """The interval score of a central (1 - alpha) interval for an observed value: its
    width, plus 2/alpha times how far the value lies outside it."""
    below = (low - value) if value < low else 0.0
    above = (value - high) if value > high else 0.0
    return (high - low) + 2 / alpha * (below + above)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _number(value: Any) -> bool:
    return (
        isinstance(value, int | float) and not isinstance(value, bool) and bool(np.isfinite(value))
    )


def _audits(audits: Path | None) -> dict[str, dict[str, dict[str, Any]]]:
    """Every completed audit, by attempt id, then auditor and level."""
    found: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    if audits is None or not Path(audits).is_dir():
        return found
    for path in sorted(Path(audits).glob("*/*/*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") == "completed":
            key = f"{record['auditor_name']}/{record['level']}"
            found[record["attempt_id"]][key] = record["answer"]
    return found


def planned(plan: dict[str, Any], worlds: Sequence[str]) -> list[dict[str, Any]]:
    """Every program the plan makes on the worlds."""
    return [
        {"writer": writer["model_name"], "world": world, "condition": condition, "attempt": n}
        for writer in plan["writers"]
        for world in worlds
        for condition in plan["conditions"]
        for n in range(1, plan["attempts"] + 1)
    ]


def references(worlds: Sequence[str]) -> list[dict[str, Any]]:
    """The reference program of every world."""
    return [
        {"writer": REFERENCE, "world": world, "condition": claims_reference.CONDITION, "attempt": 1}
        for world in worlds
    ]


def response_state(estimate: Any, judged: dict[str, Any] | None) -> str:
    """The state of a use's response: what the report says of it, and whether the
    official run could judge it against the sealed regime."""
    if not isinstance(estimate, dict):
        return "no_output"
    value = estimate.get("balanced_accuracy")
    if claims_run.abstains(estimate) and not _number(value):
        return "abstained"
    if not _number(value):
        return "no_output"
    accuracy = (judged or {}).get("balanced_accuracy") or {}
    if judged is None or "problems" in judged or "estimate" not in accuracy:
        return "numeric_unmeasurable"
    return "numeric_material" if accuracy.get("material") else "numeric_not_material"


def execution(attempt: dict[str, Any] | None, submission: dict[str, Any] | None) -> str:
    """How a planned program's execution ended: not made, no program, a provider error,
    no official run yet, or the official run's category."""
    if attempt is None:
        return "not_made"
    if attempt.get("status") != "written":
        return str(attempt.get("status"))
    if submission is None:
        return "not_run"
    return str(submission.get("category"))


def _intervals(judged: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The writer's own intervals that hold their estimates, per metric."""
    found = {}
    for metric in claims_task.METRICS:
        entry = judged.get(metric) or {}
        if entry.get("interval_scored"):
            low, high = entry["interval"]
            found[metric] = {
                "low": low,
                "high": high,
                "measured": entry["value"],
                "covers": bool(entry["interval_covers"]),
            }
    return found


def _replicated(probe: dict[str, Any], name: str) -> float | None:
    values = [
        run["estimates"][name]["balanced_accuracy"]["estimate"]
        for run in probe.get("replication", [])
        if "balanced_accuracy" in run.get("estimates", {}).get(name, {})
    ]
    return float(np.mean(values)) if values else None


def claim_rows(
    output: Path, audits: Path | None, slots: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """One row per use of every planned program, made or not."""
    answers = _audits(audits)
    rows = []
    for slot in slots:
        chosen = claims_task.spec(slot["world"])
        names = (slot["writer"], chosen.name, slot["condition"], slot["attempt"])
        directory = claims_agent.attempt_directory(Path(output), *names)
        program = claims_agent.attempt_id(*names)
        attempt = _read(directory / claims_agent.ATTEMPT_FILE)
        submission = _read(directory / claims_run.RESULT_FILE) if attempt else None
        settled = (_read(directory / claims_run.SETTLED_LABELS) or {}).get("warrant", {})
        probe = _read(directory / claims_probe.PROBE_FILE) or {}
        report = (submission or {}).get("report")
        estimates = report.get("estimates") if isinstance(report, dict) else None
        estimates = estimates if isinstance(estimates, dict) else {}
        judged_uses = (submission or {}).get("claims") or {}
        partition = (submission or {}).get("partition") or {}
        ended = execution(attempt, submission)
        for claim in chosen.claims:
            estimate = estimates.get(claim.name)
            judged = judged_uses.get(claim.name)
            state = response_state(estimate, judged)
            accuracy = (judged or {}).get("balanced_accuracy") or {}
            warrant = settled.get(claim.name) or (partition.get(claim.name) or {}).get("warrant")
            forecast = ((probe.get("forecast") or {}).get(claim.name) or {}).get(
                "balanced_accuracy"
            )
            rows.append(
                {
                    "program": program,
                    "writer": slot["writer"],
                    "world": chosen.name,
                    "dataset": chosen.dataset,
                    "condition": slot["condition"],
                    "attempt": slot["attempt"],
                    "use": claim.name,
                    "part": claim.part,
                    "execution": ended,
                    "agent_claim": (attempt or {}).get("claim"),
                    "state": state,
                    "reason": str(estimate[claims_task.NOT_ESTIMABLE])
                    if state == "abstained"
                    else None,
                    "estimate": float(estimate["balanced_accuracy"]) if state in NUMERIC else None,
                    "measured": accuracy.get("value"),
                    "measured_low": accuracy.get("low"),
                    "measured_high": accuracy.get("high"),
                    "optimism": accuracy.get("optimism") if state in MEASURED_NUMERIC else None,
                    "intervals": _intervals(judged) if state in MEASURED_NUMERIC else {},
                    "warrant": (warrant or "unknown") if state in NUMERIC else None,
                    "probe_forecast": forecast,
                    "replicated_mean": _replicated(probe, claim.name),
                    "answers": {
                        key: answer[claim.name]
                        for key, answer in answers.get(program, {}).items()
                        if claim.name in answer
                    },
                }
            )
    return rows


def _decided(row: dict[str, Any], key: str) -> str | None:
    answer = row["answers"].get(key)
    return None if answer is None else answer["decision"]


def _forecast(row: dict[str, Any], key: str) -> float | None:
    answer = row["answers"].get(key)
    return None if answer is None else float(answer["forecast"]["balanced_accuracy"])


def share(rows: Sequence[dict[str, Any]], test: Callable[[dict[str, Any]], bool]) -> float | None:
    return float(np.mean([test(row) for row in rows])) if rows else None


def strata(rows: Sequence[dict[str, Any]]) -> list[list[str]]:
    """The programs of rows by writer and arm, in the order met. Strata with fewer than
    MIN_STRATUM programs are pooled; a pool still that small joins the largest stratum."""
    by_key: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        programs = by_key[(row["writer"], row["condition"])]
        if row["program"] not in programs:
            programs.append(row["program"])
    kept = [list(programs) for programs in by_key.values() if len(programs) >= MIN_STRATUM]
    pooled = [p for programs in by_key.values() if len(programs) < MIN_STRATUM for p in programs]
    if pooled:
        if len(pooled) >= MIN_STRATUM or not kept:
            kept.append(pooled)
        else:
            max(kept, key=len).extend(pooled)
    return kept


def bootstrap(
    rows: Sequence[dict[str, Any]],
    statistic: Callable[[Sequence[dict[str, Any]]], float | None],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """A statistic of rows with a 95% interval from resampling whole programs within the
    strata; no interval over fewer than MIN_STRATUM programs."""
    by_program: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_program[row["program"]].append(row)
    value = statistic(rows)
    found: dict[str, Any] = {"value": value, "low": None, "high": None, "programs": len(by_program)}
    if value is None or len(by_program) < MIN_STRATUM:
        return found
    groups = strata(rows)
    generator = np.random.default_rng(seed)
    drawn = []
    for _ in range(samples):
        picked: list[dict[str, Any]] = []
        for programs in groups:
            for index in generator.integers(0, len(programs), len(programs)):
                picked.extend(by_program[programs[index]])
        result = statistic(picked)
        if result is not None:
            drawn.append(result)
    if drawn:
        found["low"] = float(np.quantile(drawn, ALPHA / 2))
        found["high"] = float(np.quantile(drawn, 1 - ALPHA / 2))
    return found


def sign_flip(
    differences: Sequence[float],
    *,
    draws: int = RANDOMIZATION_DRAWS,
    seed: int = RANDOMIZATION_SEED,
) -> float | None:
    """The two-sided p-value of a paired sign-flip randomization test that the mean
    difference is zero: exact over every sign pattern up to EXACT_UP_TO pairs, otherwise
    from draws random patterns with the observed one counted."""
    found = np.asarray(differences, dtype=np.float64)
    count = found.size
    if count == 0:
        return None
    observed = abs(found.mean()) - 1e-12
    if count <= EXACT_UP_TO:
        bits = (np.arange(2**count)[:, None] >> np.arange(count)) & 1
        means = np.abs((1 - 2 * bits) @ found) / count
        return float(np.mean(means >= observed))
    generator = np.random.default_rng(seed)
    extreme, left = 0, draws
    while left:
        size = min(left, 10_000)
        signs = generator.choice((-1.0, 1.0), size=(size, count))
        extreme += int(np.sum(np.abs(signs @ found) / count >= observed))
        left -= size
    return (extreme + 1) / (draws + 1)


def _arms(rows: Sequence[dict[str, Any]]) -> list[str]:
    return list(dict.fromkeys(row["condition"] for row in rows))


def _writers(rows: Sequence[dict[str, Any]], part: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["writer"] != REFERENCE and row["part"] == part]


def responses(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """P1: the state of every planned response, per arm and over both."""
    found: dict[str, Any] = {}
    for arm in [*_arms(rows), "both"]:
        chosen = list(rows) if arm == "both" else [r for r in rows if r["condition"] == arm]
        found[arm] = {
            "programs": len(chosen),
            "states": {
                state: {
                    "count": sum(r["state"] == state for r in chosen),
                    **bootstrap(
                        chosen,
                        lambda f, state=state: share(f, lambda row: row["state"] == state),
                    ),
                }
                for state in STATES
            },
            "abstained_with_measured_predictor": sum(
                r["state"] == "abstained" and r["measured"] is not None for r in chosen
            ),
        }
    return found


def endorsed_without_warrant(rows: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    """P2: the share of numeric claims without warrant an auditor at a level calls
    supported, over its answers, with the bounds for its missing answers."""
    candidates = [r for r in rows if r["state"] in NUMERIC and r["warrant"] == "not_supported"]
    answered = [r for r in candidates if key in r["answers"]]
    endorsed = sum(_decided(r, key) == "supported" for r in answered)
    missing = len(candidates) - len(answered)
    return {
        **bootstrap(answered, lambda f: share(f, lambda r: _decided(r, key) == "supported")),
        "claims": len(candidates),
        "answered": len(answered),
        "endorsed": endorsed,
        "bounds": [endorsed / len(candidates), (endorsed + missing) / len(candidates)]
        if candidates
        else None,
        "by_arm": {
            arm: {
                "claims": sum(r["condition"] == arm for r in candidates),
                "answered": sum(r["condition"] == arm for r in answered),
                "endorsed": sum(
                    r["condition"] == arm and _decided(r, key) == "supported" for r in answered
                ),
            }
            for arm in _arms(candidates)
        },
        "tested": False,
    }


def evidence_difference(rows: Sequence[dict[str, Any]], auditor: str) -> dict[str, Any]:
    """P3: the auditor's absolute forecast error with the replication table minus that
    with the held-out-site table, over every measured predictor, and its test."""
    first, second = (f"{auditor}/{level}" for level in PAIRED_LEVELS)
    measured = [r for r in rows if r["measured"] is not None]

    def difference(row: dict[str, Any]) -> float:
        errors = []
        for key in (first, second):
            forecast = _forecast(row, key)
            forecast = FALLBACK_FORECAST if forecast is None else forecast
            errors.append(abs(forecast - row["measured"]))
        return errors[0] - errors[1]

    def mean(found: Sequence[dict[str, Any]]) -> float | None:
        return float(np.mean([difference(r) for r in found])) if found else None

    complete = [
        r for r in measured if _forecast(r, first) is not None and _forecast(r, second) is not None
    ]
    return {
        **bootstrap(measured, mean),
        "claims": len(measured),
        "p": sign_flip([difference(r) for r in measured]),
        "missing": {
            level: sum(_forecast(r, key) is None for r in measured)
            for level, key in zip(PAIRED_LEVELS, (first, second), strict=True)
        },
        "complete_pairs": {
            **bootstrap(complete, mean),
            "claims": len(complete),
            "p": sign_flip([difference(r) for r in complete]),
        },
        "tested": True,
    }


def primary(
    rows: Sequence[dict[str, Any]], part: str = "new_site", *, auditor: str = PRIMARY_AUDITOR
) -> dict[str, Any]:
    """The three primary endpoints on a use; on the new lesions or patients, secondary."""
    writers = _writers(rows, part)
    return {
        "P1_responses": responses(writers),
        "P2_endorsed_without_warrant": endorsed_without_warrant(
            writers, f"{auditor}/{PRIMARY_LEVEL}"
        ),
        "P3_error_replication_minus_regime": evidence_difference(writers, auditor),
    }


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def intervals(rows: Sequence[dict[str, Any]], metric: str) -> dict[str, Any]:
    """The writers' own intervals of a metric on measured numeric claims."""
    scored = [r for r in rows if metric in r["intervals"]]

    def entry(row: dict[str, Any]) -> dict[str, Any]:
        return row["intervals"][metric]

    return {
        "claims": len(rows),
        "scored": len(scored),
        "coverage": bootstrap(scored, lambda f: share(f, lambda r: entry(r)["covers"])),
        "width": _mean([entry(r)["high"] - entry(r)["low"] for r in scored]),
        "interval_score": _mean(
            [
                interval_score(entry(r)["low"], entry(r)["high"], entry(r)["measured"])
                for r in scored
            ]
        ),
    }


def secondary(
    rows: Sequence[dict[str, Any]], part: str, *, auditor: str = PRIMARY_AUDITOR
) -> dict[str, Any]:
    """The pre-specified descriptive results on a use."""
    writers = _writers(rows, part)
    key = f"{auditor}/{PRIMARY_LEVEL}"
    measured = [r for r in writers if r["state"] in MEASURED_NUMERIC]
    material = [r for r in measured if r["state"] == "numeric_material"]
    unsupported = [r for r in material if r["warrant"] == "not_supported" and key in r["answers"]]
    supported = [
        r
        for r in measured
        if r["state"] == "numeric_not_material"
        and r["warrant"] == "supported"
        and key in r["answers"]
    ]

    def endorsed(found):
        return share(found, lambda r: _decided(r, key) == "supported")

    by_claim: dict[str, Counter] = defaultdict(Counter)
    for row in writers:
        by_claim[row["state"]][str(row["agent_claim"])] += 1
    return {
        "materially_optimistic_among_measured": {
            arm: {
                **bootstrap(
                    [r for r in measured if r["condition"] == arm],
                    lambda f: share(f, lambda row: row["state"] == "numeric_material"),
                ),
                "claims": sum(r["condition"] == arm for r in measured),
            }
            for arm in _arms(writers)
        },
        "endorsed_without_warrant_and_material": {
            **bootstrap(unsupported, endorsed),
            "claims": len(unsupported),
        },
        "endorsed_with_warrant_not_material": {
            **bootstrap(supported, endorsed),
            "claims": len(supported),
        },
        "writer_intervals": {
            arm: {
                metric: intervals([r for r in measured if r["condition"] == arm], metric)
                for metric in claims_task.METRICS
            }
            for arm in _arms(writers)
        },
        "states_by_agent_claim": {state: dict(by_claim[state]) for state in STATES},
    }


def _auroc(labels: Sequence[bool], scores: Sequence[float]) -> float | None:
    labels_array = np.asarray(labels, dtype=bool)
    if labels_array.all() or not labels_array.any():
        return None
    return claims_run.auroc(labels_array, np.asarray(scores, dtype=float))


def levels(rows: Sequence[dict[str, Any]], part: str = "new_site") -> dict[str, Any]:
    """Every auditor and level on a use of the writers' programs: endorsement of numeric
    claims without warrant, and of measured claims with warrant that are not materially
    optimistic; forecast error and interval score on measured predictors; same-vendor
    pairs; and the AUROC of risk for an optimism above MATERIAL within each dataset."""
    writers = _writers(rows, part)
    keys = sorted({key for r in writers for key in r["answers"]})
    found: dict[str, Any] = {}
    for key in keys:
        answered = [r for r in writers if key in r["answers"]]
        measured = [r for r in answered if r["measured"] is not None]
        without = [r for r in answered if r["state"] in NUMERIC and r["warrant"] == "not_supported"]
        with_warrant = [
            r
            for r in answered
            if r["state"] == "numeric_not_material" and r["warrant"] == "supported"
        ]
        scored = [r for r in answered if r["state"] in MEASURED_NUMERIC]
        auditor = key.split("/")[0]

        def endorsed(found, key=key):
            return share(found, lambda row: _decided(row, key) == "supported")

        found[key] = {
            "answered": len(answered),
            "endorsed_without_warrant": endorsed(without),
            "without_warrant": len(without),
            "endorsed_with_warrant": endorsed(with_warrant),
            "with_warrant": len(with_warrant),
            "measured": len(measured),
            "forecast_error": _mean([abs(_forecast(r, key) - r["measured"]) for r in measured]),
            "interval_score": _mean(
                [
                    interval_score(
                        r["answers"][key]["forecast"]["low"],
                        r["answers"][key]["forecast"]["high"],
                        r["measured"],
                    )
                    for r in measured
                ]
            ),
            "same_vendor": sum(VENDORS.get(r["writer"]) == VENDORS.get(auditor) for r in answered),
            "risk_auroc": {
                dataset: _auroc(
                    [r["optimism"] > MATERIAL for r in scored if r["dataset"] == dataset],
                    [r["answers"][key]["risk"] for r in scored if r["dataset"] == dataset],
                )
                for dataset in sorted({r["dataset"] for r in scored})
            },
        }
    return found


def rules(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The fixed rules as forecasters of the sealed value, on measured numeric claims."""
    forecasts = {
        "claim": lambda r: r["estimate"],
        "claim_minus_discount": lambda r: r["estimate"] - DISCOUNT,
        "probe": lambda r: (r["probe_forecast"] or {}).get("value"),
        "replicated_mean": lambda r: r["replicated_mean"],
    }
    found: dict[str, Any] = {}
    for part in USES:
        measured = [r for r in _writers(rows, part) if r["state"] in MEASURED_NUMERIC]
        found[part] = {}
        for name, forecast in forecasts.items():
            usable = [r for r in measured if forecast(r) is not None]

            def error(f, forecast=forecast):
                return _mean([abs(forecast(r) - r["measured"]) for r in f])

            found[part][name] = {**bootstrap(usable, error), "claims": len(usable)}
    return found


def section(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Every result of one part of the study: the main worlds, or development."""
    writers = [r for r in rows if r["writer"] != REFERENCE]
    endings = {r["program"]: r["execution"] for r in writers}
    return {
        "planned_programs": len(endings),
        "made_programs": sum(ended != "not_made" for ended in endings.values()),
        "executions": dict(Counter(endings.values())),
        "primary": primary(rows, "new_site"),
        "new_groups": primary(rows, "new_groups"),
        "secondary": {part: secondary(rows, part) for part in USES},
        "levels": {part: levels(rows, part) for part in USES},
        "rules": rules(rows),
        "abstentions": [
            {
                key: r[key]
                for key in ("program", "writer", "world", "condition", "use", "reason", "measured")
            }
            for r in writers
            if r["state"] == "abstained"
        ],
        "reference": [
            {
                key: r[key]
                for key in (
                    "world",
                    "use",
                    "state",
                    "estimate",
                    "measured",
                    "optimism",
                    "intervals",
                )
            }
            for r in rows
            if r["writer"] == REFERENCE
        ],
    }


def analyse(plan: dict[str, Any] | None = None) -> dict[str, Any]:
    """The whole analysis, from the outputs and audits the plan names."""
    plan = plan or claims_freeze.PLAN
    development = plan["development"]
    main_worlds = [world for world in plan["worlds"] if world not in development["worlds"]]
    found: dict[str, Any] = {
        "schema_version": 2,
        "task": "claims_analysis",
        "primary_auditor": PRIMARY_AUDITOR,
        "primary_level": PRIMARY_LEVEL,
        "arms": list(plan["conditions"]),
    }
    for name, worlds, output, audits in (
        ("main", main_worlds, plan["output"], plan["audits"]),
        ("development", development["worlds"], development["output"], development["audits"]),
    ):
        slots = planned(plan, worlds) + references(worlds)
        found[name] = {"worlds": list(worlds), **section(claim_rows(output, audits, slots))}
    return found


def _shown(entry: dict[str, Any]) -> str:
    if entry["value"] is None:
        return "none"
    if entry["low"] is None:
        return f"{entry['value']:.3f}"
    return f"{entry['value']:.3f} [{entry['low']:.3f}, {entry['high']:.3f}]"


def summary(found: dict[str, Any]) -> str:
    lines = []
    for name in ("main", "development"):
        part = found[name]
        primary_found = part["primary"]
        lines.append(
            f"{name}: {part['made_programs']} of {part['planned_programs']} planned programs made"
        )
        for arm, entry in primary_found["P1_responses"].items():
            states = ", ".join(
                f"{state} {value['count']}" for state, value in entry["states"].items()
            )
            lines.append(f"  P1 new-site responses, {arm} ({entry['programs']}): {states}")
        p2 = primary_found["P2_endorsed_without_warrant"]
        bounds = "" if p2["bounds"] is None else " bounds {:.3f} to {:.3f}".format(*p2["bounds"])
        lines.append(
            f"  P2_endorsed_without_warrant: {_shown(p2)} over {p2['answered']} answered of "
            f"{p2['claims']} claims;{bounds}"
        )
        p3 = primary_found["P3_error_replication_minus_regime"]
        test = "" if p3["p"] is None else f", p {p3['p']:.4f}"
        lines.append(
            f"  P3_error_replication_minus_regime: {_shown(p3)} over {p3['claims']} "
            f"predictors{test}; forecasts missing: replication {p3['missing']['replication']}, "
            f"regime {p3['missing']['regime']}; complete pairs "
            f"{_shown(p3['complete_pairs'])} over {p3['complete_pairs']['claims']}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The analysis of the claims study.")
    parser.add_argument("--report", type=Path, required=True, help="Where to write the analysis.")
    parser.add_argument(
        "--freeze",
        type=Path,
        help="A freeze record whose plan to analyse; by default the plan in claims_freeze.",
    )
    args = parser.parse_args(argv)
    plan = None
    if args.freeze is not None:
        plan = json.loads(args.freeze.read_text(encoding="utf-8"))["plan"]
    found = analyse(plan)
    args.report.write_text(json.dumps(found, indent=2) + "\n", encoding="utf-8")
    print(summary(found))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
