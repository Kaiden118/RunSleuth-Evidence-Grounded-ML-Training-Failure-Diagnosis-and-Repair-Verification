"""Compare LLM reviewers on the same recorded evidence, for example Gemini and Ollama.

    python -m runsleuth.llm_review_eval run --provider ollama --model qwen3:8b \n        --from-record evaluations/results/signature-matching-20261008.json
    python -m runsleuth.llm_review_eval summarize <run directory> ... --output <result.json>

Every labeled run in the experiment reports is reviewed twice, without and with its
experiment's clean run as reference: the same cases as the signature-matching
evaluation. Each review goes through diagnose_run unchanged, so citations are
validated and the matcher stays final. Records are appended per case to
cases.jsonl, so an interrupted run (a quota limit, a stopped server) resumes where
it stopped; a case that ended in an API error is retried. Replies are sampled once
per case at provider defaults, so repeated runs can differ.
"""

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from runsleuth.camelyon_optimizer_probe import file_sha256
from runsleuth.diagnose_run import (
    MAX_MODEL_CALLS,
    MAX_OUTPUT_TOKENS,
    PAYLOAD_VERSION,
    PROMPT_VERSION,
    RESPONSE_FORMATS,
    diagnose_run,
    system_prompt,
)
from runsleuth.signature_matching import labeled_runs

MODES = ("reference_free", "with_reference")
FINAL_STATUSES = ("completed", "invalid_output")
MAX_CONSECUTIVE_API_ERRORS = 3


def run_directory(output_dir: Path, provider: str, model: str) -> Path:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model)
    return output_dir / f"{provider}-{slug}-prompt-v{PROMPT_VERSION}"


def _cases(report_paths: list[Path]):
    for item in labeled_runs(report_paths):
        for mode in MODES:
            yield f"{item['report'].parent.name}:{item['variant']}:{mode}", mode, item


def record_reports(record_path: Path) -> list[Path]:
    """The experiment reports of a signature-matching record, in its order."""
    record = json.loads(record_path.read_text(encoding="utf-8"))
    paths = dict.fromkeys(case["report"].replace("\\", "/") for case in record["cases"])
    return [Path(path) for path in paths]


def _latest(directory: Path) -> dict[str, dict]:
    """The last record of every case; earlier records are API errors that were retried."""
    path = directory / "cases.jsonl"
    if not path.is_file():
        return {}
    records = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    return {record["case_id"]: record for record in records}


def _record(case_id: str, mode: str, item: dict, result: dict, latency: float) -> dict:
    review, matcher = result["llm"], result["matcher"]
    return {
        "case_id": case_id,
        "report": str(item["report"]),
        "seed": item["seed"],
        "variant": item["variant"],
        "mode": mode,
        "truth": item["truth"],
        "matcher_diagnosis": matcher["diagnosis"],
        "matcher_pending_reference": matcher["pending_reference"],
        "llm_status": review["status"],
        "llm_diagnosis": review["diagnosis"]["diagnosis"] if review["diagnosis"] else None,
        "conflict": result["conflict"],
        "conflict_kind": result.get("conflict_kind"),
        "model_calls": review["model_calls"],
        "input_tokens": review["input_tokens"],
        "output_tokens": review["output_tokens"],
        "latency_seconds": latency,
        "attempt_issues": [attempt["issues"] for attempt in review["attempts"]],
        "finish_reasons": [attempt["finish_reason"] for attempt in review["attempts"]],
        "reviewed_at": datetime.now(UTC).isoformat(),
        "review": review,
    }


def run_reviews(
    report_paths: list[Path],
    client,
    provider: str,
    model: str,
    output_dir: Path,
    *,
    limit: int | None = None,
    pause: float = 0.0,
    error_wait: float = 0.0,
    response_format: str = "json_schema",
    only_cases: set[str] | None = None,
    sleep=time.sleep,
) -> Path:
    """Review every pending case; returns the run directory.

    pause separates cases; error_wait follows an API error, so a briefly overloaded
    server can recover before the next case. only_cases limits the run to those
    case ids, for example the failed reviews of an earlier run.
    """
    cases = list(_cases(report_paths))
    if only_cases is not None:
        unknown = only_cases - {case_id for case_id, _, _ in cases}
        if not only_cases or unknown:
            raise ValueError(f"Choose known case ids; unknown: {sorted(unknown)}")
        cases = [case for case in cases if case[0] in only_cases]
    directory = run_directory(output_dir, provider, model)
    directory.mkdir(parents=True, exist_ok=True)
    settings = {
        "provider": provider,
        "model": model,
        "max_model_calls": MAX_MODEL_CALLS,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "prompt_version": PROMPT_VERSION,
        "payload_version": PAYLOAD_VERSION,
        "system_prompt_sha256": hashlib.sha256(system_prompt().encode()).hexdigest(),
        "response_format": response_format,
        "reports": [{"path": str(path), "sha256": file_sha256(path)} for path in report_paths],
    }
    if only_cases is not None:
        settings["case_ids"] = sorted(only_cases)
    settings_path = directory / "run.json"
    if not settings_path.is_file():
        settings_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    elif json.loads(settings_path.read_text(encoding="utf-8")) != settings:
        raise ValueError(f"{directory} holds a run with other settings or reports")
    done = {
        case_id
        for case_id, record in _latest(directory).items()
        if record["llm_status"] in FINAL_STATUSES
    }
    pending = [case for case in cases if case[0] not in done]
    total = len(done) + len(pending)
    if limit is not None:
        pending = pending[:limit]
    consecutive_errors, wait = 0, 0.0
    with (directory / "cases.jsonl").open("a", encoding="utf-8") as stream:
        for index, (case_id, mode, item) in enumerate(pending, start=1):
            if wait:
                sleep(wait)
            reference = item["reference"] if mode == "with_reference" else None
            started = time.perf_counter()
            result = diagnose_run(
                item["run"],
                reference,
                client=client,
                model=model,
                optimizer_name=item["optimizer"],
                response_format=response_format,
            )
            record = _record(case_id, mode, item, result, time.perf_counter() - started)
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            outcome = "conflict" if record["conflict"] else "agrees"
            status = record["llm_status"]
            if error := record["review"].get("error"):
                status += f" {error['type']} {error.get('status_code')}"
            print(
                f"[{len(done) + index}/{total}] {case_id}: "
                f"{status} {record['llm_diagnosis']} ({outcome}, "
                f"{record['latency_seconds']:.1f}s)",
                flush=True,
            )
            failed = record["llm_status"] == "api_error"
            consecutive_errors = consecutive_errors + 1 if failed else 0
            wait = max(pause, error_wait) if failed else pause
            if consecutive_errors >= MAX_CONSECUTIVE_API_ERRORS:
                print(
                    f"Stopped after {consecutive_errors} API errors in a row (quota or server); "
                    "rerun the same command to resume",
                    flush=True,
                )
                break
    return directory


def _statistics(records: list[dict]) -> dict:
    completed = [record for record in records if record["llm_status"] == "completed"]
    called = [record for record in records if record["model_calls"]]
    return {
        "cases": len(records),
        "completed": len(completed),
        "invalid_output": sum(record["llm_status"] == "invalid_output" for record in records),
        "api_error": sum(record["llm_status"] == "api_error" for record in records),
        "valid_on_first_attempt": sum(not record["attempt_issues"][0] for record in completed),
        "agrees_with_matcher": sum(not record["conflict"] for record in completed),
        "conflicts": dict(
            Counter(record["conflict_kind"] for record in completed if record["conflict"])
        ),
        "llm_matches_truth": sum(
            record["llm_diagnosis"] == record["truth"] for record in completed
        ),
        "llm_defers_with_true_fault": sum(
            record["llm_diagnosis"] == "pending_reference"
            and record["truth"] in record["matcher_pending_reference"]
            for record in completed
        ),
        "matcher_matches_truth": sum(
            record["matcher_diagnosis"] == record["truth"] for record in records
        ),
        "issue_types": dict(
            Counter(
                issue["type"]
                for record in records
                for issues in record["attempt_issues"]
                for issue in issues
            )
        ),
        "truncated_replies": sum(
            reason == "length" for record in records for reason in record["finish_reasons"]
        ),
        "input_tokens": sum(record["input_tokens"] for record in records),
        "output_tokens": sum(record["output_tokens"] for record in records),
        "mean_latency_seconds": (
            sum(record["latency_seconds"] for record in called) / len(called) if called else None
        ),
    }


def summarize(directory: Path) -> dict:
    """Statistics of one run directory, overall, by reference mode and by health."""
    # Runs before versioning used prompt 1, JSON-object replies and payload 1.
    settings = {
        "prompt_version": 1,
        "payload_version": 1,
        "response_format": "json_object",
        **json.loads((directory / "run.json").read_text(encoding="utf-8")),
    }
    expected = [case_id for case_id, _, _ in _cases([Path(r["path"]) for r in settings["reports"]])]
    if "case_ids" in settings:
        expected = [case_id for case_id in expected if case_id in settings["case_ids"]]
    latest = _latest(directory)
    records = [latest[case_id] for case_id in expected if case_id in latest]
    return {
        **{
            key: settings.get(key)
            for key in (
                "provider",
                "model",
                "prompt_version",
                "payload_version",
                "response_format",
                "max_output_tokens",
            )
        },
        "expected_cases": len(expected),
        "complete": len(records) == len(expected)
        and all(record["llm_status"] in FINAL_STATUSES for record in records),
        "overall": _statistics(records),
        "by_mode": {
            mode: _statistics([record for record in records if record["mode"] == mode])
            for mode in MODES
        },
        "by_health": {
            health: _statistics(
                [
                    record
                    for record in records
                    if (record["truth"] == "no_known_fault") == (health == "healthy")
                ]
            )
            for health in ("healthy", "faulty")
        },
        "cases": [
            {
                "case_id": record["case_id"],
                "truth": record["truth"],
                "matcher": record["matcher_diagnosis"],
                "llm_status": record["llm_status"],
                "llm": record["llm_diagnosis"],
                "conflict_kind": record["conflict_kind"],
                "valid_on_first_attempt": not record["attempt_issues"][0]
                if record["attempt_issues"]
                else None,
            }
            for record in records
        ],
        "reports": settings["reports"],
    }


def summary_table(summaries: list[dict]) -> str:
    def rate(statistics, key):
        return f"{statistics[key]}/{statistics['completed']}"

    lines = [
        f"{'model':<28}{'completed':>10}{'valid@1':>9}{'agrees':>8}{'assumed':>9}"
        f"{'differs':>9}{'truth':>7}{'defers':>8}{'tokens in/out':>16}{'latency':>9}"
    ]
    for summary in summaries:
        overall = summary["overall"]
        conflicts = overall["conflicts"]
        latency = overall["mean_latency_seconds"]
        lines.append(
            f"{summary['model'] + ' p' + str(summary['prompt_version']):<28}"
            f"{overall['completed']:>5}/{summary['expected_cases']:<4}"
            f"{rate(overall, 'valid_on_first_attempt'):>9}{rate(overall, 'agrees_with_matcher'):>8}"
            f"{conflicts.get('assumed_reference_condition', 0):>9}"
            f"{conflicts.get('different_diagnosis', 0):>9}{overall['llm_matches_truth']:>7}"
            f"{overall['llm_defers_with_true_fault']:>8}"
            f"{overall['input_tokens']:>9}/{overall['output_tokens']:<6}"
            f"{'-' if latency is None else f'{latency:.1f}s':>9}"
        )
    return "\n".join(lines)


def main() -> None:
    from runsleuth.llm_client import PROVIDERS, create_llm_client

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Review every case with one model")
    sources = run.add_mutually_exclusive_group(required=True)
    sources.add_argument("--reports", type=Path, nargs="+", help="Experiment reports")
    sources.add_argument(
        "--from-record", type=Path, help="The reports of a signature-matching evaluation"
    )
    run.add_argument("--provider", choices=PROVIDERS, required=True)
    run.add_argument("--model", help="Default: GEMINI_MODEL or OLLAMA_MODEL")
    run.add_argument("--output-dir", type=Path, default=Path("artifacts/llm_review_eval"))
    run.add_argument("--limit", type=int, help="Review at most this many pending cases")
    run.add_argument("--pause", type=float, default=0.0, help="Seconds between cases")
    run.add_argument(
        "--error-wait", type=float, default=0.0, help="Seconds to wait after an API error"
    )
    run.add_argument(
        "--response-format",
        choices=RESPONSE_FORMATS,
        default="json_schema",
        help="json_object for a provider that rejects the reply schema",
    )
    run.add_argument(
        "--only-invalid-from",
        type=Path,
        help="Review only the cases that ended invalid in this earlier run directory",
    )
    summary = commands.add_parser("summarize", help="Compare finished run directories")
    summary.add_argument("directories", type=Path, nargs="+")
    summary.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        try:
            client, model = create_llm_client(args.provider, args.model)
        except RuntimeError as error:
            raise SystemExit(str(error)) from error
        only_cases = None
        if args.only_invalid_from:
            only_cases = {
                case_id
                for case_id, record in _latest(args.only_invalid_from).items()
                if record["llm_status"] == "invalid_output"
            }
            if not only_cases:
                raise SystemExit(f"No invalid reviews in {args.only_invalid_from}")
        directory = run_reviews(
            args.reports or record_reports(args.from_record),
            client,
            args.provider,
            model,
            args.output_dir,
            limit=args.limit,
            pause=args.pause,
            error_wait=args.error_wait,
            response_format=args.response_format,
            only_cases=only_cases,
        )
        print(summary_table([summarize(directory)]))
        print(f"run_directory={directory}")
        return
    summaries = [summarize(directory) for directory in args.directories]
    if len({json.dumps(item["reports"], sort_keys=True) for item in summaries}) != 1:
        raise SystemExit("Run directories reviewed different reports")
    result = {
        "schema_version": 1,
        "task": "llm_review_evaluation",
        "scope": "development evaluation on author-injected faults; one sampled reply per "
        "case at provider defaults",
        "runs": summaries,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", "utf-8")
    print(summary_table(summaries))


if __name__ == "__main__":
    main()
