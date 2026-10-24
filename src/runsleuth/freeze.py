"""The freeze: what every result on the test split depends on, recorded before the first
test candidate is generated.

write records, into evaluations/freeze.json, the commit, the hash of every frozen file,
the settings the test runs use, the rule combining v2 with an LLM reviewer, the analysis
plan, the budget and the three records of independence. It refuses while a frozen file
has changes not committed, and once the test seeds are drawn: the seeds are drawn only
after the record is committed (challenge_set draw-seeds).

Every command on the test split checks first that no frozen file has changed since
(check, called by candidates.candidates). A change made after the freeze, such as the
fix of a crash, is allowed only once it is recorded as a deviation with its reason, and
the deviations are reported with the results.

The frozen files are every file of the package (the verifiers, the hidden tests, the
harness, the agent loop, the reviewer's prompt and the scoring), the run configurations,
the metric gate's thresholds and v2's probe settings. Hashes are of the text with LF
line ends, as git stores it.

    python -m runsleuth.freeze write
    python -m runsleuth.freeze check
    python -m runsleuth.freeze deviation --file src/runsleuth/candidates.py --reason "..."
"""

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from runsleuth import (
    challenge_set,
    llm_client,
    llm_reviewer,
    pipeline_agent,
    pipeline_contract,
    pipeline_tasks,
    verifier_v2,
)

FREEZE_FILE = Path("evaluations/freeze.json")
PACKAGE = Path("src/runsleuth")
FROZEN_DATA = (
    Path("configs/camelyon17_clean.json"),
    Path("configs/camelyon17_probe.json"),
    Path("configs/ham10000.json"),
    Path("evaluations/metric_gate.json"),
    verifier_v2.PROBES_FILE,
)
COMBINATION = (
    "RunSleuth v2 with an LLM reviewer (verifier_evaluation.combined): v2 finding a part "
    "violated rejects; v2 and the reviewer both accepting accepts; anything else goes to "
    "review."
)
ANALYSIS = {
    "main": (
        "The agent candidates of the test split. For every verifier: wrong candidates "
        "accepted (of the wrong), correct candidates rejected (of the correct), the share "
        "sent to review and the cost of checking, each with its denominator."
    ),
    "verifiers": (
        "metric_gate, stage_one_gate, runsleuth_v1, llm_reviewer/<each reviewer>, "
        "runsleuth_v2, runsleuth_v2_fixed_order and runsleuth_v2+llm_reviewer/<each reviewer>."
    ),
    "not_runnable": (
        "A candidate the harness refuses or whose hidden tests crash or time out is wrong; "
        "the rates are given again without these candidates."
    ),
    "apart": (
        "Hand-written versions and partial repairs, evidence-removed runs, each family, "
        "and for a reviewer the candidates of its maker's models are reported apart."
    ),
    "uncertainty": (
        "95% intervals by a bootstrap over cases: the candidates that start from one "
        "version (a partial repair with its fault) are drawn together, since they are not "
        "independent; settings in verifier_evaluation.BOOTSTRAP. A verifier that never "
        "errs in a kind, or always does, gets the exact bound with the cases as units. "
        "No significance tests."
    ),
    "order": (
        "v2's gap order against the fixed order: probe runs and GPU seconds per candidate, "
        "and whether their decisions agree."
    ),
    "labels": (
        "The hidden tests decide every label; the blind review of two annotators audits "
        "them and changes none."
    ),
}
BUDGET = {
    "cap_usd": 20,
    "rule": (
        "Spending on test candidates and reviews is counted from the providers' bills. "
        "Should it be about to pass the cap, every remaining model runs two repetitions "
        "(the first two test seeds) instead of three; no model is dropped."
    ),
}
INDEPENDENCE = (
    "RunSleuth v2 never sees the hidden tests, their assertions or inputs, nor any label: "
    "verifier_v2 imports neither pipeline_contract nor anything that reads labels, and "
    "its probes run the candidate's own project with the statement's numbers.",
    "v2's probe settings (evaluations/verifier_v2_probes.json, its probe data and seeds) "
    "and its validation tolerance were fixed on development material alone: the healthy "
    "project, the development versions and the development candidates.",
    "Development results served only to debug the verifiers. No verifier has seen a "
    "test candidate, and none exists before this record.",
)


def _file_sha256(path: Path) -> str:
    """A file's hash whatever its line endings, which a checkout on Windows may rewrite."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def frozen_files(root: Path = Path(".")) -> list[Path]:
    """Every frozen file, relative to the repository's root."""
    root = Path(root)
    package = [
        path.relative_to(root)
        for path in sorted((root / PACKAGE).rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    ]
    missing = [str(path) for path in FROZEN_DATA if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"Frozen files missing: {missing}")
    return [*package, *FROZEN_DATA]


def uncommitted(paths: list[Path]) -> list[str]:
    """The paths among these that git sees changed or untracked."""
    status = _git("status", "--porcelain", "--untracked-files=all", "--", *map(str, paths))
    return [line[3:] for line in status.splitlines() if line.strip()]


def committed(path: Path = FREEZE_FILE) -> bool:
    """Whether git holds this file as it is."""
    try:
        _git("ls-files", "--error-unmatch", str(path))
    except subprocess.CalledProcessError:
        return False
    return not uncommitted([path])


def settings() -> dict[str, Any]:
    """The settings the test runs use, as the frozen modules hold them."""
    from runsleuth import candidates, verifier_evaluation

    planned = challenge_set.plan("test")
    return {
        "models": list(challenge_set.MODELS),
        "repetitions": challenge_set.REPETITIONS,
        "test_seeds": challenge_set.TEST_SEED_COUNT,
        "reviewers": list(challenge_set.REVIEWERS),
        "vendors": challenge_set.VENDORS,
        "plan": dict(Counter(candidate.kind for candidate in planned)),
        "agent": {
            "prompt_version": pipeline_agent.PROMPT_VERSION,
            "max_tool_calls": pipeline_agent.MAX_TOOL_CALLS,
            "max_runs": pipeline_agent.MAX_RUNS,
            "max_output_tokens": pipeline_agent.MAX_OUTPUT_TOKENS,
            "short_run": pipeline_agent.SHORT_RUN,
        },
        "reviewer": {
            "prompt_version": llm_reviewer.PROMPT_VERSION,
            "system_prompt_sha256": hashlib.sha256(llm_reviewer.SYSTEM_PROMPT.encode()).hexdigest(),
            "max_model_calls": llm_reviewer.MAX_MODEL_CALLS,
            "max_output_tokens": llm_reviewer.MAX_OUTPUT_TOKENS,
        },
        "request_timeout_seconds": llm_client.REQUEST_TIMEOUT,
        "candidate_runs": {
            "epochs": candidates.EPOCHS,
            "interrupt_after_epochs": candidates.INTERRUPT_AFTER_EPOCHS,
        },
        "v2": {
            "orders": list(verifier_v2.ORDERS),
            "validation_tolerance": verifier_v2.VALIDATION_TOLERANCE,
            "probes": verifier_v2.load_settings(),
        },
        "contract": {
            "version": pipeline_contract.CONTRACT_VERSION,
            "family_sha256": {
                family: pipeline_tasks.family_sha256(family) for family in challenge_set.FAMILIES
            },
        },
        "bootstrap": verifier_evaluation.BOOTSTRAP,
    }


def write(
    path: Path = FREEZE_FILE,
    *,
    root: Path = Path("."),
    seeds: Path = challenge_set.SEEDS_FILE,
) -> dict[str, Any]:
    """Record the freeze; refused once frozen, once the seeds are drawn, or while a
    frozen file has changes not committed."""
    if Path(path).exists():
        raise FileExistsError(f"{path} already records the freeze")
    if Path(seeds).exists():
        raise FileExistsError(f"The test seeds are drawn already ({seeds}); a freeze comes first")
    files = frozen_files(root)
    changed = uncommitted([PACKAGE, *FROZEN_DATA])
    if changed:
        raise RuntimeError(f"Commit these before the freeze: {changed}")
    record = {
        "schema_version": 1,
        "task": "freeze",
        "frozen_on": date.today().isoformat(),
        "commit": _git("rev-parse", "HEAD").strip(),
        "files": {path.as_posix(): _file_sha256(Path(root) / path) for path in files},
        "settings": settings(),
        "combination": COMBINATION,
        "analysis": ANALYSIS,
        "budget": BUDGET,
        "independence": list(INDEPENDENCE),
        "deviations": [],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def _read(path: Path) -> dict[str, Any]:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"The test split opens only after the freeze ({path}): python -m runsleuth.freeze write"
        )
    return json.loads(Path(path).read_text(encoding="utf-8"))


def allowed(record: dict[str, Any]) -> dict[str, str | None]:
    """Each frozen file's hash as the freeze and its deviations allow it."""
    hashes = dict(record["files"])
    for deviation in record["deviations"]:
        hashes[deviation["file"]] = deviation["sha256"]
    return hashes


def changed_files(record: dict[str, Any], root: Path = Path(".")) -> list[str]:
    """The frozen files that are not what the freeze and its deviations allow."""
    found = []
    for name, digest in allowed(record).items():
        current = Path(root) / name
        if (_file_sha256(current) if current.is_file() else None) != digest:
            found.append(name)
    return found


def check(path: Path = FREEZE_FILE, *, root: Path = Path(".")) -> dict[str, Any]:
    """The freeze record, once no frozen file has changed but as a recorded deviation."""
    record = _read(path)
    changed = changed_files(record, root)
    if changed:
        raise RuntimeError(
            f"Frozen files changed since the freeze: {changed}. Record each change and "
            "its reason with python -m runsleuth.freeze deviation"
        )
    return record


def deviation(
    name: str, reason: str, path: Path = FREEZE_FILE, *, root: Path = Path(".")
) -> dict[str, Any]:
    """Allow a frozen file as it is now, with the reason it changed."""
    record = _read(path)
    name = Path(name).as_posix()
    if name not in record["files"]:
        raise ValueError(f"{name} is not a frozen file")
    if not reason.strip():
        raise ValueError("A deviation needs its reason")
    current = Path(root) / name
    record["deviations"].append(
        {
            "file": name,
            "sha256": _file_sha256(current) if current.is_file() else None,
            "reason": reason.strip(),
            "recorded_on": date.today().isoformat(),
        }
    )
    Path(path).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="The freeze before the test split.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("write", help=f"Record the freeze into {FREEZE_FILE}.")
    commands.add_parser("check", help="Say whether any frozen file has changed.")
    deviated = commands.add_parser("deviation", help="Allow a changed frozen file, with why.")
    deviated.add_argument("--file", required=True)
    deviated.add_argument("--reason", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "write":
        record = write()
        print(f"frozen {len(record['files'])} files at {record['commit']}: {FREEZE_FILE}")
    elif args.command == "check":
        record = check()
        print(
            f"no frozen file changed since {record['commit'][:12]}; "
            f"{len(record['deviations'])} deviations recorded"
        )
    else:
        record = deviation(args.file, args.reason)
        print(f"{len(record['deviations'])} deviations recorded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
