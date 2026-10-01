"""Localize a recorded optimizer defect and export a controller-owned source proposal."""

import argparse
import ast
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from runsleuth.evaluate_optimizer import Manifest, _attempt, _read, _scoped
from runsleuth.optimizer_agent_repair import _require_supported_pair
from runsleuth.optimizer_evidence import _same, inspect_optimizer_run
from runsleuth.optimizer_source_patch import propose_optimizer_order_patch

SOURCE_FILES = ("optimizer_probe.py", "camelyon_optimizer_training.py")
MAX_SOURCE_BYTES = 512 * 1024
PAIR_TASKS = {
    "camelyon17_paired_optimizer_binding_experiment",
    "camelyon17_optimizer_rebind_verification",
}


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _dispatch_evidence(source: str) -> dict:
    """Recognize the recorded caller; AST inspection is not an execution trace."""
    tree = ast.parse(source, feature_version=(3, 11))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run_variant"
    ]
    if len(functions) != 1 or functions[0].decorator_list:
        raise ValueError("Unrecognized recorded training dispatcher")
    function = functions[0]
    target = ast.dump(ast.parse('variant == "stale_head_repaired"', mode="eval").body)
    branches = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.If) and ast.dump(node.test) == target
    ]
    expected = ast.parse(
        "optimizer = make_probe_optimizer(model, variant=variant, "
        "learning_rate=config.learning_rate, weight_decay=config.weight_decay)"
    ).body[0]
    repaired = ast.parse(
        'optimizer, report["repair"] = make_repaired_probe_optimizer(model, '
        "learning_rate=config.learning_rate, weight_decay=config.weight_decay)"
    ).body[0]
    if (
        len(branches) != 1
        or len(branches[0].orelse) != 1
        or len(branches[0].body) != 1
        or ast.dump(branches[0].orelse[0]) != ast.dump(expected)
        or ast.dump(branches[0].body[0]) != ast.dump(repaired)
    ):
        raise ValueError("Recorded training dispatch does not match the supported fixture")
    calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"make_probe_optimizer", "make_repaired_probe_optimizer"}
    ]
    if len(calls) != 2:
        raise ValueError("Ambiguous optimizer factory calls in recorded training source")
    imports = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.ImportFrom)
        and node.module == "runsleuth.optimizer_probe"
        and node.level == 0
    ]
    if len(imports) != 1 or not all(
        any(alias.name == name and alias.asname is None for alias in imports[0].names)
        for name in ("make_probe_optimizer", "make_repaired_probe_optimizer")
    ):
        raise ValueError("Recorded optimizer factory import is not recognized")
    node = branches[0]
    return {
        "function": "_run_variant",
        "line": node.lineno,
        "end_line": node.end_lineno,
        "code": ast.get_source_segment(source, node),
        "interpretation": "Recorded variant metadata selects a factory branch; static evidence only.",
    }


def _source_bundle(root: Path, reference: Path, candidate: Path, runs: dict) -> dict:
    if (
        reference.parent != candidate.parent
        or reference.name != "clean"
        or candidate.name not in {"stale_head", "stale_head_repaired"}
    ):
        raise ValueError("Source localization requires clean and candidate from the same pair")
    pair = reference.parent
    parent_path = _scoped(root, pair / "optimizer_training_report.json", file=True)
    parent, parent_hash = _read(parent_path)
    if (
        parent.get("status") != "completed"
        or parent.get("error") is not None
        or parent.get("task") not in PAIR_TASKS
        or not isinstance(parent.get("variants"), dict)
    ):
        raise ValueError("A completed controlled optimizer training report is required")
    for name, directory in (("reference", reference), ("candidate", candidate)):
        child, _ = _read(_scoped(root, directory / "run_report.json", file=True))
        _same(child.get("variant"), directory.name, "Recorded variant")
        _same(parent["variants"].get(directory.name), child, "Parent/child training record")
        _same(
            parent.get("reference_initial_state_sha256"),
            runs[name]["initial_state_sha256"],
            "Pair initial state",
        )
        _same(parent.get("epochs_per_variant"), runs[name]["config"]["epochs"], "Pair epochs")
    for key in ("config", "data_identity", "initial_state_sha256"):
        _same(runs["reference"][key], runs["candidate"][key], f"Paired {key}")
    environment_path = _scoped(root, pair / "environment.json", file=True)
    environment, environment_hash = _read(environment_path)
    hashes = environment.get("source_sha256")
    if not isinstance(hashes, dict):
        raise ValueError("Recorded source snapshot hashes are missing")
    sources, files = {}, {}
    for name in SOURCE_FILES:
        expected = pair / "source_snapshot" / name
        path = _scoped(root, expected, file=True)
        if path != expected:
            raise ValueError("Source snapshots must be direct files in the recorded pair")
        with path.open("rb") as stream:
            payload = stream.read(MAX_SOURCE_BYTES + 1)
        if len(payload) > MAX_SOURCE_BYTES:
            raise ValueError("Source snapshot exceeds the size limit")
        _same(_sha(payload), hashes.get(name), "Recorded source SHA256")
        sources[name] = payload.decode("utf-8")
        files[name] = {"path": path.relative_to(root).as_posix(), "sha256": _sha(payload)}
    return {
        "provenance": {
            "kind": "recorded_snapshot_consistency",
            "authenticated_execution_provenance": False,
            "training_report": {
                "path": parent_path.relative_to(root).as_posix(),
                "sha256": parent_hash,
            },
            "environment": {
                "path": environment_path.relative_to(root).as_posix(),
                "sha256": environment_hash,
            },
            "files": files,
        },
        "dispatch": _dispatch_evidence(sources["camelyon_optimizer_training.py"]),
        "original_source": sources["optimizer_probe.py"],
    }


def prepare_source_proposals(manifest_path: Path, *, workspace_root: Path) -> dict:
    """Validate every input before any output; expected labels do not select patches."""
    root = workspace_root.resolve()
    path = _scoped(root, manifest_path, file=True)
    raw, manifest_hash = _read(path)
    manifest = Manifest.model_validate(raw)
    if len(manifest.cases) > 4:
        raise ValueError("This source proposal batch supports at most four cases")
    cases, seen_ids, seen_reports, seen_pairs = [], set(), set(), set()
    for case in manifest.cases:
        if len(case.agent_reports) != 1:
            raise ValueError("Use a fresh manifest with exactly one Agent report per case")
        reference = _scoped(root, case.reference_run)
        candidate = _scoped(root, case.candidate_run)
        agent_path = _scoped(root, case.agent_reports[0], file=True)
        if (
            case.id in seen_ids
            or agent_path in seen_reports
            or (reference, candidate) in seen_pairs
        ):
            raise ValueError("Duplicate case, report, or run pair")
        seen_ids.add(case.id)
        seen_reports.add(agent_path)
        seen_pairs.add((reference, candidate))
        runs, scoped_runs = {}, {}
        for role, directory, name in (
            ("reference", reference, case.reference_run),
            ("candidate", candidate, case.candidate_run),
        ):
            data = inspect_optimizer_run(
                directory, require_file=lambda value: _scoped(root, value, file=True)
            )
            data["run_directory"] = name
            _same(data["config"]["seed"], case.seed, "Case seed")
            runs[role] = data
            scoped_runs[name] = data
        attempt = _attempt(agent_path, root, case, scoped_runs)
        if attempt["status"] != "completed" or attempt["task_binding"] != "both_runs_refreshed":
            raise ValueError(f"{case.id}: Agent report did not pass replay: {attempt.get('error')}")
        source = _source_bundle(root, reference, candidate, runs)
        labels = attempt["labels"]
        patch = None
        if labels["implementation_status"] == "optimizer_binding_defect":
            if candidate.name != "stale_head":
                raise ValueError("Only the recorded stale_head branch can receive this proposal")
            _require_supported_pair(runs["reference"], runs["candidate"])
            patch = propose_optimizer_order_patch(source["original_source"])
            patch.update(
                owner="controller_ast",
                runtime_verified=False,
                target="isolated_source_copy",
                repair_accepted=None,
            )
            decision = "patch_proposed"
        elif (
            labels["implementation_status"] == "no_optimizer_binding_defect"
            and labels["recommended_action"] == "no_change"
        ):
            decision = "no_change"
        else:
            raise ValueError("The recorded Agent diagnosis does not support a source action")
        cases.append(
            {
                "id": case.id,
                "seed": case.seed,
                "decision": decision,
                "reference_run": case.reference_run,
                "candidate_run": case.candidate_run,
                "agent": {
                    "path": attempt["report"],
                    "sha256": attempt["sha256"],
                    "model": attempt["model"],
                    "labels": labels,
                    "first_pass_valid": attempt["first_pass_valid"],
                },
                "run_files_sha256": {
                    role: run["provenance"]["files_sha256"] for role, run in runs.items()
                },
                "source": source,
                "proposed_patch": patch,
            }
        )
    return {
        "schema_version": 1,
        "task": "optimizer_source_proposals",
        "status": "prepared",
        "manifest": {"path": path.relative_to(root).as_posix(), "sha256": manifest_hash},
        "new_execution": {"api_calls": 0, "training_runs": 0, "source_executions": 0},
        "scope": "Controlled stale-head fixture; controller AST proposals, not LLM-authored patches.",
        "limitations": [
            "Recorded hashes establish file consistency, not proof of executed instructions.",
            "Source syntax and edit shape are checked; runtime and performance remain unverified.",
            "Do not apply the proposal to the benchmark fixture: that would remove its injected fault.",
        ],
        "summary": {
            "total_cases": len(cases),
            "patch_proposals": sum(c["decision"] == "patch_proposed" for c in cases),
            "no_change": sum(c["decision"] == "no_change" for c in cases),
        },
        "cases": cases,
    }


def run_source_proposals(manifest_path: Path, *, workspace_root: Path) -> Path:
    """Export reviewable copies only; never overwrite project or recorded source."""
    root = workspace_root.resolve()
    report = prepare_source_proposals(manifest_path, workspace_root=root)
    base = _scoped(root, "artifacts/optimizer_source_proposals")
    directory = base / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    directory.mkdir(parents=True, exist_ok=False)
    for index, case in enumerate(report["cases"], 1):
        original = case["source"].pop("original_source")
        patch = case["proposed_patch"]
        if patch is None:
            continue
        case_dir = directory / "cases" / f"case-{index:03d}"
        proposed = patch.pop("proposed_source")
        for folder, text in (("original", original), ("proposed", proposed)):
            destination = case_dir / folder / "optimizer_probe.py"
            destination.parent.mkdir(parents=True)
            with destination.open("xb") as stream:
                stream.write(text.encode("utf-8"))
            patch[f"{folder}_file"] = destination.relative_to(root).as_posix()
            patch[f"{folder}_sha256"] = _sha(text.encode("utf-8"))
        diff_path = case_dir / "repair.patch"
        diff_path.write_bytes(patch["unified_diff"].encode("utf-8"))
        patch["patch_file"] = diff_path.relative_to(root).as_posix()
    report_path = directory / "report.json"
    with report_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    try:
        path = run_source_proposals(args.manifest, workspace_root=Path.cwd())
    except (OSError, ValueError, TypeError, KeyError, SyntaxError) as error:
        parser.error(str(error))
    report, _ = _read(path)
    print(f"source_proposal_report={path}")
    print(json.dumps(report["summary"]))
    for case in report["cases"]:
        patch = case["proposed_patch"]
        print(
            json.dumps(
                {
                    "case": case["id"],
                    "decision": case["decision"],
                    "patch_file": patch["patch_file"] if patch else None,
                }
            )
        )


if __name__ == "__main__":
    main()
