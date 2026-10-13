"""Reviewed correctness labels for the scripts agents leave behind.

An episode decides from the source alone whether its final script is correct when
that script is the healthy one or was left untouched (agent_episodes.semantic_label).
Any other script has to be read by a reviewer. Those decisions are kept in a file,
apart from the episodes and from the verifier, and are keyed by the program: one
review labels every episode, model and repetition that ends in the same program.

A reviewer is shown only how a script differs from the healthy one, never the task
it started from, the agent's claim or a verifier's decision. Each review says whether
it was made that way (blind_to_verifier).
"""

import ast
import difflib
import hashlib
import json
from pathlib import Path
from typing import Any

from runsleuth import agent_tasks
from runsleuth.agent_tasks import HEALTHY

REVIEWS_PATH = Path("evaluations/agent_script_reviews.json")
REVIEWED = "reviewed"


def program_sha256(source: str) -> str | None:
    """A hash of the program, ignoring comments and layout; None if it does not parse."""
    try:
        tree = ast.dump(ast.parse(source))
    except SyntaxError:
        return None
    return hashlib.sha256(tree.encode("utf-8")).hexdigest()


def difference(source: str) -> list[str]:
    """The lines in which a script differs from the healthy one, as a unified diff."""
    healthy = agent_tasks.task_source(HEALTHY).splitlines()
    return list(difflib.unified_diff(healthy, source.splitlines(), lineterm="", n=1))[2:]


def load_reviews(path: Path) -> dict[str, dict[str, Any]]:
    """The reviews of a file by program hash; a malformed review is an error."""
    reviews: dict[str, dict[str, Any]] = {}
    for review in json.loads(Path(path).read_text(encoding="utf-8"))["reviews"]:
        program = review["program_sha256"]
        if program in reviews:
            raise ValueError(f"{path}: program {program[:12]} is reviewed twice")
        for key in ("correct", "blind_to_verifier"):
            if not isinstance(review[key], bool):
                raise ValueError(f"{path}: {key} of program {program[:12]} must be true or false")
        for key in ("reason", "reviewer"):
            if not str(review[key]).strip():
                raise ValueError(f"{path}: program {program[:12]} has no {key}")
        reviews[program] = review
    return reviews


def label(episode: dict[str, Any], reviews: dict[str, dict[str, Any]]) -> tuple[bool | None, str]:
    """An episode's correctness label; a review decides only what the source left open."""
    if episode["semantically_correct"] is not None:
        return episode["semantically_correct"], episode["semantic_basis"]
    review = reviews.get(program_sha256(episode["final_script"]) or "")
    if review is None:
        return None, episode["semantic_basis"]
    return review["correct"], REVIEWED


def pending(
    episodes: list[dict[str, Any]], reviews: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """The distinct programs that still need a review, with nothing but their difference."""
    programs: dict[str, dict[str, Any]] = {}
    for episode in episodes:
        if label(episode, reviews)[0] is not None:
            continue
        source = episode["final_script"]
        program = program_sha256(source)
        entry = programs.setdefault(
            program, {"program_sha256": program, "episodes": 0, "diff": difference(source)}
        )
        entry["episodes"] += 1
    return list(programs.values())


def pending_text(programs: list[dict[str, Any]]) -> str:
    if not programs:
        return "No script needs a review."
    blocks = []
    for entry in programs:
        blocks.append(
            f"program_sha256 {entry['program_sha256']}  (ends {entry['episodes']} episodes)\n"
            + "\n".join(f"    {line}" for line in entry["diff"])
        )
    return "\n\n".join(blocks)
