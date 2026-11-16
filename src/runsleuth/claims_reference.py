"""The reference program (reference/train.py) as an attempt of every world, so that the
official run and the probes treat it as they treat an agent's program.

The reference is the control of the study: it keeps lesions or patients whole and
estimates each use on held-out data that match it, so its claims show how far careful
internal evidence reaches on each world's sealed regime.

    python -m runsleuth.claims_reference write --out artifacts/claims/dev-v2 \\
        --world ham10000-vienna_dias camelyon-1
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from runsleuth import claims_agent, claims_task

REFERENCE = Path(__file__).resolve().parent / "reference" / claims_task.PROGRAM
MODEL_NAME = "reference"
CONDITION = "neutral"


def write(output: Path, world: str) -> Path:
    """The reference attempt of a world under output, written once."""
    chosen = claims_task.spec(world)
    directory = claims_agent.attempt_directory(output, MODEL_NAME, chosen.name, CONDITION, 1)
    path = directory / claims_agent.ATTEMPT_FILE
    if path.is_file():
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    source = REFERENCE.read_text(encoding="utf-8")
    (directory / claims_task.PROGRAM).write_text(source, encoding="utf-8", newline="\n")
    record = {
        "schema_version": 1,
        "task": "claims_program",
        "id": claims_agent.attempt_id(MODEL_NAME, chosen.name, CONDITION, 1),
        "world": chosen.name,
        "dataset": chosen.dataset,
        "condition": CONDITION,
        "model_name": MODEL_NAME,
        "attempt": 1,
        "provider": None,
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": "written",
        "claim": None,
        "program_sha256": claims_agent._sha256(source),
        "program_lines": len(source.splitlines()),
        "reference": True,
    }
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return directory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The reference program as an attempt.")
    commands = parser.add_subparsers(dest="command", required=True)
    written = commands.add_parser("write", help="Write the reference attempt of worlds.")
    written.add_argument("--out", type=Path, required=True, help="The attempts' directory.")
    written.add_argument("--world", nargs="+", required=True, choices=sorted(claims_task.SPECS))
    args = parser.parse_args(argv)
    for world in args.world:
        print(write(args.out, world))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
