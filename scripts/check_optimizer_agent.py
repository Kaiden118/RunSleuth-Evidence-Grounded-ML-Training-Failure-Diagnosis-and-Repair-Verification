"""Check optimizer evidence locally, then optionally diagnose defect and control."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

FILES = (
    "src/runsleuth/agent.py",
    "src/runsleuth/agent_diagnose.py",
    "src/runsleuth/diagnostic_tools.py",
    "src/runsleuth/optimizer_evidence.py",
    "src/runsleuth/optimizer_agent_diagnosis.py",
    "src/runsleuth/optimizer_agent_profile.py",
    "tests/test_optimizer_evidence.py",
    "tests/test_optimizer_agent_diagnosis.py",
    "tests/test_optimizer_agent.py",
    "tests/test_optimizer_tool_dispatch.py",
    "scripts/check_optimizer_agent.py",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-run", type=Path, required=True)
    parser.add_argument("--diagnose", action="store_true", help="Allow two real LLM diagnoses.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    reference = (args.pair_run / "clean").as_posix()
    candidate = (args.pair_run / "stale_head").as_posix()
    commands = [
        [sys.executable, "-m", "ruff", "check", *FILES, "--fix"],
        [sys.executable, "-m", "ruff", "format", *FILES],
        [sys.executable, "-m", "pytest"],
        [sys.executable, "-m", "ruff", "check", "."],
        ["git", "diff", "--check"],
    ]
    try:
        for command in commands:
            print("Running: " + subprocess.list2cmdline(command), flush=True)
            subprocess.run(command, cwd=root, check=True)

        from runsleuth.diagnostic_tools import DiagnosticTools

        tools = DiagnosticTools(root, enable_optimizer_audit=True)
        for run in (reference, candidate):
            result = tools.execute("inspect_optimizer_run", {"run_directory": run})
            if not result.ok:
                print(result.to_json(), flush=True)
                return 1
            data = result.data
            print(
                json.dumps(
                    {
                        "preflight_run": run,
                        "optimizer_audit": data["optimizer_audit"],
                        "final_epoch": data["epochs"][-1],
                    },
                    allow_nan=False,
                ),
                flush=True,
            )

        if args.diagnose:
            for label, run in (("binding_defect", candidate), ("clean_control", reference)):
                print(f"phase=agent_diagnosis case={label}", flush=True)
                command = [
                    sys.executable,
                    "-m",
                    "runsleuth.agent_diagnose",
                    "--profile",
                    "optimizer_binding",
                    "--reference-run",
                    reference,
                    "--candidate-run",
                    run,
                    "--max-model-calls",
                    "6",
                    "--max-tool-calls",
                    "5",
                    "--max-output-tokens",
                    "3000",
                ]
                print("Running: " + subprocess.list2cmdline(command), flush=True)
                subprocess.run(command, cwd=root, check=True)
    except subprocess.CalledProcessError as error:
        print(f"Stopped: exit status {error.returncode}.", file=sys.stderr)
        return error.returncode if error.returncode > 0 else 1
    except (OSError, ValueError) as error:
        print(f"Stopped: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
