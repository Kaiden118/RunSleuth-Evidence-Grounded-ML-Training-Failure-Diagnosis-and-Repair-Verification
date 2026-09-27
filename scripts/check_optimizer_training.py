"""Check the paired-training feature, then run both fixed-budget variants."""

import argparse
import subprocess
import sys
from pathlib import Path

FILES = (
    "src/runsleuth/parameter_group_monitor.py",
    "src/runsleuth/camelyon_optimizer_training.py",
    "tests/test_parameter_group_monitor.py",
    "tests/test_camelyon_optimizer_training.py",
    "scripts/check_optimizer_training.py",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", required=True)
    args = parser.parse_args()
    commands = [
        [sys.executable, "-m", "ruff", "check", *FILES, "--fix"],
        [sys.executable, "-m", "ruff", "format", *FILES],
        [sys.executable, "-m", "pytest"],
        [sys.executable, "-m", "ruff", "check", "."],
        ["git", "diff", "--check"],
        [
            sys.executable,
            "-m",
            "runsleuth.camelyon_optimizer_training",
            "--reference-run",
            args.reference_run,
            "--epochs",
            "3",
        ],
    ]
    try:
        for command in commands:
            print("Running: " + subprocess.list2cmdline(command), flush=True)
            subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=True)
    except subprocess.CalledProcessError as error:
        print(f"Stopped: exit status {error.returncode}.", file=sys.stderr)
        return error.returncode if error.returncode > 0 else 1
    except OSError as error:
        print(f"Stopped: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
