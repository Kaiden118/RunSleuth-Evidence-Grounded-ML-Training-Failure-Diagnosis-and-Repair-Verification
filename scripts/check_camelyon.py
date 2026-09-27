"""Run Camelyon checks and optionally train only after every check succeeds."""

import argparse
import subprocess
import sys
from pathlib import Path

CAMELYON_FILES = (
    "src/runsleuth/camelyon.py",
    "src/runsleuth/camelyon_config.py",
    "src/runsleuth/camelyon_data.py",
    "src/runsleuth/camelyon_mirror.py",
    "tests/test_camelyon_config.py",
    "tests/test_camelyon_data.py",
    "tests/test_camelyon_mirror.py",
    "tests/test_camelyon_experiment.py",
    "tests/test_camelyon_checks.py",
    "scripts/check_camelyon.py",
)


def build_commands(*, train: bool, download: bool, config: str) -> list[list[str]]:
    commands = [
        [sys.executable, "-m", "ruff", "check", *CAMELYON_FILES, "--fix"],
        [sys.executable, "-m", "ruff", "format", *CAMELYON_FILES],
        [sys.executable, "-m", "pytest"],
        [sys.executable, "-m", "ruff", "check", "."],
        ["git", "diff", "--check"],
    ]
    if train:
        command = [sys.executable, "-m", "runsleuth.camelyon", "--config", config]
        if download:
            command.append("--download")
        commands.append(command)
    return commands


def run_commands(commands: list[list[str]], *, cwd: Path) -> None:
    for command in commands:
        print("Running: " + subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--config", default="configs/camelyon17_clean.json")
    args = parser.parse_args()
    if args.download and not args.train:
        parser.error("--download requires --train")
    commands = build_commands(train=args.train, download=args.download, config=args.config)
    try:
        run_commands(commands, cwd=Path(__file__).resolve().parents[1])
    except subprocess.CalledProcessError as error:
        print(
            f"Stopped: exit status {error.returncode}; remaining commands were not run.",
            file=sys.stderr,
        )
        return error.returncode if error.returncode > 0 else 1
    except OSError as error:
        print(f"Stopped: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
