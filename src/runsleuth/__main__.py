"""Command-line entry point for RunSleuth experiments."""

import argparse

from runsleuth.config import TrainingConfig
from runsleuth.experiment import run_experiment


def build_parser() -> argparse.ArgumentParser:
    defaults = TrainingConfig()

    parser = argparse.ArgumentParser(
        description="Run a bounded FashionMNIST training experiment.",
    )
    parser.add_argument("--run-name", default=defaults.run_name)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=defaults.learning_rate,
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=defaults.device,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = TrainingConfig(
        run_name=args.run_name,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
    )
    run_experiment(config)


if __name__ == "__main__":
    main()
