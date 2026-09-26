"""Command-line entry point for bounded LLM diagnosis."""

import argparse
from datetime import UTC, datetime
from pathlib import Path

from runsleuth.agent import run_diagnostic_agent
from runsleuth.diagnostic_tools import DiagnosticTools
from runsleuth.llm_client import create_llm_client


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose training failures using bounded LLM tool calls.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("src/runsleuth/train.py"),
    )
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--max-model-calls", type=positive_int, default=6)
    parser.add_argument("--max-tool-calls", type=positive_int, default=5)
    parser.add_argument("--max-output-tokens", type=positive_int, default=2000)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        client, model = create_llm_client()
    except RuntimeError as error:
        parser.error(str(error))

    with client:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        run_directory = Path("artifacts/agent_runs") / timestamp
        run_directory.mkdir(parents=True)

        report = run_diagnostic_agent(
            client,
            model,
            DiagnosticTools(Path.cwd()),
            source_path=args.source.as_posix(),
            reference_run=args.reference_run.as_posix(),
            candidate_run=args.candidate_run.as_posix(),
            max_model_calls=args.max_model_calls,
            max_tool_calls=args.max_tool_calls,
            max_output_tokens=args.max_output_tokens,
        )

        report_path = run_directory / "agent_report.json"
        report_path.write_text(report.to_json() + "\n", encoding="utf-8")

    print(f"agent_report={report_path}")
    print(f"status={report.status}")
    print(f"model_calls={report.model_calls}, tool_calls={report.tool_calls}")
    print(
        f"recorded_input_tokens={report.input_tokens}, "
        f"recorded_output_tokens={report.output_tokens}, "
        f"usage_complete={report.usage_complete}"
    )

    if report.final_text:
        print(report.final_text)
    if report.error:
        print(f"error={report.error}")
    if report.status != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
