"""Export a compact development-evaluation record from existing local artifacts."""

import argparse
import json
from pathlib import Path


def read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = read_object(args.report)
        original_path = Path(report["original_batch_summary"])
        settings_path = args.report.parent / "settings.json"
        manifest_path = args.report.parent / "manifest.json"
        inputs = (args.report, original_path, settings_path, manifest_path)
        if args.output.resolve() in {path.resolve() for path in inputs}:
            raise ValueError("Output must not overwrite a source artifact")
        original = read_object(original_path)
        settings = read_object(settings_path)
        manifest = read_object(manifest_path)
        first = report["first_attempt_evaluation"]
        final = report["final_evaluation"]
        snapshot = {
            "schema_version": 1,
            "evaluation_type": "development_regression",
            "source_report": args.report.as_posix(),
            "model": report["model"],
            "original_batch_summary": report["original_batch_summary"],
            "training_settings": original["settings"],
            "agent_settings": settings,
            "manifest": manifest,
            "first_attempt_summary": first["summary"],
            "final_summary": final["summary"],
            "case_outcomes": [
                {"id": case["id"], "outcome": case["outcome"]} for case in final["cases"]
            ],
            "all_attempt_usage": report["all_attempt_usage"],
        }
        text = json.dumps(snapshot, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output.exists():
            previous = read_object(args.output)
            if previous.get("source_report") != snapshot["source_report"]:
                raise ValueError("Output already belongs to another report; choose a new filename")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(f"evaluation_snapshot={args.output}")
    print("first_attempt_summary=" + json.dumps(snapshot["first_attempt_summary"]))
    print("all_attempt_usage=" + json.dumps(snapshot["all_attempt_usage"]))


if __name__ == "__main__":
    main()
