"""Diagnose a monitored run: deterministic signature matching, then an LLM explanation.

The signature matcher decides first. The LLM receives only the extracted evidence
and the matcher's per-condition verdicts, and returns a structured diagnosis with
cited evidence. Every citation is checked against the evidence; a format or
citation error gets one retry listing the problems. When the LLM disagrees with
the matcher the conflict is flagged for review and the matcher's diagnosis stays
final: the LLM explains and arbitrates, but is never trusted over the evidence.
"""

import argparse
import hashlib
import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from runsleuth import signature_matching as matching

MAX_MODEL_CALLS = 2
# Version 2 clarifies skipped and holding reference conditions after the first model
# comparison, and constrains replies with a JSON Schema.
PROMPT_VERSION = 2
# Payload 2 writes exponents without leading zeros (5e-5, not Python's 5e-05).
# llama.cpp's JSON grammar, which Ollama uses for schema-constrained replies,
# forbids them, so a model copying 5e-05 exactly was cut to 5e-0.
PAYLOAD_VERSION = 2
# A JSON string is matched whole, so only number tokens lose their exponent zeros.
EXPONENT_ZEROS = re.compile(r'"(?:\\.|[^"\\])*"|(?<=[0-9])([eE][+-]?)0+(?=[0-9])')
RESPONSE_FORMATS = ("json_schema", "json_object")
# Thinking models (Gemini 3, Qwen3) spend output tokens on reasoning before the JSON.
MAX_OUTPUT_TOKENS = 4000
CITATION_REL_TOLERANCE = 0.01
# Some local servers return a thinking model's reasoning inline before the answer.
REASONING_BLOCK = re.compile(r"\A\s*<think>.*?</think>\s*", re.DOTALL)
SYSTEM_PROMPT = """You are RunSleuth's diagnosis reviewer for PyTorch training runs.
You receive evidence extracted from training telemetry, a library of known failure
signatures, and a deterministic matcher's verdict for every signature condition.
Use only the supplied evidence; never assume facts that are not in it.
Each signature lists "requires" and "contradicts" conditions. A signature is
supported when every requires condition is true and no contradicts condition is
true. A contradicts condition with result false means the contradicting fact is
absent; it does not count against the signature. A result of null means the value
was unavailable (status missing, needs_reference or skipped_without_reference).
A condition with status skipped_without_reference is optional: without a reference
run the signature is decided by its other conditions, so it never makes a
signature pending.
When a healthy reference run is supplied, it shows what the setup intends: a
reference condition that holds is evidence of a fault, so do not explain it away
as an intended design choice.
Choose one diagnosis: a signature id, "no_known_fault", "insufficient_evidence", or
"pending_reference". Answer "pending_reference" when a signature is supported except
for conditions with status needs_reference: only a healthy reference run can establish
that intent, so do not assume it.
Cite 1 to 8 evidence items by their exact names with the exact values supplied,
and say whether each supports or contradicts your diagnosis.
If you disagree with the matcher, explain why in disagreement_with_matcher;
otherwise set it to null. Reply with one JSON object that follows the schema."""


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: Annotated[str, Field(min_length=1, max_length=80)]
    value: float | int | bool | str | None
    role: Literal["supports", "contradicts"]


class LLMDiagnosis(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    diagnosis: Annotated[str, Field(min_length=1, max_length=80)]
    explanation: Annotated[str, Field(min_length=1, max_length=1500)]
    citations: Annotated[list[Citation], Field(min_length=1, max_length=8)]
    repair: Annotated[str, Field(min_length=1, max_length=500)]
    confidence: Literal["high", "medium", "low"]
    disagreement_with_matcher: Annotated[str, Field(max_length=800)] | None


def _same_value(cited, actual) -> bool:
    if isinstance(actual, bool) or isinstance(cited, bool) or actual is None or cited is None:
        return cited == actual
    if isinstance(actual, (int, float)) and isinstance(cited, (int, float)):
        return math.isclose(cited, actual, rel_tol=CITATION_REL_TOLERANCE, abs_tol=1e-12)
    return cited == actual


def validate_llm_diagnosis(raw: str, evidence: dict, allowed: set[str]) -> tuple[dict | None, list]:
    """Parse the reply and check the diagnosis id and every citation against the evidence."""
    try:
        parsed = LLMDiagnosis.model_validate_json(raw)
    except ValidationError as error:
        return None, [
            {"type": "schema", "detail": item["msg"], "location": item["loc"]}
            for item in error.errors()
        ]
    issues = []
    if parsed.diagnosis not in allowed:
        issues.append({"type": "unknown_diagnosis", "detail": parsed.diagnosis})
    for citation in parsed.citations:
        if citation.name not in evidence:
            issues.append({"type": "unknown_evidence", "detail": citation.name})
        elif not _same_value(citation.value, evidence[citation.name]):
            issues.append({"type": "wrong_value", "detail": citation.name})
    return (None, issues) if issues else (parsed.model_dump(), [])


def _compact_verdicts(result: dict) -> list[dict]:
    """Keep required and contradicting conditions apart: their results mean opposite things."""
    keys = ("evidence", "op", "expected", "observed", "result", "status")

    def compact(conditions):
        flat = []
        for condition in conditions:
            if "any_of" in condition:  # an alternative group holds if any member holds
                flat += [
                    {**{key: part.get(key) for key in keys}, "any_of_group": True}
                    for part in condition["any_of"]
                ]
            else:
                flat.append({key: condition.get(key) for key in keys})
        return flat

    return [
        {
            "id": item["id"],
            "verdict": item["verdict"],
            "requires": compact(item["requires"]),
            "contradicts": compact(item["contradicts"]),
        }
        for item in result["verdicts"]
    ]


def _dumps(value) -> str:
    """JSON with grammar-safe exponents; every number keeps its value."""
    return EXPONENT_ZEROS.sub(lambda match: match[1] or match[0], json.dumps(value))


def system_prompt() -> str:
    return SYSTEM_PROMPT + "\nJSON Schema:\n" + json.dumps(LLMDiagnosis.model_json_schema())


def response_schema(evidence: dict, allowed: set[str]) -> dict:
    """Constrain decoding to known diagnosis ids and evidence names; values are checked after.

    Lengths stay in LLMDiagnosis only, since providers support different schema subsets.
    """
    citation = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "enum": sorted(evidence)},
            "value": {
                "anyOf": [{"type": kind} for kind in ("number", "boolean", "string", "null")]
            },
            "role": {"type": "string", "enum": ["supports", "contradicts"]},
        },
        "required": ["name", "value", "role"],
        "additionalProperties": False,
    }
    properties = {
        "diagnosis": {"type": "string", "enum": sorted(allowed)},
        "explanation": {"type": "string"},
        "citations": {"type": "array", "items": citation, "minItems": 1, "maxItems": 8},
        "repair": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "disagreement_with_matcher": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def request_llm_diagnosis(
    client, model: str, matcher: dict, library: dict, *, response_format: str = "json_schema"
) -> dict:
    """At most MAX_MODEL_CALLS requests; returns a JSON-safe record of every attempt."""
    if response_format not in RESPONSE_FORMATS:
        raise ValueError(f"response_format must be one of {RESPONSE_FORMATS}")
    allowed = {signature["id"] for signature in library["signatures"]} | {
        "no_known_fault",
        "insufficient_evidence",
        "pending_reference",
    }
    signatures = [
        {key: signature[key] for key in ("id", "subsystem", "scope", "description", "repair")}
        for signature in library["signatures"]
    ]
    system = system_prompt()
    request_format = (
        {
            "type": "json_schema",
            "json_schema": {
                "name": "runsleuth_diagnosis",
                "strict": True,
                "schema": response_schema(matcher["evidence"], allowed),
            },
        }
        if response_format == "json_schema"
        else {"type": "json_object"}
    )
    payload = {
        "evidence": matcher["evidence"],
        "signatures": signatures,
        "matcher_diagnosis": matcher["diagnosis"],
        "matcher_verdicts": _compact_verdicts(matcher),
    }
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": _dumps(payload)},
    ]
    record = {
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "payload_version": PAYLOAD_VERSION,
        "system_prompt_sha256": hashlib.sha256(system.encode()).hexdigest(),
        "response_format": response_format,
        "status": "running",
        "model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "attempts": [],
        "diagnosis": None,
    }
    while record["model_calls"] < MAX_MODEL_CALLS:
        record["model_calls"] += 1
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format=request_format,
                max_tokens=MAX_OUTPUT_TOKENS,
            )
        except Exception as error:  # network, quota or provider errors end the review
            # The HTTP status separates a quota limit (429) from an overloaded server (503).
            status_code = getattr(error, "status_code", None)
            record.update(
                status="api_error",
                error={"type": type(error).__name__, "status_code": status_code},
            )
            return record
        usage = getattr(response, "usage", None)
        record["input_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
        record["output_tokens"] += getattr(usage, "completion_tokens", 0) or 0
        choice = response.choices[0]
        raw = choice.message.content or ""
        parsed, issues = validate_llm_diagnosis(
            REASONING_BLOCK.sub("", raw), matcher["evidence"], allowed
        )
        record["attempts"].append(
            {"finish_reason": choice.finish_reason, "raw_text": raw, "issues": issues}
        )
        if parsed is not None:
            record.update(status="completed", diagnosis=parsed)
            return record
        messages += [
            {"role": "assistant", "content": raw},
            {
                "role": "user",
                "content": "Your reply was rejected for these problems: "
                + json.dumps(issues)
                + ". Reply again with one corrected JSON object.",
            },
        ]
    record["status"] = "invalid_output"
    return record


def diagnose_run(
    run: dict,
    reference: dict | None = None,
    *,
    client=None,
    model: str | None = None,
    optimizer_name: str | None = None,
    response_format: str = "json_schema",
) -> dict:
    """Deterministic diagnosis plus, when a client is given, a checked LLM review."""
    library = matching.load_library()
    evidence = matching.extract_evidence(run, reference, optimizer_name)
    matcher = matching.diagnose(evidence, library)
    result = {
        "matcher": matcher,
        "llm": None,
        "final_diagnosis": matcher["diagnosis"],
        "decided_by": "signature_matcher",
        "conflict": False,
    }
    if client is None:
        return result
    review = request_llm_diagnosis(client, model, matcher, library, response_format=response_format)
    result["llm"] = review
    if review["status"] != "completed":
        result["llm_note"] = f"LLM review {review['status']}; the matcher's diagnosis stands"
        return result
    llm_diagnosis = review["diagnosis"]["diagnosis"]
    agrees = llm_diagnosis == matcher["diagnosis"]
    result["conflict"] = not agrees
    result["decided_by"] = "signature_matcher_and_llm" if agrees else "signature_matcher"
    if not agrees:
        assumed = (
            matcher["diagnosis"] == "pending_reference"
            and llm_diagnosis in matcher["pending_reference"]
        )
        result["conflict_kind"] = (
            "assumed_reference_condition" if assumed else "different_diagnosis"
        )
        result["llm_note"] = (
            f"LLM chose {llm_diagnosis} without the reference it needs; flagged for human review"
            if assumed
            else "LLM disagrees with the matcher; flagged for human review"
        )
    return result


def summary_text(result: dict) -> str:
    """A short human-readable report for the terminal."""
    matcher = result["matcher"]
    verdicts = {item["id"]: item for item in matcher["verdicts"]}
    lines = [f"Diagnosis: {result['final_diagnosis']}  (decided by {result['decided_by']})"]
    chosen = verdicts.get(result["final_diagnosis"])
    if chosen:
        lines.append("Evidence:")
        for condition in chosen["requires"]:
            parts = condition.get("any_of", [condition])
            for part in parts:
                mark = {True: "+", False: "x", None: "?"}[part["result"]]
                alternative = " (any one suffices)" if len(parts) > 1 else ""
                lines.append(
                    f"  {mark} {part['evidence']} = {part['observed']} "
                    f"({part['op']} {part['expected']}){alternative}"
                )
        lines.append(f"Repair: {chosen['repair']['description']}")
    others = [
        f"{item['id']}: {item['verdict']}"
        for item in matcher["verdicts"]
        if item["id"] != result["final_diagnosis"]
    ]
    lines.append("Other signatures: " + "; ".join(others))
    review = result["llm"]
    if review and review["status"] == "completed":
        lines.append(
            f"LLM ({review['diagnosis']['confidence']}): {review['diagnosis']['explanation']}"
        )
    if result.get("llm_note"):
        lines.append(f"Note: {result['llm_note']}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True, help="run_report.json to diagnose")
    parser.add_argument("--reference", type=Path, help="run_report.json of a healthy run")
    parser.add_argument("--no-llm", action="store_true", help="Deterministic matching only")
    parser.add_argument(
        "--provider",
        choices=("gemini", "ollama"),
        default="gemini",
        help="LLM for the review: Gemini, or a local Ollama model (OLLAMA_MODEL)",
    )
    parser.add_argument(
        "--optimizer",
        help="Optimizer class name (default: the report, then the experiment's environment.json)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/diagnoses"))
    args = parser.parse_args()
    run = json.loads(args.run.read_text(encoding="utf-8"))
    reference = json.loads(args.reference.read_text(encoding="utf-8")) if args.reference else None
    optimizer_name = args.optimizer or run.get("optimizer")
    environment = args.run.resolve().parent.parent / "environment.json"
    if optimizer_name is None and environment.is_file():
        optimizer_name = json.loads(environment.read_text(encoding="utf-8")).get("optimizer")
    client = model = None
    if not args.no_llm:
        from runsleuth.llm_client import create_llm_client

        try:
            client, model = create_llm_client(args.provider)
        except RuntimeError as error:
            raise SystemExit(f"{error}; or rerun with --no-llm") from error
    result = diagnose_run(run, reference, client=client, model=model, optimizer_name=optimizer_name)
    result["inputs"] = {
        "run": str(args.run),
        "reference": str(args.reference) if args.reference else None,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    path = args.output_dir / f"diagnosis-{timestamp}.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(summary_text(result))
    print(f"diagnosis_report={path}")


if __name__ == "__main__":
    main()
