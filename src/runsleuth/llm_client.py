"""Gemini setup through the OpenAI-compatible Chat Completions API."""

import os

from openai import OpenAI

from runsleuth.diagnostic_tools import DiagnosticTools


def create_llm_client() -> tuple[OpenAI, str]:
    """Create a Gemini client without sending a request."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    model = os.environ.get("GEMINI_MODEL", "").strip()
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY before running LLM diagnosis")
    if not model:
        raise RuntimeError("Set GEMINI_MODEL before running LLM diagnosis")

    return OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        timeout=60.0,
        max_retries=0,
    ), model


def build_openai_tools(diagnostic_tools: DiagnosticTools) -> list[dict[str, object]]:
    """Describe our flat path arguments; local validation remains authoritative."""
    tools = []
    for definition in diagnostic_tools.definitions():
        schema = definition["parameters"]
        parameters = {
            "type": "object",
            "properties": {
                name: {key: value for key, value in field.items() if key in ("type", "description")}
                for name, field in schema["properties"].items()
            },
            "required": schema["required"],
        }
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": definition["name"],
                    "description": definition["description"],
                    "parameters": parameters,
                },
            }
        )
    return tools
