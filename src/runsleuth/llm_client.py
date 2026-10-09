"""LLM clients through OpenAI-compatible Chat Completions APIs: Gemini or a local Ollama."""

import os

from openai import OpenAI

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
PROVIDERS = ("gemini", "ollama")


def create_llm_client(provider: str = "gemini", model: str | None = None) -> tuple[OpenAI, str]:
    """Create a client without sending a request; the model defaults to the provider's env var."""
    if provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        model = model or os.environ.get("GEMINI_MODEL", "").strip()
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY before running LLM diagnosis")
        if not model:
            raise RuntimeError("Set GEMINI_MODEL before running LLM diagnosis")
        return OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL, timeout=60.0, max_retries=0), model
    if provider == "ollama":
        model = model or os.environ.get("OLLAMA_MODEL", "").strip()
        if not model:
            raise RuntimeError("Set OLLAMA_MODEL (for example qwen3:8b) or pass a model")
        # Ollama ignores the key, but the client requires one. Local generation is
        # slower than a hosted API, hence the longer timeout.
        base_url = os.environ.get("OLLAMA_BASE_URL", "").strip() or OLLAMA_BASE_URL
        return OpenAI(api_key="ollama", base_url=base_url, timeout=600.0, max_retries=0), model
    raise ValueError(f"Unknown provider {provider!r}; expected one of {PROVIDERS}")
