"""LLM clients through Chat Completions APIs.

Gemini, a local Ollama and OpenAI have their own settings. Any other service that
speaks the same API (provider "compatible") is named by its endpoint. Claude goes
through Anthropic's own Messages API (provider "anthropic", anthropic_chat), which
keeps its prompt caching and thinking; CLAUDE_BASE_URL and CLAUDE_API_KEY name the
endpoint and its key, apart from the ANTHROPIC_* variables Claude Code itself reads.
"""

import os

from openai import OpenAI

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
PROVIDERS = ("gemini", "ollama", "openai", "compatible", "anthropic")
# Seconds a hosted model may take over one reply. Thinking models took more than two
# minutes over a first reply in the smoke tests, so every provider gets ten.
REQUEST_TIMEOUT = 600.0


def _setting(name: str, purpose: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Set {name} {purpose}")
    return value


def create_llm_client(provider: str = "gemini", model: str | None = None) -> tuple[OpenAI, str]:
    """Create a client without sending a request; the model defaults to the provider's env var."""
    if provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        model = model or os.environ.get("GEMINI_MODEL", "").strip()
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY before running LLM diagnosis")
        if not model:
            raise RuntimeError("Set GEMINI_MODEL before running LLM diagnosis")
        return OpenAI(
            api_key=api_key, base_url=GEMINI_BASE_URL, timeout=REQUEST_TIMEOUT, max_retries=0
        ), model
    if provider == "ollama":
        model = model or os.environ.get("OLLAMA_MODEL", "").strip()
        if not model:
            raise RuntimeError("Set OLLAMA_MODEL (for example qwen3:8b) or pass a model")
        # Ollama ignores the key, but the client requires one. Local generation is
        # slower than a hosted API, hence the longer timeout.
        base_url = os.environ.get("OLLAMA_BASE_URL", "").strip() or OLLAMA_BASE_URL
        return OpenAI(api_key="ollama", base_url=base_url, timeout=600.0, max_retries=0), model
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        model = model or os.environ.get("OPENAI_MODEL", "").strip()
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY before using the OpenAI provider")
        if not model:
            raise RuntimeError("Set OPENAI_MODEL or pass a model")
        return OpenAI(api_key=api_key, timeout=REQUEST_TIMEOUT, max_retries=0), model
    if provider == "compatible":
        base_url = _setting("LLM_BASE_URL", "to the provider's OpenAI-compatible endpoint")
        api_key = _setting("LLM_API_KEY", "to the key for LLM_BASE_URL")
        model = model or _setting("LLM_MODEL", "or pass a model")
        return OpenAI(
            api_key=api_key, base_url=base_url, timeout=REQUEST_TIMEOUT, max_retries=0
        ), model
    if provider == "anthropic":
        import anthropic

        from runsleuth.anthropic_chat import AnthropicChat

        base_url = _setting("CLAUDE_BASE_URL", "to the Messages API endpoint, without /v1")
        api_key = _setting("CLAUDE_API_KEY", "to the key for CLAUDE_BASE_URL")
        model = model or _setting("LLM_MODEL", "or pass a model")
        # An explicit timeout also lets a large max_tokens go without streaming.
        client = anthropic.Anthropic(
            api_key=api_key, base_url=base_url, timeout=REQUEST_TIMEOUT, max_retries=0
        )
        return AnthropicChat(client), model
    raise ValueError(f"Unknown provider {provider!r}; expected one of {PROVIDERS}")
