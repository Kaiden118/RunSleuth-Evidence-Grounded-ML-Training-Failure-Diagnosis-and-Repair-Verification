"""Offline tests for LLM client setup; creating a client sends no request."""

import os
import unittest
from unittest.mock import patch

try:
    from runsleuth import llm_client
except ModuleNotFoundError as error:
    if error.name != "openai":
        raise
    llm_client = None


@unittest.skipIf(llm_client is None, "Install the llm extra for client tests")
class LLMClientTests(unittest.TestCase):
    def create(self, environment, *args):
        with patch.dict(os.environ, environment, clear=True):
            return llm_client.create_llm_client(*args)

    def test_gemini_needs_a_key_and_a_model(self):
        for environment, message in (
            ({"GEMINI_MODEL": "gemini-x"}, "GEMINI_API_KEY"),
            ({"GEMINI_API_KEY": "test-key"}, "GEMINI_MODEL"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, message):
                self.create(environment)
        client, model = self.create({"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "gemini-x"})
        self.assertEqual(model, "gemini-x")
        self.assertEqual(str(client.base_url), llm_client.GEMINI_BASE_URL)

    def test_ollama_is_local_and_needs_only_a_model(self):
        with self.assertRaisesRegex(RuntimeError, "OLLAMA_MODEL"):
            self.create({}, "ollama")
        client, model = self.create({"OLLAMA_MODEL": "qwen3:8b"}, "ollama")
        self.assertEqual(model, "qwen3:8b")
        self.assertEqual(str(client.base_url).rstrip("/"), llm_client.OLLAMA_BASE_URL)
        self.assertGreater(client.timeout, 60)
        _, model = self.create({}, "ollama", "llama3.1:8b")
        self.assertEqual(model, "llama3.1:8b")
        client, _ = self.create({"OLLAMA_BASE_URL": "http://gpu-box:11434/v1"}, "ollama", "m")
        self.assertEqual(str(client.base_url).rstrip("/"), "http://gpu-box:11434/v1")

    def test_unknown_provider_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown provider"):
            self.create({}, "other")


if __name__ == "__main__":
    unittest.main()
