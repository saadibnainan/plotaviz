"""Ollama provider — local models, no network egress.

This is the privacy-preserving path. Ollama runs on the user's own machine, so nothing about
their dataset leaves it, and no API key exists to leak. For sensitive data this is the provider
to recommend.
"""

from __future__ import annotations

import json

from ..errors import LLMError
from .base import DEFAULT_TIMEOUT, Provider

#: Where Ollama listens by default.
DEFAULT_HOST = "http://localhost:11434"


class OllamaProvider(Provider):
    """Talks to a local Ollama server.

    Args:
        model: Model tag, e.g. ``llama3.1`` or ``qwen2.5``.
        host: Base URL of the Ollama server.
        timeout: Per-request timeout. Local models on CPU are slow, so this defaults higher than
            the hosted providers.
    """

    name = "ollama"
    default_model = "llama3.1"
    available_models = ("llama3.1", "llama3.2", "qwen2.5", "mistral", "gemma2", "phi4")
    sends_data_off_machine = False

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT * 4,
        host: str = DEFAULT_HOST,
    ) -> None:
        super().__init__(model=model, api_key=api_key, timeout=timeout)
        self.host = host.rstrip("/")

    def is_configured(self) -> bool:
        """Ollama needs no key — it is configured if the server answers."""
        return self.is_running()

    def is_running(self) -> bool:
        """Whether an Ollama server is reachable at :attr:`host`."""
        try:
            import httpx

            response = httpx.get(f"{self.host}/api/tags", timeout=2.0)
            return response.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """Model tags the local server has pulled. Empty when it is not running."""
        try:
            import httpx

            response = httpx.get(f"{self.host}/api/tags", timeout=3.0)
            response.raise_for_status()
            return [m["name"] for m in response.json().get("models", [])]
        except Exception:
            return []

    def complete(self, system: str, user: str, *, timeout: float) -> str:
        """Send one message to the local model and return the text of the reply."""
        try:
            import httpx
        except ImportError as exc:
            raise LLMError(
                "The Ollama provider needs the httpx package.",
                hint='Install it with: pip install "plotaviz[llm]"',
            ) from exc

        payload = {
            "model": self.model,
            "system": system,
            "prompt": user,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.2},
        }

        try:
            response = httpx.post(f"{self.host}/api/generate", json=payload, timeout=timeout)
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise LLMError(
                f"No Ollama server is running at {self.host}.",
                hint="Start it with `ollama serve`, then pull a model: `ollama pull llama3.1`.",
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMError(
                f"The local model did not answer within {timeout:.0f} seconds.",
                hint="Local models can be slow on CPU. Try a smaller model, or raise the timeout.",
            ) from exc
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:200]
            if exc.response.status_code == 404:
                raise LLMError(
                    f"Ollama does not have the model {self.model!r}.",
                    hint=f"Pull it first: ollama pull {self.model}",
                ) from exc
            raise LLMError("The Ollama request failed.", hint=detail) from exc

        try:
            return str(response.json().get("response", ""))
        except json.JSONDecodeError as exc:
            raise LLMError("Ollama returned a malformed response.", hint=str(exc)) from exc
