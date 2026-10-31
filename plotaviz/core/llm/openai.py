"""OpenAI provider."""

from __future__ import annotations

from ..errors import LLMError
from .base import Provider, redact


class OpenAIProvider(Provider):
    """Talks to the OpenAI Chat Completions API.

    Requires the ``openai`` package: ``pip install "plotaviz[llm]"``.
    """

    name = "openai"
    # Suggestions only — the model box in Settings is editable. Checked against OpenAI's
    # supported-model list in October 2026.
    default_model = "gpt-5.4-mini"
    available_models = ("gpt-5.4-mini", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-6.1-sol")

    def list_models(self) -> list[str]:
        """Model IDs from ``GET /v1/models``; sends no dataset information."""
        try:
            import openai
        except ImportError as exc:
            raise LLMError(
                "The OpenAI provider needs the openai package.",
                hint='Install it with: pip install "plotaviz[llm]"',
            ) from exc
        client = openai.OpenAI(api_key=self.require_key(), timeout=self.timeout)
        try:
            return sorted(model.id for model in client.models.list())
        except openai.AuthenticationError as exc:
            raise LLMError("OpenAI rejected the API key.") from exc
        except openai.APIError as exc:
            raise LLMError("Could not reach OpenAI.", hint=redact(str(exc))) from exc

    def complete(self, system: str, user: str, *, timeout: float) -> str:
        """Send one message and return the text of the reply."""
        try:
            import openai
        except ImportError as exc:
            raise LLMError(
                "The OpenAI provider needs the openai package.",
                hint='Install it with: pip install "plotaviz[llm]"',
            ) from exc

        client = openai.OpenAI(api_key=self.require_key(), timeout=timeout)
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                # Ask for JSON explicitly; the base class still tolerates a chatty reply.
                response_format={"type": "json_object"},
                # GPT-5-era models reject `max_tokens` on Chat Completions; this works for all.
                max_completion_tokens=1024,
            )
        except openai.AuthenticationError as exc:
            raise LLMError(
                "OpenAI rejected the API key.", hint="Check the key in Settings → LLM provider."
            ) from exc
        except openai.RateLimitError as exc:
            raise LLMError(
                "OpenAI is rate limiting this key.", hint="Wait a moment and try again."
            ) from exc
        except openai.APITimeoutError as exc:
            raise LLMError(
                f"OpenAI did not respond within {timeout:.0f} seconds.",
                hint="PlotaViz will use its own rules engine instead.",
            ) from exc
        except openai.APIError as exc:
            raise LLMError("The OpenAI request failed.", hint=redact(str(exc))) from exc

        choice = response.choices[0] if response.choices else None
        return (choice.message.content or "") if choice else ""
