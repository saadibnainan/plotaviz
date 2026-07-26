"""OpenAI provider."""

from __future__ import annotations

from ..errors import LLMError
from .base import Provider, redact


class OpenAIProvider(Provider):
    """Talks to the OpenAI Chat Completions API.

    Requires the ``openai`` package: ``pip install "plotaviz[llm]"``.
    """

    name = "openai"
    default_model = "gpt-4o-mini"
    available_models = ("gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "o4-mini")

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
                max_tokens=1024,
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
