"""Anthropic (Claude) provider."""

from __future__ import annotations

from ..errors import LLMError
from .base import Provider, redact


class AnthropicProvider(Provider):
    """Talks to the Anthropic Messages API.

    Requires the ``anthropic`` package, which is an optional extra:
    ``pip install "plotaviz[llm]"``.
    """

    name = "anthropic"
    default_model = "claude-sonnet-4-5"
    available_models = (
        "claude-opus-4-5",
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
    )

    def complete(self, system: str, user: str, *, timeout: float) -> str:
        """Send one message and return the text of the reply."""
        try:
            import anthropic
        except ImportError as exc:
            raise LLMError(
                "The Anthropic provider needs the anthropic package.",
                hint='Install it with: pip install "plotaviz[llm]"',
            ) from exc

        client = anthropic.Anthropic(api_key=self.require_key(), timeout=timeout)
        try:
            message = client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except anthropic.AuthenticationError as exc:
            raise LLMError(
                "Anthropic rejected the API key.",
                hint="Check the key in Settings → LLM provider.",
            ) from exc
        except anthropic.RateLimitError as exc:
            raise LLMError(
                "Anthropic is rate limiting this key.", hint="Wait a moment and try again."
            ) from exc
        except anthropic.APITimeoutError as exc:
            raise LLMError(
                f"Anthropic did not respond within {timeout:.0f} seconds.",
                hint="PlotaViz will use its own rules engine instead.",
            ) from exc
        except anthropic.APIError as exc:
            raise LLMError("The Anthropic request failed.", hint=redact(str(exc))) from exc

        return "".join(
            block.text for block in message.content if getattr(block, "type", "") == "text"
        )
