"""Google Gemini provider."""

from __future__ import annotations

from ..errors import LLMError
from .base import Provider, redact


class GeminiProvider(Provider):
    """Talks to the Google Gemini API.

    Requires the ``google-genai`` package: ``pip install "plotaviz[llm]"``.
    """

    name = "gemini"
    default_model = "gemini-2.0-flash"
    available_models = ("gemini-2.5-pro", "gemini-2.0-flash", "gemini-2.0-flash-lite")

    def complete(self, system: str, user: str, *, timeout: float) -> str:
        """Send one message and return the text of the reply."""
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise LLMError(
                "The Gemini provider needs the google-genai package.",
                hint='Install it with: pip install "plotaviz[llm]"',
            ) from exc

        client = genai.Client(api_key=self.require_key())
        try:
            response = client.models.generate_content(
                model=self.model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type="application/json",
                    max_output_tokens=1024,
                    http_options=types.HttpOptions(timeout=int(timeout * 1000)),
                ),
            )
        except Exception as exc:
            message = str(exc)
            if "API key" in message or "PERMISSION_DENIED" in message:
                raise LLMError(
                    "Google rejected the API key.",
                    hint="Check the key in Settings → LLM provider.",
                ) from exc
            raise LLMError("The Gemini request failed.", hint=redact(message)) from exc

        return response.text or ""
