"""The LLM layer — optional intelligence on top of a fully working local engine.

PlotaViz is **offline-first**. The rules + scoring engine answers instantly with no provider
configured, and it is the fallback whenever an LLM call fails, times out, or returns something
that does not validate. A model is never on the critical path.

The layer does two jobs:

1. **A second opinion on selection** when the local scores are close enough that the ranking is
   effectively a coin flip.
2. **Natural language → chart spec**, which is the only way to get from "revenue by region over
   time" to a rendered chart.

Both return a *validated chart spec*. No code is ever generated, returned, or executed.
"""

from __future__ import annotations

import logging
from typing import Any

from ..errors import LLMError, ProviderNotConfigured
from ..profiler import DatasetProfile
from ..spec import ChartSpec
from .anthropic import AnthropicProvider
from .base import (
    KEYRING_SERVICE,
    LLMResult,
    Provider,
    build_prompt,
    delete_api_key,
    extract_json,
    get_api_key,
    redact,
    sample_rows_for,
    set_api_key,
)
from .claude_code import ClaudeCodeProvider, find_claude_binary
from .codex_cli import CodexCLIProvider, find_codex_binary
from .gemini import GeminiProvider
from .ollama import OllamaProvider
from .openai import OpenAIProvider

logger = logging.getLogger(__name__)

#: Exactly what a remote request contains. Shown in Settings and in the consent prompt, so the
#: wording has to match :func:`build_prompt` — not round it down.
CONSENT_TEXT = (
    "PlotaViz sends the column names, summary statistics (counts, min/max, mean, skew), and up "
    "to five example and most-frequent values per column. Values from identifier and free-text "
    "columns are withheld. The dataset itself is never uploaded."
)

#: Every provider PlotaViz can use, keyed by the name stored in settings and the keyring.
PROVIDERS: dict[str, type[Provider]] = {
    AnthropicProvider.name: AnthropicProvider,
    ClaudeCodeProvider.name: ClaudeCodeProvider,
    OpenAIProvider.name: OpenAIProvider,
    CodexCLIProvider.name: CodexCLIProvider,
    GeminiProvider.name: GeminiProvider,
    OllamaProvider.name: OllamaProvider,
}

__all__ = [
    "CONSENT_TEXT",
    "KEYRING_SERVICE",
    "PROVIDERS",
    "AnthropicProvider",
    "ClaudeCodeProvider",
    "CodexCLIProvider",
    "GeminiProvider",
    "LLMAssistant",
    "LLMResult",
    "OllamaProvider",
    "OpenAIProvider",
    "Provider",
    "build_prompt",
    "delete_api_key",
    "detected_provider",
    "detected_providers",
    "extract_json",
    "find_claude_binary",
    "find_codex_binary",
    "get_api_key",
    "get_provider",
    "redact",
    "resolve_provider",
    "sample_rows_for",
    "set_api_key",
]


#: Which detected provider wins when the user has never chosen one. The subscription CLIs come
#: first because they work with no setup at all; a local Ollama still needs a pulled model.
AUTO_DETECT_ORDER: tuple[str, ...] = (
    ClaudeCodeProvider.name,
    CodexCLIProvider.name,
    OllamaProvider.name,
)


def detected_providers() -> list[str]:
    """Providers installed on this machine, in :data:`AUTO_DETECT_ORDER`. No process is started."""
    return [name for name in AUTO_DETECT_ORDER if PROVIDERS[name].detect()]


def detected_provider() -> str:
    """The provider to use when the user has never chosen one, or ``""`` for the local engine.

    Consent is still asked before the first request that would leave the machine.
    """
    found = detected_providers()
    return found[0] if found else ""


def resolve_provider(saved: str | None) -> tuple[str, str]:
    """Decide which provider to use, given what Settings has stored.

    Args:
        saved: The stored ``llm/provider`` value, or ``None`` if the user never chose one.

    Returns:
        ``(name, note)``. ``name`` is the provider to use, ``""`` for the local engine only.
        ``note`` is empty when ``name`` is simply the saved choice; otherwise it explains, for
        Settings to show, why another provider was picked.

    The rules, in order:

    * Never chosen → the first provider installed on this machine (:func:`detected_provider`).
    * An explicit "None" → the local engine. That is a decision, not a gap to fill.
    * A saved provider that needs an API key and has none stored cannot make a single request,
      so an installed provider (Claude Code, Codex, Ollama) is used instead. Consent is still
      asked for that provider by name before anything is sent.
    """
    if saved is None:
        return detected_provider(), ""
    if not saved:
        return "", ""
    cls = PROVIDERS.get(saved)
    if cls is None:
        return detected_provider(), ""
    if cls.needs_api_key and not get_api_key(saved):
        found = detected_provider()
        if found:
            chosen = cls.label or saved
            used = PROVIDERS[found].label or found
            return found, (
                f"{chosen} is selected but has no API key, so PlotaViz is using {used}, which is "
                f"installed on this machine. To use {chosen} instead, select it and paste a key."
            )
    return saved, ""


def get_provider(name: str, **kwargs: Any) -> Provider:
    """Instantiate a provider by name.

    Raises:
        ProviderNotConfigured: If ``name`` is not a known provider.
    """
    cls = PROVIDERS.get(name.lower().strip())
    if cls is None:
        raise ProviderNotConfigured(
            f"Unknown LLM provider {name!r}.",
            hint=f"Available providers: {', '.join(PROVIDERS)}.",
        )
    return cls(**kwargs)


class LLMAssistant:
    """Optional LLM help, with the local engine always underneath.

    Args:
        provider: A configured provider, or ``None`` for local-only operation.
        consent: Whether the user has agreed to send schema and statistics to a remote service.
            Required before any provider that leaves the machine is used. Ollama does not need it.

    Example:
        >>> assistant = LLMAssistant(get_provider("ollama"), consent=True)
        >>> question = "revenue by region over time"
        >>> result = assistant.from_question(profile, question)  # doctest: +SKIP
    """

    def __init__(self, provider: Provider | None = None, *, consent: bool = False) -> None:
        self.provider = provider
        self.consent = consent

    # ------------------------------------------------------------------ state

    @property
    def available(self) -> bool:
        """Whether a request could be made right now — configured, and consented to if remote."""
        if self.provider is None:
            return False
        if self.provider.sends_data_off_machine and not self.consent:
            return False
        return self.provider.is_configured()

    def consent_prompt(self) -> str | None:
        """The text to show before the first remote request, or ``None`` if none is needed."""
        if self.provider is None or not self.provider.sends_data_off_machine:
            return None
        return f"{CONSENT_TEXT} It goes to {self.provider.name}.\n\n" + (
            "To keep everything on this machine, use a local Ollama server instead."
        )

    # ------------------------------------------------------------------ jobs

    def from_question(
        self,
        profile: DatasetProfile,
        question: str,
        *,
        sample_rows: list[dict[str, Any]] | None = None,
    ) -> LLMResult:
        """Turn a natural-language request into a validated chart spec.

        Raises:
            LLMError: If no provider is available, or the request fails. There is no local
                fallback for this job — a free-text question genuinely needs a model — so the
                caller should show the error and leave the current chart alone.
        """
        if self.provider is None:
            raise ProviderNotConfigured(
                "The natural-language query bar needs an LLM provider.",
                hint=(
                    "Configure one in Settings → LLM provider. For a fully local setup, install "
                    "Ollama and pull a model. The chart recommendations work without any of this."
                ),
            )
        if self.provider.sends_data_off_machine and not self.consent:
            raise LLMError(
                "Sending data to a remote provider has not been approved for this session.",
                hint=(
                    "Accept the consent prompt in Settings, or switch to the local Ollama provider."
                ),
            )

        prompt_rows = sample_rows or []
        result = self.provider.suggest_chart(profile, question=question, sample_rows=prompt_rows)
        logger.debug(
            "NL query answered by %s/%s with %d sample rows",
            result.provider,
            result.model,
            len(prompt_rows),
        )
        return result

    def refine(
        self,
        profile: DatasetProfile,
        candidates: list[ChartSpec],
        *,
        fallback: ChartSpec | None = None,
    ) -> ChartSpec:
        """Ask for a second opinion on an ambiguous ranking, falling back silently.

        This is the failure-tolerant path: a timeout, a bad key, a malformed reply, or no
        provider at all all end the same way — with the local engine's answer. The user's chart
        appears either way.

        Args:
            profile: The dataset profile.
            candidates: The local engine's ranked candidates.
            fallback: Spec to return on any failure. Defaults to the top candidate.

        Returns:
            Either the model's validated spec or the fallback.
        """
        default = fallback or (candidates[0] if candidates else None)
        if default is None:
            raise LLMError("There are no candidate charts to refine.")
        if not self.available:
            return default

        try:
            assert self.provider is not None  # narrowed by self.available
            result = self.provider.suggest_chart(profile, candidates=candidates)
        except LLMError as exc:
            logger.info("LLM refinement failed, using the local ranking: %s", redact(str(exc)))
            return default
        except Exception as exc:
            logger.warning("Unexpected LLM error, using the local ranking: %s", redact(str(exc)))
            return default

        spec = result.spec
        if not spec.why:
            spec.why = f"Suggested by {result.provider} ({result.model})."
        else:
            spec.why = f"{spec.why} (Suggested by {result.provider}.)"
        return spec
