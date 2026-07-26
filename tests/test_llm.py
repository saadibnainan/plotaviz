"""LLM layer tests — all mocked, no network, no keys.

Three properties matter more than any individual provider working:

1. A model's reply is **validated against the real schema** before it reaches a renderer.
2. The prompt carries **schema and statistics, never the dataset**.
3. A failure — no provider, bad key, timeout, garbage reply — **falls back to the local engine**
   rather than breaking the app.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pytest

from plotaviz.core.errors import LLMError, ProviderNotConfigured, SpecError
from plotaviz.core.llm import (
    PROVIDERS,
    AnthropicProvider,
    GeminiProvider,
    LLMAssistant,
    OllamaProvider,
    OpenAIProvider,
    Provider,
    build_prompt,
    extract_json,
    get_provider,
    redact,
)
from plotaviz.core.llm.base import sample_rows_for
from plotaviz.core.profiler import profile
from plotaviz.core.selector import ChartSelector


class FakeProvider(Provider):
    """A provider that returns whatever it was told to, without touching a network."""

    name = "fake"
    default_model = "fake-1"
    sends_data_off_machine = False

    def __init__(self, reply: str = "", error: Exception | None = None) -> None:
        super().__init__()
        self.reply = reply
        self.error = error
        self.prompts: list[str] = []

    def is_configured(self) -> bool:
        return True

    def complete(self, system: str, user: str, *, timeout: float) -> str:
        self.prompts.append(user)
        if self.error:
            raise self.error
        return self.reply


@pytest.fixture
def dataset_profile(timeseries_df: pd.DataFrame):
    return profile(timeseries_df)


class TestRegistry:
    def test_every_provider_is_registered(self) -> None:
        assert set(PROVIDERS) == {"anthropic", "openai", "gemini", "ollama"}

    @pytest.mark.parametrize(
        "cls", [AnthropicProvider, OpenAIProvider, GeminiProvider, OllamaProvider]
    )
    def test_providers_declare_a_default_model(self, cls: type[Provider]) -> None:
        assert cls.default_model
        assert cls.default_model in cls.available_models

    def test_only_ollama_stays_on_the_machine(self) -> None:
        assert OllamaProvider.sends_data_off_machine is False
        for cls in (AnthropicProvider, OpenAIProvider, GeminiProvider):
            assert cls.sends_data_off_machine is True

    def test_unknown_provider_lists_the_real_ones(self) -> None:
        with pytest.raises(ProviderNotConfigured, match="Available providers"):
            get_provider("skynet")

    def test_get_provider_passes_options_through(self) -> None:
        provider = get_provider("ollama", model="qwen2.5", host="http://box:1234")
        assert provider.model == "qwen2.5"
        assert provider.host == "http://box:1234"  # type: ignore[attr-defined]

    def test_missing_key_explains_where_to_put_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("plotaviz.core.llm.base.get_api_key", lambda _name: None)
        with pytest.raises(ProviderNotConfigured, match="keychain"):
            AnthropicProvider().require_key()


class TestJsonExtraction:
    def test_plain_json(self) -> None:
        assert extract_json('{"chart": "bar"}')["chart"] == "bar"

    def test_markdown_fenced(self) -> None:
        assert extract_json('```json\n{"chart": "line"}\n```')["chart"] == "line"

    def test_unlabelled_fence(self) -> None:
        assert extract_json('```\n{"chart": "box"}\n```')["chart"] == "box"

    def test_surrounded_by_chatter(self) -> None:
        reply = 'Sure! Here is the spec:\n{"chart": "scatter", "x": "a"}\nHope that helps.'
        assert extract_json(reply)["chart"] == "scatter"

    def test_empty_reply(self) -> None:
        with pytest.raises(LLMError, match="empty"):
            extract_json("   ")

    def test_no_json_at_all(self) -> None:
        with pytest.raises(LLMError, match="did not contain a chart specification"):
            extract_json("I would recommend a bar chart.")

    def test_broken_json(self) -> None:
        with pytest.raises(LLMError, match="not valid JSON"):
            extract_json('{"chart": "bar",,,}')

    def test_json_that_is_not_an_object(self) -> None:
        with pytest.raises(LLMError, match="not a chart specification"):
            extract_json("[1, 2, 3]")


class TestPromptContents:
    def test_carries_schema_and_statistics(self, dataset_profile: Any) -> None:
        prompt = build_prompt(dataset_profile)
        assert "order_date" in prompt
        assert "revenue" in prompt
        assert '"role"' in prompt

    def test_does_not_carry_the_dataset(self, dataset_profile: Any) -> None:
        prompt = build_prompt(dataset_profile)
        assert len(prompt) < 20_000
        # A prompt containing the data would mention far more values than the sample allows.
        assert prompt.count("2025-01-01") <= 3

    def test_sample_rows_are_capped(
        self, timeseries_df: pd.DataFrame, dataset_profile: Any
    ) -> None:
        rows = sample_rows_for(timeseries_df, 5)
        prompt = build_prompt(dataset_profile, sample_rows=rows)

        assert len(rows) <= 5
        assert "illustrative only" in prompt

    def test_includes_the_question(self, dataset_profile: Any) -> None:
        prompt = build_prompt(dataset_profile, question="revenue by region over time")
        assert "revenue by region over time" in prompt

    def test_includes_local_candidates_for_a_tie_break(self, dataset_profile: Any) -> None:
        candidates = ChartSelector().recommend(dataset_profile)[:3]
        prompt = build_prompt(dataset_profile, candidates=candidates)
        assert "rules engine ranked these" in prompt


class TestSpecValidation:
    def test_a_valid_reply_becomes_a_spec(self, dataset_profile: Any) -> None:
        provider = FakeProvider(json.dumps({"chart": "line", "x": "order_date", "y": "revenue"}))
        result = provider.suggest_chart(dataset_profile, question="trend please")

        assert result.spec.chart == "line"
        assert result.spec.source == "llm"
        assert result.provider == "fake"

    def test_a_hallucinated_column_is_rejected(self, dataset_profile: Any) -> None:
        provider = FakeProvider(json.dumps({"chart": "line", "x": "order_date", "y": "profit"}))
        with pytest.raises(SpecError, match="profit"):
            provider.suggest_chart(dataset_profile, question="profit over time")

    def test_a_hallucinated_chart_type_is_rejected(self, dataset_profile: Any) -> None:
        provider = FakeProvider(json.dumps({"chart": "hologram", "x": "order_date"}))
        with pytest.raises(SpecError, match="Unknown chart type"):
            provider.suggest_chart(dataset_profile)

    def test_code_in_the_reply_is_never_executed(self, dataset_profile: Any) -> None:
        """The contract is a spec, not code. A code-shaped reply is simply not JSON."""
        provider = FakeProvider("import os; os.system('rm -rf /')")
        with pytest.raises(LLMError):
            provider.suggest_chart(dataset_profile)


class TestAssistantFallback:
    def test_unavailable_without_a_provider(self) -> None:
        assert LLMAssistant(None).available is False

    def test_remote_provider_needs_consent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(AnthropicProvider, "is_configured", lambda _self: True)
        assert LLMAssistant(AnthropicProvider(), consent=False).available is False
        assert LLMAssistant(AnthropicProvider(), consent=True).available is True

    def test_ollama_needs_no_consent_prompt(self) -> None:
        assert LLMAssistant(OllamaProvider()).consent_prompt() is None

    def test_consent_prompt_names_the_provider(self) -> None:
        prompt = LLMAssistant(OpenAIProvider()).consent_prompt()
        assert prompt is not None
        assert "openai" in prompt
        assert "never uploaded" in prompt

    def test_refine_falls_back_when_there_is_no_provider(self, dataset_profile: Any) -> None:
        candidates = ChartSelector().recommend(dataset_profile)
        assert LLMAssistant(None).refine(dataset_profile, candidates) is candidates[0]

    def test_refine_falls_back_on_a_provider_error(self, dataset_profile: Any) -> None:
        candidates = ChartSelector().recommend(dataset_profile)
        broken = FakeProvider(error=LLMError("the api is on fire"))

        assert LLMAssistant(broken).refine(dataset_profile, candidates) is candidates[0]

    def test_refine_falls_back_on_a_garbage_reply(self, dataset_profile: Any) -> None:
        candidates = ChartSelector().recommend(dataset_profile)
        chatty = FakeProvider("I think a pie chart would be lovely")

        assert LLMAssistant(chatty).refine(dataset_profile, candidates) is candidates[0]

    def test_refine_falls_back_on_an_unexpected_crash(self, dataset_profile: Any) -> None:
        candidates = ChartSelector().recommend(dataset_profile)
        exploding = FakeProvider(error=RuntimeError("segfault in the vendor sdk"))

        assert LLMAssistant(exploding).refine(dataset_profile, candidates) is candidates[0]

    def test_refine_uses_a_valid_suggestion_and_credits_it(self, dataset_profile: Any) -> None:
        candidates = ChartSelector().recommend(dataset_profile)
        provider = FakeProvider(
            json.dumps({"chart": "box", "x": "region", "y": "revenue", "why": "spread matters"})
        )

        spec = LLMAssistant(provider).refine(dataset_profile, candidates)

        assert spec.chart == "box"
        assert "fake" in spec.why

    def test_nl_query_without_a_provider_points_at_settings(self, dataset_profile: Any) -> None:
        with pytest.raises(ProviderNotConfigured, match="Ollama"):
            LLMAssistant(None).from_question(dataset_profile, "revenue over time")

    def test_nl_query_without_consent_is_refused(
        self, dataset_profile: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(OpenAIProvider, "is_configured", lambda _self: True)
        with pytest.raises(LLMError, match="not been approved"):
            LLMAssistant(OpenAIProvider(), consent=False).from_question(dataset_profile, "hi")

    def test_nl_query_returns_a_validated_spec(self, dataset_profile: Any) -> None:
        provider = FakeProvider(
            json.dumps({"chart": "line", "x": "order_date", "y": "revenue", "color": "region"})
        )
        result = LLMAssistant(provider, consent=True).from_question(
            dataset_profile, "revenue by region over time"
        )

        assert result.spec.chart == "line"
        assert result.spec.color == "region"


class TestOllama:
    def test_reports_not_running_without_a_server(self) -> None:
        provider = OllamaProvider(host="http://127.0.0.1:59999")
        assert provider.is_running() is False
        assert provider.list_models() == []
        assert provider.is_configured() is False

    def test_connection_error_explains_how_to_start_it(self) -> None:
        pytest.importorskip("httpx", reason="the Ollama transport is part of the llm extra")

        provider = OllamaProvider(host="http://127.0.0.1:59999")
        with pytest.raises(LLMError, match="ollama serve"):
            provider.complete("system", "user", timeout=2.0)

    def test_missing_httpx_says_which_extra_to_install(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def blocked(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "httpx":
                raise ImportError("no httpx here")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)
        with pytest.raises(LLMError, match=r"plotaviz\[llm\]"):
            OllamaProvider().complete("system", "user", timeout=1.0)


class TestRedaction:
    @pytest.mark.parametrize(
        "secret",
        [
            "sk-ant-api03-abcdefghijklmnop",
            "sk-abcdefghijklmnopqrstuvwx",
            "AIzaSyA1234567890abcdefghijklmnopqrs",
            "Bearer abcdefghijklmnopqrstuvwxyz123456",
        ],
    )
    def test_keys_are_masked(self, secret: str) -> None:
        masked = redact(f"request failed with {secret} in the header")
        assert secret not in masked
        assert "[redacted]" in masked

    def test_ordinary_text_is_untouched(self) -> None:
        assert redact("column revenue is missing") == "column revenue is missing"
