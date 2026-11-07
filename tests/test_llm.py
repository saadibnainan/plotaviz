"""LLM layer tests — all mocked, no network, no keys.

Three properties matter more than any individual provider working:

1. A model's reply is **validated against the real schema** before it reaches a renderer.
2. The prompt carries **schema and statistics, never the dataset**.
3. A failure — no provider, bad key, timeout, garbage reply — **falls back to the local engine**
   rather than breaking the app.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from plotaviz.core.errors import LLMError, ProviderNotConfigured, SpecError
from plotaviz.core.llm import (
    PROVIDERS,
    AnthropicProvider,
    ClaudeCodeProvider,
    CodexCLIProvider,
    GeminiProvider,
    LLMAssistant,
    OllamaProvider,
    OpenAIProvider,
    Provider,
    _cli,
    build_prompt,
    claude_code,
    codex_cli,
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
        assert set(PROVIDERS) == {
            "anthropic",
            "claude-code",
            "openai",
            "codex-cli",
            "gemini",
            "ollama",
        }

    @pytest.mark.parametrize(
        "cls",
        [
            AnthropicProvider,
            ClaudeCodeProvider,
            OpenAIProvider,
            CodexCLIProvider,
            GeminiProvider,
            OllamaProvider,
        ],
    )
    def test_providers_declare_a_default_model(self, cls: type[Provider]) -> None:
        assert cls.default_model
        assert cls.default_model in cls.available_models

    def test_only_ollama_stays_on_the_machine(self) -> None:
        assert OllamaProvider.sends_data_off_machine is False
        for cls in (
            AnthropicProvider,
            ClaudeCodeProvider,
            OpenAIProvider,
            CodexCLIProvider,
            GeminiProvider,
        ):
            assert cls.sends_data_off_machine is True

    def test_only_the_sdk_providers_take_a_pasted_key(self) -> None:
        for cls in (AnthropicProvider, OpenAIProvider, GeminiProvider):
            assert cls.needs_api_key is True
        for cls in (ClaudeCodeProvider, CodexCLIProvider, OllamaProvider):
            assert cls.needs_api_key is False

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
        ok, message, models = provider.check_key()
        assert ok is False and models == [] and "No Ollama server" in message
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


def _cli_result(**fields: Any) -> str:
    """A ``claude -p --output-format json`` result object."""
    base: dict[str, Any] = {"type": "result", "subtype": "success", "is_error": False}
    base.update(fields)
    return json.dumps(base)


class FakeRun:
    """Stands in for ``subprocess.run`` and records how the CLI was invoked."""

    def __init__(
        self,
        stdout: str = "",
        stderr: str = "",
        returncode: int = 0,
        error: Exception | None = None,
    ) -> None:
        self.stdout, self.stderr, self.returncode, self.error = stdout, stderr, returncode, error
        self.calls: list[dict[str, Any]] = []

    def __call__(self, args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append({"args": args, **kwargs})
        if self.error:
            raise self.error
        return subprocess.CompletedProcess(args, self.returncode, self.stdout, self.stderr)


class TestClaudeCode:
    """The Claude Code provider shells out to the local CLI. Every test here mocks the process."""

    @pytest.fixture
    def provider(self) -> ClaudeCodeProvider:
        return ClaudeCodeProvider(binary="/fake/bin/claude")

    def test_needs_no_key_and_ignores_one(self) -> None:
        provider = ClaudeCodeProvider(api_key="sk-ant-should-be-ignored", binary="/x/claude")
        assert provider._api_key is None
        assert provider.is_configured() is True

    def test_not_configured_without_the_cli(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(claude_code, "find_claude_binary", lambda: None)
        provider = ClaudeCodeProvider()
        assert provider.is_configured() is False
        assert provider.auth_status() is None
        with pytest.raises(LLMError, match="not installed"):
            provider.complete("system", "user", timeout=5)

    def test_runs_a_tool_less_isolated_one_shot(
        self, provider: ClaudeCodeProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = FakeRun(stdout=_cli_result(result='{"chart": "bar"}'))
        monkeypatch.setattr(claude_code.subprocess, "run", run)

        assert provider.complete("SYSTEM", "the user message", timeout=12) == '{"chart": "bar"}'

        call = run.calls[0]
        args = call["args"]
        assert args[0] == "/fake/bin/claude"
        assert "--print" in args
        assert args[args.index("--tools") + 1] == ""
        assert args[args.index("--system-prompt") + 1] == "SYSTEM"
        assert args[args.index("--model") + 1] == "sonnet"
        assert args[args.index("--output-format") + 1] == "json"
        for flag in ("--safe-mode", "--strict-mcp-config", "--no-session-persistence"):
            assert flag in args
        # The prompt travels on stdin, not on the command line where `ps` would show it.
        assert call["input"] == "the user message"
        assert "the user message" not in args
        assert call["timeout"] == 12

    def test_model_choice_is_passed_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        run = FakeRun(stdout=_cli_result(result="{}"))
        monkeypatch.setattr(claude_code.subprocess, "run", run)
        get_provider("claude-code", model="opus", binary="/c").complete("s", "u", timeout=5)
        args = run.calls[0]["args"]
        assert args[args.index("--model") + 1] == "opus"

    def test_end_to_end_reply_becomes_a_validated_spec(
        self, provider: ClaudeCodeProvider, dataset_profile: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spec = {"chart": "line", "x": "order_date", "y": "revenue", "color": "region"}
        monkeypatch.setattr(
            claude_code.subprocess, "run", FakeRun(stdout=_cli_result(result=json.dumps(spec)))
        )

        result = LLMAssistant(provider, consent=True).from_question(
            dataset_profile, "revenue by region over time"
        )

        assert result.spec.chart == "line"
        assert result.provider == "claude-code"

    def test_hallucinated_column_is_still_rejected(
        self, provider: ClaudeCodeProvider, dataset_profile: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reply = json.dumps({"chart": "bar", "x": "not_a_column", "y": "revenue"})
        monkeypatch.setattr(
            claude_code.subprocess, "run", FakeRun(stdout=_cli_result(result=reply))
        )
        with pytest.raises(SpecError):
            provider.suggest_chart(dataset_profile, question="anything")

    def test_signed_out_cli_says_how_to_sign_in(
        self, provider: ClaudeCodeProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stdout = _cli_result(is_error=True, result="Not logged in · Please run /login")
        monkeypatch.setattr(claude_code.subprocess, "run", FakeRun(stdout=stdout, returncode=1))
        with pytest.raises(LLMError, match="not signed in"):
            provider.complete("s", "u", timeout=5)

    def test_other_api_errors_are_reported(
        self, provider: ClaudeCodeProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stdout = _cli_result(is_error=True, result="API Error: 529 Overloaded")
        monkeypatch.setattr(claude_code.subprocess, "run", FakeRun(stdout=stdout, returncode=1))
        with pytest.raises(LLMError, match="request failed"):
            provider.complete("s", "u", timeout=5)

    def test_non_json_output_is_an_error_not_a_crash(
        self, provider: ClaudeCodeProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = FakeRun(stdout="", stderr="error: unknown option", returncode=2)
        monkeypatch.setattr(claude_code.subprocess, "run", run)
        with pytest.raises(LLMError, match="status 2"):
            provider.complete("s", "u", timeout=5)

    def test_timeout_falls_back_to_the_local_engine(
        self, provider: ClaudeCodeProvider, dataset_profile: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = FakeRun(error=subprocess.TimeoutExpired(cmd="claude", timeout=1))
        monkeypatch.setattr(claude_code.subprocess, "run", run)
        with pytest.raises(LLMError, match="did not respond"):
            provider.complete("s", "u", timeout=1)

        candidates = ChartSelector().recommend(dataset_profile)
        assistant = LLMAssistant(provider, consent=True)
        assert assistant.refine(dataset_profile, candidates) is candidates[0]

    def test_auth_status_is_read_without_sending_a_prompt(
        self, provider: ClaudeCodeProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = FakeRun(stdout=json.dumps({"loggedIn": True, "authMethod": "claude.ai"}))
        monkeypatch.setattr(claude_code.subprocess, "run", run)
        assert provider.auth_status() == {"loggedIn": True, "authMethod": "claude.ai"}
        assert run.calls[0]["args"] == ["/fake/bin/claude", "auth", "status"]

    def test_check_key_reports_the_signed_in_account(
        self, provider: ClaudeCodeProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        status = {"loggedIn": True, "authMethod": "claude.ai", "subscriptionType": "pro"}
        run = FakeRun(stdout=json.dumps(status))
        monkeypatch.setattr(claude_code.subprocess, "run", run)
        ok, message, models = provider.check_key()
        assert ok is True
        assert "Claude Pro plan" in message
        assert models == list(ClaudeCodeProvider.available_models)

    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            ({"authMethod": "claude.ai", "subscriptionType": "pro"}, "your Claude Pro plan"),
            ({"authMethod": "claude.ai", "subscriptionType": "max"}, "your Claude Max plan"),
            ({"authMethod": "claude.ai"}, "your Claude subscription"),
            ({"authMethod": "api_key"}, "api_key"),
        ],
    )
    def test_names_the_plan(self, status: dict[str, Any], expected: str) -> None:
        assert claude_code.describe_account(status) == expected

    def test_detected_only_when_installed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from plotaviz.core.llm import detected_provider

        assert detected_provider() == ""
        monkeypatch.setattr(claude_code, "find_claude_binary", lambda: "/usr/bin/claude")
        assert detected_provider() == "claude-code"

    def test_check_key_says_how_to_sign_in(
        self, provider: ClaudeCodeProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = FakeRun(stdout=json.dumps({"loggedIn": False}), returncode=1)
        monkeypatch.setattr(claude_code.subprocess, "run", run)
        ok, message, models = provider.check_key()
        assert ok is False
        assert "/login" in message
        assert models == []

    def test_check_key_without_the_cli(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(claude_code, "find_claude_binary", lambda: None)
        ok, message, _models = ClaudeCodeProvider().check_key()
        assert ok is False
        assert "not found" in message

    def test_child_env_drops_session_markers_and_bundle_libs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CLAUDECODE", "1")
        monkeypatch.setenv("LD_LIBRARY_PATH", "/tmp/_MEI123")
        monkeypatch.setenv("LD_LIBRARY_PATH_ORIG", "/usr/lib/custom")
        monkeypatch.setattr(_cli.sys, "frozen", True, raising=False)

        env = claude_code._child_env()

        assert "CLAUDECODE" not in env
        assert env["LD_LIBRARY_PATH"] == "/usr/lib/custom"
        assert "LD_LIBRARY_PATH_ORIG" not in env

    @pytest.mark.real_cli_lookup
    def test_finds_the_cli_outside_path(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = tmp_path / "claude"
        fake.write_text("#!/bin/sh\n")
        fake.chmod(0o755)
        monkeypatch.setattr(_cli.shutil, "which", lambda _name: None)
        monkeypatch.setattr(claude_code, "_FALLBACK_LOCATIONS", (str(fake),))
        assert claude_code.find_claude_binary() == str(fake)


class TestAutoDetection:
    """Each provider says how it is found; the first one installed is used by default."""

    def test_nothing_installed_means_local_engine(self) -> None:
        from plotaviz.core.llm import detected_provider, detected_providers

        assert detected_providers() == []
        assert detected_provider() == ""

    def test_order_prefers_the_subscription_clis(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from plotaviz.core.llm import detected_provider, detected_providers, ollama

        monkeypatch.setattr(ollama, "find_ollama_binary", lambda: "/usr/bin/ollama")
        assert detected_provider() == "ollama"
        monkeypatch.setattr(codex_cli, "find_codex_binary", lambda: "/usr/bin/codex")
        assert detected_provider() == "codex-cli"
        monkeypatch.setattr(claude_code, "find_claude_binary", lambda: "/usr/bin/claude")
        assert detected_providers() == ["claude-code", "codex-cli", "ollama"]

    def test_api_key_providers_are_never_auto_detected(self) -> None:
        for cls in (AnthropicProvider, OpenAIProvider, GeminiProvider):
            assert cls.detect() is False
            assert cls.listed_only_when_detected is False


class FakeCodex(FakeRun):
    """``codex exec`` stand-in: writes the reply to the ``--output-last-message`` file."""

    def __init__(self, reply: str = "", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.reply = reply

    def __call__(self, args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        done = super().__call__(args, **kwargs)
        if "--output-last-message" in args and self.returncode == 0 and self.reply:
            Path(args[args.index("--output-last-message") + 1]).write_text(self.reply)
        return done


class TestCodexCLI:
    """The Codex CLI provider shells out to ``codex exec``. Every test mocks the process."""

    @pytest.fixture
    def provider(self) -> CodexCLIProvider:
        return CodexCLIProvider(binary="/fake/bin/codex")

    def test_listed_only_when_installed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert CodexCLIProvider.detect() is False
        monkeypatch.setattr(codex_cli, "find_codex_binary", lambda: "/usr/bin/codex")
        assert CodexCLIProvider.detect() is True
        assert CodexCLIProvider.listed_only_when_detected is True

    def test_runs_without_tools_config_or_session(
        self, provider: CodexCLIProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = FakeCodex(reply='{"chart": "bar"}')
        monkeypatch.setattr(codex_cli.subprocess, "run", run)

        assert provider.complete("SYSTEM", "the user message", timeout=12) == '{"chart": "bar"}'

        call = run.calls[0]
        args = call["args"]
        assert args[:2] == ["/fake/bin/codex", "exec"]
        for flag in ("--ephemeral", "--ignore-user-config", "--ignore-rules"):
            assert flag in args
        assert args[args.index("--sandbox") + 1] == "read-only"
        disabled = {args[i + 1] for i, a in enumerate(args) if a == "--disable"}
        assert {"shell_tool", "unified_exec", "computer_use", "browser_use"} <= disabled
        assert "--model" not in args  # "default" leaves the choice to Codex
        assert args[-1] == "-"
        # The prompt, instructions first, travels on stdin, never on the command line.
        assert call["input"] == "SYSTEM\n\nthe user message"
        assert "the user message" not in " ".join(args)
        assert call["cwd"] == args[args.index("--cd") + 1]

    def test_model_choice_is_passed_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        run = FakeCodex(reply="{}")
        monkeypatch.setattr(codex_cli.subprocess, "run", run)
        get_provider("codex-cli", model="gpt-6-luna", binary="/c").complete("s", "u", timeout=5)
        args = run.calls[0]["args"]
        assert args[args.index("--model") + 1] == "gpt-6-luna"

    def test_end_to_end_reply_becomes_a_validated_spec(
        self, provider: CodexCLIProvider, dataset_profile: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spec = {"chart": "line", "x": "order_date", "y": "revenue", "color": "region"}
        monkeypatch.setattr(codex_cli.subprocess, "run", FakeCodex(reply=json.dumps(spec)))

        result = LLMAssistant(provider, consent=True).from_question(dataset_profile, "trend")

        assert result.spec.chart == "line"
        assert result.provider == "codex-cli"

    def test_a_feature_codex_no_longer_has_is_dropped_and_retried(
        self, provider: CodexCLIProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[list[str]] = []

        def run(args: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
            calls.append(args)
            if "apps" in args:
                return subprocess.CompletedProcess(args, 1, "", "Error: Unknown feature flag: apps")
            Path(args[args.index("--output-last-message") + 1]).write_text("{}")
            return subprocess.CompletedProcess(args, 0, "", "")

        monkeypatch.setattr(codex_cli.subprocess, "run", run)
        assert provider.complete("s", "u", timeout=5) == "{}"
        assert len(calls) == 2
        assert "shell_tool" in calls[1]
        assert "apps" not in calls[1]

    def test_signed_out_cli_says_how_to_sign_in(
        self, provider: CodexCLIProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = FakeCodex(stderr="Error: Not logged in. Run codex login.", returncode=1)
        monkeypatch.setattr(codex_cli.subprocess, "run", run)
        with pytest.raises(LLMError, match="not signed in"):
            provider.complete("s", "u", timeout=5)

    def test_a_401_in_the_echoed_prompt_is_not_a_sign_in_error(
        self, provider: CodexCLIProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = FakeCodex(stderr="user\nmax: 401\nError: stream disconnected", returncode=1)
        monkeypatch.setattr(codex_cli.subprocess, "run", run)
        with pytest.raises(LLMError, match="exited with status 1"):
            provider.complete("s", "u", timeout=5)

    def test_timeout_is_an_llm_error(
        self, provider: CodexCLIProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = FakeCodex(error=subprocess.TimeoutExpired(cmd="codex", timeout=1))
        monkeypatch.setattr(codex_cli.subprocess, "run", run)
        with pytest.raises(LLMError, match="did not respond"):
            provider.complete("s", "u", timeout=1)

    @pytest.mark.parametrize(
        ("stderr", "returncode", "ok", "phrase"),
        [
            ("Logged in using ChatGPT", 0, True, "ChatGPT plan"),
            ("Logged in using an API key - sk-***", 0, True, "OpenAI API key"),
            ("Not logged in", 1, False, "codex login"),
        ],
    )
    def test_check_key_reads_login_status(
        self,
        provider: CodexCLIProvider,
        monkeypatch: pytest.MonkeyPatch,
        stderr: str,
        returncode: int,
        ok: bool,
        phrase: str,
    ) -> None:
        run = FakeCodex(stderr=stderr, returncode=returncode)
        monkeypatch.setattr(codex_cli.subprocess, "run", run)
        result_ok, message, _models = provider.check_key()
        assert result_ok is ok
        assert phrase in message
        assert run.calls[0]["args"] == ["/fake/bin/codex", "login", "status"]


class TestResolveProvider:
    """Which provider is used, given what Settings stored and what is installed."""

    @pytest.fixture
    def claude_installed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(claude_code, "find_claude_binary", lambda: "/usr/bin/claude")

    @pytest.fixture
    def no_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("plotaviz.core.llm.get_api_key", lambda _name: None)

    def test_never_chosen_uses_what_is_installed(self, claude_installed: None) -> None:
        from plotaviz.core.llm import resolve_provider

        assert resolve_provider(None) == ("claude-code", "")

    def test_explicit_none_is_kept(self, claude_installed: None) -> None:
        from plotaviz.core.llm import resolve_provider

        assert resolve_provider("") == ("", "")

    def test_key_less_saved_provider_falls_back_and_says_why(
        self, claude_installed: None, no_keys: None
    ) -> None:
        from plotaviz.core.llm import resolve_provider

        name, note = resolve_provider("anthropic")
        assert name == "claude-code"
        assert "Anthropic is selected but has no API key" in note
        assert "Claude Code" in note

    def test_saved_provider_with_a_key_is_kept(
        self, claude_installed: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from plotaviz.core.llm import resolve_provider

        monkeypatch.setattr("plotaviz.core.llm.get_api_key", lambda _name: "sk-ant-xxxxxxxx")
        assert resolve_provider("anthropic") == ("anthropic", "")

    def test_without_an_alternative_the_saved_choice_stands(self, no_keys: None) -> None:
        from plotaviz.core.llm import resolve_provider

        assert resolve_provider("openai") == ("openai", "")

    def test_key_less_providers_are_never_overridden(self, claude_installed: None) -> None:
        from plotaviz.core.llm import resolve_provider

        assert resolve_provider("ollama") == ("ollama", "")
        assert resolve_provider("codex-cli") == ("codex-cli", "")

    def test_a_removed_provider_falls_back_to_detection(self, claude_installed: None) -> None:
        from plotaviz.core.llm import resolve_provider

        assert resolve_provider("skynet") == ("claude-code", "")
