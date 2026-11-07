"""Claude Code provider — Claude through the locally installed ``claude`` CLI.

This runs the user's own Claude Code install in headless mode (``claude -p``), so requests go out
under whatever account Claude Code is signed in with — a claude.ai subscription or an API key it
already manages. PlotaViz never sees or stores a credential for this provider: there is no key to
paste, nothing in the OS keyring, and nothing to leak.

The call is deliberately stripped down. Claude Code is an agent with tools, hooks, MCP servers,
and memory files; none of that belongs in "turn this schema into a chart spec". So every request
runs with ``--safe-mode`` (no CLAUDE.md, hooks, plugins, or MCP servers), ``--tools ""`` (no
tools at all, so it cannot read files or run commands), PlotaViz's own system prompt in place of
Claude Code's, and no saved session. What the model sees is exactly what every other provider
sees: the schema, the statistics, and the user's question.

Data still leaves the machine — it goes to Anthropic — so the consent rule applies as it does to
the API-key providers.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from typing import Any

from ..errors import LLMError
from ._cli import child_env, find_executable
from .base import DEFAULT_TIMEOUT, Provider, redact

#: Executable name looked up on ``PATH``.
CLAUDE_BINARY = "claude"

#: Where installers put ``claude`` when it is not on ``PATH``.
_FALLBACK_LOCATIONS = (
    "~/.local/bin/claude",
    "~/.claude/local/claude",
    "~/.npm-global/bin/claude",
    "/usr/local/bin/claude",
    "/opt/homebrew/bin/claude",
    "/usr/bin/claude",
)

#: Environment variables that mark a process as running inside a Claude Code session. Removed so
#: PlotaViz launched from a Claude Code terminal still gets a fresh, independent CLI process.
_SESSION_ENV = ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")

#: Phrases in a failed result that mean the CLI has no usable credentials.
_SIGNED_OUT_MARKERS = (
    "/login",
    "not logged in",
    "invalid api key",
    "authentication",
    "oauth token",
)


def find_claude_binary() -> str | None:
    """Locate the ``claude`` executable, or return ``None`` if it is not installed."""
    return find_executable(CLAUDE_BINARY, _FALLBACK_LOCATIONS)


def describe_account(status: dict[str, Any]) -> str:
    """Name the account a ``claude auth status`` object describes, e.g. "your Claude Pro plan"."""
    plan = str(status.get("subscriptionType") or status.get("planTier") or "").strip()
    if status.get("authMethod") == "claude.ai":
        return f"your Claude {plan.title()} plan" if plan else "your Claude subscription"
    method = str(status.get("authMethod") or "").strip()
    return method or "its own login"


def _child_env() -> dict[str, str]:
    """The environment for the CLI process, without Claude Code session markers."""
    return child_env(drop=_SESSION_ENV)


class ClaudeCodeProvider(Provider):
    """Talks to Claude through the local Claude Code CLI.

    Args:
        model: A Claude Code model alias (``sonnet``, ``opus``, ``haiku``, ``fable``) or a full
            model name.
        api_key: Ignored. Claude Code handles its own authentication.
        timeout: Per-request timeout. Higher than the SDK providers because each request starts a
            CLI process.
        binary: Path to the ``claude`` executable. Found automatically when omitted.
    """

    name = "claude-code"
    label = "Claude Code"
    default_model = "sonnet"
    available_models = ("sonnet", "opus", "haiku", "fable")
    needs_api_key = False
    listed_only_when_detected = True
    setup_note = (
        "Runs the claude command on this machine and uses the account Claude Code is signed in "
        "with (Claude Pro, Max, or its own API key) — no key to paste. Its tools, hooks, and "
        "memory files are switched off for these requests. The summary still goes to Anthropic, "
        "so consent applies."
    )

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT * 2,
        binary: str | None = None,
    ) -> None:
        super().__init__(model=model, api_key=None, timeout=timeout)
        self._binary = binary

    # ------------------------------------------------------------------ discovery

    @classmethod
    def detect(cls) -> bool:
        """Installed on this machine — a file lookup, no process is started."""
        return find_claude_binary() is not None

    @property
    def binary(self) -> str | None:
        """The ``claude`` executable this provider will run, if one can be found."""
        return self._binary or find_claude_binary()

    def is_configured(self) -> bool:
        """Configured when the CLI is installed. Sign-in is checked when a request is made.

        This is called on every settings refresh, so it only looks for the file; running the CLI
        here would add seconds of latency to opening a dataset.
        """
        return self.binary is not None

    def auth_status(self) -> dict[str, Any] | None:
        """Ask the CLI whether it is signed in, without sending any data to a model.

        Returns:
            The CLI's status object (``loggedIn``, ``authMethod``, …), or ``None`` if the CLI is
            missing or the answer could not be read.
        """
        binary = self.binary
        if binary is None:
            return None
        try:
            done = subprocess.run(
                [binary, "auth", "status"],
                capture_output=True,
                text=True,
                timeout=15,
                env=_child_env(),
                cwd=tempfile.gettempdir(),
                stdin=subprocess.DEVNULL,
            )
            status = json.loads(done.stdout)
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
            return None
        return status if isinstance(status, dict) else None

    def list_models(self) -> list[str]:
        """The model aliases Claude Code accepts. No request is made."""
        return list(self.available_models)

    def check_key(self) -> tuple[bool, str, list[str]]:
        """Check the CLI is installed and signed in. Nothing about the data is sent.

        Settings runs this on a worker thread, so the CLI start-up time never blocks the dialog.
        """
        binary = self.binary
        if binary is None:
            return (
                False,
                "The claude command was not found. Install Claude Code, then run `claude` once "
                "in a terminal to sign in.",
                [],
            )
        status = self.auth_status()
        if status is None:
            return (
                False,
                f"Found {binary}, but could not read its sign-in status. Run `claude` in a "
                "terminal to check it works.",
                [],
            )
        if not status.get("loggedIn"):
            return (
                False,
                "Claude Code is installed but not signed in. Run `claude` in a terminal and use "
                "/login.",
                [],
            )
        account = describe_account(status)
        return True, f"Claude Code is signed in and will use {account}.", self.list_models()

    # ------------------------------------------------------------------ request

    def command(self, system: str) -> list[str]:
        """The CLI invocation for one request. The user message is sent on stdin."""
        binary = self.binary
        if binary is None:
            raise LLMError(
                "The Claude Code provider needs the claude command, and it is not installed.",
                hint=(
                    "Install Claude Code (https://claude.com/claude-code), sign in once by "
                    "running `claude` in a terminal, then try again."
                ),
            )
        return [
            binary,
            "--print",
            "--output-format",
            "json",
            "--model",
            self.model,
            "--system-prompt",
            system,
            # Tools off and customizations off: this is a one-shot text completion, not an agent.
            "--tools",
            "",
            "--safe-mode",
            "--strict-mcp-config",
            "--no-session-persistence",
        ]

    def complete(self, system: str, user: str, *, timeout: float) -> str:
        """Run one headless Claude Code request and return the text of the reply."""
        try:
            done = subprocess.run(
                self.command(system),
                input=user,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=_child_env(),
                # Run outside any project, so no project settings or memory files apply.
                cwd=tempfile.gettempdir(),
            )
        except subprocess.TimeoutExpired as exc:
            raise LLMError(
                f"Claude Code did not respond within {timeout:.0f} seconds.",
                hint="PlotaViz will use its own rules engine instead.",
            ) from exc
        except OSError as exc:
            raise LLMError("Could not start the claude command.", hint=str(exc)) from exc

        return self._parse(done)

    def _parse(self, done: subprocess.CompletedProcess[str]) -> str:
        """Turn the CLI's JSON result into reply text, or a clear error."""
        try:
            payload = json.loads(done.stdout)
        except json.JSONDecodeError:
            detail = (done.stderr or done.stdout or "").strip()[-300:]
            raise LLMError(
                f"Claude Code exited with status {done.returncode} without a result.",
                hint=redact(detail) or "Run `claude` in a terminal to check it works.",
            ) from None

        if not isinstance(payload, dict):
            raise LLMError("Claude Code returned an unexpected response.")

        text = str(payload.get("result") or "")
        if payload.get("is_error") or payload.get("subtype") not in (None, "success"):
            lowered = text.lower()
            signed_out = payload.get("api_error_status") in (401, 403) or any(
                marker in lowered for marker in _SIGNED_OUT_MARKERS
            )
            if signed_out:
                raise LLMError(
                    "Claude Code is not signed in.",
                    hint="Run `claude` in a terminal and sign in with /login, then try again.",
                )
            raise LLMError("The Claude Code request failed.", hint=redact(text[:300]))
        return text
