"""Codex CLI provider — OpenAI models through the locally installed ``codex`` CLI.

The OpenAI counterpart of the Claude Code provider: it runs ``codex exec`` headless, so requests
go out under whatever account Codex is signed in with — a ChatGPT plan or an API key Codex already
manages. PlotaViz never sees or stores a credential for it.

Codex is a coding agent, so every request is stripped down to a one-shot completion: no user
config, no rules files, no saved session, a read-only sandbox in an empty temporary folder, and its
shell, browser, computer-use, app, and plugin features switched off. Codex has no system-prompt
flag, so PlotaViz's instructions are sent at the top of the prompt on stdin.

Data still leaves the machine — it goes to OpenAI — so the consent rule applies.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from ..errors import LLMError
from ._cli import child_env, find_executable
from .base import DEFAULT_TIMEOUT, Provider, redact

#: Executable name looked up on ``PATH``.
CODEX_BINARY = "codex"

#: Where installers put ``codex`` when it is not on ``PATH``.
_FALLBACK_LOCATIONS = (
    "~/.local/bin/codex",
    "~/.npm-global/bin/codex",
    "/usr/local/bin/codex",
    "/opt/homebrew/bin/codex",
    "/usr/bin/codex",
)

#: Codex features that give the model a way to act — run commands, drive a browser or the desktop,
#: call connected apps. All off: PlotaViz wants a chart spec, and column names come from the user's
#: file, so nothing in a prompt should be able to make the model touch this machine.
DISABLED_FEATURES = (
    "shell_tool",
    "unified_exec",
    "browser_use",
    "browser_use_external",
    "in_app_browser",
    "computer_use",
    "apps",
    "plugins",
)

#: Model choice that means "whatever Codex uses by default", so no ``--model`` flag is passed.
DEFAULT_MODEL = "default"

_UNKNOWN_FEATURE = re.compile(r"Unknown feature flag: (\S+)")

#: Phrases in Codex's error output that mean it has no usable sign-in.
#: Kept specific: Codex echoes the prompt on stderr, so a bare "401" could be a dataset value.
_SIGNED_OUT_MARKERS = (
    "not logged in",
    "please log in",
    "codex login",
    "401 unauthorized",
    "invalid api key",
)


def find_codex_binary() -> str | None:
    """Locate the ``codex`` executable, or return ``None`` if it is not installed."""
    return find_executable(CODEX_BINARY, _FALLBACK_LOCATIONS)


def describe_login(status_text: str) -> str:
    """Name the account ``codex login status`` describes, e.g. "your ChatGPT plan"."""
    lowered = status_text.lower()
    if "chatgpt" in lowered:
        return "your ChatGPT plan"
    if "api key" in lowered:
        return "an OpenAI API key"
    return "its own login"


class CodexCLIProvider(Provider):
    """Talks to OpenAI models through the local Codex CLI.

    Args:
        model: A model Codex accepts, or ``"default"`` for Codex's own choice.
        api_key: Ignored. Codex handles its own authentication.
        timeout: Per-request timeout. Higher than the SDK providers because each request starts a
            CLI process.
        binary: Path to the ``codex`` executable. Found automatically when omitted.
    """

    name = "codex-cli"
    label = "Codex CLI"
    default_model = DEFAULT_MODEL
    available_models = (DEFAULT_MODEL,)
    needs_api_key = False
    listed_only_when_detected = True
    setup_note = (
        "Runs the codex command on this machine and uses the account Codex is signed in with "
        "(a ChatGPT plan or its own API key) — no key to paste. Its shell, browser, and plugin "
        "tools are switched off for these requests. The summary still goes to OpenAI, so consent "
        "applies. Model “default” means whatever Codex uses."
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
        return find_codex_binary() is not None

    @property
    def binary(self) -> str | None:
        """The ``codex`` executable this provider will run, if one can be found."""
        return self._binary or find_codex_binary()

    def is_configured(self) -> bool:
        """Configured when the CLI is installed. Sign-in is checked when a request is made."""
        return self.binary is not None

    def list_models(self) -> list[str]:
        """The model choices offered in Settings. No request is made."""
        return list(self.available_models)

    def check_key(self) -> tuple[bool, str, list[str]]:
        """Check the CLI is installed and signed in. Nothing about the data is sent."""
        binary = self.binary
        if binary is None:
            return (
                False,
                "The codex command was not found. Install the Codex CLI, then run `codex login` "
                "in a terminal.",
                [],
            )
        try:
            done = subprocess.run(
                [binary, "login", "status"],
                capture_output=True,
                text=True,
                timeout=15,
                env=child_env(),
                cwd=tempfile.gettempdir(),
                stdin=subprocess.DEVNULL,
            )
        except (OSError, subprocess.SubprocessError):
            return False, f"Found {binary}, but could not run it.", []
        # Codex prints its status on stderr.
        text = f"{done.stdout}\n{done.stderr}".strip()
        if (
            done.returncode != 0
            or "logged in" not in text.lower()
            or "not logged in" in text.lower()
        ):
            return (
                False,
                "The Codex CLI is installed but not signed in. Run `codex login` in a terminal.",
                [],
            )
        return True, f"Codex is signed in and will use {describe_login(text)}.", self.list_models()

    # ------------------------------------------------------------------ request

    def command(self, workdir: str, reply_file: str, disabled: tuple[str, ...]) -> list[str]:
        """The CLI invocation for one request. The prompt is sent on stdin."""
        binary = self.binary
        if binary is None:
            raise LLMError(
                "The Codex CLI provider needs the codex command, and it is not installed.",
                hint="Install the Codex CLI, run `codex login` in a terminal, then try again.",
            )
        args = [
            binary,
            "exec",
            "--skip-git-repo-check",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--sandbox",
            "read-only",
            "--cd",
            workdir,
            "--color",
            "never",
            "--output-last-message",
            reply_file,
        ]
        for feature in disabled:
            args += ["--disable", feature]
        if self.model and self.model != DEFAULT_MODEL:
            args += ["--model", self.model]
        args.append("-")  # read the prompt from stdin
        return args

    def complete(self, system: str, user: str, *, timeout: float) -> str:
        """Run one headless Codex request and return the text of its final message."""
        prompt = f"{system}\n\n{user}"
        disabled: tuple[str, ...] = DISABLED_FEATURES
        with tempfile.TemporaryDirectory(prefix="plotaviz-codex-") as workdir:
            reply_file = Path(workdir) / "reply.txt"
            # A feature Codex no longer has is reported as unknown and refused. Its tool is gone
            # either way, so drop that flag and run again rather than failing.
            for _attempt in range(len(DISABLED_FEATURES) + 1):
                done = self._run(self.command(workdir, str(reply_file), disabled), prompt, timeout)
                unknown = _UNKNOWN_FEATURE.search(done.stderr or "")
                if done.returncode != 0 and unknown and unknown.group(1) in disabled:
                    disabled = tuple(f for f in disabled if f != unknown.group(1))
                    continue
                break

            if done.returncode != 0:
                self._raise_for(done)
            if reply_file.is_file():
                text = reply_file.read_text(encoding="utf-8").strip()
                if text:
                    return text
            return (done.stdout or "").strip()

    def _run(
        self, args: list[str], prompt: str, timeout: float
    ) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                args,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=child_env(),
                cwd=args[args.index("--cd") + 1],
            )
        except subprocess.TimeoutExpired as exc:
            raise LLMError(
                f"Codex did not respond within {timeout:.0f} seconds.",
                hint="PlotaViz will use its own rules engine instead.",
            ) from exc
        except OSError as exc:
            raise LLMError("Could not start the codex command.", hint=str(exc)) from exc

    @staticmethod
    def _raise_for(done: subprocess.CompletedProcess[str]) -> None:
        detail = (done.stderr or done.stdout or "").strip()
        lowered = detail.lower()
        if any(marker in lowered for marker in _SIGNED_OUT_MARKERS):
            raise LLMError(
                "The Codex CLI is not signed in.",
                hint="Run `codex login` in a terminal, then try again.",
            )
        raise LLMError(
            f"Codex exited with status {done.returncode}.",
            hint=redact(detail[-300:]) or "Run `codex` in a terminal to check it works.",
        )
