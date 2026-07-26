"""Exception hierarchy for PlotaViz.

Every error the user can plausibly trigger is one of these, and every one carries a message
written for a human rather than a stack trace. The UI catches :class:`PlotaVizError` at the
boundary and shows ``str(exc)`` in a dialog; nothing else should reach the user.
"""

from __future__ import annotations


class PlotaVizError(Exception):
    """Base class for all PlotaViz errors intended to be shown to the user.

    Args:
        message: Human-readable description of what went wrong.
        hint: Optional suggestion for how to fix it, shown as secondary text.
    """

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        return f"{self.message}\n\n{self.hint}" if self.hint else self.message


class LoadError(PlotaVizError):
    """A dataset could not be read — unsupported format, corrupt file, bad encoding."""


class PreprocessError(PlotaVizError):
    """A cleaning step could not be applied to the dataframe."""


class ProfileError(PlotaVizError):
    """The dataset could not be profiled (empty, or no usable columns)."""


class SelectionError(PlotaVizError):
    """No chart could be recommended for the given data shape."""


class SpecError(PlotaVizError):
    """A chart spec is malformed or references columns that do not exist."""


class PlotError(PlotaVizError):
    """A figure could not be built from a chart spec."""


class ExportError(PlotaVizError):
    """Writing an image, script, or session file failed."""


class SessionError(PlotaVizError):
    """A ``.pviz`` session file could not be read or is from an incompatible version."""


class LLMError(PlotaVizError):
    """An LLM provider failed. Always recoverable — the caller falls back to the local engine."""


class ProviderNotConfigured(LLMError):
    """No provider selected, or the selected provider has no API key in the keyring."""
