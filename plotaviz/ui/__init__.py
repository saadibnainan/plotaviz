"""PySide6 widgets.

Everything in this package may import Qt; nothing in :mod:`plotaviz.core` may. The boundary runs
one way: the UI calls into the core through plain data objects — a
:class:`~plotaviz.core.analysis.Analysis`, a :class:`~plotaviz.core.spec.ChartSpec`, a
:class:`~plotaviz.core.profiler.DatasetProfile` — and the core knows nothing about widgets.

Imports here are lazy at the module level so ``import plotaviz`` stays cheap for CLI use, which
never touches Qt at all.
"""

from __future__ import annotations

__all__ = ["MainWindow", "launch"]


def __getattr__(name: str) -> object:
    """Import the main window on first use, so importing this package does not start Qt."""
    if name in __all__:
        from .main_window import MainWindow, launch

        return {"MainWindow": MainWindow, "launch": launch}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
