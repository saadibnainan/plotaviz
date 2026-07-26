"""The chart view — Plotly in a QWebEngineView, with a matplotlib fallback.

**Every QtWebEngine import in PlotaViz is inside this module.** That isolation is deliberate:
WebEngine drags in Chromium, which is most of the bundle size and nearly all of the packaging
pain (AppImage in particular). Keeping it behind one seam means it can be lazy-loaded, swapped,
or dropped entirely without touching the rest of the app.

When WebEngine is unavailable — a slim build, a headless box, a distro that ships PySide6 without
it — :class:`ChartView` silently falls back to a matplotlib canvas embedded in Qt. The user loses
hover and zoom, not the chart, and the status line says which renderer is active.
"""

from __future__ import annotations

import contextlib
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
from PySide6.QtCore import Qt, QUrl
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from ..core.errors import PlotError
from ..core.plotter import build_matplotlib, build_plotly, prepare
from ..core.spec import ChartSpec

#: Set once by :func:`webengine_available` so the probe import happens at most once.
_WEBENGINE_STATE: bool | None = None


def webengine_available() -> bool:
    """Whether QtWebEngine can be imported in this environment.

    The import is attempted once and the answer cached. A failure here is completely normal —
    several Linux distributions package ``pyside6`` without ``pyside6-webengine``.
    """
    global _WEBENGINE_STATE
    if _WEBENGINE_STATE is None:
        try:
            from PySide6.QtWebEngineWidgets import QWebEngineView  # noqa: F401

            _WEBENGINE_STATE = True
        except Exception:
            _WEBENGINE_STATE = False
    return _WEBENGINE_STATE


class ChartView(QWidget):
    """Displays a chart, interactively when it can and statically when it cannot.

    Args:
        parent: Parent widget.
        prefer_static: Force the matplotlib canvas even when WebEngine is present. Useful for
            very large charts and for users who dislike the browser view.
    """

    def __init__(self, parent: QWidget | None = None, *, prefer_static: bool = False) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

        self._interactive = webengine_available() and not prefer_static
        self._temp_files: list[Path] = []
        self._placeholder: QLabel | None = None
        self._canvas: Any = None
        self._web: Any = None
        self.last_notes: list[str] = []

        self._show_placeholder(
            "Open a dataset to get started.\n\nDrag a CSV, Excel, JSON, or Parquet file here."
        )

    # ------------------------------------------------------------------ properties

    @property
    def renderer(self) -> str:
        """``"interactive"`` (Plotly/WebEngine) or ``"static"`` (matplotlib)."""
        return "interactive" if self._interactive else "static"

    def renderer_note(self) -> str:
        """One line for the status bar explaining what the user is looking at."""
        if self._interactive:
            return "Interactive chart — zoom, pan, hover, and toggle series in the legend."
        if webengine_available():
            return "Static chart (interactive view disabled in settings)."
        return "Static chart — QtWebEngine is not installed, so interactive charts are unavailable."

    def set_prefer_static(self, prefer_static: bool) -> None:
        """Switch renderers. The caller re-renders afterwards."""
        self._interactive = webengine_available() and not prefer_static

    # ------------------------------------------------------------------ rendering

    def render_spec(self, df: pd.DataFrame, spec: ChartSpec) -> list[str]:
        """Draw a chart.

        Args:
            df: The cleaned, filtered data.
            spec: What to draw.

        Returns:
            User-facing notes from preparation (sampling, category capping) so the caller can
            surface them. They are also stored on :attr:`last_notes`.

        Raises:
            PlotError: If the chart cannot be built. The caller shows this in a dialog.
        """
        prepared = prepare(df, spec)
        self.last_notes = list(prepared.notes)

        if self._interactive:
            try:
                self._render_interactive(df, spec, prepared)
                return self.last_notes
            except PlotError:
                raise
            except Exception as exc:
                self._interactive = False
                self.last_notes.append(
                    f"The interactive view failed ({exc}); showing a static chart instead."
                )

        self._render_static(df, spec, prepared)
        return self.last_notes

    def _render_interactive(self, df: pd.DataFrame, spec: ChartSpec, prepared: Any) -> None:
        """Render through Plotly into a QWebEngineView."""
        from PySide6.QtWebEngineWidgets import QWebEngineView

        figure = build_plotly(df, spec, prepared=prepared)

        if not isinstance(self._web, QWebEngineView):
            self._clear()
            self._web = QWebEngineView(self)
            self._web.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self._layout.addWidget(self._web)

        # Plotly's bundled JS is several megabytes and `setHtml` truncates above ~2 MB, so the
        # figure goes to a temp file and is loaded as a URL instead.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", prefix="plotaviz_", delete=False, encoding="utf-8"
        ) as handle:
            handle.write(
                figure.to_html(include_plotlyjs=True, full_html=True, config={"responsive": True})
            )
        path = Path(handle.name)
        self._temp_files.append(path)
        self._web.load(QUrl.fromLocalFile(str(path)))
        self._prune_temp_files()

    def _render_static(self, df: pd.DataFrame, spec: ChartSpec, prepared: Any) -> None:
        """Render through matplotlib into an embedded Qt canvas."""
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.backends.backend_qtagg import (
            NavigationToolbar2QT as NavigationToolbar,
        )

        figure = build_matplotlib(df, spec, prepared=prepared, figsize=(9, 5.5), dpi=100)

        self._clear()
        self._canvas = FigureCanvasQTAgg(figure)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        toolbar = NavigationToolbar(self._canvas, self)
        self._layout.addWidget(toolbar)
        self._layout.addWidget(self._canvas)
        self._canvas.draw_idle()

    # ------------------------------------------------------------------ housekeeping

    def _show_placeholder(self, text: str) -> None:
        """Replace the chart with centred guidance text."""
        self._clear()
        self._placeholder = QLabel(text, self)
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setWordWrap(True)
        self._placeholder.setStyleSheet("color: palette(mid); font-size: 14px; padding: 40px;")
        self._layout.addWidget(self._placeholder)

    def show_message(self, text: str) -> None:
        """Show a message in place of a chart (no data, or an error the user must act on)."""
        self._show_placeholder(text)

    def _clear(self) -> None:
        """Tear down whatever widget is currently showing."""
        for attr in ("_placeholder", "_canvas", "_web"):
            widget = getattr(self, attr, None)
            if widget is not None:
                self._layout.removeWidget(widget)
                widget.setParent(None)
                widget.deleteLater()
                setattr(self, attr, None)

    def _prune_temp_files(self, keep: int = 2) -> None:
        """Delete old temp HTML files, keeping the most recent few.

        The current page cannot be deleted out from under WebEngine on Windows-style locking
        semantics, so a small tail is retained rather than deleting eagerly.
        """
        while len(self._temp_files) > keep:
            stale = self._temp_files.pop(0)
            with contextlib.suppress(OSError):
                stale.unlink(missing_ok=True)

    def cleanup(self) -> None:
        """Remove every temp file. Called when the window closes."""
        for path in self._temp_files:
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)
        self._temp_files.clear()
