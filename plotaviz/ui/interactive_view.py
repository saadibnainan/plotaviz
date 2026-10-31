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
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from PySide6.QtCore import QStandardPaths, Qt, QUrl, Signal
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from ..core.errors import PlotError
from ..core.plotter import MAX_POINTS, build_matplotlib, build_plotly, prepare
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


@dataclass
class RenderResult:
    """A chart built off the UI thread, ready for :meth:`ChartView.install`.

    Exactly one of :attr:`figure_json` (Plotly) and :attr:`figure` (matplotlib) is set.
    """

    figure_json: str | None = None
    figure: Any = None
    notes: list[str] = field(default_factory=list)
    #: The interactive build failed and this is the static fallback.
    fell_back: bool = False


def build_render(
    df: pd.DataFrame, spec: ChartSpec, *, interactive: bool, max_points: int | None = None
) -> RenderResult:
    """Prepare and build a chart. Pure and Qt-free, so it can run on a worker thread.

    Raises:
        PlotError: If the chart cannot be built at all.
    """
    prepared = prepare(df, spec, max_points=max_points or MAX_POINTS)
    notes = list(prepared.notes)
    if interactive:
        try:
            figure = build_plotly(df, spec, prepared=prepared)
            return RenderResult(figure_json=figure.to_json(), notes=notes)
        except PlotError:
            raise
        except Exception as exc:
            notes.append(f"The interactive view failed ({exc}); showing a static chart instead.")
            figure = build_matplotlib(df, spec, prepared=prepared, figsize=(9, 5.5), dpi=100)
            return RenderResult(figure=figure, notes=notes, fell_back=True)
    figure = build_matplotlib(df, spec, prepared=prepared, figsize=(9, 5.5), dpi=100)
    return RenderResult(figure=figure, notes=notes)


#: The page the interactive view loads once. Charts are pushed into it with Plotly.react.
_SHELL_HTML = """<!doctype html>
<html><head><meta charset="utf-8">
<style>html,body{{margin:0;height:100%;overflow:hidden}}#chart{{width:100%;height:100%}}</style>
<script src="{script}"></script>
</head><body><div id="chart"></div>
<script>
window.PlotaViz = {{
  render: function (json) {{
    var figure = JSON.parse(json);
    Plotly.react("chart", figure.data, figure.layout, {{responsive: true}});
  }}
}};
</script></body></html>
"""


def _shell_page() -> Path:
    """Write (once per plotly version) plotly.js and the shell page into the app cache."""
    import plotly
    from plotly.offline import get_plotlyjs

    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.CacheLocation)
    folder = Path(base or tempfile.gettempdir()) / "plotaviz-view"
    folder.mkdir(parents=True, exist_ok=True)
    script = folder / f"plotly-{plotly.__version__}.min.js"
    source = get_plotlyjs().encode("utf-8")
    # Rewrite when missing or not exactly the expected size: a write interrupted by a crash, or
    # raced by a second instance, must not leave a truncated script cached forever. The write
    # goes to a temporary name first and is renamed into place atomically.
    if not script.exists() or script.stat().st_size != len(source):
        partial = folder / f".{script.name}.{os.getpid()}.part"
        partial.write_bytes(source)
        os.replace(partial, script)
    shell = folder / "chart.html"
    shell.write_text(_SHELL_HTML.format(script=script.name), encoding="utf-8")
    return shell


class ChartView(QWidget):
    """Displays a chart, interactively when it can and statically when it cannot.

    Args:
        parent: Parent widget.
        prefer_static: Force the matplotlib canvas even when WebEngine is present. Useful for
            very large charts and for users who dislike the browser view.
    """

    #: The interactive view failed while showing a chart; the owner should re-render (the
    #: view has already switched to static).
    static_fallback_needed = Signal()

    def __init__(self, parent: QWidget | None = None, *, prefer_static: bool = False) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

        self._interactive = webengine_available() and not prefer_static
        self._temp_files: list[Path] = []
        self._placeholder: QLabel | None = None
        self._canvas: Any = None
        self._toolbar: Any = None
        self._web: Any = None
        self._shell_ready = False
        self._shell_failed = False
        self._pending_json: str | None = None
        self.last_notes: list[str] = []
        #: Counters for tests and diagnostics: full page loads vs. in-place updates.
        self.page_loads = 0
        self.react_calls = 0

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

    @property
    def interactive(self) -> bool:
        """Whether the next render uses the Plotly view."""
        return self._interactive

    def render_spec(
        self, df: pd.DataFrame, spec: ChartSpec, *, max_points: int | None = None
    ) -> list[str]:
        """Build and show a chart on the calling thread (the synchronous convenience path).

        The main window builds on a worker with :func:`build_render` and calls :meth:`install`;
        this does both for callers that do not need the chart off the UI thread.

        Returns:
            User-facing notes from preparation (sampling, category capping).

        Raises:
            PlotError: If the chart cannot be built.
        """
        result = build_render(df, spec, interactive=self._interactive, max_points=max_points)
        notes = self.install(result)
        if result.figure_json is not None and not self._interactive:
            # The interactive install failed just now: draw the static chart in its place, and
            # keep the note that says why.
            failure = [n for n in notes if "interactive view failed" in n]
            static = build_render(df, spec, interactive=False, max_points=max_points)
            notes = [*self.install(static), *failure]
            self.last_notes = notes
        return notes

    def install(self, result: RenderResult) -> list[str]:
        """Show a chart built by :func:`build_render`. Must run on the UI thread."""
        self.last_notes = list(result.notes)
        if result.fell_back:
            self._interactive = False
        if result.figure_json is not None:
            try:
                self._install_interactive(result.figure_json)
                return self.last_notes
            except Exception as exc:  # WebEngine broke at runtime
                # The result holds only Plotly JSON, so the static chart has to be rebuilt from
                # the data; the owner does that on this signal (render_spec does it inline).
                self._interactive = False
                self.last_notes.append(
                    f"The interactive view failed ({exc}); showing a static chart instead."
                )
                self.show_message("Switching to the static chart…")
                self.static_fallback_needed.emit()
                return self.last_notes
        self._install_static(result.figure)
        return self.last_notes

    def _install_interactive(self, figure_json: str) -> None:
        """Show a Plotly figure, updating the already-loaded page in place when possible.

        The page (and the ~4 MB plotly.js it loads from the app cache) is loaded once. Every
        later chart is pushed into it with ``Plotly.react``, so filtering or switching charts
        no longer re-parses plotly.js and flashes a blank view.
        """
        from PySide6.QtWebEngineWidgets import QWebEngineView

        if not isinstance(self._web, QWebEngineView):
            self._clear()
            self._web = QWebEngineView(self)
            self._web.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self._layout.addWidget(self._web)
            self._shell_ready = False
            self._web.loadFinished.connect(self._on_shell_loaded)
            self.page_loads += 1
            self._web.load(QUrl.fromLocalFile(str(_shell_page())))
        self._pending_json = figure_json
        if self._shell_failed:
            self._load_standalone()
        elif self._shell_ready:
            self._push_figure()

    def _on_shell_loaded(self, ok: bool) -> None:
        """The shell page finished loading; draw whatever chart is waiting."""
        if ok:
            self._shell_ready = True
            self._push_figure()
            return
        # The shell could not load (a broken cache directory, say): from now on every chart is
        # written as one standalone page, which is what earlier versions always did.
        self._shell_failed = True
        if self._web is not None:
            self._web.loadFinished.disconnect(self._on_shell_loaded)
        self._load_standalone()

    def _load_standalone(self) -> None:
        """Load the pending chart as a self-contained page (the no-shell fallback)."""
        if self._pending_json is None or self._web is None:
            return
        import plotly.io as pio

        figure = pio.from_json(self._pending_json)
        self._pending_json = None
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", prefix="plotaviz_", delete=False, encoding="utf-8"
        ) as handle:
            handle.write(
                figure.to_html(include_plotlyjs=True, full_html=True, config={"responsive": True})
            )
        self._temp_files.append(Path(handle.name))
        self.page_loads += 1
        self._web.load(QUrl.fromLocalFile(handle.name))
        self._prune_temp_files()

    def _push_figure(self) -> None:
        if self._pending_json is None or self._web is None:
            return
        payload = json.dumps(self._pending_json)  # a JS string literal holding the figure JSON
        self._pending_json = None
        self.react_calls += 1
        self._web.page().runJavaScript(f"window.PlotaViz && PlotaViz.render({payload});")

    def _install_static(self, figure: Any) -> None:
        """Show a matplotlib figure in an embedded Qt canvas."""
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.backends.backend_qtagg import (
            NavigationToolbar2QT as NavigationToolbar,
        )

        self._clear()
        self._canvas = FigureCanvasQTAgg(figure)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._toolbar = NavigationToolbar(self._canvas, self)
        self._layout.addWidget(self._toolbar)
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
        """Tear down whatever widget is currently showing.

        The matplotlib toolbar is listed too: it is a separate widget from the canvas, and
        leaving it behind stacked one extra toolbar above the chart on every re-render.
        """
        if self._canvas is not None:
            # pyplot keeps every figure it creates until it is closed; without this each static
            # re-render leaked the previous figure.
            import matplotlib.pyplot as plt

            plt.close(self._canvas.figure)
        for attr in ("_placeholder", "_toolbar", "_canvas", "_web"):
            widget = getattr(self, attr, None)
            if widget is not None:
                self._layout.removeWidget(widget)
                widget.setParent(None)
                widget.deleteLater()
                setattr(self, attr, None)
        self._shell_ready = False

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
