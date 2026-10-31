"""The dashboard view — every pinned chart at once, in a grid.

A dataset rarely tells its story in one chart: a trend, a breakdown, and a distribution are
usually all worth seeing together. Charts pinned from the Charts tab live here as tiles. Each tile
is just a :class:`~plotaviz.core.spec.ChartSpec`, so it redraws from the current cleaned and
filtered data whenever that changes, and the whole dashboard is saved with the session.

Tiles render with the static matplotlib canvas. An interactive WebEngine view per tile would load
plotly.js (≈4 MB) once per tile and make the dashboard the slowest screen in the app; the main
Chart tab stays interactive, and "Open" puts any tile there.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..core.errors import PlotaVizError
from ..core.plotter import TILE_MAX_POINTS, build_matplotlib, prepare
from ..core.spec import ChartSpec
from .theme import darken_figure

#: Height of one tile's chart, in pixels.
TILE_HEIGHT = 320


class DashboardView(QWidget):
    """A grid of pinned charts.

    Signals:
        open_requested: Tile index to load into the main chart view for editing.
        remove_requested: Tile index to unpin.
        move_requested: ``(index, new_index)`` to reorder.
        export_requested: The user clicked "Export dashboard…".
    """

    open_requested = Signal(int)
    remove_requested = Signal(int)
    move_requested = Signal(int, int)
    export_requested = Signal()
    columns_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._canvases: list[Any] = []
        self.tile_count = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        bar = QHBoxLayout()
        self.summary = QLabel("", self)
        self.summary.setStyleSheet("color: palette(mid);")
        bar.addWidget(self.summary, stretch=1)
        bar.addWidget(QLabel("Columns", self))
        self.columns_spin = QSpinBox(self)
        self.columns_spin.setRange(1, 3)
        self.columns_spin.setValue(2)
        self.columns_spin.valueChanged.connect(self.columns_changed.emit)
        bar.addWidget(self.columns_spin)
        self.export_button = QPushButton("Export dashboard…", self)
        self.export_button.clicked.connect(self.export_requested.emit)
        bar.addWidget(self.export_button)
        layout.addLayout(bar)

        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        # Tiles fit the pane's width and scroll vertically only; a sideways scroll hides whole
        # columns of the dashboard.
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.container = QWidget()
        self.grid = QGridLayout(self.container)
        self.grid.setSpacing(10)
        self.scroll.setWidget(self.container)
        layout.addWidget(self.scroll, stretch=1)

        self.set_tiles(None, [], [])

    @property
    def columns(self) -> int:
        """Tiles per row."""
        return self.columns_spin.value()

    def set_columns(self, columns: int) -> None:
        """Set tiles per row without emitting."""
        self.columns_spin.blockSignals(True)
        self.columns_spin.setValue(max(1, min(3, int(columns))))
        self.columns_spin.blockSignals(False)

    # ------------------------------------------------------------------ rendering

    def set_tiles(
        self,
        df: pd.DataFrame | None,
        specs: list[ChartSpec],
        statuses: list[str | None],
        built: list[Any] | None = None,
    ) -> None:
        """Rebuild every tile from the current data.

        Args:
            df: The cleaned, filtered frame.
            specs: Pinned charts in order.
            statuses: Per tile, ``None`` or why it cannot be drawn (from
                :meth:`Analysis.tile_status`).
            built: The output of :func:`build_tiles`, when it was computed on a worker thread.
                Built here, on the calling thread, when omitted.
        """
        if built is None and df is not None:
            built = build_tiles(df, specs, statuses)
        self._clear()
        self.tile_count = len(specs)
        self.export_button.setEnabled(bool(specs))
        if not specs or df is None:
            self.summary.setText("")
            empty = QLabel(
                "No charts pinned yet.\n\nPick a chart in the Charts tab and press "
                "“Pin to dashboard” (Ctrl+D) to collect several views of this dataset here.",
                self.container,
            )
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty.setWordWrap(True)
            empty.setStyleSheet("color: palette(mid); font-size: 14px; padding: 40px;")
            self.grid.addWidget(empty, 0, 0)
            return

        self.summary.setText(
            f"{len(specs)} chart{'s' if len(specs) != 1 else ''} · redrawn from the current "
            f"filters and cleaning ({len(df):,} rows)"
        )
        columns = self.columns
        for index, spec in enumerate(specs):
            content = built[index] if built and index < len(built) else "Not drawn yet."
            tile = self._tile(index, len(specs), spec, content)
            self.grid.addWidget(tile, index // columns, index % columns)
        for column in range(columns):
            self.grid.setColumnStretch(column, 1)

    def _tile(self, index: int, total: int, spec: ChartSpec, content: Any) -> QWidget:
        """One tile: a title bar with actions, then the chart or the reason it cannot show."""
        frame = QFrame(self.container)
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(6, 4, 6, 6)

        header = QHBoxLayout()
        title = QLabel(f"<b>{index + 1}.</b> {spec.display_title()}", frame)
        title.setToolTip(spec.why or spec.display_title())
        title.setTextFormat(Qt.TextFormat.RichText)
        # A long title must not set the tile's minimum width; it is clipped here (the chart shows
        # it in full, and so does the tooltip).
        title.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        header.addWidget(title, stretch=1)
        for text, tip, enabled, handler in (
            ("Open", "Show this chart in the Chart tab to edit it", True, self.open_requested),
            ("↑", "Move earlier", index > 0, None),
            ("↓", "Move later", index < total - 1, None),
            ("✕", "Remove from the dashboard", True, self.remove_requested),
        ):
            button = QToolButton(frame)
            button.setText(text)
            button.setToolTip(tip)
            button.setEnabled(enabled)
            if text == "↑":
                button.clicked.connect(lambda _=False, i=index: self.move_requested.emit(i, i - 1))
            elif text == "↓":
                button.clicked.connect(lambda _=False, i=index: self.move_requested.emit(i, i + 1))
            elif handler is not None:
                button.clicked.connect(lambda _=False, i=index, h=handler: h.emit(i))
            header.addWidget(button)
        layout.addLayout(header)

        if isinstance(content, str):
            label = QLabel(f"This chart cannot be drawn on the current data:\n{content}", frame)
            label.setWordWrap(True)
            label.setMinimumHeight(TILE_HEIGHT // 2)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("color: #A33;")
            layout.addWidget(label)
        else:
            layout.addWidget(self._canvas(content, frame))
        return frame

    def _canvas(self, figure: Any, parent: QWidget) -> Any:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        canvas = FigureCanvasQTAgg(figure)
        canvas.setParent(parent)
        canvas.setMinimumHeight(TILE_HEIGHT)
        canvas.setMinimumWidth(160)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._canvases.append(canvas)
        return canvas

    def _clear(self) -> None:
        """Remove every tile. Tile figures are plain ``Figure`` objects (not pyplot-managed),
        so dropping the canvases frees them."""
        self._canvases.clear()
        while self.grid.count():
            item = self.grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        for column in range(3):
            self.grid.setColumnStretch(column, 0)


def build_tiles(
    df: pd.DataFrame, specs: list[ChartSpec], statuses: list[str | None], *, dark: bool = False
) -> list[Any]:
    """Build every tile's figure, or the reason it cannot be drawn. Qt-free: safe on a worker.

    Returns:
        Per tile, a matplotlib ``Figure`` or an error message string.
    """
    built: list[Any] = []
    for index, spec in enumerate(specs):
        status = statuses[index] if index < len(statuses) else None
        if status is not None:
            built.append(status)
            continue
        try:
            prepared = prepare(df, spec, max_points=TILE_MAX_POINTS)
            figure = build_matplotlib(df, spec, prepared=prepared, figsize=(5.5, 3.6), dpi=90)
        except PlotaVizError as exc:
            built.append(exc.message)
            continue
        # Re-flow on every resize: tight_layout ran once at the build size, and a narrower tile
        # would otherwise clip the axis labels.
        figure.set_layout_engine("constrained")
        for axes in figure.axes:
            axes.title.set_fontsize(10)  # tiles are narrow; full-size titles overflow them
        if dark:
            darken_figure(figure)
        built.append(figure)
    return built
