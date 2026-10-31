"""A virtualized table model for the data preview.

``QTableView`` only asks for the cells it is about to paint, so backing it with a model that
reads straight from the dataframe means a million-row file costs the same to display as a
hundred-row one. The rule from the performance guardrails — *never render 1M rows* — is satisfied
by never materializing them: no list of rows is ever built.

A page cap is still applied on top, because scrolling through a million rows is not a feature
anyone wants; the view shows the first :data:`MAX_PREVIEW_ROWS` and says so.
"""

from __future__ import annotations

import html
from collections.abc import Callable
from typing import Any

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QObject, Qt, Signal

from ..core.profiler import DatasetProfile

#: Rows exposed to the preview table. Beyond this the user is looking at data, not reading it.
MAX_PREVIEW_ROWS = 5_000

#: Role-specific tints, so a glance at the header tells you how each column was classified.
_ROLE_COLOURS = {
    "numeric": "#2E6DA4",
    "categorical": "#7B4EA8",
    "datetime": "#1F7A5C",
    "boolean": "#A8681F",
    "text": "#6B6B6B",
}


def _is_missing(value: Any) -> bool:
    """``pd.isna`` for one cell. Lists and dicts (nested JSON) are values, never missing —
    ``pd.isna`` on them returns an array, which used to raise while painting the table."""
    if isinstance(value, (list, tuple, dict, set)):
        return False
    return bool(pd.isna(value))


def arrange_frame(
    df: pd.DataFrame, sort_column: str | None, ascending: bool, search: str
) -> pd.DataFrame:
    """Search and sort the *whole* frame — not just the rows the preview shows.

    Pure and Qt-free, so the window can run it on a worker for large data.

    Args:
        df: The cleaned, filtered frame.
        sort_column: Column to sort by, or ``None``.
        ascending: Sort direction.
        search: Case-insensitive text; rows where any cell contains it are kept.
    """
    out = df
    if search:
        mask = pd.Series(False, index=out.index)
        for column in out.columns:
            mask |= (
                out[column]
                .astype("string")
                .str.contains(search, case=False, regex=False, na=False)
                .astype(bool)
            )
        out = out.loc[mask]
    if sort_column is not None and sort_column in out.columns:
        out = out.sort_values(sort_column, ascending=ascending, na_position="last", kind="stable")
    return out


def column_summary_html(prof: Any) -> str:
    """A short HTML summary of one column's profile, for the stats panel."""
    rows = [
        ("Type", prof.role + (" (identifier)" if prof.is_identifier else "")),
        ("Distinct values", f"{prof.n_unique:,}"),
        ("Missing", f"{prof.n_missing:,} ({prof.pct_missing:.1f}%)"),
    ]
    for key in ("min", "max", "mean", "median", "std", "span_days"):
        if key in prof.stats:
            value = prof.stats[key]
            rows.append((key, f"{value:,.4g}" if isinstance(value, float) else str(value)))
    body = "".join(
        f"<tr><td style='color:gray;padding-right:8px'>{html.escape(k)}</td>"
        f"<td>{html.escape(str(v))}</td></tr>"
        for k, v in rows
    )
    top = ""
    if prof.top_values:
        items = "".join(
            f"<li>{html.escape(str(value))} <span style='color:gray'>({count:,})</span></li>"
            for value, count in prof.top_values[:8]
        )
        top = f"<p style='margin:4px 0 0 0'>Most frequent:</p><ul style='margin:0'>{items}</ul>"
    note = f"<p style='color:gray'>{html.escape(prof.note)}</p>" if prof.note else ""
    return f"<b>{html.escape(prof.name)}</b><table>{body}</table>{top}{note}"


#: The same tints, lightened to stay readable on a dark background.
_ROLE_COLOURS_DARK = {
    "numeric": "#7FB2E5",
    "categorical": "#C3A3E8",
    "datetime": "#6FD0A8",
    "boolean": "#E8B26F",
    "text": "#B0B0B0",
}


class DataFrameModel(QAbstractTableModel):
    """Exposes a pandas DataFrame to a ``QTableView`` without copying it.

    Args:
        df: The frame to display.
        profile: Optional profile, used to annotate headers with each column's role.
        parent: Parent object.
    """

    #: The displayed rows changed (a new frame, a sort, or a search).
    view_changed = Signal()

    def __init__(
        self,
        df: pd.DataFrame | None = None,
        profile: DatasetProfile | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._df = df if df is not None else pd.DataFrame()
        self._view = self._df
        self._profile = profile
        self._rows = min(len(self._view), MAX_PREVIEW_ROWS)
        self._sort: tuple[str | None, bool] = (None, True)
        #: Use header tints that read on a dark background.
        self.dark = False
        self._search = ""
        #: How to run :func:`arrange_frame`: ``scheduler(work, done)``. The window sets one that
        #: uses a worker thread; without it the work runs inline.
        self.scheduler: (
            Callable[[Callable[[], pd.DataFrame], Callable[[pd.DataFrame], None]], None] | None
        ) = None

    # ------------------------------------------------------------------ data access

    def set_frame(self, df: pd.DataFrame, profile: DatasetProfile | None = None) -> None:
        """Swap in a new frame and repaint, keeping the current sort and search."""
        self._df = df
        self._profile = profile
        self._arrange()

    def sort(self, column: int, order: Qt.SortOrder = Qt.SortOrder.AscendingOrder) -> None:
        """Sort the whole frame by a column (called by the view on a header click)."""
        if not 0 <= column < len(self._df.columns):
            return
        self._sort = (str(self._df.columns[column]), order == Qt.SortOrder.AscendingOrder)
        self._arrange()

    def set_search(self, text: str) -> None:
        """Show only rows where some cell contains ``text`` (case-insensitive)."""
        self._search = text.strip()
        self._arrange()

    def _arrange(self) -> None:
        column, ascending = self._sort
        if column is None and not self._search:
            self._set_view(self._df)
            return
        df, search = self._df, self._search

        def work() -> pd.DataFrame:
            return arrange_frame(df, column, ascending, search)

        if self.scheduler is None:
            self._set_view(work())
        else:
            self.scheduler(work, self._set_view)

    def _set_view(self, view: pd.DataFrame) -> None:
        self.beginResetModel()
        self._view = view
        self._rows = min(len(view), MAX_PREVIEW_ROWS)
        self.endResetModel()
        self.view_changed.emit()

    @property
    def frame(self) -> pd.DataFrame:
        """The frame being displayed."""
        return self._df

    @property
    def truncated(self) -> bool:
        """Whether the view is showing fewer rows than the frame holds."""
        return len(self._view) > self._rows

    def status_text(self) -> str:
        """A line describing what the preview is showing."""
        rows, cols = len(self._df), len(self._df.columns)
        if self._search:
            matched = len(self._view)
            shown = f"; previewing the first {self._rows:,}" if self.truncated else ""
            return f"{matched:,} of {rows:,} rows contain “{self._search}”{shown}"
        if self.truncated:
            return f"Previewing the first {self._rows:,} of {rows:,} rows × {cols} columns"
        return f"{rows:,} rows × {cols} columns"

    # ------------------------------------------------------------------ Qt model API

    def rowCount(self, parent: QModelIndex | None = None) -> int:
        """Rows the view should ask about."""
        if parent is not None and parent.isValid():
            return 0
        return self._rows

    def columnCount(self, parent: QModelIndex | None = None) -> int:
        """Column count."""
        if parent is not None and parent.isValid():
            return 0
        return len(self._df.columns)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        """Return one cell's display text or alignment."""
        if not index.isValid():
            return None

        row, col = index.row(), index.column()
        if row >= self._rows or col >= len(self._view.columns):
            return None

        if role == Qt.ItemDataRole.DisplayRole:
            value = self._view.iat[row, col]
            if _is_missing(value):
                return "—"
            if isinstance(value, float):
                return f"{value:,.4g}"
            if isinstance(value, pd.Timestamp):
                return value.strftime(
                    "%Y-%m-%d %H:%M" if value.time() != value.min.time() else "%Y-%m-%d"
                )
            return str(value)

        if role == Qt.ItemDataRole.TextAlignmentRole:
            column = self._view.columns[col]
            if pd.api.types.is_numeric_dtype(self._view[column]):
                return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            return int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        if role == Qt.ItemDataRole.ForegroundRole and _is_missing(self._view.iat[row, col]):
            from PySide6.QtGui import QColor

            return QColor("#999999")

        return None

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        """Column names with their inferred role, and row numbers down the side."""
        if orientation == Qt.Orientation.Vertical:
            if role == Qt.ItemDataRole.DisplayRole:
                return str(section + 1)
            return None

        if section >= len(self._df.columns):
            return None
        name = str(self._df.columns[section])

        if role == Qt.ItemDataRole.DisplayRole:
            return name

        if role == Qt.ItemDataRole.ToolTipRole:
            return self._column_tooltip(name)

        if role == Qt.ItemDataRole.ForegroundRole and self._profile:
            prof = self._profile.columns.get(name)
            if prof:
                from PySide6.QtGui import QColor

                colours = _ROLE_COLOURS_DARK if self.dark else _ROLE_COLOURS
                return QColor(colours.get(prof.role, "#888888"))

        return None

    def _column_tooltip(self, name: str) -> str:
        """Per-column summary shown when hovering a header."""
        if not self._profile:
            return name
        prof = self._profile.columns.get(name)
        if prof is None:
            return name

        lines = [f"{name} — {prof.role}"]
        if prof.is_identifier:
            lines.append("Looks like an identifier, so it is excluded from chart suggestions.")
        if prof.is_derived:
            lines.append("Added by PlotaViz.")
        lines.append(f"{prof.n_unique:,} distinct values")
        if prof.pct_missing:
            lines.append(f"{prof.pct_missing:.1f}% missing")
        if prof.role == "numeric" and prof.stats:
            lines.append(
                f"min {prof.stats.get('min', 0):,.4g} · "
                f"median {prof.stats.get('median', 0):,.4g} · "
                f"max {prof.stats.get('max', 0):,.4g}"
            )
        if prof.note:
            lines.append(prof.note)
        return "\n".join(lines)
