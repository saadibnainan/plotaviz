"""A virtualized table model for the data preview.

``QTableView`` only asks for the cells it is about to paint, so backing it with a model that
reads straight from the dataframe means a million-row file costs the same to display as a
hundred-row one. The rule from the performance guardrails — *never render 1M rows* — is satisfied
by never materializing them: no list of rows is ever built.

A page cap is still applied on top, because scrolling through a million rows is not a feature
anyone wants; the view shows the first :data:`MAX_PREVIEW_ROWS` and says so.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QObject, Qt

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


class DataFrameModel(QAbstractTableModel):
    """Exposes a pandas DataFrame to a ``QTableView`` without copying it.

    Args:
        df: The frame to display.
        profile: Optional profile, used to annotate headers with each column's role.
        parent: Parent object.
    """

    def __init__(
        self,
        df: pd.DataFrame | None = None,
        profile: DatasetProfile | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._df = df if df is not None else pd.DataFrame()
        self._profile = profile
        self._rows = min(len(self._df), MAX_PREVIEW_ROWS)

    # ------------------------------------------------------------------ data access

    def set_frame(self, df: pd.DataFrame, profile: DatasetProfile | None = None) -> None:
        """Swap in a new frame and repaint."""
        self.beginResetModel()
        self._df = df
        self._profile = profile
        self._rows = min(len(df), MAX_PREVIEW_ROWS)
        self.endResetModel()

    @property
    def frame(self) -> pd.DataFrame:
        """The frame being displayed."""
        return self._df

    @property
    def truncated(self) -> bool:
        """Whether the view is showing fewer rows than the frame holds."""
        return len(self._df) > self._rows

    def status_text(self) -> str:
        """A line describing what the preview is showing."""
        rows, cols = len(self._df), len(self._df.columns)
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
        if row >= self._rows or col >= len(self._df.columns):
            return None

        if role == Qt.ItemDataRole.DisplayRole:
            value = self._df.iat[row, col]
            if pd.isna(value):
                return "—"
            if isinstance(value, float):
                return f"{value:,.4g}"
            if isinstance(value, pd.Timestamp):
                return value.strftime(
                    "%Y-%m-%d %H:%M" if value.time() != value.min.time() else "%Y-%m-%d"
                )
            return str(value)

        if role == Qt.ItemDataRole.TextAlignmentRole:
            column = self._df.columns[col]
            if pd.api.types.is_numeric_dtype(self._df[column]):
                return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            return int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        if role == Qt.ItemDataRole.ForegroundRole and pd.isna(self._df.iat[row, col]):
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

                return QColor(_ROLE_COLOURS.get(prof.role, "#333333"))

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
