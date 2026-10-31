"""The join dialog — bring in columns from a second file, matched on key columns.

It shows, before anything is applied, how the join would turn out: how many rows find a match,
and whether the second file repeats a key (which would multiply rows). The join itself is a
pipeline step, so it is undoable, saved in sessions, and replayed by generated code.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core import loader
from ..core.errors import PlotaVizError
from ..core.preprocess import NormalizeColumnNames, join_preview

#: Column names that usually are join keys, preselected when the files share them.
_KEY_NAME = re.compile(r"(^|_)(id|key|code|no|number|sku|iso)$", re.IGNORECASE)


class JoinDialog(QDialog):
    """Pick a second file, the key columns, and the join type.

    Args:
        current: The frame the join will apply to (:meth:`Analysis.join_base` — renamed, with
            earlier joins, unfiltered and not yet type-coerced), used for the match preview.
        normalize: Whether the second file's names are normalized, as the join step will.
        start_dir: Folder the file picker opens in.
        parent: Parent widget.
    """

    def __init__(
        self,
        current: pd.DataFrame,
        start_dir: str | Path = ".",
        parent: QWidget | None = None,
        *,
        normalize: bool = True,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Join another dataset")
        self.setMinimumWidth(560)
        self._current = current
        self._start_dir = Path(start_dir)
        self._other: pd.DataFrame | None = None
        self._normalize = normalize

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Add columns from a second file by matching rows on one or more key columns, "
            "like a spreadsheet lookup. Names are normalized the same way as this dataset's.",
            self,
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()
        row = QHBoxLayout()
        self.path_edit = QLineEdit(self)
        self.path_edit.setReadOnly(True)
        browse = QPushButton("Choose file…", self)
        browse.clicked.connect(self._browse)
        row.addWidget(self.path_edit, stretch=1)
        row.addWidget(browse)
        form.addRow("File", row)

        self.keys_list = QListWidget(self)
        self.keys_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.keys_list.setMaximumHeight(150)
        self.keys_list.itemSelectionChanged.connect(self._update_preview)
        form.addRow("Match on", self.keys_list)

        self.how_combo = QComboBox(self)
        self.how_combo.addItem("Keep every row of the current data (left join)", "left")
        self.how_combo.addItem("Keep only rows found in both (inner join)", "inner")
        form.addRow("Rows", self.how_combo)
        layout.addLayout(form)

        self.preview_label = QLabel("Choose a file to see how many rows match.", self)
        self.preview_label.setWordWrap(True)
        layout.addWidget(self.preview_label)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        layout.addWidget(self.buttons)

    # ------------------------------------------------------------------ interaction

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Dataset to join", str(self._start_dir), "Data files (*.*)"
        )
        if path:
            self.set_file(path)

    def set_file(self, path: str | Path) -> bool:
        """Read the second file and offer the columns both datasets share as keys."""
        try:
            other = loader.load(path).df
        except PlotaVizError as exc:
            self.preview_label.setText(str(exc))
            return False
        self._other = NormalizeColumnNames().apply(other)[0] if self._normalize else other
        self.path_edit.setText(str(Path(path).expanduser().resolve()))

        shared = [c for c in self._current.columns if c in set(self._other.columns)]
        self.keys_list.clear()
        likely = [c for c in shared if _KEY_NAME.search(str(c))] or shared[:1]
        for column in shared:
            item = QListWidgetItem(str(column))
            item.setData(Qt.ItemDataRole.UserRole, column)
            self.keys_list.addItem(item)
            item.setSelected(column in likely)
        if not shared:
            self.preview_label.setText(
                "The two files share no column names, so there is nothing to match on. Rename a "
                "column in one of them so the key has the same name in both."
            )
        self._update_preview()
        return bool(shared)

    def keys(self) -> list[str]:
        return [item.data(Qt.ItemDataRole.UserRole) for item in self.keys_list.selectedItems()]

    def _update_preview(self) -> None:
        ok = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        keys = self.keys()
        if self._other is None or not keys:
            ok.setEnabled(False)
            if self._other is not None:
                self.preview_label.setText("Select at least one column to match on.")
            return
        try:
            stats = join_preview(self._current, self._other, keys)
        except (KeyError, TypeError, ValueError) as exc:
            ok.setEnabled(False)
            self.preview_label.setText(f"These columns cannot be matched: {exc}")
            return
        share = stats["matched_rows"] / stats["rows"] if stats["rows"] else 0.0
        text = (
            f"{stats['matched_rows']:,} of {stats['rows']:,} rows ({share:.0%}) find a match "
            f"in {Path(self.path_edit.text()).name}."
        )
        if stats["duplicate_keys_in_other"]:
            text += (
                f" {stats['duplicate_keys_in_other']:,} key value(s) repeat in the second file, "
                "so matching rows will be multiplied."
            )
        self.preview_label.setText(text)
        ok.setEnabled(stats["matched_rows"] > 0)

    def result_values(self) -> dict[str, Any]:
        """``path``, ``on``, and ``how`` for :meth:`Analysis.join_dataset`."""
        return {
            "path": self.path_edit.text(),
            "on": self.keys(),
            "how": self.how_combo.currentData(),
        }
