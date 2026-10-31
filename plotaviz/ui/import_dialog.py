"""The import options dialog — "File → Open with options…".

Detection gets most files right, and the ones it gets wrong are predictable: a semicolon file
with comma decimals, a Latin-1 export from an old ERP, a report with a title row above the
header, a workbook whose data is on the third sheet. This dialog lets the user say so directly.
Every choice is passed to :func:`plotaviz.core.loader.load` as ``options``, recorded in the
session, and replayed in the generated script.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPlainTextEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..core.loader import CSV_SUFFIXES, EXCEL_SUFFIXES

#: Value stored for "let PlotaViz detect it".
AUTO = None

_DELIMITERS = (
    ("Detect automatically", AUTO),
    ("Comma  ,", ","),
    ("Semicolon  ;", ";"),
    ("Tab", "\t"),
    ("Pipe  |", "|"),
)
_ENCODINGS = (
    ("Detect automatically", AUTO),
    ("UTF-8", "utf-8"),
    ("UTF-8 with BOM", "utf-8-sig"),
    ("Latin-1 (ISO-8859-1)", "latin-1"),
    ("Windows-1252", "cp1252"),
)
_DECIMALS = (
    ("Detect automatically", AUTO),
    ("Point  1.5", "."),
    ("Comma  1,5", ","),
)
_THOUSANDS = (
    ("None", AUTO),
    ("Comma  1,000", ","),
    ("Point  1.000", "."),
    ("Space  1 000", " "),
    ("Apostrophe  1'000", "'"),
)

#: Lines of the raw file shown so the user can see the delimiter and number format.
PREVIEW_LINES = 12


class ImportDialog(QDialog):
    """Collects reader settings for one file.

    Args:
        path: The file about to be opened. Its type decides which settings are offered.
        parent: Parent widget.
    """

    def __init__(self, path: str | Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.path = Path(path)
        self.setWindowTitle(f"Open {self.path.name}")
        self.setMinimumWidth(560)
        suffix = self.path.suffix.lower()
        self.is_text = suffix in CSV_SUFFIXES
        self.is_excel = suffix in EXCEL_SUFFIXES

        layout = QVBoxLayout(self)

        self.text_box = QGroupBox("Text file", self)
        text_form = QFormLayout(self.text_box)
        self.delimiter_combo = _combo(_DELIMITERS, self.text_box)
        self.encoding_combo = _combo(_ENCODINGS, self.text_box)
        self.decimal_combo = _combo(_DECIMALS, self.text_box)
        self.thousands_combo = _combo(_THOUSANDS, self.text_box)
        self.header_spin = QSpinBox(self.text_box)
        self.header_spin.setRange(1, 1000)
        self.header_spin.setToolTip(
            "Which line holds the column names. Lines above it are skipped."
        )
        self.no_header_check = QCheckBox("The file has no header row", self.text_box)
        self.no_header_check.toggled.connect(lambda on: self.header_spin.setDisabled(on))
        text_form.addRow("Delimiter", self.delimiter_combo)
        text_form.addRow("Encoding", self.encoding_combo)
        text_form.addRow("Decimal mark", self.decimal_combo)
        text_form.addRow("Thousands separator", self.thousands_combo)
        text_form.addRow("Header on line", self.header_spin)
        text_form.addRow("", self.no_header_check)

        self.preview = QPlainTextEdit(self.text_box)
        self.preview.setReadOnly(True)
        self.preview.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.preview.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.preview.setMaximumHeight(180)
        self.preview.setPlainText(_raw_preview(self.path))
        text_form.addRow(QLabel("First lines of the file:", self.text_box))
        text_form.addRow(self.preview)
        layout.addWidget(self.text_box)

        self.excel_box = QGroupBox("Workbook", self)
        excel_form = QFormLayout(self.excel_box)
        self.sheet_combo = QComboBox(self.excel_box)
        names = _sheet_names(self.path) if self.is_excel else []
        for name in names:
            self.sheet_combo.addItem(name, name)
        if len(names) > 1:
            self.sheet_combo.addItem("All sheets, stacked (adds a 'sheet' column)", "*")
        excel_form.addRow("Sheet", self.sheet_combo)
        layout.addWidget(self.excel_box)

        self.text_box.setVisible(self.is_text)
        self.excel_box.setVisible(self.is_excel)
        if not (self.is_text or self.is_excel):
            note = QLabel("This file type has no import options; it will open as usual.", self)
            note.setWordWrap(True)
            layout.addWidget(note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Open | QDialogButtonBox.StandardButton.Cancel, self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def options(self) -> dict[str, Any]:
        """The chosen settings, leaving out everything left on "detect"."""
        chosen: dict[str, Any] = {}
        if self.is_text:
            for key, combo in (
                ("sep", self.delimiter_combo),
                ("encoding", self.encoding_combo),
                ("decimal", self.decimal_combo),
                ("thousands", self.thousands_combo),
            ):
                value = combo.currentData()
                if value is not AUTO:
                    chosen[key] = value
            if self.no_header_check.isChecked():
                chosen["header"] = None
            elif self.header_spin.value() > 1:
                chosen["header"] = self.header_spin.value() - 1
        if self.is_excel and self.sheet_combo.count() and self.sheet_combo.currentIndex() > 0:
            chosen["sheet"] = self.sheet_combo.currentData()
        return chosen


def _combo(items: tuple[tuple[str, Any], ...], parent: QWidget) -> QComboBox:
    combo = QComboBox(parent)
    for label, value in items:
        combo.addItem(label, value)
    return combo


def _raw_preview(path: Path) -> str:
    """The first lines of a text file, decoded leniently — enough to see its format."""
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return "".join(handle.readline() for _ in range(PREVIEW_LINES))
    except OSError as exc:
        return f"(could not read the file: {exc})"


def _sheet_names(path: Path) -> list[str]:
    """Sheet names of a workbook, or an empty list if it cannot be opened."""
    try:
        import pandas as pd

        with pd.ExcelFile(path) as book:
            return [str(name) for name in book.sheet_names]
    except Exception:
        return []
