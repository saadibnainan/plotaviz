"""The cleaning panel — how rows are cleaned, and the full step list with removal.

Cleaning used to be fixed at load time: median imputation, outlier flags, deduplication, with no
way to change any of it from the window. This panel exposes those choices as settings and shows
every step the pipeline runs, with what it did, so a step can be removed and the effect seen
immediately. Every change goes through :class:`~plotaviz.core.analysis.Analysis`, so it is
undoable, saved in sessions, and replayed in generated code.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core.preprocess import MISSING_CHOICES, OUTLIER_ACTIONS, OUTLIER_METHODS, Step, StepReport

_MISSING_LABELS = {
    "keep": "Leave them missing (recommended)",
    "drop": "Drop rows that have any",
    "mean": "Fill numbers with the column mean",
    "median": "Fill numbers with the column median",
    "mode": "Fill with the most frequent value",
    "ffill": "Carry the previous value forward",
    "bfill": "Carry the next value backward",
    "zero": "Fill numbers with 0",
}
_OUTLIER_LABELS = {
    "off": "Leave them alone",
    "flag": "Flag them (adds <column>__outlier)",
    "drop": "Remove those rows",
}
_METHOD_LABELS = {"iqr": "IQR fences (Tukey)", "zscore": "z-score"}


class CleaningPanel(QWidget):
    """Cleaning settings plus the editable step list.

    Signals:
        cleaning_changed: Emitted with :meth:`Analysis.set_cleaning` keyword arguments.
        remove_requested: Emitted with the index of the step to delete.
    """

    cleaning_changed = Signal(dict)
    remove_requested = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        settings_box = QGroupBox("How to clean", self)
        form = QFormLayout(settings_box)
        self.missing_combo = QComboBox(settings_box)
        for choice in MISSING_CHOICES:
            self.missing_combo.addItem(_MISSING_LABELS.get(choice, choice), choice)
        self.outliers_combo = QComboBox(settings_box)
        for action in OUTLIER_ACTIONS:
            self.outliers_combo.addItem(_OUTLIER_LABELS.get(action, action), action)
        self.method_combo = QComboBox(settings_box)
        for method in OUTLIER_METHODS:
            self.method_combo.addItem(_METHOD_LABELS.get(method, method), method)
        self.threshold_spin = QDoubleSpinBox(settings_box)
        self.threshold_spin.setRange(0.5, 10.0)
        self.threshold_spin.setSingleStep(0.5)
        self.threshold_spin.setValue(1.5)
        self.threshold_spin.setToolTip("IQR multiplier (1.5 is Tukey's), or the z-score cutoff.")
        self.dedupe_check = QCheckBox("Remove fully duplicated rows", settings_box)
        self.outliers_combo.currentIndexChanged.connect(self._sync_enabled)

        form.addRow("Missing values", self.missing_combo)
        form.addRow("Outliers", self.outliers_combo)
        form.addRow("Detect with", self.method_combo)
        form.addRow("Threshold", self.threshold_spin)
        form.addRow("", self.dedupe_check)
        self.apply_button = QPushButton("Apply cleaning", settings_box)
        self.apply_button.clicked.connect(self._emit_cleaning)
        form.addRow("", self.apply_button)
        layout.addWidget(settings_box)

        steps_box = QGroupBox("Steps, in the order they run", self)
        steps_layout = QVBoxLayout(steps_box)
        self.steps_list = QListWidget(steps_box)
        self.steps_list.setWordWrap(True)
        self.steps_list.currentRowChanged.connect(self._sync_enabled)
        steps_layout.addWidget(self.steps_list, stretch=1)
        row = QHBoxLayout()
        self.remove_button = QPushButton("Remove step", steps_box)
        self.remove_button.clicked.connect(self._emit_remove)
        row.addWidget(self.remove_button)
        row.addStretch(1)
        hint = QLabel("Edit → Undo reverts any change.", steps_box)
        hint.setStyleSheet("color: palette(mid); font-size: 11px;")
        row.addWidget(hint)
        steps_layout.addLayout(row)
        layout.addWidget(steps_box, stretch=1)

        self._removable: list[bool] = []
        self._sync_enabled()

    # ------------------------------------------------------------------ state

    def set_state(
        self, cleaning: dict[str, Any], steps: list[Step], reports: list[StepReport]
    ) -> None:
        """Show the current settings and step list without emitting anything."""
        _select(self.missing_combo, cleaning.get("missing", "keep"))
        _select(self.outliers_combo, cleaning.get("outliers", "off"))
        _select(self.method_combo, cleaning.get("outlier_method", "iqr"))
        self.threshold_spin.setValue(float(cleaning.get("outlier_threshold", 1.5)))
        self.dedupe_check.setChecked(bool(cleaning.get("dedupe", True)))

        self.steps_list.clear()
        self._removable = []
        for index, step in enumerate(steps):
            summary = reports[index].summary if index < len(reports) else ""
            item = QListWidgetItem(f"{index + 1}. {step.label}\n{summary}".rstrip())
            item.setToolTip(repr(step))
            self.steps_list.addItem(item)
            # Renaming is what every other step refers to; the core refuses to remove it.
            self._removable.append(step.kind != "normalize_columns")
        self._sync_enabled()

    def settings(self) -> dict[str, Any]:
        """The chosen settings as :meth:`Analysis.set_cleaning` keyword arguments."""
        return {
            "missing": self.missing_combo.currentData(),
            "outliers": self.outliers_combo.currentData(),
            "outlier_method": self.method_combo.currentData(),
            "outlier_threshold": self.threshold_spin.value(),
            "dedupe": self.dedupe_check.isChecked(),
        }

    # ------------------------------------------------------------------ signals

    def _sync_enabled(self, *_: object) -> None:
        detecting = self.outliers_combo.currentData() != "off"
        self.method_combo.setEnabled(detecting)
        self.threshold_spin.setEnabled(detecting)
        row = self.steps_list.currentRow()
        self.remove_button.setEnabled(0 <= row < len(self._removable) and self._removable[row])

    def _emit_cleaning(self) -> None:
        self.cleaning_changed.emit(self.settings())

    def _emit_remove(self) -> None:
        row = self.steps_list.currentRow()
        if 0 <= row < len(self._removable) and self._removable[row]:
            self.remove_requested.emit(row)


def _select(combo: QComboBox, value: Any) -> None:
    index = combo.findData(value)
    if index >= 0:
        combo.setCurrentIndex(index)
