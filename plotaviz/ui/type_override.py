"""The column type override panel.

Automatic type inference gets identifiers and dates wrong constantly — a zip code becomes a
number, an order ID becomes a measure, a date column stays text — and every wrong type produces a
wrong chart. Letting the user correct the inference *before* analysis proceeds is the single
biggest frustration-preventer in the app, which is why this panel is a first-class part of the
flow rather than a settings page.

Corrections are cheap because the preprocessing pipeline is replayable: changing a type re-runs a
short list of pure steps rather than reloading anything.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.profiler import ROLES, DatasetProfile

#: Roles offered in the dropdown, with the labels users actually recognise.
_ROLE_LABELS = {
    "numeric": "Number",
    "categorical": "Category",
    "datetime": "Date / time",
    "boolean": "True / false",
    "text": "Free text",
}


class TypeOverridePanel(QWidget):
    """Shows every column's inferred type and lets the user correct it.

    Signals:
        types_changed: Emitted with ``{column: role}`` when the user applies corrections.
    """

    types_changed = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._profile: DatasetProfile | None = None
        self._combos: dict[str, QComboBox] = {}
        self._inferred: dict[str, str] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        intro = QLabel(
            "PlotaViz guessed these types. Correct any it got wrong — IDs read as numbers and "
            "dates read as text are the usual culprits."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: palette(mid);")
        layout.addWidget(intro)

        self.table = QTableWidget(0, 4, self)
        self.table.setHorizontalHeaderLabels(["Column", "Detected as", "Treat as", "Notes"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table, stretch=1)

        buttons = QHBoxLayout()
        self.reset_button = QPushButton("Reset to detected", self)
        self.reset_button.clicked.connect(self.reset)
        self.apply_button = QPushButton("Apply types", self)
        self.apply_button.setDefault(True)
        self.apply_button.clicked.connect(self._emit_changes)
        buttons.addWidget(self.reset_button)
        buttons.addStretch(1)
        buttons.addWidget(self.apply_button)
        layout.addLayout(buttons)

    # ------------------------------------------------------------------ population

    def set_profile(self, profile: DatasetProfile, overrides: dict[str, str] | None = None) -> None:
        """Rebuild the panel for a dataset.

        Args:
            profile: The current profile, whose inferred roles seed the dropdowns.
            overrides: Corrections already in force, so reopening the panel shows them.
        """
        self._profile = profile
        overrides = overrides or {}
        self._combos.clear()
        self._inferred.clear()

        columns = list(profile.columns.values())
        self.table.setRowCount(len(columns))

        for row, prof in enumerate(columns):
            self._inferred[prof.name] = prof.role

            name_item = QTableWidgetItem(prof.name)
            name_item.setToolTip(
                f"{prof.n_unique:,} distinct values, {prof.pct_missing:.1f}% missing"
            )
            self.table.setItem(row, 0, name_item)

            detected = QTableWidgetItem(_ROLE_LABELS.get(prof.role, prof.role))
            detected.setForeground(Qt.GlobalColor.gray)
            self.table.setItem(row, 1, detected)

            combo = QComboBox(self)
            for role in ROLES:
                combo.addItem(_ROLE_LABELS.get(role, role), role)
            current = overrides.get(prof.name, prof.role)
            index = combo.findData(current)
            combo.setCurrentIndex(max(0, index))
            combo.setEnabled(not prof.is_derived)
            self._combos[prof.name] = combo
            self.table.setCellWidget(row, 2, combo)

            notes = []
            if prof.note:
                notes.append(prof.note)
            if prof.is_identifier:
                notes.append("Excluded from chart suggestions.")
            if prof.is_derived:
                notes.append("Added by PlotaViz.")
            note_item = QTableWidgetItem(" ".join(notes))
            note_item.setToolTip(" ".join(notes))
            self.table.setItem(row, 3, note_item)

    # ------------------------------------------------------------------ interaction

    def overrides(self) -> dict[str, str]:
        """The corrections the user has made — only entries that differ from inference."""
        return {
            name: combo.currentData()
            for name, combo in self._combos.items()
            if combo.currentData() != self._inferred.get(name)
        }

    def reset(self) -> None:
        """Put every dropdown back to what inference chose, and apply."""
        for name, combo in self._combos.items():
            index = combo.findData(self._inferred.get(name))
            combo.setCurrentIndex(max(0, index))
        self._emit_changes()

    def _emit_changes(self) -> None:
        """Emit the current corrections."""
        self.types_changed.emit(self.overrides())
