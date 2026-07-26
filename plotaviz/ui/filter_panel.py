"""The filter panel — per-column widgets plus a raw pandas-query bar.

Filters are :class:`~plotaviz.core.preprocess.Step` objects, exactly like the cleaning steps.
That is not a detail: because a filter *is* a pipeline step, it serializes into sessions and
appears in generated code for free, with no separate code path and no chance of the exported
script disagreeing with what is on screen.

Re-rendering is debounced. Dragging a range slider fires continuously, and re-running a pipeline
plus a chart on every pixel is how an app earns a reputation for being slow.
"""

from __future__ import annotations

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDateEdit,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..core.preprocess import ColumnFilter, QueryFilter, Step
from ..core.profiler import BOOLEAN, CATEGORICAL, DATETIME, NUMERIC, DatasetProfile

#: Milliseconds of quiet before a filter change triggers a re-render.
DEBOUNCE_MS = 350

#: Categorical columns with more distinct values than this get a search box instead of a list.
MAX_LISTED_CATEGORIES = 200


class FilterPanel(QWidget):
    """Builds filter widgets from a dataset profile.

    Signals:
        filters_changed: Emitted with a list of filter steps after the debounce interval.
    """

    filters_changed = Signal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._profile: DatasetProfile | None = None
        self._numeric: dict[str, tuple[QDoubleSpinBox, QDoubleSpinBox]] = {}
        self._categorical: dict[str, QListWidget] = {}
        self._datetime: dict[str, tuple[QDateEdit, QDateEdit]] = {}
        self._boolean: dict[str, QCheckBox] = {}

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(DEBOUNCE_MS)
        self._debounce.timeout.connect(self._emit_filters)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)

        # --- free-text pandas query
        query_box = QGroupBox("Query", self)
        query_layout = QVBoxLayout(query_box)
        self.query_edit = QLineEdit(self)
        self.query_edit.setPlaceholderText("revenue > 1000 and region == 'EMEA'")
        self.query_edit.setClearButtonEnabled(True)
        self.query_edit.returnPressed.connect(self._emit_filters)
        query_layout.addWidget(self.query_edit)
        hint = QLabel("pandas query syntax. Press Enter to apply.", self)
        hint.setStyleSheet("color: palette(mid); font-size: 11px;")
        query_layout.addWidget(hint)
        self.query_error = QLabel("", self)
        self.query_error.setWordWrap(True)
        self.query_error.setStyleSheet("color: #C0392B; font-size: 11px;")
        self.query_error.hide()
        query_layout.addWidget(self.query_error)
        outer.addWidget(query_box)

        # --- per-column widgets, scrollable
        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._container = QWidget()
        self._container_layout = QVBoxLayout(self._container)
        self._container_layout.setContentsMargins(0, 0, 0, 0)
        self._container_layout.addStretch(1)
        self._scroll.setWidget(self._container)
        outer.addWidget(self._scroll, stretch=1)

        buttons = QHBoxLayout()
        self.clear_button = QPushButton("Clear all filters", self)
        self.clear_button.clicked.connect(self.clear)
        buttons.addWidget(self.clear_button)
        buttons.addStretch(1)
        outer.addLayout(buttons)

    # ------------------------------------------------------------------ population

    def set_profile(self, profile: DatasetProfile) -> None:
        """Rebuild the per-column widgets for a dataset."""
        self._profile = profile
        self._numeric.clear()
        self._categorical.clear()
        self._datetime.clear()
        self._boolean.clear()

        while self._container_layout.count():
            item = self._container_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        for name, prof in profile.columns.items():
            if prof.is_derived or prof.is_identifier:
                continue
            if prof.role == NUMERIC:
                self._container_layout.addWidget(self._numeric_widget(name, prof))
            elif prof.role == DATETIME:
                self._container_layout.addWidget(self._datetime_widget(name, prof))
            elif prof.role == BOOLEAN:
                self._container_layout.addWidget(self._boolean_widget(name))
            elif prof.role == CATEGORICAL and prof.n_unique <= MAX_LISTED_CATEGORIES:
                self._container_layout.addWidget(self._categorical_widget(name, prof))

        self._container_layout.addStretch(1)

    def _numeric_widget(self, name: str, prof: object) -> QWidget:
        """A low/high pair bounded by the column's actual range."""
        stats = getattr(prof, "stats", {}) or {}
        low_value = float(stats.get("min", 0.0))
        high_value = float(stats.get("max", 0.0))
        span = high_value - low_value or 1.0

        box = QGroupBox(name, self)
        form = QFormLayout(box)
        low = QDoubleSpinBox(box)
        high = QDoubleSpinBox(box)
        for spin in (low, high):
            spin.setRange(low_value - span, high_value + span)
            spin.setDecimals(4 if span < 10 else 2)
            spin.setSingleStep(span / 100.0)
            spin.valueChanged.connect(self._schedule)
        low.setValue(low_value)
        high.setValue(high_value)
        form.addRow("From", low)
        form.addRow("To", high)
        self._numeric[name] = (low, high)
        return box

    def _datetime_widget(self, name: str, prof: object) -> QWidget:
        """A start/end date pair."""
        box = QGroupBox(name, self)
        form = QFormLayout(box)
        start, end = QDateEdit(box), QDateEdit(box)
        for edit in (start, end):
            edit.setCalendarPopup(True)
            edit.setDisplayFormat("yyyy-MM-dd")
            edit.dateChanged.connect(self._schedule)
        form.addRow("From", start)
        form.addRow("To", end)
        self._datetime[name] = (start, end)
        return box

    def _boolean_widget(self, name: str) -> QWidget:
        """A single checkbox meaning "only rows where this is true"."""
        box = QGroupBox(name, self)
        layout = QVBoxLayout(box)
        check = QCheckBox("Only true", box)
        check.stateChanged.connect(self._schedule)
        layout.addWidget(check)
        self._boolean[name] = check
        return box

    def _categorical_widget(self, name: str, prof: object) -> QWidget:
        """A multi-select list of the column's values, everything selected by default."""
        box = QGroupBox(name, self)
        layout = QVBoxLayout(box)
        listing = QListWidget(box)
        listing.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        listing.setMaximumHeight(140)
        for value, count in getattr(prof, "top_values", []) or []:
            item = QListWidgetItem(f"{value}  ({count:,})")
            item.setData(0x0100, value)  # Qt.UserRole
            listing.addItem(item)
            item.setSelected(True)
        listing.itemSelectionChanged.connect(self._schedule)
        layout.addWidget(listing)
        self._categorical[name] = listing
        return box

    # ------------------------------------------------------------------ filter building

    def filters(self) -> list[Step]:
        """Build the current filter steps.

        Widgets left at their full range contribute nothing, so an untouched panel produces an
        empty list rather than a pile of no-op steps.
        """
        steps: list[Step] = []
        profile = self._profile

        expression = self.query_edit.text().strip()
        if expression:
            steps.append(QueryFilter(expression))

        for name, (low, high) in self._numeric.items():
            prof = profile.columns.get(name) if profile else None
            stats = getattr(prof, "stats", {}) or {}
            full_low, full_high = float(stats.get("min", 0.0)), float(stats.get("max", 0.0))
            if low.value() > full_low or high.value() < full_high:
                steps.append(ColumnFilter(name, "between", [low.value(), high.value()]))

        for name, (start, end) in self._datetime.items():
            if start.date() != start.minimumDate() or end.date() != end.maximumDate():
                steps.append(
                    ColumnFilter(
                        name,
                        "between",
                        [start.date().toString("yyyy-MM-dd"), end.date().toString("yyyy-MM-dd")],
                    )
                )

        for name, listing in self._categorical.items():
            selected = [item.data(0x0100) for item in listing.selectedItems()]
            if selected and len(selected) < listing.count():
                steps.append(ColumnFilter(name, "in", selected))

        for name, check in self._boolean.items():
            if check.isChecked():
                steps.append(ColumnFilter(name, "in", [True]))

        return steps

    def clear(self) -> None:
        """Reset every widget and emit an empty filter set."""
        self.query_edit.clear()
        self.query_error.hide()
        for name, (low, high) in self._numeric.items():
            prof = self._profile.columns.get(name) if self._profile else None
            stats = getattr(prof, "stats", {}) or {}
            low.setValue(float(stats.get("min", low.minimum())))
            high.setValue(float(stats.get("max", high.maximum())))
        for listing in self._categorical.values():
            listing.selectAll()
        for check in self._boolean.values():
            check.setChecked(False)
        self._emit_filters()

    def show_query_error(self, message: str) -> None:
        """Display a failed query message under the query bar."""
        self.query_error.setText(message)
        self.query_error.show()

    def clear_query_error(self) -> None:
        """Hide the query error."""
        self.query_error.hide()

    # ------------------------------------------------------------------ signalling

    def _schedule(self, *_: object) -> None:
        """Restart the debounce timer. Called by every widget's change signal."""
        self._debounce.start()

    def _emit_filters(self) -> None:
        """Emit the current filter steps."""
        self._debounce.stop()
        self.clear_query_error()
        self.filters_changed.emit(self.filters())
