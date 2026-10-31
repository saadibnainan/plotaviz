"""Chart customization — title, axis labels, scales, category limit, sort, bins, palette.

Everything here is stored on the :class:`~plotaviz.core.spec.ChartSpec` (``title`` and
``options``), which is the one contract every renderer, the code generator, and the session file
read. So a customized chart looks the same on screen, in an exported image, in the generated
script, and after reopening a session — with no extra plumbing.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QWidget,
)

from ..core.spec import ChartSpec

_PALETTES = (
    ("Default", ""),
    ("Colour-blind safe (Okabe–Ito)", "colorblind"),
)

#: Option keys this panel owns. Others in ``spec.options`` (trendline, pinned columns…) are kept.
OWNED_OPTIONS = (
    "x_label",
    "y_label",
    "log_x",
    "log_y",
    "top_n",
    "sort_ascending",
    "bins",
    "palette",
)


class CustomizePanel(QGroupBox):
    """Edits the presentation of the current chart.

    Signals:
        customized: Emitted with ``{"title": str | None, "options": dict}`` — the title to set
            and the full replacement for the options this panel owns (absent keys are cleared).
    """

    customized = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Customize", parent)
        self.setCheckable(True)
        self.setChecked(False)  # collapsed by default; ticking the title expands it
        form = QFormLayout(self)

        self.title_edit = QLineEdit(self)
        self.x_label_edit = QLineEdit(self)
        self.y_label_edit = QLineEdit(self)
        self.log_x_check = QCheckBox("Log x", self)
        self.log_y_check = QCheckBox("Log y", self)
        scales = QHBoxLayout()
        scales.addWidget(self.log_x_check)
        scales.addWidget(self.log_y_check)
        scales.addStretch(1)

        self.top_n_spin = QSpinBox(self)
        self.top_n_spin.setRange(0, 500)
        self.top_n_spin.setSpecialValueText("Automatic (30)")
        self.top_n_spin.setToolTip("Show this many categories; the rest are grouped as 'Other'.")
        self.sort_check = QCheckBox("Smallest first", self)
        self.bins_spin = QSpinBox(self)
        self.bins_spin.setRange(0, 500)
        self.bins_spin.setSpecialValueText("Automatic")
        self.palette_combo = QComboBox(self)
        for label, value in _PALETTES:
            self.palette_combo.addItem(label, value)

        form.addRow("Title", self.title_edit)
        form.addRow("X label", self.x_label_edit)
        form.addRow("Y label", self.y_label_edit)
        form.addRow("Scales", scales)
        form.addRow("Categories", self.top_n_spin)
        form.addRow("Bar order", self.sort_check)
        form.addRow("Histogram bins", self.bins_spin)
        form.addRow("Colours", self.palette_combo)

        self.apply_button = QPushButton("Apply", self)
        self.apply_button.clicked.connect(self._emit)
        form.addRow("", self.apply_button)

        self.toggled.connect(self._show_fields)
        self._show_fields(False)

    def _show_fields(self, expanded: bool) -> None:
        """Collapse to just the title bar when unticked."""
        for child in self.findChildren(QWidget):
            child.setVisible(expanded)
        self.setFlat(not expanded)  # no empty frame under the title when collapsed
        self.setMaximumHeight(16_777_215 if expanded else self.fontMetrics().height() * 2)

    def set_spec(self, spec: ChartSpec | None) -> None:
        """Show the customization of the chart on screen."""
        opts = spec.options if spec else {}
        self.title_edit.setText(spec.title or "" if spec else "")
        self.title_edit.setPlaceholderText(spec.display_title() if spec else "")
        self.x_label_edit.setText(str(opts.get("x_label") or ""))
        self.x_label_edit.setPlaceholderText(str(spec.x or "") if spec else "")
        self.y_label_edit.setText(str(opts.get("y_label") or ""))
        self.y_label_edit.setPlaceholderText(str(spec.y or "") if spec else "")
        self.log_x_check.setChecked(bool(opts.get("log_x")))
        self.log_y_check.setChecked(bool(opts.get("log_y")))
        self.top_n_spin.setValue(int(opts.get("top_n") or 0))
        self.sort_check.setChecked(bool(opts.get("sort_ascending")))
        self.bins_spin.setValue(int(opts.get("bins") or 0))
        index = self.palette_combo.findData(str(opts.get("palette") or ""))
        self.palette_combo.setCurrentIndex(max(0, index))

    def values(self) -> dict[str, Any]:
        """The current settings as the payload of :attr:`customized`."""
        options: dict[str, Any] = {}
        for key, edit in (("x_label", self.x_label_edit), ("y_label", self.y_label_edit)):
            if edit.text().strip():
                options[key] = edit.text().strip()
        if self.log_x_check.isChecked():
            options["log_x"] = True
        if self.log_y_check.isChecked():
            options["log_y"] = True
        if self.top_n_spin.value():
            options["top_n"] = self.top_n_spin.value()
        if self.sort_check.isChecked():
            options["sort_ascending"] = True
        if self.bins_spin.value():
            options["bins"] = self.bins_spin.value()
        if self.palette_combo.currentData():
            options["palette"] = self.palette_combo.currentData()
        return {"title": self.title_edit.text().strip() or None, "options": options}

    def _emit(self) -> None:
        self.customized.emit(self.values())


def apply_customization(spec: ChartSpec, values: dict[str, Any]) -> ChartSpec:
    """A copy of ``spec`` with the panel's title and options applied."""
    kept = {k: v for k, v in spec.options.items() if k not in OWNED_OPTIONS}
    return spec.copy(title=values.get("title"), options={**kept, **values.get("options", {})})
