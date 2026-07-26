"""The export dialog — image formats, size, DPI, or the generated Python code.

Exporting code sits in the same dialog as exporting an image on purpose. Reproducibility is a
headline feature, not a hidden menu item, and putting the two side by side makes "you can take
the script with you" discoverable.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..core.exporter import DEFAULT_DPI, ExportOptions

#: Image formats offered, with a note on when each is the right choice.
_FORMATS = (
    ("png", "PNG — raster, good for slides and documents"),
    ("svg", "SVG — vector, scales without blurring"),
    ("pdf", "PDF — vector, best for print and LaTeX"),
    ("jpg", "JPEG — smaller files, lossy"),
    ("webp", "WebP — small raster files for the web"),
)


class ExportDialog(QDialog):
    """Collects export settings.

    Args:
        parent: Parent widget.
        default_name: Suggested file stem, usually derived from the dataset name.
        start_dir: Directory the file picker opens in.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        default_name: str = "chart",
        start_dir: str | Path = ".",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export")
        self.setMinimumWidth(520)
        self._start_dir = Path(start_dir)
        self._default_name = default_name

        layout = QVBoxLayout(self)

        # --- what to export
        what = QGroupBox("What to export", self)
        what_layout = QVBoxLayout(what)
        self.image_radio = QRadioButton("Image of the chart", what)
        self.image_radio.setChecked(True)
        self.code_radio = QRadioButton("Python code that reproduces it", what)
        self.html_radio = QRadioButton("Interactive HTML page", what)
        for radio in (self.image_radio, self.code_radio, self.html_radio):
            radio.toggled.connect(self._sync_enabled)
            what_layout.addWidget(radio)
        code_note = QLabel(
            "The generated script is standalone — it reads your data file, replays the exact "
            "cleaning steps, and draws the chart. It does not import PlotaViz.",
            what,
        )
        code_note.setWordWrap(True)
        code_note.setStyleSheet("color: palette(mid); font-size: 11px;")
        what_layout.addWidget(code_note)
        layout.addWidget(what)

        # --- image settings
        self.image_box = QGroupBox("Image settings", self)
        image_form = QFormLayout(self.image_box)

        self.format_combo = QComboBox(self.image_box)
        for value, label in _FORMATS:
            self.format_combo.addItem(label, value)
        self.format_combo.currentIndexChanged.connect(self._sync_enabled)
        image_form.addRow("Format", self.format_combo)

        self.dpi_spin = QSpinBox(self.image_box)
        self.dpi_spin.setRange(72, 1200)
        self.dpi_spin.setSingleStep(50)
        self.dpi_spin.setValue(DEFAULT_DPI)
        self.dpi_spin.setSuffix(" dpi")
        image_form.addRow("Resolution", self.dpi_spin)

        size_row = QHBoxLayout()
        self.width_spin = QDoubleSpinBox(self.image_box)
        self.width_spin.setRange(2.0, 40.0)
        self.width_spin.setValue(10.0)
        self.width_spin.setSuffix(" in")
        self.height_spin = QDoubleSpinBox(self.image_box)
        self.height_spin.setRange(2.0, 40.0)
        self.height_spin.setValue(6.0)
        self.height_spin.setSuffix(" in")
        size_row.addWidget(self.width_spin)
        size_row.addWidget(QLabel("×", self.image_box))
        size_row.addWidget(self.height_spin)
        size_row.addStretch(1)
        image_form.addRow("Size", size_row)

        self.transparent_check = QCheckBox("Transparent background", self.image_box)
        image_form.addRow("", self.transparent_check)

        self.notice_check = QCheckBox(
            "Include the sampling notice when data was downsampled", self.image_box
        )
        self.notice_check.setChecked(True)
        self.notice_check.setToolTip(
            "An exported chart that hides its own sampling misleads whoever reads it. "
            "Leave this on unless you have a reason not to."
        )
        image_form.addRow("", self.notice_check)
        layout.addWidget(self.image_box)

        # --- code settings
        self.code_box = QGroupBox("Code settings", self)
        code_form = QFormLayout(self.code_box)
        self.flavour_combo = QComboBox(self.code_box)
        self.flavour_combo.addItem("matplotlib — static figure, best for reports", "matplotlib")
        self.flavour_combo.addItem("Plotly — interactive figure, best for notebooks", "plotly")
        code_form.addRow("Library", self.flavour_combo)
        layout.addWidget(self.code_box)

        # --- destination
        destination = QGroupBox("Save to", self)
        dest_layout = QHBoxLayout(destination)
        self.path_edit = QLineEdit(destination)
        browse = QPushButton("Browse…", destination)
        browse.clicked.connect(self._browse)
        dest_layout.addWidget(self.path_edit, stretch=1)
        dest_layout.addWidget(browse)
        layout.addWidget(destination)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel, self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._sync_enabled()

    # ------------------------------------------------------------------ behaviour

    @property
    def mode(self) -> str:
        """``"image"``, ``"code"``, or ``"html"``."""
        if self.code_radio.isChecked():
            return "code"
        if self.html_radio.isChecked():
            return "html"
        return "image"

    def _suggested_suffix(self) -> str:
        """The extension implied by the current mode and format."""
        if self.mode == "code":
            return ".py"
        if self.mode == "html":
            return ".html"
        return "." + str(self.format_combo.currentData())

    def _sync_enabled(self, *_: object) -> None:
        """Enable only the settings that apply to the selected mode, and fix the extension."""
        is_image = self.mode == "image"
        self.image_box.setEnabled(is_image)
        self.code_box.setEnabled(self.mode == "code")

        raster = str(self.format_combo.currentData()) in {"png", "jpg", "jpeg", "webp", "tiff"}
        self.dpi_spin.setEnabled(is_image and raster)

        current = self.path_edit.text().strip()
        stem = Path(current).stem if current else self._default_name
        parent = Path(current).parent if current else self._start_dir
        self.path_edit.setText(str(Path(parent) / f"{stem}{self._suggested_suffix()}"))

    def _browse(self) -> None:
        """Open a save dialog seeded with the current path."""
        suffix = self._suggested_suffix()
        chosen, _ = QFileDialog.getSaveFileName(
            self,
            "Export",
            self.path_edit.text() or str(self._start_dir / f"{self._default_name}{suffix}"),
            f"*{suffix}",
        )
        if chosen:
            self.path_edit.setText(chosen)

    # ------------------------------------------------------------------ results

    def path(self) -> Path:
        """The destination the user chose, with the right extension enforced."""
        raw = self.path_edit.text().strip() or f"{self._default_name}{self._suggested_suffix()}"
        target = Path(raw).expanduser()
        if target.suffix.lower() != self._suggested_suffix():
            target = target.with_suffix(self._suggested_suffix())
        return target

    def options(self) -> ExportOptions:
        """Image settings as an :class:`~plotaviz.core.exporter.ExportOptions`."""
        return ExportOptions(
            dpi=self.dpi_spin.value(),
            width=self.width_spin.value(),
            height=self.height_spin.value(),
            transparent=self.transparent_check.isChecked(),
            include_notice=self.notice_check.isChecked(),
        )

    def flavour(self) -> str:
        """``"matplotlib"`` or ``"plotly"`` for code export."""
        return str(self.flavour_combo.currentData())
