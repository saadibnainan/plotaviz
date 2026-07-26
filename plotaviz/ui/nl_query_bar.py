"""The natural-language query bar.

Type "revenue by region over time", get a chart. The important design decision is what happens
in between: the model returns a **chart spec**, PlotaViz validates it against the real schema,
and then *shows the user the spec it interpreted* before drawing. The mapping is never opaque,
and it stays editable — if the model picked the wrong column, that is one dropdown away from
fixed rather than a reason to give up on the feature.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core.spec import AGGREGATIONS, CHART_TYPES, ChartSpec

#: Example prompts shown as placeholder text, rotated so the bar suggests what it can do.
EXAMPLES = (
    "revenue by region over time",
    "how does price relate to size?",
    "distribution of response times",
    "average score per department",
)


class NLQueryBar(QWidget):
    """Free-text input plus an editable view of the interpreted chart spec.

    Signals:
        question_asked: Emitted with the user's text when they submit a query.
        spec_edited: Emitted with a :class:`ChartSpec` when the user tweaks the interpretation.
        settings_requested: Emitted when the user clicks through to configure a provider.
    """

    question_asked = Signal(str)
    spec_edited = Signal(object)
    settings_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._columns: list[str] = []
        self._spec: ChartSpec | None = None
        self._updating = False
        self._configured = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        # --- the question row
        row = QHBoxLayout()
        self.input = QLineEdit(self)
        self.input.setPlaceholderText(f"Ask for a chart — e.g. {EXAMPLES[0]}")
        self.input.setClearButtonEnabled(True)
        self.input.returnPressed.connect(self._submit)
        self.ask_button = QPushButton("Ask", self)
        self.ask_button.clicked.connect(self._submit)
        row.addWidget(self.input, stretch=1)
        row.addWidget(self.ask_button)
        layout.addLayout(row)

        self.busy = QProgressBar(self)
        self.busy.setRange(0, 0)
        self.busy.setTextVisible(False)
        self.busy.setMaximumHeight(3)
        self.busy.hide()
        layout.addWidget(self.busy)

        self.status = QLabel("", self)
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: palette(mid); font-size: 11px;")
        self.status.hide()
        layout.addWidget(self.status)

        # --- the interpreted spec, editable
        self.interpretation = QFrame(self)
        self.interpretation.setFrameShape(QFrame.Shape.StyledPanel)
        interp_layout = QHBoxLayout(self.interpretation)
        interp_layout.setContentsMargins(8, 4, 8, 4)

        caption = QLabel("Interpreted as", self.interpretation)
        caption.setStyleSheet("color: palette(mid); font-size: 11px;")
        interp_layout.addWidget(caption)

        self.chart_combo = self._combo(CHART_TYPES)
        self.x_combo = self._combo([])
        self.y_combo = self._combo([])
        self.color_combo = self._combo([])
        self.agg_combo = self._combo(("none", *AGGREGATIONS))

        for label, combo in (
            ("chart", self.chart_combo),
            ("x", self.x_combo),
            ("y", self.y_combo),
            ("colour", self.color_combo),
            ("aggregate", self.agg_combo),
        ):
            tag = QLabel(label, self.interpretation)
            tag.setStyleSheet("color: palette(mid); font-size: 11px;")
            interp_layout.addWidget(tag)
            interp_layout.addWidget(combo)

        interp_layout.addStretch(1)
        self.interpretation.hide()
        layout.addWidget(self.interpretation)

    def _combo(self, values: object) -> QComboBox:
        """Build a combo box wired to emit an edited spec."""
        combo = QComboBox(self)
        combo.addItems([str(v) for v in values])  # type: ignore[arg-type]
        combo.currentIndexChanged.connect(self._on_edited)
        return combo

    # ------------------------------------------------------------------ state

    def set_columns(self, columns: list[str]) -> None:
        """Populate the column dropdowns for the current dataset."""
        self._columns = list(columns)
        self._updating = True
        for combo in (self.x_combo, self.y_combo, self.color_combo):
            combo.clear()
            combo.addItem("none")
            combo.addItems(self._columns)
        self._updating = False

    def set_enabled_for_provider(self, configured: bool, provider_name: str = "") -> None:
        """Enable or disable the bar depending on whether an LLM provider is set up.

        Disabled is the honest state when there is no provider — the rest of PlotaViz works
        offline, but turning free text into a chart genuinely needs a model.
        """
        self._configured = configured
        self.input.setEnabled(configured)
        self.ask_button.setEnabled(configured)
        if configured:
            provider = f" via {provider_name}" if provider_name else ""
            self.input.setPlaceholderText(f"Ask for a chart{provider} — e.g. {EXAMPLES[0]}")
            self.status.hide()
        else:
            self.input.setPlaceholderText("Set up an LLM provider to ask for charts in words")
            self.show_status(
                "No LLM provider is configured, so the query bar is off. Chart recommendations "
                "work without one. Configure a provider in Settings, or install Ollama to keep "
                "everything on this machine."
            )

    def set_busy(self, busy: bool) -> None:
        """Show or hide the indeterminate progress strip and lock the input while waiting."""
        self.busy.setVisible(busy)
        enabled = self._configured and not busy
        self.input.setEnabled(enabled)
        self.ask_button.setEnabled(enabled)

    def show_status(self, message: str, *, error: bool = False) -> None:
        """Show a status or error line under the input."""
        self.status.setText(message)
        self.status.setStyleSheet(
            "color: #C0392B; font-size: 11px;" if error else "color: palette(mid); font-size: 11px;"
        )
        self.status.show()

    def show_spec(self, spec: ChartSpec) -> None:
        """Display the interpreted spec so the mapping is visible and editable."""
        self._spec = spec
        self._updating = True

        self.chart_combo.setCurrentText(spec.chart)
        self.x_combo.setCurrentText(spec.x or "none")
        self.y_combo.setCurrentText(spec.y or "none")
        self.color_combo.setCurrentText(spec.color or "none")
        self.agg_combo.setCurrentText(spec.agg or "none")

        self._updating = False
        self.interpretation.show()
        if spec.why:
            self.show_status(spec.why)

    def hide_interpretation(self) -> None:
        """Collapse the interpreted-spec row."""
        self.interpretation.hide()

    # ------------------------------------------------------------------ signals

    def _submit(self) -> None:
        """Emit the user's question."""
        question = self.input.text().strip()
        if question:
            self.question_asked.emit(question)

    def _on_edited(self, *_: object) -> None:
        """Emit a spec reflecting the user's manual tweak to the interpretation."""
        if self._updating or self._spec is None:
            return

        def value(combo: QComboBox) -> str | None:
            text = combo.currentText()
            return None if text == "none" else text

        edited = self._spec.copy(
            chart=self.chart_combo.currentText(),
            x=value(self.x_combo),
            y=value(self.y_combo),
            color=value(self.color_combo),
            agg=value(self.agg_combo),
            source="user",
            why="Adjusted by you.",
        )
        self._spec = edited
        self.spec_edited.emit(edited)
