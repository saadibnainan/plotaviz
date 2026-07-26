"""Settings — LLM provider, API keys, and performance thresholds.

The key handling here is the part worth reading. Keys go into the **OS keyring** and nowhere
else: not into a config file, not into the session, not into the window's own state after the
dialog closes. The field shows a masked placeholder when a key is already stored, so the user can
tell it is set without the value being displayed or re-read.

The consent checkbox is not decoration. Nothing is sent to a remote provider until it is ticked,
and the Ollama path skips the question entirely because nothing leaves the machine.
"""

from __future__ import annotations

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..core.llm import PROVIDERS, OllamaProvider, delete_api_key, get_api_key, set_api_key
from ..core.loader import DEFAULT_LARGE_FILE_MB
from ..core.plotter import MAX_POINTS

#: Placeholder shown when a key is already in the keyring. The real value is never displayed.
_STORED_PLACEHOLDER = "•" * 24

_PROVIDER_LABELS = {
    "anthropic": "Anthropic (Claude)",
    "openai": "OpenAI",
    "gemini": "Google Gemini",
    "ollama": "Ollama (local — nothing leaves this machine)",
}


class SettingsDialog(QDialog):
    """Configures the LLM provider and performance thresholds.

    Args:
        parent: Parent widget.
        settings: The app's ``QSettings``. Non-secret preferences are stored here; keys are not.
    """

    def __init__(self, parent: QWidget | None = None, *, settings: QSettings | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("PlotaViz settings")
        self.setMinimumWidth(560)
        self._settings = settings or QSettings("plotaviz", "plotaviz")

        layout = QVBoxLayout(self)

        # ------------------------------------------------------------------ LLM
        llm_box = QGroupBox("LLM assistance (optional)", self)
        llm_form = QFormLayout(llm_box)

        intro = QLabel(
            "PlotaViz recommends charts on its own, offline, with no provider configured. "
            "A model adds two things: a second opinion when the ranking is close, and the "
            "natural-language query bar.",
            llm_box,
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: palette(mid); font-size: 11px;")
        llm_form.addRow(intro)

        self.provider_combo = QComboBox(llm_box)
        self.provider_combo.addItem("None — use the local engine only", "")
        for name in PROVIDERS:
            self.provider_combo.addItem(_PROVIDER_LABELS.get(name, name), name)
        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        llm_form.addRow("Provider", self.provider_combo)

        self.model_combo = QComboBox(llm_box)
        self.model_combo.setEditable(True)
        llm_form.addRow("Model", self.model_combo)

        key_row = QHBoxLayout()
        self.key_edit = QLineEdit(llm_box)
        self.key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_edit.setPlaceholderText("Paste an API key")
        self.clear_key_button = QPushButton("Forget key", llm_box)
        self.clear_key_button.clicked.connect(self._forget_key)
        key_row.addWidget(self.key_edit, stretch=1)
        key_row.addWidget(self.clear_key_button)
        llm_form.addRow("API key", key_row)

        key_note = QLabel(
            "Keys are stored in your operating system's keychain, never in a config file, a "
            "session file, or this project's folder.",
            llm_box,
        )
        key_note.setWordWrap(True)
        key_note.setStyleSheet("color: palette(mid); font-size: 11px;")
        llm_form.addRow("", key_note)

        self.host_edit = QLineEdit(llm_box)
        self.host_edit.setPlaceholderText("http://localhost:11434")
        llm_form.addRow("Ollama host", self.host_edit)

        self.consent_check = QCheckBox(
            "Allow sending column names, summary statistics, and up to 5 sample rows", llm_box
        )
        self.consent_check.setToolTip(
            "The dataset itself is never uploaded. Only schema and statistics are sent, and only "
            "when you ask for a suggestion."
        )
        llm_form.addRow("Consent", self.consent_check)

        self.status_label = QLabel("", llm_box)
        self.status_label.setWordWrap(True)
        llm_form.addRow("", self.status_label)

        test_button = QPushButton("Test connection", llm_box)
        test_button.clicked.connect(self._test_connection)
        llm_form.addRow("", test_button)

        layout.addWidget(llm_box)

        # ------------------------------------------------------------------ performance
        perf_box = QGroupBox("Performance", self)
        perf_form = QFormLayout(perf_box)

        self.large_file_spin = QSpinBox(perf_box)
        self.large_file_spin.setRange(10, 20_000)
        self.large_file_spin.setSingleStep(50)
        self.large_file_spin.setSuffix(" MB")
        self.large_file_spin.setToolTip(
            "Files above this size are read with polars and profiled on a sample."
        )
        perf_form.addRow("Use the fast reader above", self.large_file_spin)

        self.max_points_spin = QSpinBox(perf_box)
        self.max_points_spin.setRange(1_000, 2_000_000)
        self.max_points_spin.setSingleStep(10_000)
        self.max_points_spin.setGroupSeparatorShown(True)
        self.max_points_spin.setToolTip(
            "Scatter and line charts above this many points are downsampled. Raising it past "
            "about 100,000 will make the interactive view sluggish."
        )
        perf_form.addRow("Downsample charts above", self.max_points_spin)

        self.static_check = QCheckBox(
            "Always use the static renderer (no interactive view)", perf_box
        )
        self.static_check.setToolTip(
            "Useful on machines without QtWebEngine, or when you prefer matplotlib output."
        )
        perf_form.addRow("Renderer", self.static_check)

        layout.addWidget(perf_box)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel, self
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._load()

    # ------------------------------------------------------------------ load / save

    def _load(self) -> None:
        """Populate the dialog from QSettings and the keyring."""
        provider = str(self._settings.value("llm/provider", "") or "")
        index = self.provider_combo.findData(provider)
        self.provider_combo.setCurrentIndex(max(0, index))

        self.host_edit.setText(
            str(self._settings.value("llm/ollama_host", "http://localhost:11434"))
        )
        self.consent_check.setChecked(self._settings.value("llm/consent", False, type=bool))
        self.large_file_spin.setValue(
            int(self._settings.value("performance/large_file_mb", DEFAULT_LARGE_FILE_MB))
        )
        self.max_points_spin.setValue(
            int(self._settings.value("performance/max_points", MAX_POINTS))
        )
        self.static_check.setChecked(
            self._settings.value("performance/prefer_static", False, type=bool)
        )

        self._on_provider_changed()
        model = str(self._settings.value("llm/model", "") or "")
        if model:
            self.model_combo.setCurrentText(model)

    def _save(self) -> None:
        """Write settings, and put any newly entered key into the keyring."""
        provider = str(self.provider_combo.currentData() or "")
        self._settings.setValue("llm/provider", provider)
        self._settings.setValue("llm/model", self.model_combo.currentText().strip())
        self._settings.setValue("llm/ollama_host", self.host_edit.text().strip())
        self._settings.setValue("llm/consent", self.consent_check.isChecked())
        self._settings.setValue("performance/large_file_mb", self.large_file_spin.value())
        self._settings.setValue("performance/max_points", self.max_points_spin.value())
        self._settings.setValue("performance/prefer_static", self.static_check.isChecked())

        key = self.key_edit.text().strip()
        if provider and key and key != _STORED_PLACEHOLDER:
            try:
                set_api_key(provider, key)
            except Exception as exc:
                self.status_label.setText(str(exc))
                self.status_label.setStyleSheet("color: #C0392B;")
                return

        self._settings.sync()
        self.accept()

    # ------------------------------------------------------------------ interaction

    def _on_provider_changed(self, *_: object) -> None:
        """Update the model list, key field, and consent state for the chosen provider."""
        provider = str(self.provider_combo.currentData() or "")
        is_ollama = provider == "ollama"
        needs_key = bool(provider) and not is_ollama

        self.key_edit.setEnabled(needs_key)
        self.clear_key_button.setEnabled(needs_key)
        self.host_edit.setEnabled(is_ollama)
        self.consent_check.setEnabled(needs_key)

        self.model_combo.clear()
        if provider:
            cls = PROVIDERS[provider]
            self.model_combo.addItems(list(cls.available_models))
            self.model_combo.setCurrentText(cls.default_model)

        if needs_key:
            stored = get_api_key(provider)
            self.key_edit.setText(_STORED_PLACEHOLDER if stored else "")
            self.status_label.setText(
                "A key is stored in your keychain." if stored else "No key stored yet."
            )
            self.status_label.setStyleSheet("color: palette(mid); font-size: 11px;")
        elif is_ollama:
            self.key_edit.clear()
            self.consent_check.setChecked(True)
            self.status_label.setText(
                "Ollama runs locally. No API key and no network access are needed, and no data "
                "leaves this machine."
            )
            self.status_label.setStyleSheet("color: #1F7A5C; font-size: 11px;")
        else:
            self.key_edit.clear()
            self.status_label.setText(
                "PlotaViz will use its own rules engine. Everything except the natural-language "
                "query bar works normally."
            )
            self.status_label.setStyleSheet("color: palette(mid); font-size: 11px;")

    def _forget_key(self) -> None:
        """Delete the stored key for the selected provider."""
        provider = str(self.provider_combo.currentData() or "")
        if not provider:
            return
        delete_api_key(provider)
        self.key_edit.clear()
        self.status_label.setText("Key removed from your keychain.")
        self.status_label.setStyleSheet("color: palette(mid); font-size: 11px;")

    def _test_connection(self) -> None:
        """Check the provider looks usable without sending any data."""
        provider = str(self.provider_combo.currentData() or "")
        if not provider:
            self.status_label.setText(
                "No provider selected — the local engine is always available."
            )
            self.status_label.setStyleSheet("color: palette(mid); font-size: 11px;")
            return

        if provider == "ollama":
            client = OllamaProvider(host=self.host_edit.text().strip() or "http://localhost:11434")
            if client.is_running():
                models = client.list_models()
                listed = ", ".join(models[:5]) if models else "no models pulled yet"
                self.status_label.setText(f"Ollama is running. Models: {listed}")
                self.status_label.setStyleSheet("color: #1F7A5C; font-size: 11px;")
            else:
                self.status_label.setText(
                    f"No Ollama server answered at {client.host}. Start it with `ollama serve`."
                )
                self.status_label.setStyleSheet("color: #C0392B; font-size: 11px;")
            return

        typed = self.key_edit.text().strip()
        has_key = bool(typed and typed != _STORED_PLACEHOLDER) or bool(get_api_key(provider))
        if has_key:
            self.status_label.setText(
                "A key is available. It is checked for real the first time you ask for a "
                "suggestion."
            )
            self.status_label.setStyleSheet("color: #1F7A5C; font-size: 11px;")
        else:
            self.status_label.setText("No API key for this provider yet.")
            self.status_label.setStyleSheet("color: #C0392B; font-size: 11px;")

    # ------------------------------------------------------------------ results

    def provider_name(self) -> str:
        """The provider the user selected, or an empty string for local-only."""
        return str(self.provider_combo.currentData() or "")
