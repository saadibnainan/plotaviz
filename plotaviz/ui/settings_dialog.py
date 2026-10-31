"""Settings — LLM provider, API keys, and performance thresholds.

The key handling here is the part worth reading. Keys go into the **OS keyring** and nowhere
else: not into a config file, not into the session, not into the window's own state after the
dialog closes. The field shows a masked placeholder when a key is already stored, so the user can
tell it is set without the value being displayed or re-read.

The consent checkbox is not decoration. Nothing is sent to a remote provider until it is ticked,
and the Ollama path skips the question entirely because nothing leaves the machine.
"""

from __future__ import annotations

from typing import Any

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
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..core.errors import PlotaVizError
from ..core.llm import (
    CONSENT_TEXT,
    PROVIDERS,
    delete_api_key,
    get_api_key,
    get_provider,
    set_api_key,
)
from ..core.llm.ollama import DEFAULT_HOST, is_loopback
from ..core.loader import DEFAULT_LARGE_FILE_MB
from ..core.plotter import MAX_POINTS
from ..core.profiler import DatasetProfile
from .workers import TaskRunner

#: Placeholder shown when a key is already in the keyring. The real value is never displayed.
_STORED_PLACEHOLDER = "•" * 24

_PROVIDER_LABELS = {
    "anthropic": "Anthropic (Claude)",
    "openai": "OpenAI",
    "gemini": "Google Gemini",
    "ollama": "Ollama (local server — nothing leaves this machine)",
}


class SettingsDialog(QDialog):
    """Configures the LLM provider and performance thresholds.

    Args:
        parent: Parent widget.
        settings: The app's ``QSettings``. Non-secret preferences are stored here; keys are not.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        settings: QSettings | None = None,
        profile: DatasetProfile | None = None,
    ) -> None:
        super().__init__(parent)
        self._profile = profile
        self._runner = TaskRunner(self)
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
        self.host_edit.setPlaceholderText(DEFAULT_HOST)
        self.host_edit.textChanged.connect(self._sync_consent)
        llm_form.addRow("Ollama host", self.host_edit)

        self.consent_check = QCheckBox(
            "Allow sending this summary to the selected provider when I ask for a chart", llm_box
        )
        self.consent_check.setToolTip(CONSENT_TEXT)
        llm_form.addRow("Consent", self.consent_check)
        consent_note = QLabel(CONSENT_TEXT, llm_box)
        consent_note.setWordWrap(True)
        consent_note.setStyleSheet("color: palette(mid); font-size: 11px;")
        llm_form.addRow("", consent_note)
        # Consent is remembered per provider and only for the session of this dialog until saved.
        self._consent: dict[str, bool] = {}
        self._consent_provider = ""
        self.consent_check.toggled.connect(self._remember_consent)

        self.status_label = QLabel("", llm_box)
        self.status_label.setWordWrap(True)
        llm_form.addRow("", self.status_label)

        self.test_button = QPushButton("Check key / connection", llm_box)
        self.test_button.setToolTip("Lists the models this key can use. No data is sent.")
        self.test_button.clicked.connect(self._test_connection)
        self.preview_button = QPushButton("Preview what will be sent…", llm_box)
        self.preview_button.clicked.connect(self._preview_payload)
        self.preview_button.setEnabled(profile is not None)
        if profile is None:
            self.preview_button.setToolTip("Open a dataset to see its exact request.")
        buttons_row = QHBoxLayout()
        buttons_row.addWidget(self.test_button)
        buttons_row.addWidget(self.preview_button)
        buttons_row.addStretch(1)
        llm_form.addRow("", buttons_row)

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

        self.host_edit.setText(str(self._settings.value("llm/ollama_host", DEFAULT_HOST)))
        self._consent = {
            name: self._settings.value(f"llm/consent/{name}", False, type=bool)
            for name in PROVIDERS
        }
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
        # The old single, global consent flag is retired: it carried agreement for one provider
        # over to every other, so anyone who had it set is asked again. QSettings.remove() also
        # drops sub-keys, so this has to happen before the per-provider values are written.
        self._settings.remove("llm/consent")
        for name, granted in self._consent.items():
            self._settings.setValue(f"llm/consent/{name}", granted)
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
        self._consent_provider = provider
        self._sync_consent()

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
            self._describe_ollama_host()
        else:
            self.key_edit.clear()
            self.status_label.setText(
                "PlotaViz will use its own rules engine. Everything except the natural-language "
                "query bar works normally."
            )
            self.status_label.setStyleSheet("color: palette(mid); font-size: 11px;")

    def _ollama_is_remote(self) -> bool:
        """Whether the Ollama host field points somewhere other than this machine."""
        return not is_loopback(self.host_edit.text().strip() or DEFAULT_HOST)

    def _sync_consent(self, *_: object) -> None:
        """Show the selected provider's own consent, and only where data would leave the machine.

        Selecting Ollama used to tick the shared consent box, which then stayed ticked when the
        user switched to a hosted provider — consent nobody gave. Consent is now per provider and
        is never ticked on the user's behalf.
        """
        provider = self._consent_provider
        leaves_machine = bool(provider) and (provider != "ollama" or self._ollama_is_remote())
        self.consent_check.blockSignals(True)
        self.consent_check.setChecked(bool(self._consent.get(provider, False)))
        self.consent_check.blockSignals(False)
        self.consent_check.setEnabled(leaves_machine)
        if provider == "ollama":
            self._describe_ollama_host()

    def _remember_consent(self, checked: bool) -> None:
        """Record the consent box for the provider currently selected."""
        if self._consent_provider:
            self._consent[self._consent_provider] = bool(checked)

    def _describe_ollama_host(self) -> None:
        """Say plainly whether the configured Ollama server is local."""
        if self._ollama_is_remote():
            self.status_label.setText(
                "This Ollama host is not on this machine, so the summary below is sent over the "
                "network. Consent is required, as for a hosted provider."
            )
            self.status_label.setStyleSheet("color: #A8681F; font-size: 11px;")
        else:
            self.status_label.setText(
                "Ollama runs locally. No API key and no network access are needed, and no data "
                "leaves this machine."
            )
            self.status_label.setStyleSheet("color: #1F7A5C; font-size: 11px;")

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
        """Really check the provider: list its models. Nothing about the data is sent.

        Runs on a worker so a slow or unreachable endpoint cannot freeze the dialog. On success
        the model box is filled with the models this key can actually use.
        """
        provider_name = str(self.provider_combo.currentData() or "")
        if not provider_name:
            self._status("No provider selected — the local engine is always available.", "mid")
            return
        typed = self.key_edit.text().strip()
        kwargs: dict[str, Any] = {}
        if provider_name == "ollama":
            kwargs["host"] = self.host_edit.text().strip() or DEFAULT_HOST
        elif typed and typed != _STORED_PLACEHOLDER:
            kwargs["api_key"] = typed  # check the key as typed, before it is saved
        try:
            provider = get_provider(provider_name, **kwargs)
        except PlotaVizError as exc:
            self._status(exc.message, "error")
            return
        if self._runner.busy:
            return
        self.test_button.setEnabled(False)
        self._status("Checking… (lists the available models; no data is sent)", "mid")
        self._runner.start(
            provider.check_key,
            on_finished=self._on_checked,
            on_failed=lambda message, _trace: self._on_checked((False, message, [])),
        )

    def done(self, result: int) -> None:
        """Let a running key check finish before the dialog (and its thread) goes away."""
        self._runner.cancel()
        self._runner.wait()
        super().done(result)

    def _on_checked(self, result: tuple[bool, str, list[str]]) -> None:
        ok, message, models = result
        self.test_button.setEnabled(True)
        self._status(message, "ok" if ok else "error")
        if ok and models:
            current = self.model_combo.currentText()
            self.model_combo.clear()
            self.model_combo.addItems(models)
            self.model_combo.setCurrentText(current)

    def _status(self, text: str, tone: str) -> None:
        colours = {"ok": "#1F7A5C", "error": "#C0392B", "warn": "#A8681F"}
        colour = colours.get(tone, "palette(mid)")
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {colour}; font-size: 11px;")

    def _preview_payload(self) -> None:
        """Show exactly what a request would send for the open dataset."""
        if self._profile is None:
            return
        from ..core.llm.base import _SYSTEM_PROMPT, build_prompt

        dialog = QDialog(self)
        dialog.setWindowTitle("What PlotaViz would send")
        dialog.resize(760, 620)
        layout = QVBoxLayout(dialog)
        note = QLabel(
            "This is the complete request for the open dataset — the instructions, then the "
            "summary. A question you type in the query bar is appended at the end. Nothing else "
            "is sent.",
            dialog,
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        text = QPlainTextEdit(dialog)
        text.setReadOnly(True)
        text.setPlainText(
            "=== Instructions (system prompt) ===\n"
            + _SYSTEM_PROMPT
            + "\n=== Request ===\n"
            + build_prompt(self._profile)
        )
        layout.addWidget(text, stretch=1)
        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, dialog)
        close.rejected.connect(dialog.reject)
        layout.addWidget(close)
        dialog.exec()

    # ------------------------------------------------------------------ results

    def provider_name(self) -> str:
        """The provider the user selected, or an empty string for local-only."""
        return str(self.provider_combo.currentData() or "")
