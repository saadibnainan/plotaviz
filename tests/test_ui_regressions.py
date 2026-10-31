"""UI regression tests for the defects found in the v0.1.0 audit (see ``docs/AUDIT.md``).

CI runs these under xvfb; locally they also work with ``QT_QPA_PLATFORM=offscreen``. Settings are
redirected to a temporary directory so the tests never read or write the developer's own config.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from PySide6.QtCore import QDate, QSettings
from PySide6.QtWidgets import QMessageBox

from plotaviz.core.analysis import Analysis
from plotaviz.core.preprocess import ColumnFilter, QueryFilter
from plotaviz.core.profiler import profile
from plotaviz.ui.export_dialog import ExportDialog
from plotaviz.ui.filter_panel import DEBOUNCE_MS, FilterPanel
from plotaviz.ui.main_window import MainWindow, _same_chart
from plotaviz.ui.settings_dialog import SettingsDialog
from plotaviz.ui.type_override import TypeOverridePanel
from plotaviz.ui.workers import TaskRunner

LOAD_TIMEOUT_MS = 30_000


@pytest.fixture
def settings(tmp_path: Path) -> QSettings:
    """App settings stored under ``tmp_path``, with the static renderer forced on."""
    QSettings.setPath(
        QSettings.Format.NativeFormat, QSettings.Scope.UserScope, str(tmp_path / "config")
    )
    store = QSettings("plotaviz", "plotaviz")
    store.clear()
    store.setValue("performance/prefer_static", True)
    store.sync()
    return store


@pytest.fixture
def window(qtbot, settings: QSettings) -> Iterator[MainWindow]:
    win = MainWindow()
    qtbot.addWidget(win)
    yield win
    win.runner.cancel()
    win.runner.wait()


def wait_idle(qtbot, win: MainWindow) -> None:
    """Block until the worker thread has fully stopped and its callbacks have run."""
    qtbot.waitUntil(lambda: win.runner._thread is None, timeout=LOAD_TIMEOUT_MS)


def open_and_wait(qtbot, win: MainWindow, path: Path) -> None:
    wait_idle(qtbot, win)
    win.open_path(path)
    wait_idle(qtbot, win)
    assert win.analysis is not None


# ---------------------------------------------------------------------------- filter panel


class TestFilterPanel:
    """P0-1: untouched date pickers sat at 2000-01-01 and filtered every row away."""

    @pytest.fixture
    def panel(self, qtbot, timeseries_df: pd.DataFrame) -> FilterPanel:
        widget = FilterPanel()
        qtbot.addWidget(widget)
        widget.set_profile(profile(timeseries_df))
        return widget

    def test_building_the_panel_emits_nothing(self, qtbot, timeseries_df: pd.DataFrame) -> None:
        widget = FilterPanel()
        qtbot.addWidget(widget)
        with qtbot.assertNotEmitted(widget.filters_changed, wait=DEBOUNCE_MS + 300):
            widget.set_profile(profile(timeseries_df))
        assert widget.filters() == []

    def test_date_pickers_span_the_data(self, panel: FilterPanel) -> None:
        start, end = panel._datetime["order_date"]
        assert start.date() == QDate(2025, 1, 1)
        assert end.date() == QDate(2025, 6, 29)

    def test_numeric_edit_adds_only_that_filter(self, qtbot, panel: FilterPanel) -> None:
        low, _ = panel._numeric["revenue"]
        with qtbot.waitSignal(panel.filters_changed, timeout=3000) as blocker:
            low.setValue(low.value() + 100)
        steps = blocker.args[0]
        assert [(s.column, s.op) for s in steps] == [("revenue", "between")]

    def test_clear_all_yields_no_filters(self, qtbot, panel: FilterPanel) -> None:
        with qtbot.waitSignal(panel.filters_changed, timeout=3000) as blocker:
            panel.clear()
        assert blocker.args[0] == []

    def test_narrowed_date_range_is_emitted(self, qtbot, panel: FilterPanel) -> None:
        _, end = panel._datetime["order_date"]
        with qtbot.waitSignal(panel.filters_changed, timeout=3000) as blocker:
            end.setDate(QDate(2025, 3, 31))
        (step,) = blocker.args[0]
        assert isinstance(step, ColumnFilter)
        assert step.value == ["2025-01-01", "2025-03-31"]

    def test_rounding_alone_does_not_filter(self, qtbot) -> None:
        widget = FilterPanel()
        qtbot.addWidget(widget)
        widget.set_profile(profile(pd.DataFrame({"ratio": [0.1, 0.5, 1.005] * 20})))
        assert widget.filters() == []

    def test_every_category_is_listed(self, qtbot) -> None:
        widget = FilterPanel()
        qtbot.addWidget(widget)
        widget.set_profile(profile(pd.DataFrame({"team": [f"team {i}" for i in range(15)] * 4})))
        assert widget._categorical["team"].count() == 15


# ---------------------------------------------------------------------------- workers


def test_cancelled_task_reports_back(qtbot) -> None:
    """P0-4: a cancelled task emitted nothing, so the UI never left its busy state."""
    runner = TaskRunner()
    finished: list[object] = []
    cancelled: list[bool] = []

    def slow() -> int:
        time.sleep(0.3)
        return 1

    runner.start(slow, on_finished=finished.append, on_cancelled=lambda: cancelled.append(True))
    runner.cancel()

    qtbot.waitUntil(lambda: bool(cancelled), timeout=5000)
    qtbot.waitUntil(lambda: not runner.busy, timeout=5000)
    assert finished == []


# ---------------------------------------------------------------------------- main window


class TestMainWindow:
    def test_cancel_hides_the_progress_widgets(self, qtbot, window: MainWindow) -> None:
        window._run(lambda: time.sleep(0.3), label="Working")
        assert not window.progress.isHidden()

        window._cancel_task()

        qtbot.waitUntil(lambda: window.progress.isHidden(), timeout=5000)
        assert window.cancel_button.isHidden()

    def test_opening_a_dataset_forgets_the_session_file(
        self, qtbot, window: MainWindow, csv_path: Path, tmp_path: Path
    ) -> None:
        """P0-5: Ctrl+S after session → dataset overwrote the old .pviz with the new data."""
        session_file = Analysis.from_file(csv_path).to_session().save(tmp_path / "s.pviz")

        open_and_wait(qtbot, window, session_file)
        assert window.session_path == session_file

        open_and_wait(qtbot, window, csv_path)
        assert window.session_path is None

    def test_filter_change_keeps_the_chosen_chart(
        self, qtbot, window: MainWindow, csv_path: Path
    ) -> None:
        """P0-6: every replay re-selected row 0 and replaced the user's chart."""
        open_and_wait(qtbot, window, csv_path)
        assert window.analysis is not None
        window.recommendations.setCurrentRow(2)
        chosen = window.analysis.spec
        assert chosen is not None

        window._on_filters_changed([QueryFilter("revenue > 0")])
        wait_idle(qtbot, window)

        assert window.analysis.spec is not None
        assert _same_chart(window.analysis.spec, chosen)
        assert window.recommendations.currentRow() == 2

    def test_session_queries_can_be_dropped(
        self,
        qtbot,
        window: MainWindow,
        csv_path: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Security 18: query filters in a .pviz are confirmed before pandas evaluates them."""
        analysis = Analysis.from_file(csv_path)
        analysis.set_filters([QueryFilter("revenue > 2000")])
        session_file = analysis.to_session().save(tmp_path / "q.pviz")

        shown: list[str] = []

        def fake_exec(box: QMessageBox) -> int:
            shown.append(box.informativeText())
            return 0

        monkeypatch.setattr(QMessageBox, "exec", fake_exec)
        monkeypatch.setattr(
            QMessageBox,
            "clickedButton",
            lambda box: next(b for b in box.buttons() if b.text() == "Open without them"),
        )

        open_and_wait(qtbot, window, session_file)

        assert shown and "revenue > 2000" in shown[0]
        assert window.analysis is not None
        assert not window.analysis.pipeline.filters


# ---------------------------------------------------------------------------- dialogs


class TestSettingsConsent:
    """Security 16/17: consent was shared across providers and auto-ticked by Ollama."""

    def select(self, dialog: SettingsDialog, provider: str) -> None:
        dialog.provider_combo.setCurrentIndex(dialog.provider_combo.findData(provider))

    def test_consent_is_per_provider_and_never_preset(self, qtbot, settings: QSettings) -> None:
        dialog = SettingsDialog(settings=settings)
        qtbot.addWidget(dialog)

        self.select(dialog, "ollama")
        assert not dialog.consent_check.isEnabled()
        self.select(dialog, "anthropic")
        assert dialog.consent_check.isEnabled()
        assert not dialog.consent_check.isChecked()

        dialog.consent_check.setChecked(True)
        self.select(dialog, "openai")
        assert not dialog.consent_check.isChecked()
        self.select(dialog, "anthropic")
        assert dialog.consent_check.isChecked()

        dialog._save()
        assert settings.value("llm/consent/anthropic", type=bool) is True
        assert settings.value("llm/consent/openai", type=bool) is False

    def test_remote_ollama_host_requires_consent(self, qtbot, settings: QSettings) -> None:
        dialog = SettingsDialog(settings=settings)
        qtbot.addWidget(dialog)
        self.select(dialog, "ollama")

        dialog.host_edit.setText("http://192.168.1.50:11434")

        assert dialog.consent_check.isEnabled()
        assert not dialog.consent_check.isChecked()


def test_export_asks_before_overwriting(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P1-15: the default code export path silently replaced an existing ``<dataset>.py``."""
    (tmp_path / "sales.py").write_text("# my own script\n")
    dialog = ExportDialog(default_name="sales", start_dir=tmp_path)
    qtbot.addWidget(dialog)
    dialog.code_radio.setChecked(True)

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
    dialog.accept()
    assert dialog.result() != ExportDialog.DialogCode.Accepted

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    dialog.accept()
    assert dialog.result() == ExportDialog.DialogCode.Accepted


def test_type_panel_keeps_restored_overrides(qtbot, csv_path: Path) -> None:
    """P1-12: a restored override looked "detected" and was dropped on the next Apply."""
    analysis = Analysis.from_file(csv_path)
    analysis.set_type_overrides({"units": "categorical"})
    panel = TypeOverridePanel()
    qtbot.addWidget(panel)

    panel.set_profile(analysis.profile, analysis.type_overrides, inferred=analysis.inferred_types)  # type: ignore[arg-type]

    assert panel.overrides() == {"units": "categorical"}
