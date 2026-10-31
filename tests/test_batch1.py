"""Batch 1: faster filtering, background and in-place rendering, real sampling, portable
sessions, the unsaved-changes prompt, and LLM transparency."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import open_and_wait, wait_idle

from plotaviz.core import loader
from plotaviz.core.analysis import Analysis
from plotaviz.core.errors import LLMError
from plotaviz.core.llm import OpenAIProvider
from plotaviz.core.preprocess import FlagOutliers, Pipeline, QueryFilter
from plotaviz.core.session import Session
from plotaviz.core.spec import ChartSpec

# ---------------------------------------------------------------------------- filter cache


class TestFilterCache:
    def test_filter_change_reuses_the_cleaned_frame(self, csv_path: Path) -> None:
        analysis = Analysis.from_file(csv_path)
        cached = analysis._clean_cache
        analysis.set_filters([QueryFilter("revenue > 2000")])
        analysis.set_filters([QueryFilter("units > 10")])
        assert analysis._clean_cache is cached  # same object: cleaning did not run again

    def test_result_matches_a_full_run(self, csv_path: Path) -> None:
        analysis = Analysis.from_file(csv_path)
        analysis.set_cleaning(missing="median", outliers="flag")
        analysis.set_filters([QueryFilter("revenue > 2000")])

        full = analysis.pipeline.run(analysis.raw)

        pd.testing.assert_frame_equal(analysis.df, full.df)
        assert [r.summary for r in analysis.result.reports] == [r.summary for r in full.reports]

    def test_cleaning_change_misses_the_cache(self, csv_path: Path) -> None:
        analysis = Analysis.from_file(csv_path)
        cached = analysis._clean_cache
        analysis.set_cleaning(missing="zero")
        assert analysis._clean_cache is not cached

    def test_filter_before_a_cleaning_step_still_correct(self, csv_path: Path) -> None:
        analysis = Analysis.from_file(csv_path)
        analysis.pipeline = Pipeline(
            [*analysis.pipeline.steps[:2], QueryFilter("revenue > 2000"), FlagOutliers()]
        )
        analysis.rerun()
        pd.testing.assert_frame_equal(analysis.df, analysis.pipeline.run(analysis.raw).df)


# ---------------------------------------------------------------------------- sampling


@pytest.fixture
def sorted_csv(tmp_path: Path) -> Path:
    target = tmp_path / "sorted.csv"
    pd.DataFrame({"day": np.arange(20_000), "value": np.arange(20_000) * 2.0}).to_csv(
        target, index=False
    )
    return target


class TestSampling:
    @pytest.mark.parametrize("options", [None, {"sep": ","}])
    def test_sample_spans_the_whole_file(self, sorted_csv: Path, options) -> None:
        result = loader.load(sorted_csv, large_file_mb=0.0001, sample_rows=1_000, options=options)
        days = result.df["day"]
        assert result.sampled and len(days) == 1_000 and result.total_rows == 20_000
        assert days.min() < 1_000 and days.max() > 19_000  # not just the first 1,000 rows
        assert days.is_monotonic_increasing  # file order kept
        assert "random sample" in result.notes[-1]

    def test_sample_is_deterministic(self, sorted_csv: Path) -> None:
        first = loader.load(sorted_csv, large_file_mb=0.0001, sample_rows=500)
        second = loader.load(sorted_csv, large_file_mb=0.0001, sample_rows=500)
        pd.testing.assert_frame_equal(first.df, second.df)

    def test_ndjson_is_sampled(self, tmp_path: Path) -> None:
        target = tmp_path / "events.ndjson"
        target.write_text("".join(json.dumps({"i": i}) + "\n" for i in range(5_000)))
        result = loader.load(target, large_file_mb=0.0001, sample_rows=300)
        assert result.sampled and len(result.df) == 300 and result.df["i"].max() > 4_000

    def test_large_plain_json_says_it_was_loaded_in_full(self, tmp_path: Path) -> None:
        target = tmp_path / "big.json"
        target.write_text(json.dumps([{"i": i} for i in range(2_000)]))
        result = loader.load(target, large_file_mb=0.0001)
        assert not result.sampled and any("loaded in full" in n for n in result.notes)


# ---------------------------------------------------------------------------- sessions


class TestPortableSessions:
    def test_project_folder_can_be_moved(self, tmp_path: Path, timeseries_df) -> None:
        project = tmp_path / "project"
        (project / "data").mkdir(parents=True)
        timeseries_df.to_csv(project / "data" / "sales.csv", index=False)
        analysis = Analysis.from_file(project / "data" / "sales.csv")
        saved = analysis.to_session().save(project / "analysis.pviz")
        assert Session.load(saved).source_relative == os.path.join("data", "sales.csv")

        moved = tmp_path / "elsewhere"
        shutil.move(str(project), str(moved))
        session = Session.load(moved / "analysis.pviz")
        restored, warning = Analysis.from_session(session)

        assert Path(session.source_path) == (moved / "data" / "sales.csv").resolve()
        assert warning is not None and "was not at" in warning
        assert len(restored.df) == len(analysis.df)

    def test_relocate_reports_changed_contents(self, csv_path: Path, tmp_path: Path) -> None:
        saved = Analysis.from_file(csv_path).to_session().save(tmp_path / "s.pviz")
        other = tmp_path / "other.csv"
        pd.read_csv(csv_path).head(50).to_csv(other, index=False)
        session = Session.load(saved)
        session.relocate(other)
        warning = session.check_source()
        assert warning is not None and "has changed" in warning and "was not at" in warning

    def test_gui_offers_to_locate_a_missing_file(
        self, qtbot, window, csv_path: Path, tmp_path: Path, monkeypatch
    ) -> None:
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        saved = Analysis.from_file(csv_path).to_session().save(tmp_path / "s.pviz")
        moved = tmp_path / "renamed" / "data.csv"
        moved.parent.mkdir()
        shutil.move(str(csv_path), str(moved))
        monkeypatch.setattr(
            QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Open
        )
        monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(moved), ""))
        monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)

        open_and_wait(qtbot, window, saved)

        assert window.analysis.load_result.path == moved.resolve()


# ---------------------------------------------------------------------------- unsaved changes


class TestUnsavedChanges:
    def test_title_marks_changes_and_save_clears_them(
        self, qtbot, window, csv_path: Path, tmp_path: Path, monkeypatch
    ) -> None:
        open_and_wait(qtbot, window, csv_path)
        assert not window._dirty and "*" not in window.windowTitle()

        window._on_filters_changed([QueryFilter("revenue > 2000")])
        wait_idle(qtbot, window)
        assert window._dirty and " *" in window.windowTitle()

        window.session_path = tmp_path / "s.pviz"
        assert window.save_session()
        assert not window._dirty

    @pytest.mark.parametrize(("answer", "closes"), [("Cancel", False), ("Discard", True)])
    def test_close_asks_first(
        self, qtbot, window, csv_path: Path, monkeypatch, answer: str, closes: bool
    ) -> None:
        from PySide6.QtGui import QCloseEvent
        from PySide6.QtWidgets import QMessageBox

        open_and_wait(qtbot, window, csv_path)
        window.pin_current()
        window.ask_before_discarding = True
        asked: list[bool] = []

        def question(*_a, **_k):
            asked.append(True)
            return getattr(QMessageBox.StandardButton, answer)

        monkeypatch.setattr(QMessageBox, "question", question)
        event = QCloseEvent()
        window.closeEvent(event)

        assert asked and event.isAccepted() is closes
        window.ask_before_discarding = False


# ---------------------------------------------------------------------------- LLM transparency


class TestKeyCheck:
    def test_success_lists_models(self, monkeypatch) -> None:
        provider = OpenAIProvider(api_key="sk-test")
        monkeypatch.setattr(provider, "list_models", lambda: ["m-a", "m-b"])
        ok, message, models = provider.check_key()
        assert ok and models == ["m-a", "m-b"] and "2 models" in message

    def test_failure_is_reported_and_redacted(self, monkeypatch) -> None:
        provider = OpenAIProvider(api_key="sk-test")

        def fail() -> list[str]:
            raise LLMError("Rejected.", hint="key sk-abcdefghijklmnopqrstu was bad")

        monkeypatch.setattr(provider, "list_models", fail)
        ok, message, models = provider.check_key()
        assert not ok and models == [] and "sk-abcdefghijklmnop" not in message

    def test_settings_check_fills_the_model_list(self, qtbot, settings, monkeypatch) -> None:
        from plotaviz.ui import settings_dialog
        from plotaviz.ui.settings_dialog import SettingsDialog

        class Fake:
            def check_key(self):
                return True, "The key works.", ["live-model-1", "live-model-2"]

        monkeypatch.setattr(settings_dialog, "get_provider", lambda *a, **k: Fake())
        dialog = SettingsDialog(settings=settings)
        qtbot.addWidget(dialog)
        dialog.provider_combo.setCurrentIndex(dialog.provider_combo.findData("openai"))

        dialog.test_button.click()
        qtbot.waitUntil(lambda: "works" in dialog.status_label.text(), timeout=5000)

        items = [dialog.model_combo.itemText(i) for i in range(dialog.model_combo.count())]
        assert items == ["live-model-1", "live-model-2"]

    def test_preview_shows_the_exact_request(self, qtbot, settings, monkeypatch, csv_path) -> None:
        from PySide6.QtWidgets import QDialog, QPlainTextEdit

        from plotaviz.ui.settings_dialog import SettingsDialog

        profile = Analysis.from_file(csv_path).profile
        shown: list[str] = []

        def capture(dialog: QDialog) -> int:
            shown.append(dialog.findChild(QPlainTextEdit).toPlainText())
            return 0

        monkeypatch.setattr(QDialog, "exec", capture)
        dialog = SettingsDialog(settings=settings, profile=profile)
        qtbot.addWidget(dialog)
        dialog.preview_button.click()

        assert shown and "Dataset schema and summary statistics" in shown[0]
        assert '"name": "revenue"' in shown[0]


# ---------------------------------------------------------------------------- rendering


class TestBackgroundRendering:
    def test_bursts_of_redraws_are_coalesced(self, qtbot, window, csv_path, monkeypatch) -> None:
        from plotaviz.ui import main_window

        open_and_wait(qtbot, window, csv_path)
        builds: list[str] = []
        real = main_window.build_render

        def slow(df, spec, **kwargs):
            builds.append(spec.chart)
            time.sleep(0.2)
            return real(df, spec, **kwargs)

        monkeypatch.setattr(main_window, "build_render", slow)
        for chart in ("histogram", "kde", "histogram", "kde", "histogram"):
            window.analysis.choose(ChartSpec(chart, x="units"))
            window._render_current()
        wait_idle(qtbot, window)

        assert len(builds) <= 3  # first, plus one redo for the burst — not five
        assert builds[-1] == "histogram"  # the last request is what is shown
        assert window.chart_view._canvas is not None

    def test_render_error_is_shown_not_raised(self, qtbot, window, csv_path) -> None:
        open_and_wait(qtbot, window, csv_path)
        window.analysis.spec = ChartSpec("correlation_heatmap", options={"columns": ["units"]})
        window._render_current()
        wait_idle(qtbot, window)
        assert window.chart_view._placeholder is not None


def test_interactive_view_updates_in_place(qtbot, monkeypatch) -> None:
    """The Plotly page loads once; later charts are pushed with Plotly.react."""
    from plotaviz.ui.interactive_view import ChartView, webengine_available

    if not webengine_available():
        pytest.skip("QtWebEngine is not installed")
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        monkeypatch.setenv("QTWEBENGINE_DISABLE_SANDBOX", "1")  # Chromium refuses root otherwise
    view = ChartView()
    qtbot.addWidget(view)
    view.resize(600, 400)
    view.show()
    df = pd.DataFrame({"a": range(30), "b": range(30), "c": list("xyz") * 10})

    view.render_spec(df, ChartSpec("scatter", x="a", y="b"))
    try:
        qtbot.waitUntil(lambda: view._shell_ready, timeout=30_000)
    except Exception:
        pytest.skip("QtWebEngine could not start in this environment")
    view.render_spec(df, ChartSpec("bar", x="c", y="b", agg="sum"))

    types: list[str] = []
    qtbot.wait(500)
    view._web.page().runJavaScript("document.getElementById('chart').data[0].type", 0, types.append)
    qtbot.waitUntil(lambda: bool(types), timeout=10_000)
    assert view.page_loads == 1 and view.react_calls == 2
    assert types == ["bar"]


# ---------------------------------------------------------------------------- review fixes


class TestReviewFixes:
    def test_interactive_failure_falls_back_to_a_static_chart(self, qtbot, monkeypatch) -> None:
        from plotaviz.ui.interactive_view import ChartView

        view = ChartView()
        qtbot.addWidget(view)
        view._interactive = True

        def broken(_json: str) -> None:
            raise RuntimeError("no GPU")

        monkeypatch.setattr(view, "_install_interactive", broken)
        df = pd.DataFrame({"a": range(20), "b": range(20)})
        notes = view.render_spec(df, ChartSpec("scatter", x="a", y="b"))

        assert view._canvas is not None and not view.interactive
        assert any("static chart instead" in n for n in notes)

    def test_failed_shell_keeps_updating_with_standalone_pages(self, qtbot, monkeypatch) -> None:
        from plotaviz.ui.interactive_view import ChartView, webengine_available

        if not webengine_available():
            pytest.skip("QtWebEngine is not installed")
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            monkeypatch.setenv("QTWEBENGINE_DISABLE_SANDBOX", "1")
        view = ChartView()
        qtbot.addWidget(view)
        df = pd.DataFrame({"a": range(20), "b": range(20)})
        view.render_spec(df, ChartSpec("scatter", x="a", y="b"))
        view._on_shell_loaded(False)  # simulate the shell page failing to load
        loads = view.page_loads
        view.render_spec(df, ChartSpec("line", x="a", y="b"))
        assert view.page_loads == loads + 1  # a new standalone page, not a silent no-op

    def test_key_check_result_for_another_provider_is_not_applied(
        self, qtbot, settings, monkeypatch
    ) -> None:
        from plotaviz.ui.settings_dialog import SettingsDialog

        dialog = SettingsDialog(settings=settings)
        qtbot.addWidget(dialog)
        dialog.provider_combo.setCurrentIndex(dialog.provider_combo.findData("anthropic"))
        before = [dialog.model_combo.itemText(i) for i in range(dialog.model_combo.count())]

        dialog._on_checked((True, "Connected.", ["gpt-x"]), "openai")

        after = [dialog.model_combo.itemText(i) for i in range(dialog.model_combo.count())]
        assert after == before and "Result for openai" in dialog.status_label.text()

    def test_stale_render_does_not_cover_a_newer_message(
        self, qtbot, window, csv_path, monkeypatch
    ) -> None:
        from plotaviz.ui import main_window

        open_and_wait(qtbot, window, csv_path)
        real = main_window.build_render

        def slow(df, spec, **kwargs):
            time.sleep(0.3)
            return real(df, spec, **kwargs)

        monkeypatch.setattr(main_window, "build_render", slow)
        window._render_current()  # slow build in flight
        window.analysis.spec = None
        window._render_current()  # now there is nothing to draw
        wait_idle(qtbot, window)

        assert window.chart_view._placeholder is not None
        assert "No chart" in window.chart_view._placeholder.text()

    def test_runner_stays_busy_until_results_are_delivered(self, qtbot) -> None:
        from plotaviz.ui.workers import TaskRunner

        runner = TaskRunner()
        results: list[int] = []
        runner.start(lambda: 1, on_finished=results.append)
        thread = runner._thread
        assert thread is not None
        thread.wait(5000)  # the thread is done, but its result is still queued
        assert runner.busy and not results
        qtbot.waitUntil(lambda: not runner.busy, timeout=5000)
        assert results == [1]

    def test_ndjson_polars_failure_falls_back_to_pandas(self, tmp_path, monkeypatch) -> None:
        import polars as pl

        target = tmp_path / "mixed.ndjson"
        target.write_text("".join(json.dumps({"v": i}) + "\n" for i in range(100)))

        def refuse(*_a, **_k):
            raise pl.exceptions.ComputeError("schema mismatch")

        monkeypatch.setattr(pl, "scan_ndjson", refuse)
        result = loader.load(target, large_file_mb=0.0001)
        assert len(result.df) == 100

    def test_cache_is_skipped_for_huge_frames(self, csv_path, monkeypatch) -> None:
        from plotaviz.core import analysis as analysis_module

        monkeypatch.setattr(analysis_module, "CLEAN_CACHE_MAX_BYTES", 0)
        analysis = Analysis.from_file(csv_path)
        analysis.set_filters([QueryFilter("revenue > 2000")])
        assert analysis._clean_cache is None
        pd.testing.assert_frame_equal(analysis.df, analysis.pipeline.run(analysis.raw).df)

    def test_no_models_is_not_reported_as_zero_listed(self, monkeypatch) -> None:
        provider = OpenAIProvider(api_key="sk-test")
        monkeypatch.setattr(provider, "list_models", lambda: [])
        ok, message, _ = provider.check_key()
        assert ok and "no models are available" in message
