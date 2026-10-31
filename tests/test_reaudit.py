"""Regression tests for the findings of the second review (of the audit branch itself)."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest
from conftest import open_and_wait, wait_idle

from plotaviz.core import loader
from plotaviz.core.analysis import Analysis
from plotaviz.core.preprocess import ColumnFilter, QueryFilter
from plotaviz.core.session import Session
from plotaviz.core.spec import ChartSpec


class TestDecimalCommaHeuristic:
    def test_dotted_dates_survive(self, tmp_path: Path) -> None:
        target = tmp_path / "de.csv"
        rows = "".join(f"{d:02d}.05.2024;1.234,{d:02d}\n" for d in range(1, 29))
        target.write_text("datum;betrag\n" + rows, encoding="utf-8")

        result = loader.load(target)

        # Dates intact; grouped amounts stay text (a "." thousands mark would break the dates),
        # and the user is told how to fix it.
        assert result.df["datum"].iloc[0] == "01.05.2024"
        assert "thousands" not in result.read_options
        assert result.df["betrag"].iloc[0] == "1.234,01"
        assert any("dotted dates" in note for note in result.notes)

    def test_us_thousands_are_not_read_as_decimals(self, tmp_path: Path) -> None:
        target = tmp_path / "us.tsv"
        values = ["1,250", "3,400", "12,000", "9,500"] * 6
        target.write_text("revenue\tunits\n" + "".join(f"{v}\t{i}\n" for i, v in enumerate(values)))

        result = loader.load(target)

        assert "decimal" not in result.read_options
        assert not pd.api.types.is_float_dtype(result.df["revenue"])

    def test_user_options_kept_on_reread(self, tmp_path: Path) -> None:
        target = tmp_path / "noheader.csv"
        target.write_text("".join(f"a{i};{i},5\n" for i in range(20)))

        result = loader.load(target, options={"sep": ";", "header": None})

        assert len(result.df) == 20
        assert result.read_options["header"] is None
        assert result.read_options["decimal"] == ","

    def test_large_file_with_options_is_capped(self, tmp_path: Path) -> None:
        target = tmp_path / "big.csv"
        target.write_text("a;b\n" + "".join(f"{i};{i}\n" for i in range(5_000)))

        result = loader.load(target, options={"sep": ";"}, large_file_mb=0.001, sample_rows=1_000)

        assert len(result.df) == 1_000
        assert result.sampled and result.total_rows == 5_000


def test_session_with_filters_matching_nothing_still_opens(csv_path: Path, tmp_path: Path) -> None:
    analysis = Analysis.from_file(csv_path)
    analysis.set_filters([QueryFilter("region == 'north'")])
    saved = analysis.to_session().save(tmp_path / "s.pviz")
    df = pd.read_csv(csv_path)
    df[df["region"] != "north"].to_csv(csv_path, index=False)

    restored, warning = Analysis.from_session(Session.load(saved))

    assert not restored.pipeline.filters
    assert warning is not None and "opened without them" in warning


def test_generated_date_filter_handles_timezones() -> None:
    df = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")})
    step = ColumnFilter("ts", "between", ["2024-01-02", "2024-01-03"])
    namespace: dict[str, object] = {"pd": pd, "df": df}
    exec("\n".join(step.to_code()), namespace)
    assert len(namespace["df"]) == 2  # type: ignore[arg-type]


def test_matrix_columns_follow_type_changes(tmp_path: Path, rng) -> None:
    target = tmp_path / "m.csv"
    pd.DataFrame(
        {"a": rng.normal(size=100), "b": rng.normal(size=100), "c": rng.normal(size=100)}
    ).to_csv(target, index=False)
    analysis = Analysis.from_file(target)
    analysis.choose(ChartSpec("correlation_heatmap"))
    assert analysis.spec.options["columns"] == ["a", "b", "c"]

    analysis.set_type_overrides({"a": "categorical"})

    assert analysis.spec.options["columns"] == ["b", "c"]
    analysis.prepared()  # still drawable


def test_untouched_bound_uses_the_exact_value(qtbot) -> None:
    from plotaviz.core.profiler import profile
    from plotaviz.ui.filter_panel import FilterPanel

    panel = FilterPanel()
    qtbot.addWidget(panel)
    panel.set_profile(profile(pd.DataFrame({"v": [0.123456, 1.0, 2.0, 3.5] * 10})))
    _, high = panel._numeric["v"]
    high.setValue(3.0)

    (step,) = panel.filters()
    assert step.value[0] == pytest.approx(0.123456)


def test_queue_keeps_different_kinds_of_change(qtbot, window, csv_path: Path) -> None:
    open_and_wait(qtbot, window, csv_path)
    window._run(lambda: time.sleep(0.4), label="Busy", on_finished=lambda _: None)

    window._on_types_changed({"units": "categorical"})
    window._on_filters_changed([QueryFilter("revenue > 0")])
    window._on_filters_changed([QueryFilter("revenue > 2000")])  # replaces the queued filter

    assert len(window._pending) == 2
    wait_idle(qtbot, window)
    qtbot.waitUntil(lambda: not window._pending, timeout=30_000)
    wait_idle(qtbot, window)

    assert window.analysis.type_overrides == {"units": "categorical"}
    assert [f.expression for f in window.analysis.pipeline.filters] == ["revenue > 2000"]


def test_chart_choice_waits_for_a_running_replay(qtbot, window, csv_path: Path) -> None:
    open_and_wait(qtbot, window, csv_path)
    window._run(lambda: time.sleep(0.4), label="Busy", on_finished=lambda _: None)
    target = window.analysis.recommendations[2]

    window.recommendations.setCurrentRow(2)
    assert window.analysis.spec is not target  # deferred, not applied mid-task

    wait_idle(qtbot, window)
    qtbot.waitUntil(lambda: window.analysis.spec is target, timeout=10_000)


def test_static_rerenders_do_not_leak_figures(qtbot, categorical_df) -> None:
    import matplotlib.pyplot as plt

    from plotaviz.ui.interactive_view import ChartView

    view = ChartView(prefer_static=True)
    qtbot.addWidget(view)
    spec = ChartSpec("bar", x="department", y="tenure", agg="mean")
    view.render_spec(categorical_df, spec)
    before = len(plt.get_fignums())

    for _ in range(5):
        view.render_spec(categorical_df, spec)

    assert len(plt.get_fignums()) == before
