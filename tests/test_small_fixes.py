"""Tests for the smaller open items from the audit (``docs/AUDIT.md``, "P2")."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from plotaviz.core import codegen, exporter, loader, plotter
from plotaviz.core.analysis import Analysis
from plotaviz.core.preprocess import ColumnFilter, QueryFilter
from plotaviz.core.profiler import profile
from plotaviz.core.selector import ChartSelector
from plotaviz.core.session import Session
from plotaviz.core.spec import ChartSpec
from plotaviz.main import main


class TestCli:
    def test_positional_file_with_export_runs_headless(
        self, csv_path: Path, tmp_path: Path
    ) -> None:
        target = tmp_path / "chart.png"
        assert main([str(csv_path), "--export", str(target), "-q"]) == 0
        assert target.stat().st_size > 0

    @pytest.mark.parametrize("args", [["--dpi", "0"], ["--dpi", "5000"], ["--size", "0x5"]])
    def test_out_of_range_output_settings_are_rejected(
        self, csv_path: Path, tmp_path: Path, args: list[str]
    ) -> None:
        with pytest.raises(SystemExit) as exit_info:
            main(["--input", str(csv_path), "--export", str(tmp_path / "c.png"), *args])
        assert exit_info.value.code == 2


def test_html_export_works_offline(timeseries_df: pd.DataFrame, tmp_path: Path) -> None:
    written = exporter.export_html(
        timeseries_df, ChartSpec("line", x="order_date", y="revenue"), tmp_path / "c.html"
    )
    html = written.read_text(encoding="utf-8")
    assert '<script src="https://cdn' not in html
    assert len(html) > 1_000_000  # plotly.js is embedded


def test_log_axis_not_suggested_when_values_include_zero(rng: np.random.Generator) -> None:
    values = np.concatenate([np.zeros(50), rng.lognormal(3, 2, 150)])
    df = pd.DataFrame({"region": rng.choice(list("ABCD"), 200), "revenue": values})
    for spec in ChartSelector().recommend(profile(df)):
        assert not spec.options.get("log_y")


def test_other_bucket_ranks_by_the_charted_aggregation() -> None:
    # Category "big" has many small rows (largest sum), "tall" one huge row (largest mean).
    rows = [("big", 1.0)] * 100 + [("tall", 50.0)] + [(f"c{i}", 2.0) for i in range(5)]
    df = pd.DataFrame(rows, columns=["cat", "value"])
    spec = ChartSpec("bar", x="cat", y="value", agg="mean")

    kept = set(plotter.prepare(df, spec, max_categories=2).df["cat"])

    assert "tall" in kept and "big" not in kept


def test_generated_cap_helper_ranks_by_aggregation() -> None:
    code = codegen.generate(ChartSpec("bar", x="a", y="b", agg="mean"), _pipeline(), "x.csv")
    assert "cap_categories(plot_df, 'a', 30, 'b', 'mean')" in code


def test_session_restore_hashes_the_source_once(
    csv_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved = Analysis.from_file(csv_path).to_session().save(tmp_path / "s.pviz")
    calls: list[Path] = []
    real = loader.file_hash

    def counting(path: str | Path, **kwargs: object) -> str:
        calls.append(Path(path))
        return real(path)

    monkeypatch.setattr("plotaviz.core.session.file_hash", counting)
    monkeypatch.setattr("plotaviz.core.loader.file_hash", counting)

    Analysis.from_session(Session.load(saved))

    assert len(calls) == 1


def test_filter_panel_shows_restored_filters(qtbot, timeseries_df: pd.DataFrame) -> None:
    from PySide6.QtCore import QDate

    from plotaviz.ui.filter_panel import FilterPanel

    panel = FilterPanel()
    qtbot.addWidget(panel)
    panel.set_profile(profile(timeseries_df))
    steps = [
        QueryFilter("units > 10"),
        ColumnFilter("revenue", "between", [1500.0, 2500.0]),
        ColumnFilter("order_date", "between", ["2025-02-01", "2025-03-31"]),
        ColumnFilter("region", "in", ["north", "east"]),
    ]

    with qtbot.assertNotEmitted(panel.filters_changed, wait=500):
        panel.set_filters(steps)

    assert panel.query_edit.text() == "units > 10"
    assert panel._numeric["revenue"][0].value() == pytest.approx(1500.0)
    assert panel._datetime["order_date"][1].date() == QDate(2025, 3, 31)
    selected = {i.data(0x0100) for i in panel._categorical["region"].selectedItems()}
    assert selected == {"north", "east"}
    assert [s.to_dict() for s in panel.filters()] == [s.to_dict() for s in steps]


def test_chart_view_honours_max_points(qtbot, rng: np.random.Generator) -> None:
    from plotaviz.ui.interactive_view import ChartView

    view = ChartView(prefer_static=True)
    qtbot.addWidget(view)
    df = pd.DataFrame({"a": rng.normal(size=5_000), "b": rng.normal(size=5_000)})

    notes = view.render_spec(df, ChartSpec("scatter", x="a", y="b"), max_points=1_000)

    assert any("1,000 of 5,000" in note for note in notes)


def _pipeline():  # type: ignore[no-untyped-def]
    from plotaviz.core.preprocess import Pipeline

    return Pipeline()


def test_session_restores_the_active_tab(qtbot, window, csv_path: Path, tmp_path: Path) -> None:
    from conftest import open_and_wait

    session = Analysis.from_file(csv_path).to_session(view={"active_tab": 2})
    saved = session.save(tmp_path / "v.pviz")

    open_and_wait(qtbot, window, saved)

    assert window.analysis_tabs.currentIndex() == 2
