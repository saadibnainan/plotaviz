"""Batch 2: data preview (sort/search/stats), dark mode, one panel per category."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import open_and_wait, wait_idle

from plotaviz.core import codegen, exporter, plotter
from plotaviz.core.analysis import Analysis
from plotaviz.core.errors import SpecError
from plotaviz.core.spec import ChartSpec

# ---------------------------------------------------------------------------- preview


class TestPreview:
    @pytest.fixture
    def model(self, qtbot):
        from plotaviz.ui.table_model import DataFrameModel

        frame = pd.DataFrame({"n": np.arange(9_000), "label": [f"row {i}" for i in range(9_000)]})
        model = DataFrameModel()
        model.set_frame(frame)
        return model

    def test_sort_covers_rows_beyond_the_preview(self, model) -> None:
        from PySide6.QtCore import Qt

        model.sort(0, Qt.SortOrder.DescendingOrder)
        first = model.data(model.index(0, 0))
        assert first == "8999"  # the maximum lives far past the first 5,000 previewed rows

    def test_search_filters_all_rows(self, model) -> None:
        model.set_search("ROW 8123")
        assert model.rowCount() == 1
        assert model.data(model.index(0, 1)) == "row 8123"
        assert "1 of 9,000 rows" in model.status_text()

    def test_nested_values_do_not_break_painting(self, qtbot) -> None:
        from plotaviz.ui.table_model import DataFrameModel

        model = DataFrameModel(pd.DataFrame({"tags": [["a", "b"], None]}))
        assert model.data(model.index(0, 0)) == "['a', 'b']"
        assert model.data(model.index(1, 0)) == "—"

    def test_window_search_and_column_stats(self, qtbot, window, csv_path: Path) -> None:
        open_and_wait(qtbot, window, csv_path)
        window.preview_search.setText("north")
        qtbot.waitUntil(lambda: "contain" in window.preview_status.text(), timeout=5000)
        wait_idle(qtbot, window)
        rows = window.preview_model.rowCount()
        assert 0 < rows < len(window.analysis.df)

        column = list(window.analysis.df.columns).index("revenue")
        window.preview.horizontalHeader().sectionClicked.emit(column)
        assert "revenue" in window.column_stats.toPlainText()
        assert "Missing" in window.column_stats.toPlainText()


# ---------------------------------------------------------------------------- theme


class TestTheme:
    def test_dark_theme_applies_to_window_and_on_screen_charts(
        self, qtbot, window, csv_path: Path
    ) -> None:
        from PySide6.QtGui import QPalette
        from PySide6.QtWidgets import QApplication

        open_and_wait(qtbot, window, csv_path)
        try:
            window.set_theme("dark")
            wait_idle(qtbot, window)
            window_colour = QApplication.instance().palette().color(QPalette.ColorRole.Window)
            assert window_colour.lightness() < 100
            assert window.preview_model.dark
            figure = window.chart_view._canvas.figure
            assert figure.get_facecolor()[0] < 0.3  # dark chart background
        finally:
            window.set_theme("light")
            wait_idle(qtbot, window)
        assert QApplication.instance().palette().color(QPalette.ColorRole.Window).lightness() > 150

    def test_exports_stay_light(self, timeseries_df: pd.DataFrame, tmp_path: Path) -> None:
        import matplotlib.image as mpimg

        target = exporter.export_image(
            timeseries_df, ChartSpec("histogram", x="units"), tmp_path / "c.png"
        )
        corner = mpimg.imread(str(target))[2, 2]
        assert corner[:3].min() > 0.9  # white page

    def test_plotly_dark_template_for_screen(self, timeseries_df: pd.DataFrame) -> None:
        import json

        from plotaviz.ui.interactive_view import build_render

        result = build_render(
            timeseries_df, ChartSpec("histogram", x="units"), interactive=True, dark=True
        )
        layout = json.loads(result.figure_json)["layout"]
        assert layout["template"]["layout"]["paper_bgcolor"] == "rgb(17,17,17)"


# ---------------------------------------------------------------------------- panels


class TestFacets:
    @pytest.fixture
    def spec(self) -> ChartSpec:
        return ChartSpec("bar", x="region", y="revenue", agg="mean", options={"facet": "region"})

    def test_unknown_facet_column_is_rejected(self, timeseries_df: pd.DataFrame) -> None:
        with pytest.raises(SpecError):
            ChartSpec("histogram", x="units", options={"facet": "nope"}).validate(
                list(timeseries_df.columns)
            )

    def test_both_renderers_draw_one_panel_per_category(self, timeseries_df, spec) -> None:
        spec = ChartSpec("line", x="order_date", y="revenue", options={"facet": "region"})
        fig = plotter.build_plotly(timeseries_df, spec)
        assert len({trace.xaxis for trace in fig.data}) == 3
        assert len(plotter.build_matplotlib(timeseries_df, spec, dpi=50).axes) == 3

    def test_aggregation_is_per_panel(self, timeseries_df, spec) -> None:
        prepared = plotter.prepare(
            timeseries_df,
            ChartSpec("bar", x="units", y="revenue", agg="sum", options={"facet": "region"}),
        )
        assert "region" in prepared.df.columns
        assert prepared.df.groupby("region").size().min() > 1

    def test_unsupported_chart_says_so(self, timeseries_df: pd.DataFrame) -> None:
        prepared = plotter.prepare(
            timeseries_df,
            ChartSpec("pie", x="region", agg="count", options={"facet": "region"}),
        )
        assert any("cannot be split into panels" in n for n in prepared.notes)

    @pytest.mark.parametrize(("flavour", "suffix"), [("matplotlib", "png"), ("plotly", "html")])
    def test_generated_script_draws_panels(
        self, csv_path: Path, tmp_path: Path, flavour: str, suffix: str
    ) -> None:
        analysis = Analysis.from_file(csv_path)
        analysis.choose(ChartSpec("box", x="region", y="revenue", options={"facet": "region"}))
        script = codegen.write(tmp_path / "plot.py", analysis.generate_code(flavour=flavour))
        result = subprocess.run(
            [sys.executable, str(script), "--no-show", "--save", str(tmp_path / f"o.{suffix}")],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, result.stderr
        assert "facet" in script.read_text() or "panels" in script.read_text()

    def test_dashboard_grid_draws_faceted_tile(self, timeseries_df, tmp_path: Path) -> None:
        specs = [
            ChartSpec("histogram", x="units", options={"facet": "region"}),
            ChartSpec("histogram", x="revenue"),
        ]
        figure = plotter.build_matplotlib_grid(timeseries_df, specs, dpi=40)
        assert len(figure.axes) == 4  # 3 panels + 1 plain chart

    def test_customize_panel_round_trip(self, qtbot, window, csv_path: Path) -> None:
        open_and_wait(qtbot, window, csv_path)
        panel = window.customize_panel
        panel.setChecked(True)
        panel.facet_combo.setCurrentIndex(panel.facet_combo.findData("region"))
        window.analysis.choose(ChartSpec("histogram", x="units"))
        panel.apply_button.click()
        wait_idle(qtbot, window)

        assert window.analysis.spec.options["facet"] == "region"
        assert len(window.chart_view._canvas.figure.axes) == 3


# ---------------------------------------------------------------------------- review fixes


class TestReviewFixes:
    def test_cleared_search_is_not_overwritten_by_a_late_result(self, qtbot) -> None:
        from plotaviz.ui.table_model import DataFrameModel

        model = DataFrameModel()
        queued: list = []
        model.scheduler = lambda work, done: queued.append((work, done))
        model.set_frame(pd.DataFrame({"a": ["x", "y", "z"]}))
        model.set_search("x")  # goes to the "worker"
        model.set_search("")  # cleared at once
        work, done = queued[-1]
        done(work())  # the old search finishes late
        assert model.rowCount() == 3

    def test_new_frame_is_shown_at_once_while_arranging(self, qtbot) -> None:
        from PySide6.QtCore import Qt

        from plotaviz.ui.table_model import DataFrameModel

        model = DataFrameModel()
        model.scheduler = lambda work, done: None  # never finishes
        model.set_frame(pd.DataFrame({"a": [1, 2]}))
        model.sort(0, Qt.SortOrder.DescendingOrder)
        model.set_frame(pd.DataFrame({"b": [1, 2, 3], "c": [4, 5, 6]}))
        assert model.columnCount() == 2 and model.rowCount() == 3  # headers and cells agree
        assert "sorting/searching" in model.status_text()

    def test_mixed_type_column_sorts_as_text(self) -> None:
        from plotaviz.ui.table_model import arrange_frame

        out = arrange_frame(pd.DataFrame({"m": ["x", 1, None, "a"]}), "m", True, "")
        assert list(out["m"])[:3] == [1, "a", "x"]

    def test_array_cells_paint(self, qtbot) -> None:
        from plotaviz.ui.table_model import DataFrameModel

        model = DataFrameModel(pd.DataFrame({"v": [np.array([1, 2]), np.array([3])]}))
        assert model.data(model.index(0, 0)) == "[1 2]"

    def test_facet_options_must_name_one_column(self) -> None:
        with pytest.raises(SpecError):
            ChartSpec("histogram", x="a", options={"facet": ["a"]}).validate(["a"])

    def test_missing_facet_values_get_their_own_panel(self) -> None:
        df = pd.DataFrame(
            {
                "v": np.arange(30.0),
                "flag": pd.array([True, False, None] * 10, dtype="boolean"),
            }
        )
        figure = plotter.build_matplotlib(
            df, ChartSpec("histogram", x="v", options={"facet": "flag"}), dpi=40
        )
        titles = sorted(ax.get_title() for ax in figure.axes)
        assert titles == ["flag = (missing)", "flag = False", "flag = True"]

    def test_generated_facet_script_survives_all_missing(self, tmp_path: Path) -> None:
        source = tmp_path / "d.csv"
        pd.DataFrame({"v": np.arange(20.0), "g": [None] * 20}).to_csv(source, index=False)
        analysis = Analysis.from_file(source)
        analysis.set_type_overrides({"g": "categorical"})
        spec = ChartSpec("histogram", x="v", options={"facet": "g"})
        analysis.spec = spec
        script = codegen.write(tmp_path / "p.py", analysis.generate_code())
        result = subprocess.run(
            [sys.executable, str(script), "--no-show", "--save", str(tmp_path / "o.png")],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, result.stderr

    def test_dark_figure_recolours_background_text(self, timeseries_df) -> None:
        from plotaviz.ui.theme import darken_figure

        figure = plotter.build_matplotlib(
            timeseries_df, ChartSpec("pie", x="region", agg="count"), dpi=40
        )
        darken_figure(figure)
        colours = {t.get_color() for t in figure.axes[0].texts}
        assert "#e6e6e6" in colours

    def test_explicit_light_uses_a_light_palette(self, qtbot) -> None:
        from PySide6.QtGui import QPalette
        from PySide6.QtWidgets import QApplication

        from plotaviz.ui.theme import apply_theme

        assert apply_theme("light") is False
        window = QApplication.instance().palette().color(QPalette.ColorRole.Window)
        assert window.lightness() > 150
        apply_theme("system")
