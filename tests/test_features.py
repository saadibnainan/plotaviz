"""Tests for the features added after the audit: chart builder, import options, cleaning
controls, undo/redo, and chart customization."""

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import open_and_wait, wait_idle

from plotaviz.core import codegen, loader, plotter
from plotaviz.core.analysis import Analysis
from plotaviz.core.errors import PreprocessError
from plotaviz.core.preprocess import (
    Deduplicate,
    FillMissing,
    FlagOutliers,
    NormalizeColumnNames,
    QueryFilter,
)
from plotaviz.core.session import Session
from plotaviz.core.spec import ChartSpec


def run_script(path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(path), "--no-show", *args],
        capture_output=True,
        text=True,
        timeout=180,
    )


# ---------------------------------------------------------------------------- chart builder


def test_builder_row_works_without_a_provider(qtbot, window, csv_path: Path) -> None:
    open_and_wait(qtbot, window, csv_path)
    bar = window.nl_bar
    assert not bar.input.isEnabled()  # no LLM configured
    assert not bar.interpretation.isHidden()
    assert bar.chart_combo.currentText() == window.analysis.spec.chart

    bar.chart_combo.setCurrentText("histogram")
    bar.x_combo.setCurrentText("units")
    bar.y_combo.setCurrentText("none")
    bar.color_combo.setCurrentText("none")
    bar.agg_combo.setCurrentText("none")

    assert window.analysis.spec.chart == "histogram"
    assert window.analysis.spec.x == "units"


# ---------------------------------------------------------------------------- import options


@pytest.fixture
def report_csv(tmp_path: Path) -> Path:
    """A semicolon file with a title line above the header and comma decimals."""
    target = tmp_path / "report.csv"
    body = "".join(f"R{i % 3};{i},25;{i}\n" for i in range(30))
    target.write_text("Monthly report\nRegion;Amount;Units\n" + body, encoding="latin-1")
    return target


class TestImportOptions:
    OPTIONS = {"sep": ";", "decimal": ",", "header": 1, "encoding": "latin-1"}

    def test_explicit_options_are_used_and_recorded(self, report_csv: Path) -> None:
        result = loader.load(report_csv, options=self.OPTIONS)
        assert list(result.df.columns) == ["Region", "Amount", "Units"]
        assert result.df["Amount"].iloc[1] == pytest.approx(1.25)
        assert result.read_options == self.OPTIONS

    def test_generated_script_reads_with_the_same_options(
        self, report_csv: Path, tmp_path: Path
    ) -> None:
        analysis = Analysis.from_file(report_csv, options=self.OPTIONS)
        script = codegen.write(tmp_path / "plot.py", analysis.generate_code())
        assert run_script(script, "--save", str(tmp_path / "o.png")).returncode == 0

    def test_options_survive_a_session(self, report_csv: Path, tmp_path: Path) -> None:
        analysis = Analysis.from_file(report_csv, options=self.OPTIONS)
        saved = analysis.to_session().save(tmp_path / "s.pviz")

        restored, _ = Analysis.from_session(Session.load(saved))

        assert restored.load_options == self.OPTIONS
        assert list(restored.df.columns) == ["region", "amount", "units"]

    def test_excel_sheet_choice(self, tmp_path: Path) -> None:
        target = tmp_path / "book.xlsx"
        with pd.ExcelWriter(target) as writer:
            pd.DataFrame({"a": [1, 2]}).to_excel(writer, sheet_name="Cover", index=False)
            pd.DataFrame({"b": range(30)}).to_excel(writer, sheet_name="Data", index=False)

        result = loader.load(target, options={"sheet": "Data"})

        assert list(result.df.columns) == ["b"]
        assert "sheet_name='Data'" in loader.read_code_for(target, result.read_options)

    def test_dialog_reports_only_non_default_choices(self, qtbot, report_csv: Path) -> None:
        from plotaviz.ui.import_dialog import ImportDialog

        dialog = ImportDialog(report_csv)
        qtbot.addWidget(dialog)
        assert dialog.options() == {}
        assert "Monthly report" in dialog.preview.toPlainText()

        dialog.delimiter_combo.setCurrentIndex(dialog.delimiter_combo.findData(";"))
        dialog.decimal_combo.setCurrentIndex(dialog.decimal_combo.findData(","))
        dialog.header_spin.setValue(2)
        assert dialog.options() == {"sep": ";", "decimal": ",", "header": 1}


# ---------------------------------------------------------------------------- cleaning


class TestCleaning:
    def test_missing_values_are_kept_by_default(self, messy_df: pd.DataFrame, tmp_path) -> None:
        source = tmp_path / "m.csv"
        messy_df.to_csv(source, index=False)
        analysis = Analysis.from_file(source)
        assert not any(isinstance(s, FillMissing) for s in analysis.pipeline)
        assert analysis.df["total_revenue"].isna().sum() == 1

    def test_set_cleaning_swaps_only_the_cleaning_steps(self, csv_path: Path) -> None:
        analysis = Analysis.from_file(csv_path)
        analysis.set_filters([QueryFilter("revenue > 0")])

        analysis.set_cleaning(
            missing="median", outliers="drop", outlier_method="zscore", dedupe=False
        )

        kinds = [type(s) for s in analysis.pipeline]
        assert kinds[0] is NormalizeColumnNames
        assert FillMissing in kinds and FlagOutliers in kinds and Deduplicate not in kinds
        assert isinstance(analysis.pipeline.steps[-1], QueryFilter)  # filters still last
        assert analysis.cleaning() == {
            "missing": "median",
            "outliers": "drop",
            "outlier_method": "zscore",
            "outlier_threshold": 1.5,
            "dedupe": False,
        }

    def test_unknown_choice_is_rejected(self, csv_path: Path) -> None:
        with pytest.raises(PreprocessError):
            Analysis.from_file(csv_path).set_cleaning(missing="guess")

    def test_cli_keeps_missing_values_unless_asked(self) -> None:
        from plotaviz.main import build_parser

        assert build_parser().parse_args(["--input", "x.csv"]).missing == "keep"

    def test_panel_applies_settings(self, qtbot, window, csv_path: Path) -> None:
        open_and_wait(qtbot, window, csv_path)
        panel = window.cleaning_panel
        panel.missing_combo.setCurrentIndex(panel.missing_combo.findData("zero"))
        panel.outliers_combo.setCurrentIndex(panel.outliers_combo.findData("off"))

        panel.apply_button.click()
        wait_idle(qtbot, window)

        assert window.analysis.cleaning()["missing"] == "zero"
        assert window.analysis.cleaning()["outliers"] == "off"
        assert panel.steps_list.count() == len(window.analysis.pipeline)


# ---------------------------------------------------------------------------- undo / redo


class TestUndoRedo:
    def test_filters_types_and_cleaning_undo_in_order(self, csv_path: Path) -> None:
        analysis = Analysis.from_file(csv_path)
        rows = len(analysis.df)
        assert not analysis.can_undo

        analysis.set_filters([QueryFilter("revenue > 2000")])
        analysis.set_type_overrides({"units": "categorical"})
        analysis.set_cleaning(missing="zero", outliers="off")

        analysis.undo()
        assert analysis.cleaning()["outliers"] == "flag"
        analysis.undo()
        assert analysis.type_overrides == {}
        assert pd.api.types.is_numeric_dtype(analysis.df["units"])
        analysis.undo()
        assert len(analysis.df) == rows and not analysis.can_undo

        analysis.redo()
        assert len(analysis.df) < rows
        assert analysis.can_redo

    def test_new_change_clears_redo(self, csv_path: Path) -> None:
        analysis = Analysis.from_file(csv_path)
        analysis.set_filters([QueryFilter("revenue > 2000")])
        analysis.undo()
        analysis.set_filters([QueryFilter("units > 10")])
        assert not analysis.can_redo

    def test_failed_change_is_not_recorded(self, csv_path: Path) -> None:
        analysis = Analysis.from_file(csv_path)
        with pytest.raises(PreprocessError):
            analysis.set_filters([QueryFilter("nonsense >")])
        assert not analysis.can_undo

    def test_remove_step_and_undo(self, csv_path: Path) -> None:
        analysis = Analysis.from_file(csv_path)
        index = next(i for i, s in enumerate(analysis.pipeline) if isinstance(s, FlagOutliers))
        analysis.remove_step(index)
        assert not any(isinstance(s, FlagOutliers) for s in analysis.pipeline)
        analysis.undo()
        assert any(isinstance(s, FlagOutliers) for s in analysis.pipeline)

    def test_renaming_step_cannot_be_removed(self, csv_path: Path) -> None:
        with pytest.raises(PreprocessError):
            Analysis.from_file(csv_path).remove_step(0)

    def test_edit_menu_undo_restores_filter_widgets(self, qtbot, window, csv_path: Path) -> None:
        open_and_wait(qtbot, window, csv_path)
        assert not window.undo_action.isEnabled()

        window._on_filters_changed([QueryFilter("revenue > 2000")])
        wait_idle(qtbot, window)
        assert window.undo_action.isEnabled()

        window.undo_action.trigger()
        wait_idle(qtbot, window)

        assert not window.analysis.pipeline.filters
        assert window.filter_panel.query_edit.text() == ""
        assert window.redo_action.isEnabled()


# ---------------------------------------------------------------------------- customization


class TestCustomization:
    @pytest.fixture
    def custom(self) -> ChartSpec:
        return ChartSpec(
            "bar",
            x="region",
            y="revenue",
            agg="mean",
            title="Average revenue",
            options={
                "palette": "colorblind",
                "x_label": "Sales region",
                "y_label": "EUR",
                "sort_ascending": True,
            },
        )

    def test_plotly_applies_palette_and_labels(self, timeseries_df, custom) -> None:
        fig = plotter.build_plotly(timeseries_df, custom)
        assert fig.data[0].marker.color == plotter.COLORBLIND_PALETTE[0]
        assert fig.layout.xaxis.title.text == "Sales region"
        assert fig.layout.title.text == "Average revenue"

    def test_matplotlib_applies_palette_and_labels(self, timeseries_df, custom) -> None:
        import matplotlib.colors as mcolors

        fig = plotter.build_matplotlib(timeseries_df, custom, dpi=50)
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Sales region" and ax.get_ylabel() == "EUR"
        assert (
            mcolors.to_hex(ax.patches[0].get_facecolor()) == plotter.COLORBLIND_PALETTE[0].lower()
        )

    def test_ascending_sort(self, timeseries_df, custom) -> None:
        values = plotter.prepare(timeseries_df, custom).df["revenue"].tolist()
        assert values == sorted(values)

    @pytest.mark.parametrize("flavour", ["matplotlib", "plotly"])
    def test_generated_script_carries_customization(
        self, csv_path: Path, tmp_path: Path, custom: ChartSpec, flavour: str
    ) -> None:
        analysis = Analysis.from_file(csv_path)
        analysis.choose(custom)
        code = analysis.generate_code(flavour=flavour)
        script = codegen.write(tmp_path / "plot.py", code)

        assert "PALETTE = " in code and "Sales region" in code
        suffix = "png" if flavour == "matplotlib" else "html"
        result = run_script(script, "--save", str(tmp_path / f"o.{suffix}"))
        assert result.returncode == 0, result.stderr

        if flavour == "matplotlib":
            generated = runpy.run_path(str(script), run_name="generated")
            fig = generated["build_chart"](generated["preprocess"](generated["load"]()))  # type: ignore[operator]
            assert fig.axes[0].get_xlabel() == "Sales region"

    def test_histogram_bins(self, rng: np.random.Generator) -> None:
        df = pd.DataFrame({"v": rng.normal(size=500)})
        fig = plotter.build_matplotlib(df, ChartSpec("histogram", x="v", options={"bins": 7}))
        assert len(fig.axes[0].patches) == 7

    def test_panel_round_trip(self, qtbot, window, csv_path: Path) -> None:
        from plotaviz.ui.customize_panel import apply_customization

        open_and_wait(qtbot, window, csv_path)
        panel = window.customize_panel
        panel.setChecked(True)
        panel.title_edit.setText("My chart")
        panel.palette_combo.setCurrentIndex(panel.palette_combo.findData("colorblind"))
        panel.apply_button.click()

        spec = window.analysis.spec
        assert spec.title == "My chart"
        assert spec.options["palette"] == "colorblind"

        cleared = apply_customization(spec.with_options(trendline=True), {"title": None})
        assert cleared.title is None
        assert "palette" not in cleared.options and cleared.options["trendline"] is True
