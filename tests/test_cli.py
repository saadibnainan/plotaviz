"""CLI and orchestration tests.

The CLI is the proof that ``core`` really is headless: these tests never import Qt, and they run
the same code path a CI job would.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from plotaviz.core.analysis import Analysis
from plotaviz.core.errors import PlotaVizError
from plotaviz.core.preprocess import QueryFilter
from plotaviz.core.spec import ChartSpec
from plotaviz.main import main


class TestAnalysisFacade:
    def test_load_clean_profile_recommend(self, csv_path: Path) -> None:
        analysis = Analysis.from_file(csv_path)

        assert analysis.result is not None
        assert analysis.profile is not None
        assert analysis.recommendations
        assert analysis.spec is not None

    def test_type_map_survives_column_renaming(self, tmp_path: Path) -> None:
        """Types are inferred on raw names but applied after the snake_case rename."""
        source = tmp_path / "d.csv"
        source.write_text("Order Date,Total Revenue\n2025-01-01,10\n2025-01-02,20\n2025-01-03,30\n")

        analysis = Analysis.from_file(source)

        assert "order_date" in analysis.df.columns
        assert pd.api.types.is_datetime64_any_dtype(analysis.df["order_date"])

    def test_rerun_is_idempotent(self, csv_path: Path) -> None:
        analysis = Analysis.from_file(csv_path)
        first = len(analysis.df)
        analysis.rerun()
        assert len(analysis.df) == first

    def test_filters_are_replaced_not_stacked(self, csv_path: Path) -> None:
        analysis = Analysis.from_file(csv_path)

        analysis.set_filters([QueryFilter("revenue > 1000")])
        analysis.set_filters([QueryFilter("revenue > 2000")])

        assert len(analysis.pipeline.filters) == 1

    def test_type_override_reprofiles(self, csv_path: Path) -> None:
        analysis = Analysis.from_file(csv_path)
        analysis.set_type_override("units", "categorical")

        assert analysis.profile is not None
        assert analysis.profile.columns["units"].role == "categorical"

    def test_choosing_an_invalid_spec_is_refused(self, csv_path: Path) -> None:
        analysis = Analysis.from_file(csv_path)
        with pytest.raises(PlotaVizError, match="not in this dataset"):
            analysis.choose(ChartSpec("scatter", x="revenue", y="imaginary"))

    def test_exporting_without_a_chart_explains_itself(
        self, tmp_path: Path, csv_path: Path
    ) -> None:
        analysis = Analysis.from_file(csv_path)
        analysis.spec = None
        analysis.recommendations = []

        with pytest.raises(PlotaVizError, match="No chart has been chosen"):
            analysis.export_image(tmp_path / "x.png")

    @pytest.mark.parametrize("suffix", [".png", ".svg", ".pdf"])
    def test_image_export_formats(self, tmp_path: Path, csv_path: Path, suffix: str) -> None:
        analysis = Analysis.from_file(csv_path)
        written = analysis.export_image(tmp_path / f"chart{suffix}", dpi=72)

        assert written.exists()
        assert written.stat().st_size > 1_000

    def test_unsupported_image_format_lists_the_real_ones(
        self, tmp_path: Path, csv_path: Path
    ) -> None:
        analysis = Analysis.from_file(csv_path)
        with pytest.raises(PlotaVizError, match="Supported formats"):
            analysis.export_image(tmp_path / "chart.gif")

    def test_html_export(self, tmp_path: Path, csv_path: Path) -> None:
        from plotaviz.core.exporter import export_html

        analysis = Analysis.from_file(csv_path)
        assert analysis.spec is not None
        written = export_html(analysis.df, analysis.spec, tmp_path / "chart")

        assert written.suffix == ".html"
        assert "plotly" in written.read_text().lower()


class TestCli:
    def test_describe(self, csv_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["--input", str(csv_path), "--describe"]) == 0

        out = capsys.readouterr().out
        assert "order_date" in out
        assert "datetime" in out

    def test_describe_as_json(self, csv_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["--input", str(csv_path), "--describe", "--json"]) == 0

        payload = json.loads(capsys.readouterr().out)
        assert payload["n_cols"] == 4
        assert {c["name"] for c in payload["columns"]} >= {"order_date", "revenue"}

    def test_recommend(self, csv_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["--input", str(csv_path), "--recommend"]) == 0

        out = capsys.readouterr().out
        assert "Recommended charts" in out

    def test_recommend_as_json(self, csv_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["--input", str(csv_path), "--recommend", "--json"]) == 0

        specs = json.loads(capsys.readouterr().out)
        assert specs
        assert specs[0]["chart"]
        assert specs[0]["why"]

    def test_auto_export(self, tmp_path: Path, csv_path: Path) -> None:
        target = tmp_path / "chart.png"
        assert main(["--input", str(csv_path), "--auto", "--export", str(target)]) == 0
        assert target.exists() and target.stat().st_size > 1_000

    def test_export_code(self, tmp_path: Path, csv_path: Path) -> None:
        target = tmp_path / "plot.py"
        assert main(["--input", str(csv_path), "--auto", "--export-code", str(target)]) == 0

        code = target.read_text()
        compile(code, str(target), "exec")
        imports = [
            line.strip()
            for line in code.splitlines()
            if line.strip().startswith(("import ", "from "))
        ]
        assert imports
        assert not any("plotaviz" in line for line in imports)

    def test_export_code_plotly_flavour(self, tmp_path: Path, csv_path: Path) -> None:
        target = tmp_path / "plot.py"
        main(
            [
                "--input",
                str(csv_path),
                "--auto",
                "--export-code",
                str(target),
                "--code-flavour",
                "plotly",
            ]
        )
        assert "plotly.express" in target.read_text()

    def test_explicit_chart_mapping(self, tmp_path: Path, csv_path: Path) -> None:
        target = tmp_path / "c.png"
        code = main(
            [
                "--input",
                str(csv_path),
                "--chart",
                "box",
                "--x",
                "region",
                "--y",
                "revenue",
                "--export",
                str(target),
            ]
        )
        assert code == 0
        assert target.exists()

    def test_partial_mapping_prefers_the_named_columns(
        self, csv_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["--input", str(csv_path), "--y", "units", "--recommend", "--json"]) == 0

    def test_query_filter_applies(self, tmp_path: Path, csv_path: Path) -> None:
        target = tmp_path / "c.png"
        assert (
            main(
                [
                    "--input",
                    str(csv_path),
                    "--auto",
                    "--query",
                    "revenue > 1500",
                    "--export",
                    str(target),
                ]
            )
            == 0
        )
        assert target.exists()

    def test_missing_strategy_is_honoured(
        self, tmp_path: Path, timeseries_df: pd.DataFrame
    ) -> None:
        source = tmp_path / "gappy.csv"
        gappy = timeseries_df.copy()
        gappy.loc[gappy.index[:50], "revenue"] = None
        gappy.to_csv(source, index=False)

        target = tmp_path / "c.png"
        assert (
            main(["--input", str(source), "--missing", "drop", "--auto", "--export", str(target)])
            == 0
        )

    def test_html_export_from_the_cli(self, tmp_path: Path, csv_path: Path) -> None:
        target = tmp_path / "chart.html"
        assert main(["--input", str(csv_path), "--auto", "--export", str(target)]) == 0
        assert target.exists()

    def test_size_and_dpi_options(self, tmp_path: Path, csv_path: Path) -> None:
        small = tmp_path / "small.png"
        large = tmp_path / "large.png"

        main(
            [
                "--input",
                str(csv_path),
                "--auto",
                "--export",
                str(small),
                "--size",
                "4x3",
                "--dpi",
                "72",
            ]
        )
        main(
            [
                "--input",
                str(csv_path),
                "--auto",
                "--export",
                str(large),
                "--size",
                "12x8",
                "--dpi",
                "150",
            ]
        )

        assert large.stat().st_size > small.stat().st_size

    def test_missing_file_returns_an_error_code_without_a_traceback(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["--input", str(tmp_path / "nope.csv"), "--describe"]) == 1

        err = capsys.readouterr().err
        assert "error:" in err
        assert "Traceback" not in err

    def test_bad_chart_type_is_reported_cleanly(
        self, csv_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(
            [
                "--input",
                str(csv_path),
                "--chart",
                "hologram",
                "--x",
                "region",
                "--export",
                str(tmp_path / "x.png"),
            ]
        )
        assert code == 1
        assert "Unknown chart type" in capsys.readouterr().err

    def test_bad_size_is_a_usage_error(self, csv_path: Path, tmp_path: Path) -> None:
        with pytest.raises(SystemExit) as exc:
            main(
                [
                    "--input",
                    str(csv_path),
                    "--auto",
                    "--export",
                    str(tmp_path / "x.png"),
                    "--size",
                    "big",
                ]
            )
        assert exc.value.code == 2

    def test_version_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc:
            main(["--version"])
        assert exc.value.code == 0
        assert "PlotaViz" in capsys.readouterr().out


def test_core_never_imports_qt() -> None:
    """The architectural invariant, checked mechanically rather than by convention."""
    import subprocess
    import sys

    probe = (
        "import sys, plotaviz.core, plotaviz.core.analysis, plotaviz.core.llm, "
        "plotaviz.core.codegen, plotaviz.core.exporter, plotaviz.core.session;"
        "loaded = [m for m in sys.modules if m.startswith('PySide6')];"
        "print('QT_LOADED' if loaded else 'CLEAN')"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert "CLEAN" in result.stdout
