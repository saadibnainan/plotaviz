"""Code generation tests.

The bar for this feature is specific and testable: the emitted script must **run on a machine
that has never heard of PlotaViz**. So these tests do not inspect strings for plausibility — they
write the script to disk and execute it in a subprocess, and assert an image came out.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from plotaviz.core import codegen
from plotaviz.core.analysis import Analysis
from plotaviz.core.errors import ExportError
from plotaviz.core.preprocess import Pipeline, QueryFilter, default_pipeline
from plotaviz.core.spec import ChartSpec


def run_script(path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Execute a generated script the way a user would."""
    return subprocess.run(
        [sys.executable, str(path), "--no-show", *args],
        capture_output=True,
        text=True,
        timeout=180,
    )


class TestGeneratedSource:
    def test_is_valid_python(self, csv_path: Path) -> None:
        code = codegen.generate(
            ChartSpec("line", x="order_date", y="revenue"), default_pipeline({}), csv_path
        )
        compile(code, "<generated>", "exec")

    @pytest.mark.parametrize("flavour", ["matplotlib", "plotly"])
    def test_never_imports_plotaviz(self, csv_path: Path, flavour: str) -> None:
        code = codegen.generate(
            ChartSpec("line", x="order_date", y="revenue"),
            default_pipeline({}),
            csv_path,
            flavour=flavour,
        )
        for line in code.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                assert "plotaviz" not in stripped, stripped

    def test_records_provenance_in_the_header(self, csv_path: Path) -> None:
        code = codegen.generate(
            ChartSpec("histogram", x="revenue"), default_pipeline({}), csv_path, version="9.9.9"
        )
        assert "PlotaViz 9.9.9" in code
        assert "standalone" in code

    def test_includes_the_preprocessing_steps(self, csv_path: Path) -> None:
        pipeline = default_pipeline({"revenue": "numeric"})
        pipeline.add(QueryFilter("revenue > 100"))
        code = codegen.generate(ChartSpec("histogram", x="revenue"), pipeline, csv_path)

        assert "drop_duplicates" in code
        assert "df.query('revenue > 100')" in code

    def test_rejects_an_unknown_flavour(self, csv_path: Path) -> None:
        with pytest.raises(ExportError, match="flavour"):
            codegen.generate(
                ChartSpec("histogram", x="revenue"), Pipeline(), csv_path, flavour="ggplot"
            )


class TestGeneratedScriptRuns:
    CASES = [
        ChartSpec("histogram", x="revenue"),
        ChartSpec("kde", x="revenue"),
        ChartSpec("line", x="order_date", y="revenue", color="region"),
        ChartSpec("area", x="order_date", y="revenue"),
        ChartSpec("scatter", x="revenue", y="units", options={"trendline": True}),
        ChartSpec("bar", x="region", agg="count"),
        ChartSpec("bar", x="region", y="revenue", agg="mean"),
        ChartSpec("grouped_bar", x="region", y="revenue", color="region", agg="mean"),
        ChartSpec("stacked_bar", x="region", color="region", agg="count"),
        ChartSpec("box", x="region", y="revenue"),
        ChartSpec("violin", x="region", y="revenue"),
        ChartSpec("treemap", x="region", agg="count"),
        ChartSpec("pie", x="region", agg="count"),
        ChartSpec("heatmap", x="region", y="region", agg="count"),
        ChartSpec("correlation_heatmap"),
        ChartSpec("pair_plot"),
    ]

    @pytest.mark.parametrize("spec", CASES, ids=lambda s: s.chart)
    def test_matplotlib_script_produces_an_image(
        self, tmp_path: Path, csv_path: Path, spec: ChartSpec
    ) -> None:
        script = tmp_path / f"plot_{spec.chart}.py"
        image = tmp_path / f"{spec.chart}.png"
        codegen.write(script, codegen.generate(spec, default_pipeline({}), csv_path))

        result = run_script(script, "--save", str(image))

        assert result.returncode == 0, result.stderr[-1500:]
        assert image.exists() and image.stat().st_size > 1_000

    @pytest.mark.parametrize("spec", CASES, ids=lambda s: s.chart)
    def test_plotly_script_runs(self, tmp_path: Path, csv_path: Path, spec: ChartSpec) -> None:
        script = tmp_path / f"plotly_{spec.chart}.py"
        page = tmp_path / f"{spec.chart}.html"
        codegen.write(
            script, codegen.generate(spec, default_pipeline({}), csv_path, flavour="plotly")
        )

        result = run_script(script, "--save", str(page))

        assert result.returncode == 0, result.stderr[-1500:]
        assert page.exists()

    def test_script_replays_the_whole_pipeline(self, tmp_path: Path, csv_path: Path) -> None:
        """The script's cleaned frame must match what PlotaViz produced, not approximate it."""
        analysis = Analysis.from_file(csv_path)
        analysis.set_filters([QueryFilter("revenue > 1500")])

        script = tmp_path / "repro.py"
        codegen.write(script, analysis.generate_code())

        probe = tmp_path / "probe.py"
        probe.write_text(
            script.read_text() + "\n\nif True:\n"
            "    _df = preprocess(load())\n"
            "    print('ROWS', len(_df))\n"
            "    print('COLS', ','.join(map(str, _df.columns)))\n"
        )
        result = subprocess.run(
            [sys.executable, str(probe), "--no-show"], capture_output=True, text=True, timeout=180
        )

        assert result.returncode == 0, result.stderr[-1500:]
        rows = int(
            next(line for line in result.stdout.splitlines() if line.startswith("ROWS")).split()[1]
        )
        cols = next(line for line in result.stdout.splitlines() if line.startswith("COLS")).split(
            " ", 1
        )[1]

        assert rows == len(analysis.df)
        assert set(cols.split(",")) == set(map(str, analysis.df.columns))

    def test_script_accepts_no_show_without_a_save_path(
        self, tmp_path: Path, csv_path: Path
    ) -> None:
        script = tmp_path / "plain.py"
        codegen.write(
            script,
            codegen.generate(ChartSpec("histogram", x="revenue"), default_pipeline({}), csv_path),
        )
        assert run_script(script).returncode == 0


class TestWrite:
    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        target = codegen.write(tmp_path / "nested" / "deeper" / "out.py", "print('hi')\n")
        assert target.exists()

    def test_unwritable_path_raises_a_readable_error(self, tmp_path: Path) -> None:
        blocker = tmp_path / "afile"
        blocker.write_text("not a directory")
        with pytest.raises(ExportError, match="Could not write"):
            codegen.write(blocker / "out.py", "print('hi')\n")


def test_excel_and_parquet_sources_get_the_right_reader(
    tmp_path: Path, timeseries_df: pd.DataFrame
) -> None:
    parquet = tmp_path / "d.parquet"
    timeseries_df.to_parquet(parquet)

    code = codegen.generate(ChartSpec("histogram", x="revenue"), Pipeline(), parquet)
    assert "pd.read_parquet" in code
