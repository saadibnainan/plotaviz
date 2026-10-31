"""Batch 3: the analysis report, joining a second dataset, Excel sheet stacking, CLI extras."""

from __future__ import annotations

import re
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import open_and_wait, wait_idle

from plotaviz.core import codegen, loader
from plotaviz.core.analysis import Analysis
from plotaviz.core.errors import ExportError, PreprocessError
from plotaviz.core.preprocess import JoinDataset, QueryFilter, join_preview
from plotaviz.core.report import build_report_html, export_report
from plotaviz.core.session import Session
from plotaviz.main import main

# ---------------------------------------------------------------------------- report


class TestReport:
    def test_html_has_every_section_and_works_offline(self, csv_path: Path) -> None:
        analysis = Analysis.from_file(csv_path)
        analysis.pin(analysis.recommendations[1])
        page = build_report_html(analysis, charts=3)
        for heading in ("Summary", "Cleaning", "Columns", "Correlations", "Recommended charts"):
            assert f"<h2>{heading}</h2>" in page
        assert "<h2>Dashboard</h2>" in page
        assert page.count('class="plotly-graph-div"') >= 5
        assert '<script src="https://cdn' not in page

    def test_report_reflects_filters(self, csv_path: Path) -> None:
        analysis = Analysis.from_file(csv_path)
        analysis.set_filters([QueryFilter("region == 'north'")])
        page = build_report_html(analysis, charts=1)
        assert f"<td>{len(analysis.df):,}</td>" in page

    def test_pdf(self, csv_path: Path, tmp_path: Path) -> None:
        written = export_report(Analysis.from_file(csv_path), tmp_path / "r.pdf", charts=2)
        pages = len(re.findall(rb"/Type\s*/Page[^s]", written.read_bytes()))
        assert pages >= 4  # summary, columns, and charts

    def test_unknown_format(self, csv_path: Path, tmp_path: Path) -> None:
        with pytest.raises(ExportError):
            export_report(Analysis.from_file(csv_path), tmp_path / "r.docx")

    def test_cli_report(self, csv_path: Path, tmp_path: Path) -> None:
        target = tmp_path / "r.html"
        assert main(["--input", str(csv_path), "--report", str(target), "-q"]) == 0
        assert target.stat().st_size > 1_000_000

    def test_gui_export(self, qtbot, window, csv_path: Path, tmp_path: Path, monkeypatch) -> None:
        from PySide6.QtWidgets import QFileDialog

        open_and_wait(qtbot, window, csv_path)
        target = tmp_path / "gui_report.html"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), ""))
        window.export_report()
        wait_idle(qtbot, window)
        assert target.exists()


# ---------------------------------------------------------------------------- join


@pytest.fixture
def regions_file(tmp_path: Path) -> Path:
    target = tmp_path / "regions.csv"
    pd.DataFrame(
        {
            "Region": ["north", "south", "east", "west"],
            "Manager": ["Ana", "Ben", "Cho", "Dee"],
            "Opened": ["2019-01-05", "2020-03-01", "2018-07-15", "2021-02-02"],
        }
    ).to_csv(target, index=False)
    return target


class TestJoin:
    def test_left_join_adds_columns_and_reports_matches(self, csv_path, regions_file) -> None:
        analysis = Analysis.from_file(csv_path)
        rows = len(analysis.df)
        analysis.join_dataset(regions_file, ["region"])

        assert {"manager", "opened"} <= set(analysis.df.columns)
        assert len(analysis.df) == rows
        assert pd.api.types.is_datetime64_any_dtype(analysis.df["opened"])  # typed like the rest
        report = next(r for r in analysis.result.reports if r.step == "join_dataset")
        assert f"{rows:,} of {rows:,} rows matched" in report.summary

    def test_inner_join_drops_unmatched(self, csv_path, tmp_path) -> None:
        other = tmp_path / "two.csv"
        pd.DataFrame({"region": ["north"], "target": [1.0]}).to_csv(other, index=False)
        analysis = Analysis.from_file(csv_path)
        analysis.join_dataset(other, ["region"], how="inner")
        assert set(analysis.df["region"]) == {"north"}

    def test_repeated_keys_are_reported(self, csv_path, tmp_path) -> None:
        other = tmp_path / "dup.csv"
        pd.DataFrame({"region": ["north", "north"], "x": [1, 2]}).to_csv(other, index=False)
        analysis = Analysis.from_file(csv_path)
        rows = len(analysis.df)
        analysis.join_dataset(other, ["region"])
        assert len(analysis.df) > rows
        assert "multiplied" in analysis.result.reports[1].summary

    def test_undo_and_session_replay(self, csv_path, regions_file, tmp_path) -> None:
        analysis = Analysis.from_file(csv_path)
        analysis.join_dataset(regions_file, ["region"])
        saved = analysis.to_session().save(tmp_path / "j.pviz")
        restored, _ = Analysis.from_session(Session.load(saved))
        assert "manager" in restored.df.columns

        analysis.undo()
        assert "manager" not in analysis.df.columns

    def test_generated_script_reproduces_the_join(self, csv_path, regions_file, tmp_path) -> None:
        analysis = Analysis.from_file(csv_path)
        analysis.join_dataset(regions_file, ["region"])
        script = codegen.write(tmp_path / "p.py", analysis.generate_code())
        generated = runpy.run_path(str(script), run_name="generated")
        frame = generated["preprocess"](generated["load"]())  # type: ignore[operator]
        assert frame.shape == analysis.df.shape
        assert list(frame.columns) == list(analysis.df.columns)

    def test_bad_key_and_missing_file(self, csv_path, regions_file, tmp_path) -> None:
        analysis = Analysis.from_file(csv_path)
        with pytest.raises(PreprocessError):
            analysis.join_dataset(regions_file, ["manager"])
        assert not any(isinstance(s, JoinDataset) for s in analysis.pipeline)

        step = JoinDataset(str(tmp_path / "gone.csv"), ["region"])
        with pytest.raises(PreprocessError, match="missing"):
            step.apply(analysis.df)

    def test_preview_numbers(self) -> None:
        left = pd.DataFrame({"k": [1, 2, 3, 3]})
        right = pd.DataFrame({"k": [3, 3, 4]})
        assert join_preview(left, right, ["k"]) == {
            "rows": 4,
            "matched_rows": 2,
            "duplicate_keys_in_other": 1,
        }

    def test_dialog_suggests_shared_keys(self, qtbot, csv_path, regions_file) -> None:
        from plotaviz.ui.join_dialog import JoinDialog

        dialog = JoinDialog(Analysis.from_file(csv_path).df)
        qtbot.addWidget(dialog)
        assert dialog.set_file(regions_file)
        assert dialog.keys() == ["region"]
        assert "rows (100%) find a match" in dialog.preview_label.text()

    def test_window_join(self, qtbot, window, csv_path, regions_file, monkeypatch) -> None:
        from plotaviz.ui.join_dialog import JoinDialog

        open_and_wait(qtbot, window, csv_path)
        monkeypatch.setattr(JoinDialog, "exec", lambda self: JoinDialog.DialogCode.Accepted)
        monkeypatch.setattr(
            JoinDialog,
            "result_values",
            lambda self: {"path": str(regions_file), "on": ["region"], "how": "left"},
        )
        window.join_dataset()
        wait_idle(qtbot, window)
        assert "manager" in window.analysis.df.columns
        assert "manager" in window.filter_panel._categorical


# ---------------------------------------------------------------------------- Excel sheets


def test_all_sheets_stacked(tmp_path: Path) -> None:
    target = tmp_path / "months.xlsx"
    with pd.ExcelWriter(target) as writer:
        pd.DataFrame({"sales": [1, 2]}).to_excel(writer, sheet_name="Jan", index=False)
        pd.DataFrame({"sales": [3, 4, 5]}).to_excel(writer, sheet_name="Feb", index=False)

    result = loader.load(target, options={"sheet": "*"})
    assert list(result.df["sheet"]) == ["Jan", "Jan", "Feb", "Feb", "Feb"]

    analysis = Analysis.from_file(target, options={"sheet": "*"})
    script = codegen.write(tmp_path / "p.py", analysis.generate_code())
    generated = runpy.run_path(str(script), run_name="generated")
    frame = generated["preprocess"](generated["load"]())  # type: ignore[operator]
    assert frame.shape == analysis.df.shape


# ---------------------------------------------------------------------------- CLI


class TestCliExtras:
    def test_sheet(self, tmp_path: Path) -> None:
        target = tmp_path / "book.xlsx"
        with pd.ExcelWriter(target) as writer:
            pd.DataFrame({"a": [1]}).to_excel(writer, sheet_name="Cover", index=False)
            pd.DataFrame({"v": np.arange(40.0)}).to_excel(writer, sheet_name="Data", index=False)
        out = tmp_path / "c.png"
        assert main(["--input", str(target), "--sheet", "Data", "--export", str(out), "-q"]) == 0
        assert out.exists()

    def test_custom_rules(self, csv_path: Path, tmp_path: Path, capsys) -> None:
        rules = tmp_path / "rules.yaml"
        rules.write_text(
            "rules:\n  - name: only_hist\n    requires: { numeric: 1 }\n    candidates:\n"
            "      - chart: histogram\n        base: 0.9\n        map: { x: numeric.0 }\n"
            "        why: custom\n"
        )
        assert main(["--input", str(csv_path), "--recommend", "--json", "--rules", str(rules)]) == 0
        import json

        specs = json.loads(capsys.readouterr().out)
        assert {s["chart"] for s in specs} == {"histogram"}

    def test_batch_mode(self, tmp_path: Path, timeseries_df, categorical_df) -> None:
        folder = tmp_path / "in"
        folder.mkdir()
        timeseries_df.to_csv(folder / "a.csv", index=False)
        categorical_df.to_csv(folder / "b.csv", index=False)
        (folder / "c.csv").write_text("only_a_header\n")
        out = tmp_path / "out"

        code = main(
            ["--input", str(folder / "*.csv"), "--output-dir", str(out), "--export", ".png"]
        )

        assert code == 1  # c.csv failed, the others still ran
        assert sorted(p.name for p in out.iterdir()) == ["a.png", "b.png"]

    def test_batch_needs_an_output_dir(self, csv_path: Path, tmp_path: Path) -> None:
        second = tmp_path / "second.csv"
        second.write_bytes(csv_path.read_bytes())
        with pytest.raises(SystemExit):
            main(["--input", str(csv_path), str(second), "--export", ".png"])

    def test_cli_runs_as_a_script(self, csv_path: Path, tmp_path: Path) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "plotaviz.main", "--input", str(csv_path), "--describe"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0 and "rows ×" in result.stdout
