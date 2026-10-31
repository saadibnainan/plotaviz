"""Dashboard tests: pinning several charts, exporting them together, and the Dashboard tab."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import open_and_wait, wait_idle

from plotaviz.core.analysis import MAX_TILES, Analysis
from plotaviz.core.errors import SpecError
from plotaviz.core.exporter import export_dashboard_html, export_dashboard_image
from plotaviz.core.preprocess import QueryFilter
from plotaviz.core.session import Session
from plotaviz.core.spec import ChartSpec
from plotaviz.main import main


@pytest.fixture
def pinned(csv_path: Path) -> Analysis:
    analysis = Analysis.from_file(csv_path)
    for spec in analysis.recommendations[:3]:
        analysis.pin(spec)
    return analysis


# ---------------------------------------------------------------------------- model


class TestModel:
    def test_pin_unpin_move(self, pinned: Analysis) -> None:
        first, second, third = (t.chart for t in pinned.dashboard)
        pinned.move_tile(0, 2)
        assert [t.chart for t in pinned.dashboard] == [second, third, first]
        removed = pinned.unpin(1)
        assert removed.chart == third and len(pinned.dashboard) == 2

    def test_tiles_are_independent_of_the_main_chart(self, pinned: Analysis) -> None:
        pinned.choose(pinned.dashboard[0])
        pinned.spec.title = "edited"  # type: ignore[union-attr]
        assert pinned.dashboard[0].title is None

    def test_limit(self, csv_path: Path) -> None:
        analysis = Analysis.from_file(csv_path)
        spec = ChartSpec("histogram", x="units")
        for _ in range(MAX_TILES):
            analysis.pin(spec)
        with pytest.raises(SpecError):
            analysis.pin(spec)

    def test_tiles_follow_filters_and_are_not_undone(self, pinned: Analysis) -> None:
        pinned.set_filters([QueryFilter("revenue > 2000")])
        pinned.undo()
        assert len(pinned.dashboard) == 3

    def test_invalid_tile_is_flagged_not_dropped(self, pinned: Analysis) -> None:
        pinned.pin(ChartSpec("histogram", x="units"))
        pinned.set_type_overrides({"units": "text"})  # still a column, so still drawable…
        pinned.dashboard.append(ChartSpec("histogram", x="gone"))  # …but this one is not

        statuses = pinned.tile_status()

        assert len(pinned.dashboard) == 5
        assert statuses[-1] is not None and statuses[0] is None

    def test_session_round_trip(self, pinned: Analysis, tmp_path: Path) -> None:
        saved = pinned.to_session().save(tmp_path / "d.pviz")
        restored, _ = Analysis.from_session(Session.load(saved))
        assert [t.to_dict() for t in restored.dashboard] == [t.to_dict() for t in pinned.dashboard]

    def test_old_session_without_dashboard_loads(self, csv_path: Path, tmp_path: Path) -> None:
        saved = Analysis.from_file(csv_path).to_session().save(tmp_path / "old.pviz")
        payload = json.loads(saved.read_text())
        payload.pop("dashboard")
        saved.write_text(json.dumps(payload))
        assert Session.load(saved).dashboard == []


# ---------------------------------------------------------------------------- output


class TestExport:
    @pytest.mark.parametrize("suffix", ["png", "pdf", "svg"])
    def test_grid_image(self, pinned: Analysis, tmp_path: Path, suffix: str) -> None:
        written = export_dashboard_image(pinned.df, pinned.dashboard, tmp_path / f"d.{suffix}")
        assert written.stat().st_size > 1000

    def test_unrenderable_tile_does_not_break_the_grid(self, pinned: Analysis, tmp_path) -> None:
        specs = [*pinned.dashboard, ChartSpec("histogram", x="gone")]
        assert export_dashboard_image(pinned.df, specs, tmp_path / "d.png").exists()

    def test_html_embeds_plotly_once(self, pinned: Analysis, tmp_path: Path) -> None:
        page = export_dashboard_html(pinned.df, pinned.dashboard, tmp_path / "d.html").read_text()
        assert page.count('class="plotly-graph-div"') == 3
        assert page.count("Plotly.newPlot") >= 3
        assert '<script src="https://cdn' not in page

    @pytest.mark.parametrize(("flavour", "suffix"), [("matplotlib", "png"), ("plotly", "html")])
    def test_generated_script_writes_every_chart(
        self, pinned: Analysis, tmp_path: Path, flavour: str, suffix: str
    ) -> None:
        script = tmp_path / "dash.py"
        script.write_text(pinned.generate_dashboard_code(flavour=flavour))
        out = tmp_path / "out"
        result = subprocess.run(
            [sys.executable, str(script), "--no-show", "--save-dir", str(out)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, result.stderr
        assert sorted(p.name for p in out.iterdir()) == [f"chart_{i}.{suffix}" for i in (1, 2, 3)]

    def test_cli(self, csv_path: Path, tmp_path: Path) -> None:
        target = tmp_path / "d.pdf"
        assert (
            main(["--input", str(csv_path), "--dashboard", "3", "--export", str(target), "-q"]) == 0
        )
        assert target.stat().st_size > 1000

    def test_cli_needs_an_output(self, csv_path: Path) -> None:
        with pytest.raises(SystemExit):
            main(["--input", str(csv_path), "--dashboard", "3"])


# ---------------------------------------------------------------------------- GUI


class TestDashboardTab:
    def show_dashboard(self, window, qtbot=None) -> None:
        window.central_tabs.setCurrentIndex(1)
        if qtbot is not None:
            wait_idle(qtbot, window)  # tiles are built on a worker

    def test_pin_updates_the_tab(self, qtbot, window, csv_path: Path) -> None:
        open_and_wait(qtbot, window, csv_path)
        window.pin_current()
        window.recommendations.setCurrentRow(1)
        window.pin_action.trigger()

        assert window.central_tabs.tabText(1) == "Dashboard (2)"
        self.show_dashboard(window, qtbot)
        assert window.dashboard_view.tile_count == 2
        assert len(window.dashboard_view._canvases) == 2

    def test_open_then_update_tile(self, qtbot, window, csv_path: Path) -> None:
        open_and_wait(qtbot, window, csv_path)
        window.pin_current()
        window._open_tile(0)
        assert window.update_tile_button.isVisibleTo(window)

        window.analysis.choose(ChartSpec("histogram", x="units"))
        window._update_tile()

        assert window.analysis.dashboard[0].chart == "histogram"

    def test_filter_redraws_visible_tiles(self, qtbot, window, csv_path: Path) -> None:
        open_and_wait(qtbot, window, csv_path)
        window.pin_current()
        self.show_dashboard(window, qtbot)
        before = window.dashboard_view.summary.text()

        window._on_filters_changed([QueryFilter("revenue > 2000")])
        wait_idle(qtbot, window)

        after = window.dashboard_view.summary.text()
        assert before != after and f"{len(window.analysis.df):,} rows" in after

    def test_hidden_dashboard_redraws_when_shown(self, qtbot, window, csv_path: Path) -> None:
        open_and_wait(qtbot, window, csv_path)
        window.pin_current()
        window._on_filters_changed([QueryFilter("revenue > 2000")])
        wait_idle(qtbot, window)
        assert window._dashboard_stale

        self.show_dashboard(window, qtbot)

        assert not window._dashboard_stale
        assert f"{len(window.analysis.df):,} rows" in window.dashboard_view.summary.text()

    def test_remove_and_reorder(self, qtbot, window, csv_path: Path) -> None:
        open_and_wait(qtbot, window, csv_path)
        for spec in window.analysis.recommendations[:3]:
            window.analysis.pin(spec)
        charts = [t.chart for t in window.analysis.dashboard]

        window._move_tile(2, 1)
        window._remove_tile(0)

        assert [t.chart for t in window.analysis.dashboard] == [charts[2], charts[1]]
        assert window.central_tabs.tabText(1) == "Dashboard (2)"

    def test_session_restores_tiles_and_tab(
        self, qtbot, window, pinned: Analysis, tmp_path: Path
    ) -> None:
        saved = pinned.to_session(view={"central_tab": 1, "dashboard_columns": 3}).save(
            tmp_path / "d.pviz"
        )
        open_and_wait(qtbot, window, saved)

        assert window.central_tabs.currentIndex() == 1
        assert window.dashboard_view.columns == 3
        assert window.dashboard_view.tile_count == 3
