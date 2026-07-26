"""Session tests — round-tripping a project, and noticing when the source data moved."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from plotaviz.core.analysis import Analysis
from plotaviz.core.errors import SessionError
from plotaviz.core.preprocess import QueryFilter, default_pipeline
from plotaviz.core.session import SESSION_SUFFIX, SESSION_VERSION, Session
from plotaviz.core.spec import ChartSpec


class TestRoundTrip:
    def test_saves_and_reloads(self, tmp_path: Path, csv_path: Path) -> None:
        session = Session(
            source_path=str(csv_path),
            type_overrides={"revenue": "numeric"},
            pipeline=default_pipeline({"revenue": "numeric"}),
            spec=ChartSpec("line", x="order_date", y="revenue", why="because"),
            view={"renderer": "interactive"},
        )
        written = session.save(tmp_path / "project")

        restored = Session.load(written)

        assert restored.source_path == str(csv_path)
        assert restored.type_overrides == {"revenue": "numeric"}
        assert restored.pipeline.to_list() == session.pipeline.to_list()
        assert restored.spec is not None
        assert restored.spec.chart == "line"
        assert restored.view["renderer"] == "interactive"

    def test_adds_the_pviz_suffix(self, tmp_path: Path, csv_path: Path) -> None:
        written = Session(source_path=str(csv_path)).save(tmp_path / "project.txt")
        assert written.suffix == SESSION_SUFFIX

    def test_records_the_source_hash_and_a_timestamp(self, tmp_path: Path, csv_path: Path) -> None:
        written = Session(source_path=str(csv_path)).save(tmp_path / "s")
        payload = json.loads(written.read_text())

        assert len(payload["source"]["sha256"]) == 64
        assert payload["saved_at"]
        assert payload["version"] == SESSION_VERSION

    def test_is_plain_readable_json(self, tmp_path: Path, csv_path: Path) -> None:
        written = Session(source_path=str(csv_path)).save(tmp_path / "s")
        assert isinstance(json.loads(written.read_text()), dict)

    def test_never_contains_credentials(self, tmp_path: Path, csv_path: Path) -> None:
        """Keys live in the OS keyring. A session must not be able to leak one."""
        written = Session(
            source_path=str(csv_path),
            pipeline=default_pipeline({}),
            spec=ChartSpec("histogram", x="revenue"),
            view={"renderer": "static"},
        ).save(tmp_path / "s")

        text = written.read_text().lower()
        for forbidden in ("api_key", "apikey", "sk-", "token", "secret", "password"):
            assert forbidden not in text


class TestSourceChecks:
    def test_unchanged_source_produces_no_warning(self, tmp_path: Path, csv_path: Path) -> None:
        written = Session(source_path=str(csv_path)).save(tmp_path / "s")
        assert Session.load(written).check_source() is None

    def test_changed_source_is_reported(self, tmp_path: Path, timeseries_df: pd.DataFrame) -> None:
        source = tmp_path / "data.csv"
        timeseries_df.to_csv(source, index=False)
        written = Session(source_path=str(source)).save(tmp_path / "s")

        timeseries_df.head(10).to_csv(source, index=False)
        warning = Session.load(written).check_source()

        assert warning is not None
        assert "has changed" in warning

    def test_missing_source_is_reported(self, tmp_path: Path, csv_path: Path) -> None:
        written = Session(source_path=str(csv_path)).save(tmp_path / "s")
        csv_path.unlink()

        warning = Session.load(written).check_source()
        assert warning is not None
        assert "missing" in warning

    def test_session_without_a_source_says_so(self) -> None:
        assert "does not record" in (Session(source_path="").check_source() or "")


class TestFailureModes:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(SessionError, match="No session file"):
            Session.load(tmp_path / "nope.pviz")

    def test_malformed_json(self, tmp_path: Path) -> None:
        target = tmp_path / "bad.pviz"
        target.write_text("{not json")
        with pytest.raises(SessionError, match="not a valid PlotaViz session"):
            Session.load(target)

    def test_newer_format_suggests_updating(self, tmp_path: Path) -> None:
        target = tmp_path / "future.pviz"
        target.write_text(json.dumps({"version": SESSION_VERSION + 5, "source": {"path": "x"}}))

        with pytest.raises(SessionError, match="Update PlotaViz"):
            Session.load(target)

    def test_unknown_step_is_reported_clearly(self, tmp_path: Path) -> None:
        target = tmp_path / "odd.pviz"
        target.write_text(
            json.dumps(
                {
                    "version": SESSION_VERSION,
                    "source": {"path": "x"},
                    "pipeline": [{"kind": "warp"}],
                }
            )
        )
        with pytest.raises(SessionError, match="could not be restored"):
            Session.load(target)


class TestAnalysisIntegration:
    def test_restores_a_full_analysis(self, tmp_path: Path, csv_path: Path) -> None:
        analysis = Analysis.from_file(csv_path)
        analysis.set_filters([QueryFilter("revenue > 1200")])
        analysis.choose(ChartSpec("line", x="order_date", y="revenue"))
        rows_before, spec_before = len(analysis.df), analysis.spec

        written = analysis.to_session({"renderer": "static"}).save(tmp_path / "s")
        restored, warning = Analysis.from_session(Session.load(written))

        assert warning is None
        assert len(restored.df) == rows_before
        assert restored.spec is not None
        assert spec_before is not None
        assert restored.spec.chart == spec_before.chart
        assert restored.spec.x == spec_before.x

    def test_restoring_after_the_source_changed_warns_but_still_works(
        self, tmp_path: Path, timeseries_df: pd.DataFrame
    ) -> None:
        source = tmp_path / "data.csv"
        timeseries_df.to_csv(source, index=False)

        analysis = Analysis.from_file(source)
        written = analysis.to_session().save(tmp_path / "s")

        timeseries_df.head(60).to_csv(source, index=False)
        restored, warning = Analysis.from_session(Session.load(written))

        assert warning is not None
        assert len(restored.df) < len(analysis.df)

    def test_a_chart_that_no_longer_fits_is_reported_not_fatal(
        self, tmp_path: Path, timeseries_df: pd.DataFrame
    ) -> None:
        source = tmp_path / "data.csv"
        timeseries_df.to_csv(source, index=False)
        session = Session(
            source_path=str(source),
            pipeline=default_pipeline({}),
            spec=ChartSpec("scatter", x="revenue", y="a_column_that_went_away"),
        )
        session.save(tmp_path / "s")

        restored, warning = Analysis.from_session(session)

        assert warning is not None
        assert "could not be restored" in warning
        assert restored.df is not None  # the data still loaded
