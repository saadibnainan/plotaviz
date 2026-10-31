"""Regression tests for the defects found in the v0.1.0 audit (see ``docs/AUDIT.md``).

Each test reproduces one reported failure against the headless core. They are grouped by the
audit's identifiers so a failure points straight at the write-up.
"""

from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from plotaviz.core import codegen, loader, plotter
from plotaviz.core.analysis import Analysis
from plotaviz.core.errors import PreprocessError
from plotaviz.core.llm import CONSENT_TEXT, LLMAssistant, OpenAIProvider, build_prompt
from plotaviz.core.llm.ollama import OllamaProvider, is_loopback
from plotaviz.core.preprocess import (
    CoerceTypes,
    ColumnFilter,
    FillMissing,
    Pipeline,
    QueryFilter,
    Step,
)
from plotaviz.core.profiler import infer_role, looks_like_datetime, profile
from plotaviz.core.spec import ChartSpec


def load_generated(path: Path) -> dict[str, object]:
    """Import a generated script's functions without running its ``__main__`` block."""
    return runpy.run_path(str(path), run_name="generated")


# ---------------------------------------------------------------------------- P0


class TestBooleanWithMissingValues:
    """P0-2: a yes/no column with a blank cell aborted the whole load."""

    def test_file_loads_and_blank_stays_missing(self, tmp_path: Path) -> None:
        target = tmp_path / "flags.csv"
        pd.DataFrame({"active": ["yes", "no", None, "yes"] * 10, "score": np.arange(40.0)}).to_csv(
            target, index=False
        )

        analysis = Analysis.from_file(target)

        assert str(analysis.df["active"].dtype) == "boolean"
        assert int(analysis.df["active"].isna().sum()) == 10

    @pytest.mark.parametrize("strategy", ["mean", "median", "zero"])
    def test_numeric_strategies_skip_booleans(self, strategy: str) -> None:
        df = pd.DataFrame({"flag": pd.array([True, None, False], dtype="boolean")})
        out, report = FillMissing(strategy).apply(df)
        assert out["flag"].isna().sum() == 1
        assert "flag" not in report.details["filled"]


class TestEuropeanNumbers:
    """P0-3: ``1,5`` in a semicolon CSV was classified as a date and coerced into timestamps."""

    def test_comma_decimals_are_not_dates(self) -> None:
        values = pd.Series(["1,5", "2,75", "3,1", "12,4", "7,25"] * 10, name="umsatz")
        assert not looks_like_datetime(values)
        assert infer_role(values)[0] != "datetime"

    @pytest.mark.parametrize("codes", [[f"t{i}" for i in range(15)], ["M1", "M2", "M3"]])
    def test_short_codes_are_not_dates(self, codes: list[str]) -> None:
        """dateutil reads "t13" as a day and "M2" as seconds; neither is a date column."""
        assert not looks_like_datetime(pd.Series(codes * 4, name="team"))

    @pytest.mark.parametrize(
        "values",
        [
            ["2024-01-05", "2024-02-07"],
            ["05/02/2024", "17/03/2024"],
            ["March 3 2024", "4 Apr 2024"],
            ["20240105", "20240207"],
            ["2024-01-05T10:30:00", "2024-01-06T11:00:00"],
        ],
    )
    def test_real_date_formats_still_parse(self, values: list[str]) -> None:
        assert looks_like_datetime(pd.Series(values * 10, name="when"))

    def test_day_first_dates_still_parse(self) -> None:
        values = pd.Series(["01.05.2024", "02.05.2024", "15.06.2024"] * 10, name="datum")
        assert infer_role(values)[0] == "datetime"

    def test_semicolon_csv_reads_numbers_as_numbers(self, tmp_path: Path) -> None:
        target = tmp_path / "umsatz.csv"
        target.write_text(
            "Region;Umsatz;Menge\n"
            + "".join(f"Nord;1.234,{i:02d};{i}\nSüd;{i},5;{i + 1}\n" for i in range(30)),
            encoding="utf-8",
        )

        result = loader.load(target)

        assert pd.api.types.is_float_dtype(result.df["Umsatz"])
        assert result.df["Umsatz"].iloc[0] == pytest.approx(1234.0)
        assert result.read_options == {"sep": ";", "decimal": ",", "thousands": "."}
        assert any("European" in note for note in result.notes)

    def test_generated_script_reads_the_file_the_same_way(self, tmp_path: Path) -> None:
        target = tmp_path / "umsatz.csv"
        target.write_text(
            "Region;Umsatz\n" + "".join(f"R{i % 3};{i},5\n" for i in range(40)),
            encoding="utf-8",
        )
        analysis = Analysis.from_file(target)
        script = codegen.write(tmp_path / "plot.py", analysis.generate_code())

        generated = load_generated(script)
        frame = generated["preprocess"](generated["load"]())  # type: ignore[operator]

        assert "decimal=','" in script.read_text()
        assert pd.api.types.is_numeric_dtype(frame["umsatz"])

    def test_mixed_conventions_are_left_alone(self, tmp_path: Path) -> None:
        target = tmp_path / "mixed.csv"
        target.write_text("a;b\n" + "".join(f"{i},5;{i}.25\n" for i in range(20)), encoding="utf-8")
        assert loader.load(target).read_options == {"sep": ";"}


# ---------------------------------------------------------------------------- P1


class TestFilterRollback:
    """P1-7: a rejected query stayed in the pipeline and broke every later replay."""

    def test_bad_query_restores_previous_filters(self, csv_path: Path) -> None:
        analysis = Analysis.from_file(csv_path)
        good = [QueryFilter("revenue > 0")]
        analysis.set_filters(good)
        rows = len(analysis.df)

        with pytest.raises(PreprocessError):
            analysis.set_filters([QueryFilter("no_such_column >")])

        assert [s.to_dict() for s in analysis.pipeline.filters] == [g.to_dict() for g in good]
        analysis.rerun()  # still replays cleanly
        assert len(analysis.df) == rows

    def test_filter_matching_nothing_keeps_the_previous_data(self, csv_path: Path) -> None:
        analysis = Analysis.from_file(csv_path)
        rows, spec = len(analysis.df), analysis.spec

        with pytest.raises(PreprocessError, match="No rows match"):
            analysis.set_filters([QueryFilter("revenue < -1e12")])

        assert len(analysis.df) == rows
        assert analysis.spec is spec
        assert not analysis.pipeline.filters


class TestChosenChartSurvivesReplays:
    """P0-6 (core half): a replay reset any lower-ranked chart to the top recommendation."""

    def test_user_choice_is_kept_and_auto_choice_follows(self, csv_path: Path) -> None:
        analysis = Analysis.from_file(csv_path)
        assert analysis.auto_spec
        chosen = analysis.recommendations[2]
        analysis.choose(chosen)

        analysis.set_filters([QueryFilter("revenue > 0")])

        assert analysis.spec is chosen
        assert not analysis.auto_spec

    def test_falls_back_when_the_chosen_column_disappears(self, csv_path: Path) -> None:
        analysis = Analysis.from_file(csv_path)
        analysis.choose(ChartSpec("histogram", x="units"))
        analysis.pipeline.steps.append(_DropColumn("units"))

        analysis.rerun()

        assert analysis.spec is not None and "units" not in analysis.spec.columns_used
        assert analysis.auto_spec


class TestTypeOverrideReset:
    """P1-12: removing an override left its coercion in place."""

    def test_reset_restores_inferred_coercion(self, csv_path: Path) -> None:
        analysis = Analysis.from_file(csv_path)
        assert pd.api.types.is_numeric_dtype(analysis.df["units"])

        analysis.set_type_overrides({"units": "categorical"})
        assert analysis.df["units"].dtype == object

        analysis.set_type_overrides({})
        coerce = next(s for s in analysis.pipeline if isinstance(s, CoerceTypes))
        assert coerce.types["units"] == analysis.inferred_types["units"]
        assert pd.api.types.is_numeric_dtype(analysis.df["units"])


class TestCategoryDtypeCapping:
    """P1-10: "Other" bucketing raised on a pandas ``category`` column."""

    def test_more_than_the_cap_renders(self) -> None:
        df = pd.DataFrame(
            {"city": pd.Categorical([f"c{i}" for i in range(60)]), "pop": np.arange(60.0)}
        )
        prepared = plotter.prepare(df, ChartSpec("bar", x="city", y="pop", agg="sum"))
        assert plotter.OTHER_LABEL in set(prepared.df["city"])
        assert len(prepared.df) == plotter.MAX_CATEGORIES + 1


class TestMatrixChartsExcludeIdentifiers:
    """P1-13: ``order_id`` appeared in the correlation heatmap."""

    @pytest.fixture
    def with_ids(self, rng: np.random.Generator) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "order_id": np.arange(200),
                "a": rng.normal(size=200),
                "b": rng.normal(size=200),
                "c": rng.normal(size=200),
            }
        )

    def test_unpinned_spec(self, with_ids: pd.DataFrame) -> None:
        prepared = plotter.prepare(with_ids, ChartSpec("correlation_heatmap"))
        assert "order_id" not in prepared.df.columns

    def test_selector_pins_measure_columns(self, with_ids: pd.DataFrame) -> None:
        from plotaviz.core.selector import ChartSelector

        specs = ChartSelector().recommend(profile(with_ids))
        matrix = next(s for s in specs if s.chart == "correlation_heatmap")
        assert matrix.options["columns"] == ["a", "b", "c"]


class TestGeneratedCodeMatchesTheApp:
    """P1-11: the exported script read, coerced, and grouped data differently from the app."""

    def test_mixed_dates_and_yes_no_replay_identically(self, tmp_path: Path) -> None:
        target = tmp_path / "events.csv"
        pd.DataFrame(
            {
                "event_date": ["2024-01-05", "05/02/2024", "March 3 2024", "2024-04-01"] * 15,
                "active": ["yes", "no", "yes", "no"] * 15,
                "amount": np.arange(60.0),
            }
        ).to_csv(target, index=False)
        analysis = Analysis.from_file(target)
        script = codegen.write(tmp_path / "plot.py", analysis.generate_code())

        generated = load_generated(script)
        frame = generated["preprocess"](generated["load"]())  # type: ignore[operator]

        assert frame["event_date"].notna().sum() == analysis.df["event_date"].notna().sum()
        assert str(frame["active"].dtype) == "boolean"
        assert len(frame) == len(analysis.df)
        result = subprocess.run(
            [sys.executable, str(script), "--no-show", "--save", str(tmp_path / "o.png")],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, result.stderr

    def test_category_cap_is_replayed(self, tmp_path: Path) -> None:
        target = tmp_path / "many.csv"
        pd.DataFrame({"product": [f"p{i:02d}" for i in range(40)] * 3}).to_csv(target, index=False)
        analysis = Analysis.from_file(target)
        analysis.choose(ChartSpec("bar", x="product", agg="count"))
        app_rows = len(analysis.prepared().df)
        script = codegen.write(tmp_path / "plot.py", analysis.generate_code())

        generated = load_generated(script)
        figure = generated["build_chart"](  # type: ignore[operator]
            generated["preprocess"](generated["load"]())  # type: ignore[operator]
        )

        assert app_rows == plotter.MAX_CATEGORIES + 1
        assert len(figure.axes[0].patches) == app_rows

    def test_colour_nan_rows_are_kept(self) -> None:
        spec = ChartSpec("bar", x="a", y="b", color="c", agg="sum")
        code = codegen.generate(spec, Pipeline(), "x.csv")
        assert "dropna(subset=['a', 'b'])" in code
        assert "dropna=False" in code

    def test_fill_missing_code_honours_columns(self) -> None:
        df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [1.0, None, 5.0]})
        step = FillMissing("median", columns=["a"])
        namespace: dict[str, object] = {"pd": pd, "df": df.copy()}
        exec("\n".join(step.to_code()), namespace)
        replayed = namespace["df"]
        expected, _ = step.apply(df)
        pd.testing.assert_frame_equal(replayed, expected)  # type: ignore[arg-type]

    @pytest.mark.parametrize("title", ['Revenue """ breakout', "C:\\Users\\data", 'ends with "'])
    def test_hostile_titles_still_compile(self, title: str) -> None:
        code = codegen.generate(ChartSpec("histogram", x="v", title=title), Pipeline(), "x.csv")
        compile(code, "<generated>", "exec")


class TestDateFilterBounds:
    """P0-1 (core half): a date-only end bound excluded rows later that day."""

    def test_end_date_is_inclusive(self) -> None:
        df = pd.DataFrame(
            {"ts": pd.to_datetime(["2024-03-30 09:00", "2024-03-31 14:00", "2024-04-01 00:00"])}
        )
        step = ColumnFilter("ts", "between", ["2024-03-30", "2024-03-31"])

        out, _ = step.apply(df)
        namespace: dict[str, object] = {"pd": pd, "df": df.copy()}
        exec("\n".join(step.to_code()), namespace)

        assert len(out) == 2
        assert len(namespace["df"]) == 2  # type: ignore[arg-type]

    def test_timezone_aware_column(self) -> None:
        df = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")})
        out, _ = ColumnFilter("ts", "between", ["2024-01-02", "2024-01-03"]).apply(df)
        assert len(out) == 2

    def test_profiler_reports_the_date_range(self, timeseries_df: pd.DataFrame) -> None:
        stats = profile(timeseries_df).columns["order_date"].stats
        assert stats["min"].startswith("2025-01-01")
        assert stats["max"].startswith("2025-06-29")


class TestCategoricalFilterValues:
    """P1-9: only the ten most frequent values were offered in the filter list."""

    def test_all_values_up_to_the_listing_cap(self) -> None:
        df = pd.DataFrame({"team": [f"t{i}" for i in range(15)] * 4})
        assert len(profile(df).columns["team"].top_values) == 15


# ---------------------------------------------------------------------------- loading


def test_json_object_of_arrays(tmp_path: Path) -> None:
    """P2: ``{"col": [...]}`` loaded as a single row of lists."""
    target = tmp_path / "cols.json"
    target.write_text(json.dumps({"a": [1, 2, 3], "b": ["x", "y", "z"]}))
    assert loader.load(target).df.shape == (3, 2)


# ---------------------------------------------------------------------------- privacy


class TestLLMPayload:
    """Security 19: the prompt and the consent text disagreed about what is sent."""

    def test_identifier_and_text_values_are_withheld(self, messy_df: pd.DataFrame) -> None:
        prof = profile(messy_df)
        payload = json.loads(build_prompt(prof).split("\n", 1)[1].split("\n\n")[0])
        by_name = {c["name"]: c for c in payload["columns"]}

        for name, column in prof.columns.items():
            if column.withholds_values:
                assert by_name[name]["sample_values"] == []
                assert by_name[name]["top_values"] == []

    def test_consent_prompt_describes_the_real_payload(self) -> None:
        prompt = LLMAssistant(OpenAIProvider()).consent_prompt()
        assert prompt is not None and CONSENT_TEXT in prompt
        assert "sample rows" not in CONSENT_TEXT


class TestOllamaHost:
    """Security 17: a remote Ollama host skipped the consent gate."""

    @pytest.mark.parametrize(
        ("host", "local"),
        [
            ("http://localhost:11434", True),
            ("http://127.0.0.1:11434", True),
            ("http://[::1]:11434", True),
            ("http://192.168.1.20:11434", False),
            ("https://ollama.example.com", False),
        ],
    )
    def test_loopback_detection(self, host: str, local: bool) -> None:
        assert is_loopback(host) is local
        assert OllamaProvider(host=host).sends_data_off_machine is (not local)

    def test_remote_ollama_needs_consent(self) -> None:
        assistant = LLMAssistant(OllamaProvider(host="http://10.0.0.5:11434"), consent=False)
        assert assistant.consent_prompt() is not None
        assert assistant.available is False


# ---------------------------------------------------------------------------- helpers


class _DropColumn(Step):
    """Test-only step that removes a column, to simulate a mapping becoming invalid."""

    kind = "test_drop_column"

    def __init__(self, column: str) -> None:
        self.column = column

    def apply(self, df: pd.DataFrame):  # type: ignore[override]
        from plotaviz.core.preprocess import StepReport

        out = df.drop(columns=[self.column])
        return out, StepReport(self.kind, "dropped", len(df), len(out))
