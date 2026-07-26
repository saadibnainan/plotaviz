"""Preprocessing tests — step behaviour, replay, serialization, and generated code.

The properties that matter most here are not "does fillna work" but the structural guarantees the
step design is supposed to provide: steps do not mutate their input, a pipeline replays to the
same result every time, and every step round-trips through a session file.
"""

from __future__ import annotations

import pandas as pd
import pytest

from plotaviz.core.errors import PreprocessError
from plotaviz.core.preprocess import (
    OUTLIER_FLAG_SUFFIX,
    CoerceTypes,
    ColumnFilter,
    Deduplicate,
    FillMissing,
    FlagOutliers,
    NormalizeColumnNames,
    Pipeline,
    QueryFilter,
    Step,
    default_pipeline,
)


class TestNormalizeColumnNames:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (" Total Revenue ", "total_revenue"),
            ("customerName", "customer_name"),
            ("Signup-Date", "signup_date"),
            ("A  B", "a_b"),
            ("2024 Q1", "2024_q1"),
            ("!!!", "column"),
        ],
    )
    def test_normalizes(self, raw: str, expected: str) -> None:
        assert NormalizeColumnNames.normalize(raw) == expected

    def test_deduplicates_collisions(self) -> None:
        df = pd.DataFrame([[1, 2, 3]], columns=["a b", "a_b", "A B"])
        out, _ = NormalizeColumnNames().apply(df)
        assert list(out.columns) == ["a_b", "a_b_2", "a_b_3"]

    def test_reports_what_changed(self, messy_df: pd.DataFrame) -> None:
        _, report = NormalizeColumnNames().apply(messy_df)
        assert report.details["renamed"][" Total Revenue "] == "total_revenue"


class TestCoerceTypes:
    def test_parses_dates_and_numbers(self) -> None:
        df = pd.DataFrame({"when": ["2025-01-01", "2025-01-02"], "n": ["1.5", "2.5"]})
        out, report = CoerceTypes({"when": "datetime", "n": "numeric"}).apply(df)

        assert pd.api.types.is_datetime64_any_dtype(out["when"])
        assert pd.api.types.is_numeric_dtype(out["n"])
        assert report.details["applied"] == {"when": "datetime", "n": "numeric"}

    def test_unconvertible_column_is_left_alone_not_fatal(self) -> None:
        df = pd.DataFrame({"n": ["abc", "def"]})
        out, _ = CoerceTypes({"n": "numeric"}).apply(df)
        assert out["n"].isna().all()  # coerced to NaN rather than raising

    def test_ignores_columns_that_are_not_present(self) -> None:
        df = pd.DataFrame({"a": [1]})
        out, report = CoerceTypes({"missing": "numeric"}).apply(df)
        assert list(out.columns) == ["a"]
        assert report.details["applied"] == {}


class TestFillMissing:
    def test_median_fill(self) -> None:
        df = pd.DataFrame({"v": [1.0, None, 3.0]})
        out, report = FillMissing("median").apply(df)

        assert out["v"].tolist() == [1.0, 2.0, 3.0]
        assert report.details["missing_before"]["v"] == 1

    def test_drop_removes_rows_and_reports_the_count(self) -> None:
        df = pd.DataFrame({"v": [1.0, None, 3.0]})
        out, report = FillMissing("drop").apply(df)

        assert len(out) == 2
        assert report.rows_removed == 1

    @pytest.mark.parametrize("strategy", ["mean", "median", "mode", "ffill", "bfill", "zero"])
    def test_every_strategy_removes_the_nulls(self, strategy: str) -> None:
        df = pd.DataFrame({"v": [1.0, None, 3.0, None, 5.0]})
        out, _ = FillMissing(strategy).apply(df)
        assert out["v"].isna().sum() == 0

    def test_unknown_strategy_is_rejected_at_construction(self) -> None:
        with pytest.raises(PreprocessError, match="strategy"):
            FillMissing("interpolate-with-vibes")

    def test_no_missing_values_is_a_no_op(self) -> None:
        df = pd.DataFrame({"v": [1.0, 2.0]})
        out, report = FillMissing("median").apply(df)
        assert out.equals(df)
        assert "No missing values" in report.summary


class TestFlagOutliers:
    def test_flags_without_removing(self) -> None:
        df = pd.DataFrame({"v": [1.0] * 20 + [500.0]})
        out, report = FlagOutliers("iqr").apply(df)

        assert len(out) == len(df)
        assert f"v{OUTLIER_FLAG_SUFFIX}" in out.columns
        assert out[f"v{OUTLIER_FLAG_SUFFIX}"].sum() == 1
        assert report.details["per_column"]["v"] == 1

    def test_drop_mode_removes_them(self) -> None:
        df = pd.DataFrame({"v": [1.0] * 20 + [500.0]})
        out, _ = FlagOutliers("iqr", drop=True).apply(df)

        assert len(out) == 20
        assert f"v{OUTLIER_FLAG_SUFFIX}" not in out.columns

    def test_zscore_method(self) -> None:
        df = pd.DataFrame({"v": [1.0] * 50 + [1000.0]})
        _, report = FlagOutliers("zscore", threshold=3.0).apply(df)
        assert report.details["per_column"].get("v", 0) >= 1

    def test_constant_column_has_no_outliers(self) -> None:
        df = pd.DataFrame({"v": [5.0] * 10})
        out, report = FlagOutliers().apply(df)
        assert report.details["per_column"] == {}
        assert list(out.columns) == ["v"]

    def test_unknown_method_is_rejected(self) -> None:
        with pytest.raises(PreprocessError, match="method"):
            FlagOutliers("isolation-forest")


class TestFilters:
    def test_query_filter(self, timeseries_df: pd.DataFrame) -> None:
        out, report = QueryFilter("revenue > 2000").apply(timeseries_df)
        assert (out["revenue"] > 2000).all()
        assert report.rows_after == len(out)

    def test_invalid_query_explains_itself(self, timeseries_df: pd.DataFrame) -> None:
        with pytest.raises(PreprocessError) as exc:
            QueryFilter("revenue >>> nonsense").apply(timeseries_df)
        assert "pandas query syntax" in str(exc.value)

    def test_empty_query_is_a_no_op(self, timeseries_df: pd.DataFrame) -> None:
        out, _ = QueryFilter("   ").apply(timeseries_df)
        assert len(out) == len(timeseries_df)

    def test_between_filter(self, numeric_pair_df: pd.DataFrame) -> None:
        out, _ = ColumnFilter("width", "between", [40, 60]).apply(numeric_pair_df)
        assert out["width"].between(40, 60).all()

    def test_in_filter(self, categorical_df: pd.DataFrame) -> None:
        out, _ = ColumnFilter("department", "in", ["eng"]).apply(categorical_df)
        assert set(out["department"]) == {"eng"}

    def test_filter_on_a_missing_column_is_skipped_not_fatal(
        self, categorical_df: pd.DataFrame
    ) -> None:
        out, report = ColumnFilter("gone", "in", ["x"]).apply(categorical_df)
        assert len(out) == len(categorical_df)
        assert "not present" in report.summary


class TestPipeline:
    def test_steps_do_not_mutate_their_input(self, messy_df: pd.DataFrame) -> None:
        before = messy_df.copy(deep=True)
        default_pipeline({"total_revenue": "numeric"}).run(messy_df)
        pd.testing.assert_frame_equal(messy_df, before)

    def test_replay_is_deterministic(self, messy_df: pd.DataFrame) -> None:
        pipeline = default_pipeline({"total_revenue": "numeric"})
        first = pipeline.run(messy_df)
        second = pipeline.run(messy_df)
        pd.testing.assert_frame_equal(first.df, second.df)

    def test_default_pipeline_cleans_a_messy_frame(self, messy_df: pd.DataFrame) -> None:
        result = default_pipeline({"total_revenue": "numeric", "signup_date": "datetime"}).run(
            messy_df
        )

        assert "total_revenue" in result.df.columns  # renamed
        assert result.df["total_revenue"].isna().sum() == 0  # filled
        assert len(result.df) == 9  # one duplicate removed
        assert result.rows_removed == 1
        assert "Removed 1 duplicate row" in result.cleaning_report()

    def test_report_covers_every_step(self, messy_df: pd.DataFrame) -> None:
        pipeline = default_pipeline({})
        result = pipeline.run(messy_df)
        assert len(result.reports) == len(pipeline)

    def test_replace_filters_leaves_cleaning_steps_alone(self) -> None:
        pipeline = default_pipeline({})
        cleaning_before = len([s for s in pipeline if not isinstance(s, QueryFilter)])

        pipeline.replace_filters([QueryFilter("a > 1")])
        pipeline.replace_filters([QueryFilter("a > 2")])

        assert len(pipeline.filters) == 1
        assert len([s for s in pipeline if not isinstance(s, QueryFilter)]) == cleaning_before

    def test_failing_step_names_itself(self, timeseries_df: pd.DataFrame) -> None:
        pipeline = Pipeline([QueryFilter("this is not valid")])
        with pytest.raises(PreprocessError):
            pipeline.run(timeseries_df)


class TestSerialization:
    @pytest.mark.parametrize(
        "step",
        [
            NormalizeColumnNames(),
            CoerceTypes({"a": "numeric"}),
            FillMissing("mean", ["a"]),
            FlagOutliers("zscore", 2.5, ["a"], drop=True),
            Deduplicate(["a"], "last"),
            QueryFilter("a > 1"),
            ColumnFilter("a", "between", [1, 2]),
        ],
    )
    def test_every_step_round_trips(self, step: Step) -> None:
        restored = Step.from_dict(step.to_dict())
        assert type(restored) is type(step)
        assert restored.to_dict() == step.to_dict()

    def test_pipeline_round_trips(self) -> None:
        pipeline = default_pipeline({"a": "numeric"})
        pipeline.add(QueryFilter("a > 0"))

        restored = Pipeline.from_list(pipeline.to_list())
        assert restored.to_list() == pipeline.to_list()

    def test_unknown_step_kind_suggests_updating(self) -> None:
        with pytest.raises(PreprocessError, match="newer version"):
            Step.from_dict({"kind": "quantum_normalize"})

    def test_step_without_a_kind_is_rejected(self) -> None:
        with pytest.raises(PreprocessError, match="kind"):
            Step.from_dict({"strategy": "mean"})


class TestGeneratedCode:
    @pytest.mark.parametrize(
        "step",
        [
            NormalizeColumnNames(),
            CoerceTypes({"revenue": "numeric"}),
            FillMissing("median"),
            FillMissing("drop"),
            FillMissing("mode"),
            FlagOutliers("iqr"),
            FlagOutliers("zscore", drop=True),
            Deduplicate(),
            QueryFilter("revenue > 1000"),
            ColumnFilter("revenue", "between", [1, 2]),
        ],
    )
    def test_step_code_is_valid_python(self, step: Step) -> None:
        code = "\n".join(step.to_code())
        compile(code, "<step>", "exec")

    def test_pipeline_code_runs_and_matches_the_pipeline(self, messy_df: pd.DataFrame) -> None:
        pipeline = Pipeline([NormalizeColumnNames(), Deduplicate(), FillMissing("median")])
        expected = pipeline.run(messy_df).df

        namespace: dict[str, object] = {"df": messy_df.copy(), "pd": pd}
        exec("\n".join(pipeline.to_code()), namespace)

        pd.testing.assert_frame_equal(
            namespace["df"].reset_index(drop=True),  # type: ignore[union-attr]
            expected.reset_index(drop=True),
        )
