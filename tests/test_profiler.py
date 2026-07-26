"""Profiler tests — role inference, identifier detection, and statistics.

Identifier detection gets the most attention here because it is the classification most likely to
be quietly wrong in either direction: mistake a measure for a key and it vanishes from every
chart; mistake a key for a measure and PlotaViz recommends plotting customer IDs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plotaviz.core.errors import ProfileError
from plotaviz.core.profiler import (
    BOOLEAN,
    CATEGORICAL,
    DATETIME,
    NUMERIC,
    TEXT,
    infer_role,
    looks_like_boolean,
    looks_like_datetime,
    profile,
)


class TestRoleInference:
    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            ([1.5, 2.5, 3.5, 4.5] * 10, NUMERIC),
            (["a", "b", "c", "a", "b"] * 8, CATEGORICAL),
            (["2025-01-01", "2025-02-01", "2025-03-01"] * 10, DATETIME),
            ([True, False, True, True] * 10, BOOLEAN),
            (["yes", "no", "yes", "no"] * 10, BOOLEAN),
        ],
    )
    def test_basic_roles(self, values: list, expected: str) -> None:
        role, _, _ = infer_role(pd.Series(values, name="col"))
        assert role == expected

    def test_high_cardinality_strings_are_text(self) -> None:
        series = pd.Series([f"a free text comment number {i}" for i in range(200)], name="notes")
        role, _, _ = infer_role(series)
        assert role == TEXT

    def test_few_distinct_integers_read_as_categories(self) -> None:
        series = pd.Series([1, 2, 3] * 40, name="tier")
        role, is_id, note = infer_role(series)
        assert role == CATEGORICAL
        assert not is_id
        assert "category" in note

    def test_empty_column_does_not_crash(self) -> None:
        role, _, note = infer_role(pd.Series([None] * 10, name="blank"))
        assert role == CATEGORICAL
        assert "empty" in note


class TestIdentifierDetection:
    def test_id_named_column_is_an_identifier(self) -> None:
        _, is_id, note = infer_role(pd.Series(range(100), name="customer_id"))
        assert is_id
        assert "identifier" in note

    def test_sequential_unique_integers_are_an_identifier(self) -> None:
        _, is_id, _ = infer_role(pd.Series(range(1000, 1500), name="row"))
        assert is_id

    def test_a_unique_measure_is_not_an_identifier(self) -> None:
        """Population is all-unique but spans orders of magnitude — it is data, not a key."""
        rng = np.random.default_rng(0)
        populations = pd.Series(
            rng.lognormal(13.5, 1.0, 300).astype(int), name="population"
        ).drop_duplicates()

        role, is_id, _ = infer_role(populations)
        assert role == NUMERIC
        assert not is_id

    def test_float_measures_are_never_identifiers(self) -> None:
        rng = np.random.default_rng(0)
        _, is_id, _ = infer_role(pd.Series(rng.normal(0, 1, 500), name="reading"))
        assert not is_id

    def test_identifiers_are_excluded_from_role_queries(self) -> None:
        df = pd.DataFrame({"order_id": range(100), "amount": np.arange(100.0) * 1.5})
        result = profile(df)

        assert "amount" in result.numeric
        assert "order_id" not in result.numeric
        assert "order_id" in result.by_role(NUMERIC, include_identifiers=True)


class TestDatetimeDetection:
    @pytest.mark.parametrize(
        "values",
        [
            ["2025-01-01", "2025-06-15", "2025-12-31"],
            ["2025-01-01 08:30:00", "2025-01-02 09:15:00"],
            ["01/02/2025", "03/04/2025"],
        ],
    )
    def test_recognises_date_strings(self, values: list[str]) -> None:
        assert looks_like_datetime(pd.Series(values, name="when"))

    def test_plain_integers_are_never_dates(self) -> None:
        """Pandas will happily read integers as 1970 nanosecond timestamps. It must not here."""
        assert not looks_like_datetime(pd.Series([1, 2, 3, 4], name="year"))
        assert not looks_like_datetime(pd.Series([2020, 2021, 2022], name="date"))

    def test_ordinary_words_are_not_dates(self) -> None:
        assert not looks_like_datetime(pd.Series(["alpha", "beta", "gamma"], name="label"))


class TestBooleanDetection:
    @pytest.mark.parametrize(
        "values",
        [[True, False], ["yes", "no"], ["Y", "N"], ["true", "false"], ["on", "off"]],
    )
    def test_recognises_boolean_tokens(self, values: list) -> None:
        assert looks_like_boolean(pd.Series(values * 10, name="flag"))

    def test_two_arbitrary_categories_are_not_boolean(self) -> None:
        assert not looks_like_boolean(pd.Series(["ok", "degraded"] * 10, name="status"))


class TestDatasetProfile:
    def test_counts_roles(self, timeseries_df: pd.DataFrame) -> None:
        result = profile(timeseries_df)
        counts = result.role_counts()

        assert counts[DATETIME] == 1
        assert counts[CATEGORICAL] == 1
        assert counts[NUMERIC] == 2

    def test_reports_missing_and_cardinality(self) -> None:
        df = pd.DataFrame({"v": [1.0, None, 3.0, None], "c": ["a", "a", "b", "b"]})
        result = profile(df)

        assert result.columns["v"].n_missing == 2
        assert result.columns["v"].pct_missing == 50.0
        assert result.columns["c"].n_unique == 2

    def test_finds_the_strongest_correlation(self, numeric_pair_df: pd.DataFrame) -> None:
        best = profile(numeric_pair_df).strongest_correlation()

        assert best is not None
        assert {best[0], best[1]} == {"width", "height"}
        assert best[2] > 0.9

    def test_no_correlation_matrix_with_one_numeric_column(self) -> None:
        assert profile(pd.DataFrame({"v": [1.0, 2.0], "c": ["a", "b"]})).correlations is None

    def test_counts_duplicate_rows(self) -> None:
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        assert profile(df).n_duplicate_rows == 1

    def test_user_overrides_win_over_inference(self) -> None:
        df = pd.DataFrame({"code": [1, 2, 3, 4, 5] * 20})
        inferred = profile(df).columns["code"].role

        overridden = profile(df, overrides={"code": NUMERIC}).columns["code"]
        assert overridden.role == NUMERIC
        assert overridden.role != inferred or inferred == NUMERIC
        assert "by you" in overridden.note or inferred == NUMERIC

    def test_derived_columns_are_excluded_from_chart_roles(self) -> None:
        df = pd.DataFrame({"v": [1.0, 2.0, 3.0], "v__outlier": [False, False, True]})
        result = profile(df)

        assert result.columns["v__outlier"].is_derived
        assert "v__outlier" not in result.by_role(BOOLEAN)
        assert "v__outlier" in result.by_role(BOOLEAN, include_derived=True)

    def test_samples_large_frames(self) -> None:
        df = pd.DataFrame({"v": np.arange(5_000.0), "c": ["a"] * 5_000})
        result = profile(df, sample_rows=1_000)

        assert result.sampled
        assert result.n_rows == 1_000
        assert result.total_rows == 5_000

    def test_empty_frame_raises(self) -> None:
        with pytest.raises(ProfileError):
            profile(pd.DataFrame())

    def test_schema_summary_is_json_safe(self, timeseries_df: pd.DataFrame) -> None:
        import json

        summary = profile(timeseries_df).schema_summary()
        json.dumps(summary)  # must not raise

        assert summary["n_cols"] == 4
        assert {c["name"] for c in summary["columns"]} == set(timeseries_df.columns)

    def test_schema_summary_contains_no_row_data(self, timeseries_df: pd.DataFrame) -> None:
        """The LLM payload carries statistics, not the dataset."""
        summary = profile(timeseries_df).schema_summary()
        for column in summary["columns"]:
            assert len(column["sample_values"]) <= 5
            assert len(column["top_values"]) <= 5
