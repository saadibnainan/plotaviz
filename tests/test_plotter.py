"""Plotter tests — data preparation, the performance guardrails, and both renderers.

The guardrails get the most attention because they are the ones that fail silently: a chart that
quietly drops 90% of its points still *looks* fine, which is exactly why the sampling notice is
asserted on rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plotaviz.core.errors import PlotError
from plotaviz.core.plotter import (
    COUNT_COLUMN,
    MAX_CATEGORIES,
    OTHER_LABEL,
    build_matplotlib,
    build_plotly,
    effective_spec,
    kde_curve,
    prepare,
)
from plotaviz.core.spec import ChartSpec


class TestPreparation:
    def test_does_not_mutate_the_callers_spec(self, categorical_df: pd.DataFrame) -> None:
        spec = ChartSpec("bar", x="department", agg="count")
        before = spec.to_dict()

        prepare(categorical_df, spec)

        assert spec.to_dict() == before
        spec.validate(list(categorical_df.columns))  # still valid against the source frame

    def test_count_aggregation_creates_the_measure_column(
        self, categorical_df: pd.DataFrame
    ) -> None:
        data = prepare(categorical_df, ChartSpec("bar", x="department", agg="count"))

        assert COUNT_COLUMN in data.df.columns
        assert data.spec.y == COUNT_COLUMN
        assert data.df[COUNT_COLUMN].sum() == len(categorical_df)

    def test_mean_aggregation_collapses_to_one_row_per_group(
        self, categorical_df: pd.DataFrame
    ) -> None:
        data = prepare(categorical_df, ChartSpec("bar", x="department", y="tenure", agg="mean"))
        assert len(data.df) == categorical_df["department"].nunique()

    def test_heatmap_count_does_not_collide_with_the_axis_name(
        self, categorical_df: pd.DataFrame
    ) -> None:
        data = prepare(
            categorical_df, ChartSpec("heatmap", x="department", y="rating", agg="count")
        )
        assert COUNT_COLUMN in data.df.columns
        assert data.spec.color == COUNT_COLUMN

    def test_the_same_column_mapped_twice_does_not_break_grouping(
        self, categorical_df: pd.DataFrame
    ) -> None:
        """Colouring a bar chart by its own x axis is a normal thing to want."""
        spec = ChartSpec("grouped_bar", x="department", color="department", y="tenure", agg="mean")
        data = prepare(categorical_df, spec)
        assert len(data.df) == categorical_df["department"].nunique()

    def test_time_series_is_sorted(self, timeseries_df: pd.DataFrame) -> None:
        shuffled = timeseries_df.sample(frac=1, random_state=0)
        data = prepare(shuffled, ChartSpec("line", x="order_date", y="revenue"))
        assert data.df["order_date"].is_monotonic_increasing

    def test_reports_rows_dropped_for_missing_values(self) -> None:
        df = pd.DataFrame({"x": [1.0, 2.0, None, 4.0], "y": [1.0, None, 3.0, 4.0]})
        data = prepare(df, ChartSpec("scatter", x="x", y="y"))

        assert len(data.df) == 2
        assert any("missing values" in note for note in data.notes)

    def test_all_rows_missing_raises_a_useful_error(self) -> None:
        df = pd.DataFrame({"x": [None, None], "y": [1.0, 2.0]})
        with pytest.raises(PlotError, match="No rows are left"):
            prepare(df, ChartSpec("scatter", x="x", y="y"))

    def test_matrix_charts_need_two_numeric_columns(self) -> None:
        with pytest.raises(PlotError, match="at least two numeric"):
            prepare(
                pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]}), ChartSpec("correlation_heatmap")
            )

    def test_matrix_charts_exclude_outlier_flag_columns(self) -> None:
        df = pd.DataFrame(
            {"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0], "a__outlier": [False, False, True]}
        )
        data = prepare(df, ChartSpec("correlation_heatmap"))
        assert "a__outlier" not in data.df.columns


class TestGuardrails:
    def test_caps_categories_and_says_so(self) -> None:
        df = pd.DataFrame(
            {
                "label": [f"cat_{i}" for i in range(200)] * 2,
                "value": list(range(200)) * 2,
            }
        )
        data = prepare(df, ChartSpec("bar", x="label", y="value", agg="sum"), max_categories=10)

        assert data.df["label"].nunique() <= 11  # ten kept plus the Other bucket
        assert OTHER_LABEL in set(data.df["label"])
        assert any("top 10" in note for note in data.notes)

    def test_default_category_cap_is_applied(self) -> None:
        df = pd.DataFrame({"label": [f"c{i}" for i in range(MAX_CATEGORIES + 20)]})
        data = prepare(df, ChartSpec("bar", x="label", agg="count"))
        assert data.df["label"].nunique() <= MAX_CATEGORIES + 1

    def test_downsamples_dense_scatter_and_announces_it(self, rng) -> None:
        df = pd.DataFrame({"x": rng.normal(0, 1, 20_000), "y": rng.normal(0, 1, 20_000)})
        data = prepare(df, ChartSpec("scatter", x="x", y="y"), max_points=1_000)

        assert data.sampled
        assert len(data.df) == 1_000
        assert data.rows_original == 20_000
        assert "20,000" in data.sampling_notice
        assert any("sample" in note for note in data.notes)

    def test_no_sampling_notice_when_nothing_was_dropped(
        self, numeric_pair_df: pd.DataFrame
    ) -> None:
        data = prepare(numeric_pair_df, ChartSpec("scatter", x="width", y="height"))
        assert not data.sampled
        assert data.sampling_notice == ""

    def test_lttb_keeps_the_extremes_of_a_time_series(self) -> None:
        n = 5_000
        values = np.sin(np.linspace(0, 40, n))
        values[2_500] = 100.0  # a spike a random sample would probably lose
        df = pd.DataFrame({"t": pd.date_range("2025-01-01", periods=n, freq="min"), "v": values})

        data = prepare(df, ChartSpec("line", x="t", y="v"), max_points=500)

        assert data.sampled
        assert len(data.df) <= 500
        assert data.df["v"].max() == pytest.approx(100.0)
        assert data.df["t"].iloc[0] == df["t"].iloc[0]
        assert data.df["t"].iloc[-1] == df["t"].iloc[-1]

    def test_downsampling_is_deterministic(self, rng) -> None:
        df = pd.DataFrame({"x": rng.normal(0, 1, 5_000), "y": rng.normal(0, 1, 5_000)})
        spec = ChartSpec("scatter", x="x", y="y")

        first = prepare(df, spec, max_points=500).df
        second = prepare(df, spec, max_points=500).df
        pd.testing.assert_frame_equal(first, second)


class TestEffectiveSpec:
    def test_leaves_unaggregated_specs_alone(self) -> None:
        spec = ChartSpec("scatter", x="a", y="b")
        assert effective_spec(spec).y == "b"

    def test_freezes_the_title_before_renaming_the_measure(self) -> None:
        eff = effective_spec(ChartSpec("bar", x="region", agg="count"))
        assert eff.title == "Count by region"
        assert eff.y == COUNT_COLUMN


class TestKde:
    def test_produces_a_normalised_curve(self, rng) -> None:
        grid, density = kde_curve(pd.Series(rng.normal(0, 1, 1_000)))

        assert len(grid) == len(density) == 200
        assert (density >= 0).all()
        area = np.trapezoid(density, grid) if hasattr(np, "trapezoid") else np.trapz(density, grid)
        assert 0.9 < area < 1.1

    def test_needs_at_least_two_values(self) -> None:
        with pytest.raises(PlotError, match="at least two"):
            kde_curve(pd.Series([1.0]))


class TestRenderers:
    CASES = [
        ("histogram", {"x": "tenure"}),
        ("kde", {"x": "tenure"}),
        ("bar", {"x": "department", "agg": "count"}),
        ("bar", {"x": "department", "y": "tenure", "agg": "mean"}),
        ("grouped_bar", {"x": "department", "y": "tenure", "color": "rating", "agg": "mean"}),
        ("stacked_bar", {"x": "department", "color": "rating", "agg": "count"}),
        ("treemap", {"x": "department", "agg": "count"}),
        ("pie", {"x": "department", "agg": "count"}),
        ("box", {"x": "department", "y": "tenure"}),
        ("violin", {"x": "department", "y": "tenure"}),
        ("heatmap", {"x": "department", "y": "rating", "agg": "count"}),
    ]

    @pytest.mark.parametrize(("chart", "mapping"), CASES)
    def test_plotly_builds(self, categorical_df: pd.DataFrame, chart: str, mapping: dict) -> None:
        figure = build_plotly(categorical_df, ChartSpec(chart, **mapping))
        assert figure is not None

    @pytest.mark.parametrize(("chart", "mapping"), CASES)
    def test_matplotlib_builds(
        self, categorical_df: pd.DataFrame, chart: str, mapping: dict
    ) -> None:
        import matplotlib.pyplot as plt

        figure = build_matplotlib(categorical_df, ChartSpec(chart, **mapping), dpi=60)
        assert figure.axes
        plt.close(figure)

    @pytest.mark.parametrize("chart", ["scatter", "line", "area"])
    def test_dense_charts_build(self, timeseries_df: pd.DataFrame, chart: str) -> None:
        import matplotlib.pyplot as plt

        spec = ChartSpec(chart, x="order_date", y="revenue", color="region")
        assert build_plotly(timeseries_df, spec) is not None
        figure = build_matplotlib(timeseries_df, spec, dpi=60)
        plt.close(figure)

    @pytest.mark.parametrize("chart", ["correlation_heatmap", "pair_plot"])
    def test_matrix_charts_build(self, numeric_pair_df: pd.DataFrame, chart: str) -> None:
        import matplotlib.pyplot as plt

        spec = ChartSpec(chart)
        assert build_plotly(numeric_pair_df, spec) is not None
        figure = build_matplotlib(numeric_pair_df, spec, dpi=60)
        plt.close(figure)

    def test_log_option_is_applied(self, categorical_df: pd.DataFrame) -> None:
        import matplotlib.pyplot as plt

        spec = ChartSpec("bar", x="department", y="tenure", agg="mean", options={"log_y": True})
        figure = build_matplotlib(categorical_df, spec, dpi=60)
        assert figure.axes[0].get_yscale() == "log"
        plt.close(figure)

    def test_sampling_notice_is_stamped_onto_the_static_figure(self, rng) -> None:
        import matplotlib.pyplot as plt

        df = pd.DataFrame({"x": rng.normal(0, 1, 5_000), "y": rng.normal(0, 1, 5_000)})
        prepared = prepare(df, ChartSpec("scatter", x="x", y="y"), max_points=500)
        figure = build_matplotlib(df, ChartSpec("scatter", x="x", y="y"), prepared=prepared, dpi=60)

        assert any("sample" in text.get_text() for text in figure.texts)
        plt.close(figure)

    def test_a_non_numeric_measure_explains_itself(self, categorical_df: pd.DataFrame) -> None:
        spec = ChartSpec("bar", x="department", y="rating", agg="mean")
        with pytest.raises(PlotError, match="type override"):
            prepare(categorical_df, spec)
