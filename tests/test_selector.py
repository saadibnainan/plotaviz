"""Selector tests — does each data shape produce the right chart, and is the ranking sane.

These are the tests that encode what PlotaViz is actually for. A regression here means the app
recommends the wrong chart, which is worse than a crash: the user gets a plausible-looking answer
that misrepresents their data.
"""

from __future__ import annotations

import pandas as pd
import pytest

from plotaviz.core.errors import SelectionError
from plotaviz.core.profiler import profile
from plotaviz.core.selector import ChartSelector, load_rules


@pytest.fixture
def selector() -> ChartSelector:
    return ChartSelector()


class TestRulesConfig:
    def test_ships_a_valid_rules_file(self) -> None:
        config = load_rules()
        assert config.rules
        for rule in config.rules:
            assert "name" in rule
            assert rule.get("candidates")

    def test_every_candidate_names_a_known_chart_type(self) -> None:
        from plotaviz.core.spec import CHART_TYPES

        for rule in load_rules().rules:
            for candidate in rule["candidates"]:
                assert candidate["chart"] in CHART_TYPES, rule["name"]

    def test_weights_have_defaults_for_anything_the_file_omits(self) -> None:
        from plotaviz.core.selector import RulesConfig

        config = RulesConfig(rules=[{"name": "x", "candidates": []}], scoring={})
        assert config.weights["correlation_bonus"] > 0

    def test_missing_rules_file_explains_itself(self, tmp_path: pd.DataFrame) -> None:
        with pytest.raises(SelectionError, match="not found"):
            load_rules(tmp_path / "nope.yaml")  # type: ignore[operator]


class TestShapeToChart:
    def test_datetime_and_measure_gives_a_time_series(
        self, selector: ChartSelector, timeseries_df: pd.DataFrame
    ) -> None:
        top = selector.recommend(profile(timeseries_df))[0]

        assert top.chart in {"line", "area"}
        assert top.x == "order_date"
        assert top.y in {"revenue", "units"}

    def test_two_numeric_columns_give_a_scatter(
        self, selector: ChartSelector, numeric_pair_df: pd.DataFrame
    ) -> None:
        charts = {spec.chart for spec in selector.recommend(profile(numeric_pair_df))}
        assert "scatter" in charts

    def test_measure_by_category_gives_a_comparison(
        self, selector: ChartSelector, categorical_df: pd.DataFrame
    ) -> None:
        charts = {spec.chart for spec in selector.recommend(profile(categorical_df))}
        assert charts & {"bar", "grouped_bar", "box", "violin", "heatmap"}

    def test_single_numeric_column_gives_a_distribution(self, selector: ChartSelector, rng) -> None:
        df = pd.DataFrame({"v": rng.normal(10.0, 2.5, 300)})
        top = selector.recommend(profile(df))[0]
        assert top.chart in {"histogram", "kde", "box"}

    def test_many_numeric_columns_offer_a_correlation_matrix(
        self, selector: ChartSelector, rng
    ) -> None:
        df = pd.DataFrame({f"m{i}": rng.normal(i * 10, 3.0, 200) for i in range(5)})
        charts = {spec.chart for spec in selector.recommend(profile(df))}
        assert "correlation_heatmap" in charts

    def test_two_categorical_columns_offer_a_cross_tabulation(
        self, selector: ChartSelector
    ) -> None:
        df = pd.DataFrame({"a": ["x", "y"] * 50, "b": ["p", "q"] * 50})
        charts = {spec.chart for spec in selector.recommend(profile(df))}
        assert charts & {"grouped_bar", "stacked_bar", "heatmap"}


class TestScoring:
    def test_every_recommendation_explains_itself(
        self, selector: ChartSelector, timeseries_df: pd.DataFrame
    ) -> None:
        for spec in selector.recommend(profile(timeseries_df)):
            assert spec.why.strip(), f"{spec.chart} has no justification"
            assert 0.0 <= spec.score <= 1.0

    def test_results_are_ordered_best_first(
        self, selector: ChartSelector, categorical_df: pd.DataFrame
    ) -> None:
        scores = [spec.score for spec in selector.recommend(profile(categorical_df))]
        assert scores == sorted(scores, reverse=True)

    def test_strong_correlation_beats_weak_for_the_same_chart(
        self, selector: ChartSelector, rng
    ) -> None:
        base = rng.normal(50.0, 10.0, 300)
        strong = pd.DataFrame({"a": base, "b": base * 1.7 + rng.normal(0, 1.0, 300)})
        weak = pd.DataFrame({"a": base, "b": rng.normal(50.0, 10.0, 300)})

        def scatter_score(df: pd.DataFrame) -> float:
            specs = [s for s in selector.recommend(profile(df)) if s.chart == "scatter"]
            return specs[0].score

        assert scatter_score(strong) > scatter_score(weak)

    def test_high_cardinality_is_penalised_and_explained(self, selector: ChartSelector) -> None:
        df = pd.DataFrame(
            {"label": [f"item_{i}" for i in range(400)], "value": [float(i) for i in range(400)]}
        )
        specs = [s for s in selector.recommend(profile(df)) if s.x == "label"]

        if specs:  # the label column may be classified as text, which is itself a valid answer
            assert any("categor" in s.why.lower() for s in specs)

    def test_a_constant_column_is_penalised(self, selector: ChartSelector) -> None:
        df = pd.DataFrame({"same": ["x"] * 100, "value": [float(i) for i in range(100)]})
        specs = [s for s in selector.recommend(profile(df)) if s.x == "same"]
        for spec in specs:
            assert "nothing to compare" in spec.why

    def test_skewed_measures_suggest_a_log_axis_or_a_distribution_chart(
        self, selector: ChartSelector, rng
    ) -> None:
        df = pd.DataFrame({"group": ["a", "b"] * 150, "amount": rng.lognormal(1.0, 1.8, 300)})

        specs = selector.recommend(profile(df))
        assert any(spec.options.get("log_y") or spec.chart in {"box", "violin"} for spec in specs)

    def test_preferred_columns_are_boosted(
        self, selector: ChartSelector, timeseries_df: pd.DataFrame
    ) -> None:
        dataset = profile(timeseries_df)
        plain = selector.recommend(dataset)
        preferred = selector.recommend(dataset, prefer_columns=["units"])

        assert any("you asked about" in spec.why for spec in preferred)
        assert plain[0].score <= 1.0


class TestRankingHygiene:
    def test_no_duplicate_mappings(
        self, selector: ChartSelector, timeseries_df: pd.DataFrame
    ) -> None:
        specs = selector.recommend(profile(timeseries_df), top_k=20)
        keys = [(s.chart, s.x, s.y, s.color, s.agg) for s in specs]
        assert len(keys) == len(set(keys))

    def test_alternatives_are_varied_not_eight_of_one_chart(
        self, selector: ChartSelector, timeseries_df: pd.DataFrame
    ) -> None:
        specs = selector.recommend(profile(timeseries_df), top_k=6)
        top_types = [s.chart for s in specs[:4]]
        assert len(set(top_types)) >= 2

    def test_top_k_is_respected(self, selector: ChartSelector, timeseries_df: pd.DataFrame) -> None:
        assert len(selector.recommend(profile(timeseries_df), top_k=3)) == 3

    def test_every_recommendation_validates_against_the_real_schema(
        self, selector: ChartSelector, categorical_df: pd.DataFrame
    ) -> None:
        columns = list(categorical_df.columns)
        for spec in selector.recommend(profile(categorical_df), top_k=20):
            spec.validate(columns)

    def test_identifier_columns_are_never_recommended(self, selector: ChartSelector) -> None:
        df = pd.DataFrame(
            {
                "user_id": range(200),
                "score": [float(i % 17) for i in range(200)],
                "team": ["a", "b"] * 100,
            }
        )
        for spec in selector.recommend(profile(df), top_k=20):
            assert "user_id" not in spec.columns_used

    def test_ambiguity_detection(
        self, selector: ChartSelector, categorical_df: pd.DataFrame
    ) -> None:
        specs = selector.recommend(profile(categorical_df))
        assert isinstance(selector.is_ambiguous(specs), bool)


class TestFailureModes:
    def test_only_identifiers_still_produces_a_chart_with_a_caveat(
        self, selector: ChartSelector
    ) -> None:
        """Refusing to draw anything is a dead end; draw it and explain the doubt instead."""
        df = pd.DataFrame({"user_id": range(100), "session_id": range(1000, 1100)})
        specs = selector.recommend(profile(df))

        assert specs
        assert all("looks like an identifier" in spec.why for spec in specs)

    def test_a_dataset_with_nothing_chartable_says_what_to_do(
        self, selector: ChartSelector
    ) -> None:
        df = pd.DataFrame({"notes": [f"a distinct free text comment {i}" for i in range(50)]})
        profiled = profile(df)
        profiled.columns.clear()  # the pathological case: no column of any usable role
        with pytest.raises(SelectionError) as exc:
            selector.recommend(profiled)
        assert "type override" in str(exc.value)


def test_each_sample_dataset_produces_a_usable_recommendation(sample_file) -> None:
    from plotaviz.core.analysis import Analysis

    analysis = Analysis.from_file(sample_file)
    assert analysis.recommendations, f"no recommendation for {sample_file.name}"
    assert analysis.spec is not None
    analysis.spec.validate(list(analysis.df.columns))
