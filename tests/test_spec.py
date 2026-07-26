"""Chart spec tests — validation, immutability, and tolerant deserialization.

Validation is the security boundary for the LLM layer: a model-supplied spec is untrusted input,
and the only thing standing between it and the renderer is :meth:`ChartSpec.validate`.
"""

from __future__ import annotations

import pytest

from plotaviz.core.errors import SpecError
from plotaviz.core.spec import CHART_TYPES, ChartSpec


class TestValidation:
    def test_accepts_a_well_formed_spec(self) -> None:
        spec = ChartSpec("scatter", x="a", y="b")
        assert spec.validate(["a", "b", "c"]) is spec

    def test_rejects_an_unknown_chart_type(self) -> None:
        with pytest.raises(SpecError, match="Unknown chart type"):
            ChartSpec("sankey-of-dreams", x="a", y="b").validate()

    def test_rejects_an_unknown_aggregation(self) -> None:
        with pytest.raises(SpecError, match="aggregation"):
            ChartSpec("bar", x="a", y="b", agg="vibe").validate()

    def test_rejects_a_column_that_does_not_exist(self) -> None:
        with pytest.raises(SpecError) as exc:
            ChartSpec("scatter", x="a", y="ghost").validate(["a", "b"])
        assert "ghost" in str(exc.value)
        assert "Available columns" in str(exc.value)

    def test_names_the_role_of_the_bad_column(self) -> None:
        with pytest.raises(SpecError, match="used as color"):
            ChartSpec("scatter", x="a", y="b", color="ghost").validate(["a", "b"])

    def test_requires_x(self) -> None:
        with pytest.raises(SpecError, match="needs an x column"):
            ChartSpec("bar").validate()

    def test_requires_y_for_bivariate_charts(self) -> None:
        with pytest.raises(SpecError, match="needs a y column"):
            ChartSpec("scatter", x="a").validate()

    def test_counting_charts_need_no_y(self) -> None:
        ChartSpec("stacked_bar", x="a", color="b", agg="count").validate(["a", "b"])

    def test_matrix_charts_need_no_mapping(self) -> None:
        ChartSpec("correlation_heatmap").validate(["a", "b"])

    def test_rejects_a_score_outside_the_range(self) -> None:
        with pytest.raises(SpecError, match="outside the range"):
            ChartSpec("histogram", x="a", score=1.5).validate()

    def test_validation_without_columns_skips_the_schema_check(self) -> None:
        ChartSpec("scatter", x="whatever", y="anything").validate()


class TestImmutability:
    def test_copy_is_independent(self) -> None:
        original = ChartSpec("bar", x="a", options={"top_n": 5})
        clone = original.copy()
        clone.options["top_n"] = 99
        clone.x = "b"

        assert original.options["top_n"] == 5
        assert original.x == "a"

    def test_copy_applies_changes(self) -> None:
        spec = ChartSpec("bar", x="a").copy(chart="line", y="b")
        assert (spec.chart, spec.x, spec.y) == ("line", "a", "b")

    def test_with_options_merges(self) -> None:
        spec = ChartSpec("scatter", x="a", y="b", options={"trendline": True})
        merged = spec.with_options(log_y=True)

        assert merged.options == {"trendline": True, "log_y": True}
        assert spec.options == {"trendline": True}


class TestSerialization:
    def test_round_trips(self) -> None:
        spec = ChartSpec("line", x="d", y="v", color="g", agg="sum", why="because", score=0.8)
        assert ChartSpec.from_dict(spec.to_dict()).to_dict() == spec.to_dict()

    @pytest.mark.parametrize(
        ("payload", "field", "expected"),
        [
            ({"chart_type": "bar", "x": "a"}, "chart", "bar"),
            ({"chart": "bar", "x": "a", "group": "g"}, "color", "g"),
            ({"chart": "bar", "x": "a", "aggregation": "sum", "y": "v"}, "agg", "sum"),
            ({"type": "bar", "x": "a"}, "chart", "bar"),
        ],
    )
    def test_tolerates_the_aliases_models_emit(
        self, payload: dict, field: str, expected: str
    ) -> None:
        assert getattr(ChartSpec.from_dict(payload), field) == expected

    def test_ignores_unknown_keys(self) -> None:
        spec = ChartSpec.from_dict({"chart": "bar", "x": "a", "hallucinated": "nonsense"})
        assert spec.chart == "bar"

    def test_requires_a_chart_field(self) -> None:
        with pytest.raises(SpecError, match="missing the 'chart' field"):
            ChartSpec.from_dict({"x": "a", "y": "b"})

    def test_rejects_non_objects(self) -> None:
        with pytest.raises(SpecError, match="JSON object"):
            ChartSpec.from_dict(["bar"])  # type: ignore[arg-type]

    def test_survives_a_non_numeric_score(self) -> None:
        assert ChartSpec.from_dict({"chart": "bar", "x": "a", "score": "high"}).score == 0.0

    def test_survives_a_non_dict_options(self) -> None:
        assert ChartSpec.from_dict({"chart": "bar", "x": "a", "options": "none"}).options == {}


class TestTitles:
    @pytest.mark.parametrize(
        ("spec", "fragment"),
        [
            (ChartSpec("histogram", x="revenue"), "Distribution of revenue"),
            (ChartSpec("bar", x="region", agg="count"), "Count by region"),
            (ChartSpec("line", x="order_date", y="revenue"), "revenue by order date"),
            (ChartSpec("bar", x="region", y="revenue", agg="mean"), "Mean of revenue"),
            (ChartSpec("scatter", x="a", y="b", color="g"), "split by g"),
            (ChartSpec("correlation_heatmap"), "Correlation matrix"),
        ],
    )
    def test_generated_titles_read_naturally(self, spec: ChartSpec, fragment: str) -> None:
        assert fragment in spec.display_title()

    def test_an_explicit_title_wins(self) -> None:
        assert ChartSpec("bar", x="a", title="Q3 results").display_title() == "Q3 results"


def test_columns_used_is_ordered_and_deduplicated() -> None:
    assert ChartSpec("bar", x="a", y="b", color="a").columns_used == ["a", "b"]


def test_every_declared_chart_type_can_be_constructed() -> None:
    for chart in CHART_TYPES:
        spec = ChartSpec(chart, x="a", y="b", color="c")
        spec.validate(["a", "b", "c"])
