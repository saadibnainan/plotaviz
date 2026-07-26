"""The chart spec — the single contract shared by every part of PlotaViz.

``plotter.py`` renders one, ``codegen.py`` emits a script for one, ``session.py`` serializes one,
the selector produces them, and the LLM layer returns one as JSON. Nothing passes chart state
through side channels; if a chart capability needs new information, it goes here.

A spec is *declarative and validated*: :meth:`ChartSpec.validate` checks it against a real
dataframe's schema before anything is rendered, which is what makes it safe to accept a spec from
a language model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .errors import SpecError

#: Chart types the plotter and code generator both understand.
CHART_TYPES: tuple[str, ...] = (
    "histogram",
    "kde",
    "bar",
    "grouped_bar",
    "stacked_bar",
    "treemap",
    "pie",
    "scatter",
    "line",
    "area",
    "box",
    "violin",
    "heatmap",
    "correlation_heatmap",
    "pair_plot",
)

#: Aggregations usable when ``y`` is collapsed over ``x``/``color``.
AGGREGATIONS: tuple[str, ...] = ("sum", "mean", "median", "min", "max", "count", "nunique")

#: Charts that consume a single column and need no ``y``.
UNIVARIATE_CHARTS: frozenset[str] = frozenset({"histogram", "kde", "bar", "treemap", "pie"})

#: Charts that ignore ``x``/``y`` and operate on the whole numeric frame.
MATRIX_CHARTS: frozenset[str] = frozenset({"correlation_heatmap", "pair_plot"})


@dataclass
class ChartSpec:
    """A declarative description of one chart.

    Attributes:
        chart: One of :data:`CHART_TYPES`.
        x: Column mapped to the x axis (or the single column for univariate charts).
        y: Column mapped to the y axis. ``None`` for univariate and matrix charts.
        color: Column used for grouping / series colour. Optional.
        agg: Aggregation applied to ``y`` grouped by ``x`` (and ``color``). ``None`` means the
            rows are plotted as they are.
        title: Chart title. Generated from the mapping when omitted.
        options: Renderer hints — ``log_y``, ``trendline``, ``orientation``, ``bins``,
            ``top_n``, ``stacked``, ``sort``, ``palette``. Unknown keys are ignored, so this is
            the safe place for renderer-specific extras.
        why: Human-readable justification. The selector always fills this in; it is what the UI
            shows next to a recommendation.
        score: Selector confidence in ``[0, 1]``. ``0.0`` for hand-built or LLM-built specs that
            were never scored.
        source: Where the spec came from — ``"rules"``, ``"llm"``, ``"user"``, or ``"session"``.
    """

    chart: str
    x: str | None = None
    y: str | None = None
    color: str | None = None
    agg: str | None = None
    title: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    why: str = ""
    score: float = 0.0
    source: str = "rules"

    # ---------------------------------------------------------------- validation

    def validate(self, columns: list[str] | None = None) -> ChartSpec:
        """Check the spec is internally coherent and, if given, matches a real schema.

        Args:
            columns: The dataframe's column names. When provided, every referenced column must
                exist — this is the check that makes an LLM-returned spec safe to render.

        Returns:
            The spec itself, so calls can be chained.

        Raises:
            SpecError: If the chart type is unknown, the aggregation is unknown, a required
                mapping is missing, or a referenced column is absent from ``columns``.
        """
        if self.chart not in CHART_TYPES:
            raise SpecError(
                f"Unknown chart type {self.chart!r}.",
                hint=f"Supported types: {', '.join(CHART_TYPES)}.",
            )

        if self.agg is not None and self.agg not in AGGREGATIONS:
            raise SpecError(
                f"Unknown aggregation {self.agg!r}.",
                hint=f"Supported aggregations: {', '.join(AGGREGATIONS)}.",
            )

        if self.chart not in MATRIX_CHARTS and self.x is None:
            raise SpecError(
                f"A {self.chart} chart needs an x column.",
                hint="Pick a column for the x axis and try again.",
            )

        # A counting chart's measure is implicit — "count of rows per x" needs no y column.
        needs_y = (
            self.chart not in UNIVARIATE_CHARTS
            and self.chart not in MATRIX_CHARTS
            and self.agg != "count"
        )
        if needs_y and self.y is None:
            raise SpecError(
                f"A {self.chart} chart needs a y column.",
                hint="Pick a column for the y axis and try again.",
            )

        if columns is not None:
            known = set(columns)
            for role, col in (("x", self.x), ("y", self.y), ("color", self.color)):
                if col is not None and col not in known:
                    preview = ", ".join(sorted(known)[:12])
                    raise SpecError(
                        f"Column {col!r} (used as {role}) is not in this dataset.",
                        hint=f"Available columns: {preview}" + (" …" if len(known) > 12 else ""),
                    )

        if not 0.0 <= self.score <= 1.0:
            raise SpecError(f"Score {self.score} is outside the range 0.0–1.0.")

        return self

    # ---------------------------------------------------------------- convenience

    @property
    def columns_used(self) -> list[str]:
        """The dataset columns this spec references, in mapping order, without duplicates."""
        seen: list[str] = []
        for col in (self.x, self.y, self.color):
            if col and col not in seen:
                seen.append(col)
        return seen

    def display_title(self) -> str:
        """The title to render — the explicit one, or a readable fallback from the mapping."""
        if self.title:
            return self.title

        def pretty(name: str | None) -> str:
            return (name or "").replace("_", " ").strip()

        if self.chart in MATRIX_CHARTS:
            return "Correlation matrix" if self.chart == "correlation_heatmap" else "Pair plot"
        if self.chart in UNIVARIATE_CHARTS:
            base = f"Distribution of {pretty(self.x)}"
            if self.chart in {"bar", "treemap", "pie"}:
                base = f"Count by {pretty(self.x)}"
            if self.agg and self.y:
                base = f"{self.agg.title()} of {pretty(self.y)} by {pretty(self.x)}"
            return base

        lead = f"{self.agg.title()} of {pretty(self.y)}" if self.agg else pretty(self.y)
        title = f"{lead} by {pretty(self.x)}"
        if self.color:
            title += f", split by {pretty(self.color)}"
        return title

    def copy(self, **changes: Any) -> ChartSpec:
        """Return an independent copy, optionally with fields replaced.

        Specs are passed around freely — into the plotter, the code generator, a session file —
        so anything that needs to adjust one takes a copy instead of mutating a spec the caller
        still holds. ``options`` is copied too, not shared.
        """
        fields: dict[str, Any] = {**asdict(self), **changes}
        fields["options"] = dict(fields.get("options") or {})
        return ChartSpec(**fields)

    def with_options(self, **kwargs: Any) -> ChartSpec:
        """Return a copy with ``options`` merged — the spec itself is treated as immutable."""
        return self.copy(options={**self.options, **kwargs})

    # ---------------------------------------------------------------- (de)serialization

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict (used by session files and the LLM layer)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChartSpec:
        """Build a spec from a dict, ignoring unknown keys.

        Models and older session files both produce extra or missing keys; tolerating that here
        keeps the failure mode a clear validation message rather than a ``TypeError``.

        Raises:
            SpecError: If ``data`` is not a mapping or has no ``chart`` key.
        """
        if not isinstance(data, dict):
            raise SpecError("Chart spec must be a JSON object.")

        known = set(cls.__dataclass_fields__)
        # Accept a couple of friendly aliases models tend to emit.
        aliases = {"chart_type": "chart", "group": "color", "aggregation": "agg", "type": "chart"}
        cleaned: dict[str, Any] = {}
        for key, value in data.items():
            key = aliases.get(key, key)
            if key in known:
                cleaned[key] = value

        if "chart" not in cleaned:
            raise SpecError(
                "Chart spec is missing the 'chart' field.",
                hint=f"Expected one of: {', '.join(CHART_TYPES)}.",
            )

        options = cleaned.get("options") or {}
        if not isinstance(options, dict):
            options = {}
        cleaned["options"] = options

        try:
            cleaned["score"] = float(cleaned.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            cleaned["score"] = 0.0

        return cls(**cleaned)
