"""Figure construction — Plotly for the interactive view, matplotlib for static export.

Both renderers consume the same :class:`~plotaviz.core.spec.ChartSpec` and, crucially, the same
*prepared* data. :func:`prepare` does the aggregating, sorting, category capping, and
downsampling once, so the interactive chart and the exported PNG cannot disagree with each other.

The performance guardrails live here rather than in the UI because they have to apply to CLI
exports too. Above roughly 100k points a scatter or line will lock up a WebEngine view, so the
data is downsampled — LTTB for time series, which preserves visual shape far better than random
sampling — and the caller is handed a note saying exactly what was shown. A chart that quietly
drops 90% of its data is worse than no chart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .errors import PlotError
from .spec import MATRIX_CHARTS, ChartSpec

#: Above this many points a scatter/line becomes unresponsive in a browser view.
MAX_POINTS = 100_000

#: Default cap on distinct categories drawn on a categorical axis.
MAX_CATEGORIES = 30

#: Label used for the bucket holding categories beyond the cap.
OTHER_LABEL = "Other"

#: Charts whose point count scales with row count, so they need downsampling.
_DENSE_CHARTS = frozenset({"scatter", "line", "area"})


@dataclass
class PreparedData:
    """Plot-ready data plus everything the user needs told about it.

    Attributes:
        df: The frame to render.
        spec: The **effective** spec — the caller's spec after aggregation renames the measure
            column. Renderers use this one; it describes :attr:`df`, not the source frame.
        source_spec: The spec as the caller supplied it, untouched.
        notes: User-facing remarks — sampling, category bucketing, dropped nulls.
        sampled: Whether rows were dropped for performance.
        rows_original: Row count before preparation.
    """

    df: pd.DataFrame
    spec: ChartSpec
    source_spec: ChartSpec | None = None
    notes: list[str] = field(default_factory=list)
    sampled: bool = False
    rows_original: int = 0

    @property
    def sampling_notice(self) -> str:
        """The "showing N of M points" line, or an empty string when nothing was dropped."""
        if not self.sampled:
            return ""
        return f"Showing a sample of {len(self.df):,} of {self.rows_original:,} points."


# ---------------------------------------------------------------------------- preparation


#: Name given to the measure column produced by a counting aggregation.
COUNT_COLUMN = "count"


def effective_spec(spec: ChartSpec) -> ChartSpec:
    """The spec as it describes the data *after* aggregation.

    Aggregating changes the frame's shape: ``count`` invents a measure column that does not exist
    in the source data. Renderers need a spec that matches what they are handed, but the caller's
    spec must keep matching the source frame — it still has to validate, serialize into a session,
    and generate code. So aggregation produces a separate spec instead of editing one in place.

    The title is resolved here too, before the measure is renamed, so a counted bar chart is
    titled "Count by department" rather than "Count of count by department".
    """
    out = spec.copy(title=spec.display_title())
    if not spec.agg:
        return out

    if spec.chart == "heatmap":
        # For a heatmap x and y are both axes, so the measure lives in `color`.
        if not out.color:
            out.color = COUNT_COLUMN
            out.agg = "count"
    elif spec.agg == "count" or not spec.y:
        out.y = COUNT_COLUMN
    return out


def prepare(
    df: pd.DataFrame,
    spec: ChartSpec,
    *,
    max_points: int = MAX_POINTS,
    max_categories: int | None = None,
) -> PreparedData:
    """Aggregate, cap, and downsample a frame for one chart spec.

    Args:
        df: The cleaned, filtered data.
        spec: What to draw. Never modified.
        max_points: Downsampling threshold for dense charts.
        max_categories: Category cap. Defaults to ``spec.options['top_n']`` or
            :data:`MAX_CATEGORIES`.

    Returns:
        A :class:`PreparedData` whose ``spec`` is the effective spec describing ``df``.

    Raises:
        PlotError: If the spec references missing columns or nothing plottable survives.
    """
    spec.validate(list(df.columns))
    eff = effective_spec(spec)
    notes: list[str] = []
    rows_original = len(df)
    cap = int(max_categories or spec.options.get("top_n") or MAX_CATEGORIES)

    if spec.chart in MATRIX_CHARTS:
        numeric = df.select_dtypes(include="number")
        # Outlier flag columns are booleans-as-numbers; they pollute a correlation matrix.
        numeric = numeric[[c for c in numeric.columns if not str(c).endswith("__outlier")]]
        if numeric.shape[1] < 2:
            raise PlotError(
                "This chart needs at least two numeric columns.",
                hint="Use the type override panel if a numeric column was read as text.",
            )
        if numeric.shape[1] > 12:
            keep = list(numeric.columns[:12])
            notes.append(f"Showing the first 12 of {numeric.shape[1]} numeric columns.")
            numeric = numeric[keep]
        return PreparedData(numeric, eff, spec, notes, False, rows_original)

    work = df[[c for c in spec.columns_used if c in df.columns]].copy()

    # Drop rows that cannot be plotted, and say how many.
    before = len(work)
    work = work.dropna(subset=[c for c in (spec.x, spec.y) if c])
    if len(work) < before:
        notes.append(
            f"Skipped {before - len(work):,} row(s) with missing values in the plotted columns."
        )

    if work.empty:
        raise PlotError(
            "No rows are left to plot once missing values are excluded.",
            hint="Loosen the filters, or choose a different missing-value strategy.",
        )

    # Cap categories on the x axis (and on the colour split) before aggregating.
    for column, is_axis in ((spec.x, True), (spec.color, False)):
        if not column or column not in work.columns:
            continue
        if pd.api.types.is_numeric_dtype(work[column]) or pd.api.types.is_datetime64_any_dtype(
            work[column]
        ):
            continue
        n_unique = work[column].nunique()
        limit = cap if is_axis else min(cap, 12)
        if n_unique > limit:
            work, dropped = _cap_categories(work, column, limit, spec)
            notes.append(
                f"{column!r} has {n_unique:,} categories; showing the top {limit} and grouping "
                f"the remaining {dropped:,} as {OTHER_LABEL!r}."
            )

    # Aggregate if the spec asks for it.
    if spec.agg:
        work = _aggregate(work, spec, eff)

    # Time series must be ordered, or the line zig-zags.
    if eff.x and eff.x in work.columns and pd.api.types.is_datetime64_any_dtype(work[eff.x]):
        work = work.sort_values(eff.x)
    elif eff.chart in {"bar", "treemap", "pie"} and eff.y and eff.y in work.columns:
        ascending = bool(eff.options.get("sort_ascending", False))
        work = work.sort_values(eff.y, ascending=ascending)

    # Downsample dense charts.
    sampled = False
    if eff.chart in _DENSE_CHARTS and len(work) > max_points:
        work = _downsample(work, eff, max_points)
        sampled = True
        notes.append(
            f"Showing a sample of {len(work):,} of {rows_original:,} points to keep the chart "
            "responsive. The exported code uses the full dataset."
        )

    return PreparedData(work, eff, spec, notes, sampled, rows_original)


def _cap_categories(
    df: pd.DataFrame, column: str, limit: int, spec: ChartSpec
) -> tuple[pd.DataFrame, int]:
    """Keep the ``limit`` largest categories in ``column``; bucket the rest as "Other"."""
    if spec.y and spec.y in df.columns and pd.api.types.is_numeric_dtype(df[spec.y]):
        ranking = df.groupby(column, observed=True)[spec.y].sum().nlargest(limit)
    else:
        ranking = df[column].value_counts().nlargest(limit)
    keep = set(ranking.index)
    out = df.copy()
    mask = ~out[column].isin(keep)
    dropped = int(out.loc[mask, column].nunique())
    out[column] = out[column].where(~mask, OTHER_LABEL)
    return out, dropped


def group_keys(*columns: str | None, available: list[str] | None = None) -> list[str]:
    """Group-by keys from a spec's mapping — present, ordered, and deduplicated.

    A spec may legitimately map the same column to two roles (a bar chart coloured by its own x
    axis is a common way to get per-category colours). Passing that column to ``groupby`` twice
    makes ``reset_index`` fail with "cannot insert region, already exists", so the duplicate is
    dropped here rather than surfacing as a crash mid-render.
    """
    seen: list[str] = []
    for column in columns:
        if not column or column in seen:
            continue
        if available is not None and column not in available:
            continue
        seen.append(column)
    return seen


def _group_keys(df: pd.DataFrame, *columns: str | None) -> list[str]:
    """:func:`group_keys` restricted to the columns a frame actually has."""
    return group_keys(*columns, available=list(df.columns))


def _aggregate(df: pd.DataFrame, spec: ChartSpec, eff: ChartSpec) -> pd.DataFrame:
    """Group by the spec's keys and collapse the measure.

    Args:
        df: Frame holding the source columns.
        spec: The caller's spec, which names real source columns.
        eff: The effective spec, which names the measure column to produce.
    """
    keys = _group_keys(df, spec.x, spec.color)
    measure = eff.y
    if spec.chart == "heatmap":
        keys = _group_keys(df, spec.x, spec.y)
        measure = eff.color
    if not keys or not measure:
        return df

    counting = spec.agg == "count" or not spec.y or spec.y not in df.columns
    if spec.chart == "heatmap":
        counting = eff.agg == "count" or not spec.color or spec.color not in df.columns

    if counting:
        return df.groupby(keys, observed=True, dropna=False).size().reset_index(name=measure)

    source_measure = spec.color if spec.chart == "heatmap" else spec.y
    if not source_measure:
        return df
    try:
        grouped = df.groupby(keys, observed=True, dropna=False)[source_measure].agg(spec.agg)
    except (TypeError, ValueError, KeyError) as exc:
        raise PlotError(
            f"Cannot compute the {spec.agg} of {source_measure!r} grouped by {', '.join(keys)}.",
            hint="The measure column is probably not numeric. Check the type override panel.",
        ) from exc
    out = grouped.reset_index()
    if source_measure != measure:
        out = out.rename(columns={source_measure: measure})
    return out


def _downsample(df: pd.DataFrame, spec: ChartSpec, max_points: int) -> pd.DataFrame:
    """Reduce a dense frame, preserving shape.

    Time-ordered data uses LTTB (largest-triangle-three-buckets), which keeps peaks and troughs
    a random sample would erase. Everything else takes a reproducible random sample.
    """
    is_time_series = (
        spec.x is not None
        and spec.x in df.columns
        and pd.api.types.is_datetime64_any_dtype(df[spec.x])
        and spec.y is not None
        and spec.y in df.columns
    )
    if is_time_series and spec.color is None:
        return _lttb(df, str(spec.x), str(spec.y), max_points)
    return df.sample(max_points, random_state=0).sort_index()


def _lttb(df: pd.DataFrame, x: str, y: str, threshold: int) -> pd.DataFrame:
    """Largest-triangle-three-buckets downsampling for time series.

    Keeps the first and last points, then picks from each bucket the point forming the largest
    triangle with its neighbours — the standard way to shrink a series without flattening spikes.
    """
    n = len(df)
    if threshold >= n or threshold < 3:
        return df

    ordered = df.sort_values(x)
    xs = pd.to_datetime(ordered[x]).astype("int64").to_numpy(dtype=float)
    ys = pd.to_numeric(ordered[y], errors="coerce").to_numpy(dtype=float)

    bucket_size = (n - 2) / (threshold - 2)
    picked = [0]
    a = 0
    for i in range(threshold - 2):
        start = int(np.floor((i + 1) * bucket_size)) + 1
        end = min(int(np.floor((i + 2) * bucket_size)) + 1, n)
        next_start = end
        next_end = min(int(np.floor((i + 3) * bucket_size)) + 1, n)
        if next_start >= next_end:
            next_start, next_end = max(n - 1, start), n

        avg_x = float(np.nanmean(xs[next_start:next_end])) if next_end > next_start else xs[-1]
        avg_y = float(np.nanmean(ys[next_start:next_end])) if next_end > next_start else ys[-1]

        if start >= end:
            continue
        areas = np.abs(
            (xs[a] - avg_x) * (ys[start:end] - ys[a]) - (xs[a] - xs[start:end]) * (avg_y - ys[a])
        )
        offset = int(np.nanargmax(areas)) if np.isfinite(areas).any() else 0
        a = start + offset
        picked.append(a)

    picked.append(n - 1)
    return ordered.iloc[sorted(set(picked))]


def kde_curve(values: pd.Series, *, points: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian kernel density estimate on a fixed grid.

    Implemented directly on numpy rather than pulling in scipy — the dependency is not worth one
    curve, and Silverman's rule is a perfectly good default bandwidth here.

    Returns:
        ``(grid, density)`` arrays suitable for a line plot.
    """
    data = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if data.size < 2:
        raise PlotError("A density curve needs at least two numeric values.")

    std = float(np.std(data, ddof=1)) or 1.0
    iqr = float(np.subtract(*np.percentile(data, [75, 25])))
    spread = min(std, iqr / 1.349) if iqr > 0 else std
    bandwidth = 0.9 * spread * data.size ** (-1 / 5) or 1.0

    grid = np.linspace(data.min() - 3 * bandwidth, data.max() + 3 * bandwidth, points)
    # (grid, data) distance matrix; fine for the sample sizes a KDE is meaningful on.
    if data.size > 20_000:
        data = np.random.default_rng(0).choice(data, 20_000, replace=False)
    z = (grid[:, None] - data[None, :]) / bandwidth
    density = np.exp(-0.5 * z**2).sum(axis=1) / (data.size * bandwidth * np.sqrt(2 * np.pi))
    return grid, density


# ---------------------------------------------------------------------------- Plotly


def build_plotly(df: pd.DataFrame, spec: ChartSpec, *, prepared: PreparedData | None = None) -> Any:
    """Build an interactive Plotly figure.

    Args:
        df: Cleaned, filtered data.
        spec: What to draw.
        prepared: Reuse an existing :func:`prepare` result instead of recomputing it.

    Returns:
        A ``plotly.graph_objects.Figure``.

    Raises:
        PlotError: If Plotly is unavailable or the chart cannot be built.
    """
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise PlotError(
            "Interactive charts need the plotly package.",
            hint="Install it with: pip install plotly",
        ) from exc

    data = prepared or prepare(df, spec)
    frame, spec = data.df, data.spec
    title = spec.display_title()
    opts = spec.options

    try:
        chart = spec.chart
        if chart == "histogram":
            fig = px.histogram(
                frame, x=spec.x, color=spec.color, nbins=opts.get("bins"), title=title
            )
        elif chart == "kde":
            grid, density = kde_curve(frame[spec.x])
            fig = go.Figure(go.Scatter(x=grid, y=density, fill="tozeroy", mode="lines"))
            fig.update_layout(title=title, xaxis_title=spec.x, yaxis_title="density")
        elif chart in {"bar", "grouped_bar", "stacked_bar"}:
            fig = px.bar(
                frame,
                x=spec.x,
                y=spec.y,
                color=spec.color,
                barmode="stack" if chart == "stacked_bar" or opts.get("stacked") else "group",
                orientation=opts.get("orientation", "v"),
                title=title,
            )
        elif chart == "pie":
            fig = px.pie(frame, names=spec.x, values=spec.y, title=title)
        elif chart == "treemap":
            path = [spec.x] if not spec.color else [spec.color, spec.x]
            fig = px.treemap(frame, path=path, values=spec.y, title=title)
        elif chart == "scatter":
            fig = px.scatter(
                frame,
                x=spec.x,
                y=spec.y,
                color=spec.color,
                opacity=opts.get("opacity", 0.75),
                trendline="ols" if opts.get("trendline") and _has_statsmodels() else None,
                title=title,
            )
        elif chart == "line":
            fig = px.line(
                frame, x=spec.x, y=spec.y, color=spec.color, markers=len(frame) < 200, title=title
            )
        elif chart == "area":
            fig = px.area(frame, x=spec.x, y=spec.y, color=spec.color, title=title)
        elif chart == "box":
            fig = px.box(
                frame, x=spec.x, y=spec.y, color=spec.color, points="outliers", title=title
            )
        elif chart == "violin":
            fig = px.violin(frame, x=spec.x, y=spec.y, color=spec.color, box=True, title=title)
        elif chart == "heatmap":
            value = spec.color or "count"
            value = value if value in frame.columns else frame.columns[-1]
            matrix = frame.pivot_table(index=spec.y, columns=spec.x, values=value, aggfunc="mean")
            fig = px.imshow(matrix, aspect="auto", text_auto=matrix.size <= 100, title=title)
        elif chart == "correlation_heatmap":
            corr = frame.corr(numeric_only=True)
            fig = px.imshow(
                corr,
                zmin=-1,
                zmax=1,
                color_continuous_scale="RdBu_r",
                text_auto=".2f" if corr.size <= 100 else False,
                aspect="auto",
                title=title,
            )
        elif chart == "pair_plot":
            fig = px.scatter_matrix(frame, dimensions=list(frame.columns)[:6], title=title)
            fig.update_traces(diagonal_visible=False, showupperhalf=False)
        else:  # pragma: no cover - validate() rejects unknown types first
            raise PlotError(f"No Plotly renderer for chart type {chart!r}.")
    except PlotError:
        raise
    except Exception as exc:
        raise PlotError(f"Could not draw the {spec.chart} chart.", hint=str(exc)) from exc

    if opts.get("log_y"):
        fig.update_yaxes(type="log")
    if opts.get("log_x"):
        fig.update_xaxes(type="log")

    fig.update_layout(
        template=opts.get("template", "plotly_white"),
        margin={"l": 60, "r": 30, "t": 60, "b": 60},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    if data.sampled:
        fig.add_annotation(
            text=data.sampling_notice,
            xref="paper",
            yref="paper",
            x=0,
            y=-0.16,
            showarrow=False,
            font={"size": 11, "color": "#666"},
        )
    return fig


def _has_statsmodels() -> bool:
    """Plotly's OLS trendline needs statsmodels; degrade quietly when it is absent."""
    try:
        import statsmodels.api  # noqa: F401
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------- matplotlib


def build_matplotlib(
    df: pd.DataFrame,
    spec: ChartSpec,
    *,
    prepared: PreparedData | None = None,
    figsize: tuple[float, float] = (10.0, 6.0),
    dpi: int = 300,
) -> Any:
    """Build a static matplotlib figure for export.

    Uses the ``Agg`` backend explicitly so this works headless — CLI mode and CI both rely on it.

    Args:
        df: Cleaned, filtered data.
        spec: What to draw.
        prepared: Reuse an existing :func:`prepare` result.
        figsize: Figure size in inches.
        dpi: Dots per inch. 300 is print quality and the export default.

    Returns:
        A ``matplotlib.figure.Figure``.

    Raises:
        PlotError: If the chart cannot be built.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    data = prepared or prepare(df, spec)
    frame, spec = data.df, data.spec
    opts = spec.options
    chart = spec.chart

    try:
        if chart == "pair_plot":
            fig = _mpl_pair_plot(frame, figsize=figsize, dpi=dpi)
            fig.suptitle(spec.display_title())
            return fig

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        if chart == "histogram":
            ax.hist(
                pd.to_numeric(frame[spec.x], errors="coerce").dropna(),
                bins=opts.get("bins", "auto"),
                color="#4C78A8",
                edgecolor="white",
            )
            ax.set_xlabel(str(spec.x))
            ax.set_ylabel("count")
        elif chart == "kde":
            grid, density = kde_curve(frame[spec.x])
            ax.plot(grid, density, color="#4C78A8")
            ax.fill_between(grid, density, alpha=0.3, color="#4C78A8")
            ax.set_xlabel(str(spec.x))
            ax.set_ylabel("density")
        elif chart in {"bar", "grouped_bar", "stacked_bar"}:
            _mpl_bar(ax, frame, spec, stacked=chart == "stacked_bar" or bool(opts.get("stacked")))
        elif chart == "pie":
            values = frame[spec.y] if spec.y and spec.y in frame else frame.iloc[:, -1]
            ax.pie(values, labels=frame[spec.x].astype(str).tolist(), autopct="%1.1f%%")
            ax.set_ylabel("")
        elif chart == "treemap":
            _mpl_treemap(ax, frame, spec)
        elif chart == "scatter":
            _mpl_scatter(ax, frame, spec)
        elif chart in {"line", "area"}:
            _mpl_line(ax, frame, spec, area=chart == "area")
        elif chart in {"box", "violin"}:
            _mpl_distribution(ax, frame, spec, violin=chart == "violin")
        elif chart == "heatmap":
            value = spec.color if spec.color in frame.columns else frame.columns[-1]
            matrix = frame.pivot_table(index=spec.y, columns=spec.x, values=value, aggfunc="mean")
            image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap="viridis")
            ax.set_xticks(
                range(len(matrix.columns)),
                [str(c) for c in matrix.columns],
                rotation=45,
                ha="right",
            )
            ax.set_yticks(range(len(matrix.index)), [str(i) for i in matrix.index])
            fig.colorbar(image, ax=ax, label=str(value))
        elif chart == "correlation_heatmap":
            corr = frame.corr(numeric_only=True)
            image = ax.imshow(corr.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="RdBu_r")
            ax.set_xticks(range(len(corr.columns)), list(corr.columns), rotation=45, ha="right")
            ax.set_yticks(range(len(corr.index)), list(corr.index))
            if corr.size <= 100:
                for i in range(len(corr.index)):
                    for j in range(len(corr.columns)):
                        ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=8)
            fig.colorbar(image, ax=ax, label="Pearson r")
        else:  # pragma: no cover
            raise PlotError(f"No matplotlib renderer for chart type {chart!r}.")

        if opts.get("log_y"):
            ax.set_yscale("log")
        if opts.get("log_x"):
            ax.set_xscale("log")

        ax.set_title(spec.display_title())
        if chart not in {"pie", "treemap", "heatmap", "correlation_heatmap"}:
            if spec.x:
                ax.set_xlabel(str(spec.x))
            if spec.y and chart not in {"histogram", "kde"}:
                ax.set_ylabel(str(spec.y))
            ax.grid(True, alpha=0.25, linestyle="--")
            ax.set_axisbelow(True)

        if data.sampled:
            fig.text(0.01, 0.01, data.sampling_notice, fontsize=8, color="#666")

        fig.tight_layout()
        return fig
    except PlotError:
        raise
    except Exception as exc:
        raise PlotError(f"Could not draw the {spec.chart} chart.", hint=str(exc)) from exc


def _mpl_bar(ax: Any, frame: pd.DataFrame, spec: ChartSpec, *, stacked: bool) -> None:
    """Draw a plain, grouped, or stacked bar chart on ``ax``."""
    x, y = str(spec.x), str(spec.y) if spec.y else None
    if y is None or y not in frame.columns:
        counts = frame[x].value_counts()
        ax.bar([str(i) for i in counts.index], counts.to_numpy(), color="#4C78A8")
        ax.set_ylabel("count")
    elif spec.color and spec.color in frame.columns:
        pivot = frame.pivot_table(index=x, columns=spec.color, values=y, aggfunc="mean")
        pivot.plot(kind="bar", stacked=stacked, ax=ax, width=0.8)
        ax.legend(title=str(spec.color), fontsize=8)
    else:
        ax.bar(frame[x].astype(str), pd.to_numeric(frame[y], errors="coerce"), color="#4C78A8")
    if len(ax.get_xticklabels()) > 6:
        ax.tick_params(axis="x", rotation=45)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")


def _mpl_scatter(ax: Any, frame: pd.DataFrame, spec: ChartSpec) -> None:
    """Draw a scatter, optionally split by colour, with an optional least-squares trendline."""
    x, y = str(spec.x), str(spec.y)
    if spec.color and spec.color in frame.columns:
        for name, group in frame.groupby(spec.color, observed=True):
            ax.scatter(group[x], group[y], s=14, alpha=0.7, label=str(name))
        ax.legend(title=str(spec.color), fontsize=8)
    else:
        ax.scatter(frame[x], frame[y], s=14, alpha=0.7, color="#4C78A8")

    if spec.options.get("trendline"):
        xs = pd.to_numeric(frame[x], errors="coerce")
        ys = pd.to_numeric(frame[y], errors="coerce")
        mask = xs.notna() & ys.notna()
        if int(mask.sum()) > 2:
            slope, intercept = np.polyfit(xs[mask], ys[mask], 1)
            line = np.linspace(xs[mask].min(), xs[mask].max(), 100)
            ax.plot(line, slope * line + intercept, color="#E45756", linewidth=1.5, label="trend")


def _mpl_line(ax: Any, frame: pd.DataFrame, spec: ChartSpec, *, area: bool) -> None:
    """Draw a line or filled area, one series per colour value."""
    x, y = str(spec.x), str(spec.y)
    if spec.color and spec.color in frame.columns:
        for name, group in frame.groupby(spec.color, observed=True):
            group = group.sort_values(x)
            ax.plot(group[x], group[y], label=str(name), linewidth=1.6)
            if area:
                ax.fill_between(group[x], group[y], alpha=0.25)
        ax.legend(title=str(spec.color), fontsize=8)
    else:
        ordered = frame.sort_values(x)
        ax.plot(ordered[x], ordered[y], color="#4C78A8", linewidth=1.6)
        if area:
            ax.fill_between(ordered[x], ordered[y], alpha=0.3, color="#4C78A8")
    if pd.api.types.is_datetime64_any_dtype(frame[x]):
        ax.figure.autofmt_xdate()


def _mpl_distribution(ax: Any, frame: pd.DataFrame, spec: ChartSpec, *, violin: bool) -> None:
    """Draw a box or violin plot, one body per category on the x axis."""
    y = str(spec.y)
    if spec.x and spec.x in frame.columns:
        groups = [
            (str(name), pd.to_numeric(group[y], errors="coerce").dropna().to_numpy())
            for name, group in frame.groupby(spec.x, observed=True)
        ]
        groups = [(name, values) for name, values in groups if values.size]
    else:
        groups = [(y, pd.to_numeric(frame[y], errors="coerce").dropna().to_numpy())]

    if not groups:
        raise PlotError("No numeric values are available to summarize.")

    labels = [name for name, _ in groups]
    values = [vals for _, vals in groups]
    if violin:
        ax.violinplot(values, showmedians=True)
        ax.set_xticks(range(1, len(labels) + 1), labels)
    else:
        ax.boxplot(values, tick_labels=labels)
    if len(labels) > 6:
        ax.tick_params(axis="x", rotation=45)


def _mpl_treemap(ax: Any, frame: pd.DataFrame, spec: ChartSpec) -> None:
    """Draw a simple slice-and-dice treemap without pulling in a dedicated dependency."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    labels = frame[str(spec.x)].astype(str).tolist()
    if spec.y and spec.y in frame.columns:
        sizes = pd.to_numeric(frame[spec.y], errors="coerce").fillna(0).tolist()
    else:
        counts = frame[str(spec.x)].value_counts()
        labels, sizes = [str(i) for i in counts.index], counts.tolist()

    pairs = sorted(zip(labels, sizes, strict=True), key=lambda p: p[1], reverse=True)
    pairs = [(label, size) for label, size in pairs if size > 0][:40]
    if not pairs:
        raise PlotError("The treemap has no positive values to lay out.")

    total = sum(size for _, size in pairs)
    cmap = plt.get_cmap("tab20")
    x = y = 0.0
    width = height = 1.0
    horizontal = True
    for i, (label, size) in enumerate(pairs):
        fraction = size / total if total else 0
        if horizontal:
            rect_w, rect_h = width * fraction / (fraction + _remaining(pairs, i, total)), height
        else:
            rect_w, rect_h = width, height * fraction / (fraction + _remaining(pairs, i, total))
        ax.add_patch(
            Rectangle(
                (x, y), rect_w, rect_h, facecolor=cmap(i % 20), edgecolor="white", linewidth=1.5
            )
        )
        if rect_w > 0.06 and rect_h > 0.05:
            ax.text(
                x + rect_w / 2,
                y + rect_h / 2,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                weight="bold",
            )
        if horizontal:
            x += rect_w
            width -= rect_w
        else:
            y += rect_h
            height -= rect_h
        horizontal = not horizontal
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def _remaining(pairs: list[tuple[str, float]], index: int, total: float) -> float:
    """Fraction of the total still to be laid out after ``index``."""
    rest = sum(size for _, size in pairs[index + 1 :])
    return rest / total if total else 0.0


def _mpl_pair_plot(frame: pd.DataFrame, *, figsize: tuple[float, float], dpi: int) -> Any:
    """Scatter-matrix of up to five numeric columns, histograms on the diagonal."""
    import matplotlib.pyplot as plt

    cols = list(frame.columns)[:5]
    n = len(cols)
    fig, axes = plt.subplots(n, n, figsize=figsize, dpi=dpi, squeeze=False)
    for i, row_col in enumerate(cols):
        for j, col_col in enumerate(cols):
            ax = axes[i][j]
            if i == j:
                ax.hist(frame[row_col].dropna(), bins=20, color="#4C78A8", edgecolor="white")
            else:
                ax.scatter(frame[col_col], frame[row_col], s=6, alpha=0.5, color="#4C78A8")
            if i == n - 1:
                ax.set_xlabel(str(col_col), fontsize=8)
            if j == 0:
                ax.set_ylabel(str(row_col), fontsize=8)
            ax.tick_params(labelsize=6)
    fig.tight_layout()
    return fig
