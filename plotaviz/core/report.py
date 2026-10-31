"""The analysis report — everything PlotaViz learned about a dataset, in one file.

One self-contained HTML page (plotly.js embedded once, works offline) or a PDF. It covers:
- what was loaded and how it was cleaned (the same step reports the Cleaning tab shows);
- every column's role, missingness, cardinality, and summary statistics;
- a missing-values chart and the correlation matrix;
- the top recommended charts, each with the reasoning behind it;
- the dashboard, when charts are pinned.

The report reflects the analysis exactly as it stands — current filters, cleaning, and types.
"""

from __future__ import annotations

import html
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .errors import ExportError, PlotaVizError
from .plotter import build_matplotlib, build_plotly
from .spec import ChartSpec

if TYPE_CHECKING:
    from .analysis import Analysis

#: Recommended charts included, best first.
REPORT_CHARTS = 6

#: Report formats, by file extension.
REPORT_FORMATS: tuple[str, ...] = ("html", "pdf")


def export_report(analysis: Analysis, path: str | Path, *, charts: int = REPORT_CHARTS) -> Path:
    """Write the report; the format comes from the extension (``.html`` or ``.pdf``).

    Raises:
        ExportError: For another extension, or if the file cannot be written.
    """
    target = Path(path).expanduser()
    fmt = target.suffix.lower().lstrip(".") or "html"
    if not target.suffix:
        target = target.with_suffix(".html")
    if fmt not in REPORT_FORMATS:
        raise ExportError(f"PlotaViz cannot write a {fmt!r} report.", hint="Use .html or .pdf.")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "html":
            target.write_text(build_report_html(analysis, charts=charts), encoding="utf-8")
        else:
            _write_pdf(analysis, target, charts=charts)
    except OSError as exc:
        raise ExportError(f"Could not write {target}.", hint=str(exc)) from exc
    return target


# ---------------------------------------------------------------------------- content


def report_charts(analysis: Analysis, limit: int = REPORT_CHARTS) -> list[ChartSpec]:
    """The charts a report shows: the active one first, then the best other recommendations."""
    chosen: list[ChartSpec] = []
    seen: set[tuple[Any, ...]] = set()
    for spec in [*([analysis.spec] if analysis.spec else []), *analysis.recommendations]:
        key = (spec.chart, spec.x, spec.y, spec.color, spec.agg)
        if key not in seen:
            seen.add(key)
            chosen.append(spec)
        if len(chosen) >= limit:
            break
    return chosen


def _summary_rows(analysis: Analysis) -> list[tuple[str, str]]:
    load = analysis.load_result
    profile = analysis.profile
    rows = [
        ("File", str(load.path)),
        ("Rows after cleaning and filters", f"{len(analysis.df):,}"),
        ("Rows in the file", f"{load.total_rows:,}"),
        ("Columns", f"{len(analysis.df.columns):,}"),
    ]
    if load.sampled:
        rows.append(("Sample", f"{load.rows_loaded:,} of {load.total_rows:,} rows (random)"))
    if profile is not None and profile.n_duplicate_rows:
        rows.append(("Duplicate rows remaining", f"{profile.n_duplicate_rows:,}"))
    best = profile.strongest_correlation() if profile else None
    if best:
        rows.append(("Strongest correlation", f"{best[0]} ↔ {best[1]} (r = {best[2]:.2f})"))
    return rows


def _column_rows(analysis: Analysis) -> list[list[str]]:
    profile = analysis.profile
    if profile is None:
        return []
    rows = []
    for prof in profile.columns.values():
        stats = prof.stats
        if "mean" in stats:
            summary = (
                f"min {stats['min']:,.4g} · median {stats['median']:,.4g} · max {stats['max']:,.4g}"
            )
        elif "min" in stats and isinstance(stats["min"], str):
            summary = f"{stats['min'][:10]} → {stats['max'][:10]}"
        elif prof.top_values:
            summary = ", ".join(f"{v} ({c:,})" for v, c in prof.top_values[:3])
        else:
            summary = ""
        role = prof.role + (" · identifier" if prof.is_identifier else "")
        rows.append(
            [prof.name, role, f"{prof.pct_missing:.1f}%", f"{prof.n_unique:,}", summary, prof.note]
        )
    return rows


def _missing_spec(analysis: Analysis) -> tuple[Any, ChartSpec] | None:
    """A frame + spec for the share of missing values per column, or ``None`` if none."""
    import pandas as pd

    profile = analysis.profile
    if profile is None:
        return None
    frame = pd.DataFrame(
        {
            "column": [p.name for p in profile.columns.values()],
            "missing_pct": [p.pct_missing for p in profile.columns.values()],
        }
    )
    frame = frame[frame["missing_pct"] > 0]
    if frame.empty:
        return None
    spec = ChartSpec(
        "bar",
        x="column",
        y="missing_pct",
        title="Missing values per column (%)",
        options={"sort_ascending": False},
    )
    return frame, spec


def _correlation_spec(analysis: Analysis) -> ChartSpec | None:
    profile = analysis.profile
    if profile is None or len(profile.numeric) < 2:
        return None
    return ChartSpec("correlation_heatmap", options={"columns": profile.numeric[:12]})


# ---------------------------------------------------------------------------- HTML


def build_report_html(analysis: Analysis, *, charts: int = REPORT_CHARTS) -> str:
    """The report as one self-contained HTML page."""
    from plotly.offline import get_plotlyjs

    def figure_html(df: Any, spec: ChartSpec) -> str:
        try:
            fig = build_plotly(df, spec)
            return str(fig.to_html(full_html=False, include_plotlyjs=False))
        except PlotaVizError as exc:
            return f'<p class="error">{html.escape(exc.message)}</p>'

    name = analysis.load_result.path.name
    summary = "".join(
        f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>"
        for k, v in _summary_rows(analysis)
    )
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in analysis.load_result.notes)
    cleaning = ""
    if analysis.result:
        cleaning = "".join(
            f"<li><b>{html.escape(r.step)}</b> — {html.escape(r.summary)}</li>"
            for r in analysis.result.reports
        )
    header = ["Column", "Role", "Missing", "Distinct", "Summary", "Note"]
    columns = "".join(
        "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>"
        for row in _column_rows(analysis)
    )
    sections: list[str] = []
    missing = _missing_spec(analysis)
    if missing is not None:
        sections.append("<h2>Missing values</h2>" + figure_html(missing[0], missing[1]))
    correlation = _correlation_spec(analysis)
    if correlation is not None:
        sections.append("<h2>Correlations</h2>" + figure_html(analysis.df, correlation))
    chart_blocks = []
    for number, spec in enumerate(report_charts(analysis, charts), start=1):
        why = f"<p class='why'>{html.escape(spec.why)}</p>" if spec.why else ""
        chart_blocks.append(
            f"<section class='chart'><h3>{number}. {html.escape(spec.display_title())}</h3>"
            f"{why}{figure_html(analysis.df, spec)}</section>"
        )
    if chart_blocks:
        sections.append("<h2>Recommended charts</h2>" + "".join(chart_blocks))
    if analysis.dashboard:
        tiles = "".join(
            f"<section class='chart'><h3>{html.escape(t.display_title())}</h3>"
            f"{figure_html(analysis.df, t)}</section>"
            for t in analysis.dashboard
        )
        sections.append("<h2>Dashboard</h2><div class='grid'>" + tiles + "</div>")

    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(name)} — PlotaViz report</title>
<script type="text/javascript">{get_plotlyjs()}</script>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px auto; max-width: 1100px;
       padding: 0 16px; color: #222; background: #fff; }}
table {{ border-collapse: collapse; margin: 8px 0 16px; font-size: 14px; }}
th, td {{ border-bottom: 1px solid #e5e5e5; padding: 4px 10px; text-align: left;
          vertical-align: top; }}
th {{ color: #555; font-weight: 600; }}
.columns td:nth-child(3), .columns td:nth-child(4) {{ text-align: right; }}
.why {{ color: #555; margin: 0 0 4px; }}
.error {{ color: #a33; }}
.chart {{ margin: 16px 0 28px; }}
.grid {{ display: grid; gap: 16px; grid-template-columns: repeat(2, minmax(0, 1fr)); }}
@media (max-width: 800px) {{ .grid {{ grid-template-columns: 1fr; }} }}
.muted {{ color: #777; font-size: 13px; }}
</style></head><body>
<h1>{html.escape(name)}</h1>
<p class="muted">PlotaViz report, generated {generated}. It reflects the current cleaning, types
and filters.</p>
<h2>Summary</h2><table>{summary}</table>
{f"<ul>{notes}</ul>" if notes else ""}
<h2>Cleaning</h2><ol>{cleaning}</ol>
<h2>Columns</h2>
<table class="columns"><tr>{"".join(f"<th>{h}</th>" for h in header)}</tr>{columns}</table>
{"".join(sections)}
</body></html>
"""


# ---------------------------------------------------------------------------- PDF


def _write_pdf(analysis: Analysis, target: Path, *, charts: int) -> None:
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure

    a4 = (8.27, 11.69)
    with PdfPages(target) as pdf:
        # Page 1: summary, cleaning, notes.
        page = Figure(figsize=a4)
        y = 0.95
        page.text(0.07, y, analysis.load_result.path.name, fontsize=18, weight="bold")
        y -= 0.04
        for key, value in _summary_rows(analysis):
            page.text(0.07, y, f"{key}:", fontsize=9, color="#555")
            page.text(0.40, y, value[:80], fontsize=9)
            y -= 0.022
        y -= 0.02
        page.text(0.07, y, "Cleaning", fontsize=13, weight="bold")
        y -= 0.03
        for report in analysis.result.reports if analysis.result else []:
            page.text(0.07, y, f"• {report.summary}"[:110], fontsize=8.5)
            y -= 0.02
        for note in analysis.load_result.notes:
            page.text(0.07, y, f"Note: {note}"[:110], fontsize=8.5, color="#555")
            y -= 0.02
        pdf.savefig(page)

        # Page 2: the columns table.
        rows = _column_rows(analysis)
        for start in range(0, len(rows), 34):
            page = Figure(figsize=a4)
            ax = page.add_axes((0.04, 0.05, 0.92, 0.88))
            ax.axis("off")
            page.text(0.04, 0.95, "Columns", fontsize=13, weight="bold")
            chunk = [[cell[:40] for cell in row[:5]] for row in rows[start : start + 34]]
            table = ax.table(
                cellText=chunk,
                colLabels=["Column", "Role", "Missing", "Distinct", "Summary"],
                loc="upper left",
                cellLoc="left",
                colWidths=[0.2, 0.17, 0.09, 0.1, 0.44],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(7)
            table.scale(1, 1.25)
            pdf.savefig(page)

        # Chart pages.
        figures: list[tuple[Any, ChartSpec, str]] = []
        missing = _missing_spec(analysis)
        if missing is not None:
            figures.append((missing[0], missing[1], ""))
        correlation = _correlation_spec(analysis)
        if correlation is not None:
            figures.append((analysis.df, correlation, ""))
        for spec in report_charts(analysis, charts):
            figures.append((analysis.df, spec, spec.why))
        for tile in analysis.dashboard:
            figures.append((analysis.df, tile, "Pinned to the dashboard."))
        for frame, spec, why in figures:
            try:
                figure = build_matplotlib(frame, spec, figsize=(8.27, 5.8), dpi=100)
            except PlotaVizError as exc:
                figure = Figure(figsize=(8.27, 5.8))
                figure.text(0.5, 0.5, f"{spec.display_title()}\n\n{exc.message}", ha="center")
            if why:
                figure.text(0.02, 0.005, _wrap(why, 140), fontsize=7, color="#555", va="bottom")
            pdf.savefig(figure)


def _wrap(text: str, width: int) -> str:
    import textwrap

    return "\n".join(textwrap.wrap(text, width)[:3])
