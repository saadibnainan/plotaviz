"""Static image export — PNG, SVG, PDF.

Export always goes through matplotlib rather than the Plotly figure on screen. Plotly's static
export needs the kaleido binary, which is another large dependency and another packaging problem;
matplotlib is already present, renders headless, and produces better print output. The two
renderers share :func:`~plotaviz.core.plotter.prepare`, so what gets saved matches what was seen.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .errors import ExportError, PlotaVizError
from .plotter import build_matplotlib, build_matplotlib_grid, prepare
from .spec import ChartSpec

#: Formats :func:`export_image` can write.
IMAGE_FORMATS: tuple[str, ...] = ("png", "svg", "pdf", "jpg", "jpeg", "webp", "tiff")

#: Export DPI. 300 is print quality.
DEFAULT_DPI = 300

#: Default figure size in inches.
DEFAULT_SIZE = (10.0, 6.0)


@dataclass
class ExportOptions:
    """Settings from the export dialog.

    Attributes:
        dpi: Dots per inch for raster formats. Ignored by SVG and PDF, which are vector.
        width: Figure width in inches.
        height: Figure height in inches.
        transparent: Whether to leave the background transparent.
        include_notice: Whether to stamp the "showing a sample of N" note onto the image. On by
            default — an exported chart that hides its own sampling is a chart that misleads.
    """

    dpi: int = DEFAULT_DPI
    width: float = DEFAULT_SIZE[0]
    height: float = DEFAULT_SIZE[1]
    transparent: bool = False
    include_notice: bool = True

    @property
    def figsize(self) -> tuple[float, float]:
        """Size as the ``(width, height)`` tuple matplotlib wants."""
        return (float(self.width), float(self.height))


def export_image(
    df: pd.DataFrame,
    spec: ChartSpec,
    path: str | Path,
    *,
    options: ExportOptions | None = None,
) -> Path:
    """Render a spec and write it to an image file.

    The format comes from the file extension.

    Args:
        df: Cleaned, filtered data.
        spec: What to draw.
        path: Destination file.
        options: Size, DPI, and background settings.

    Returns:
        The path written.

    Raises:
        ExportError: If the extension is unsupported or the file cannot be written.
    """
    options = options or ExportOptions()
    target = Path(path).expanduser()
    fmt = target.suffix.lower().lstrip(".")

    if not fmt:
        target = target.with_suffix(".png")
        fmt = "png"
    if fmt not in IMAGE_FORMATS:
        raise ExportError(
            f"PlotaViz cannot export {fmt!r} images.",
            hint=f"Supported formats: {', '.join(IMAGE_FORMATS)}.",
        )

    prepared = prepare(df, spec)
    if not options.include_notice:
        prepared.sampled = False

    figure = build_matplotlib(df, spec, prepared=prepared, figsize=options.figsize, dpi=options.dpi)

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            target,
            dpi=options.dpi,
            bbox_inches="tight",
            transparent=options.transparent,
            format=fmt if fmt not in {"jpg"} else "jpeg",
        )
    except OSError as exc:
        raise ExportError(f"Could not write {target}.", hint=str(exc)) from exc
    except ValueError as exc:
        raise ExportError(f"Could not render the chart as {fmt}.", hint=str(exc)) from exc
    finally:
        _close(figure)

    return target


def export_html(df: pd.DataFrame, spec: ChartSpec, path: str | Path) -> Path:
    """Write the interactive Plotly figure as a self-contained HTML file.

    Useful for sharing a chart someone can actually zoom and hover, without them installing
    anything — or having internet access, since plotly.js is embedded.

    Raises:
        ExportError: If Plotly is missing or the file cannot be written.
    """
    from .plotter import build_plotly

    target = Path(path).expanduser().with_suffix(".html")
    figure = build_plotly(df, spec)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        # Embed plotly.js (~4 MB) so the page works offline, like the rest of PlotaViz.
        figure.write_html(str(target), include_plotlyjs=True, full_html=True)
    except OSError as exc:
        raise ExportError(f"Could not write {target}.", hint=str(exc)) from exc
    return target


def export_dashboard_image(
    df: pd.DataFrame,
    specs: list[ChartSpec],
    path: str | Path,
    *,
    options: ExportOptions | None = None,
    columns: int = 2,
    title: str | None = None,
) -> Path:
    """Write every dashboard tile into one image or PDF.

    ``options.width``/``height`` are the size of *one tile*; the page grows with the grid. SVG
    and PDF stay vector.

    Raises:
        ExportError: If the extension is unsupported or the file cannot be written.
    """
    options = options or ExportOptions(width=6.0, height=4.0)
    target = Path(path).expanduser()
    fmt = target.suffix.lower().lstrip(".") or "png"
    if not target.suffix:
        target = target.with_suffix(".png")
    if fmt not in IMAGE_FORMATS:
        raise ExportError(
            f"PlotaViz cannot export {fmt!r} images.",
            hint=f"Supported formats: {', '.join(IMAGE_FORMATS)}.",
        )
    figure = build_matplotlib_grid(
        df,
        specs,
        columns=columns,
        tile_size=options.figsize,
        dpi=options.dpi,
        title=title,
    )
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            target,
            dpi=options.dpi,
            transparent=options.transparent,
            format="jpeg" if fmt == "jpg" else fmt,
        )
    except (OSError, ValueError) as exc:
        raise ExportError(f"Could not write {target}.", hint=str(exc)) from exc
    finally:
        _close(figure)
    return target


def export_dashboard_html(
    df: pd.DataFrame,
    specs: list[ChartSpec],
    path: str | Path,
    *,
    columns: int = 2,
    title: str = "Dashboard",
) -> Path:
    """Write every dashboard tile as interactive Plotly charts on one self-contained page.

    plotly.js is embedded once in the page head, not once per chart, so the file stays the size
    of a single-chart export plus the data, and it works offline.

    Raises:
        ExportError: If the file cannot be written.
    """
    import html

    from plotly.offline import get_plotlyjs

    from .plotter import build_plotly

    if not specs:
        raise ExportError("The dashboard has no charts.", hint="Pin a chart first.")
    target = Path(path).expanduser().with_suffix(".html")
    tiles: list[str] = []
    for spec in specs:
        try:
            figure = build_plotly(df, spec)
            tiles.append(
                figure.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})
            )
        except PlotaVizError as exc:
            tiles.append(
                f'<div class="error"><b>{html.escape(spec.display_title())}</b>'
                f"<p>{html.escape(exc.message)}</p></div>"
            )
    cells = "\n".join(f'<div class="tile">{tile}</div>' for tile in tiles)
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<script type="text/javascript">{get_plotlyjs()}</script>
<style>
body {{ font-family: system-ui, sans-serif; margin: 16px; background: #fff; color: #222; }}
.grid {{ display: grid; gap: 16px;
         grid-template-columns: repeat({max(1, int(columns))}, minmax(0, 1fr)); }}
.tile {{ border: 1px solid #ddd; border-radius: 6px; padding: 4px; min-height: 380px; }}
.error {{ padding: 24px; color: #a33; }}
@media (max-width: 800px) {{ .grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<div class="grid">
{cells}
</div>
</body>
</html>
"""
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(page, encoding="utf-8")
    except OSError as exc:
        raise ExportError(f"Could not write {target}.", hint=str(exc)) from exc
    return target


def _close(figure: Any) -> None:
    """Close a matplotlib figure so long sessions do not leak them."""
    try:
        import matplotlib.pyplot as plt

        plt.close(figure)
    except Exception:
        pass
