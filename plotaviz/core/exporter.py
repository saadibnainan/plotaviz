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

from .errors import ExportError
from .plotter import build_matplotlib, prepare
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
    anything.

    Raises:
        ExportError: If Plotly is missing or the file cannot be written.
    """
    from .plotter import build_plotly

    target = Path(path).expanduser().with_suffix(".html")
    figure = build_plotly(df, spec)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        figure.write_html(str(target), include_plotlyjs="cdn", full_html=True)
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
