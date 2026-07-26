"""Entry point — launches the GUI, or runs headless from the command line.

``plotaviz`` with no arguments opens the window. ``plotaviz --input data.csv --auto --export
chart.png`` does the whole pipeline without a display, which is nearly free because ``core`` has
no Qt in it: load, clean, profile, rank, render, write. That makes PlotaViz usable from a
Makefile or a CI job, not just from a desktop.

Qt is imported only on the GUI path. A headless machine never touches it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import __app_name__, __version__

logger = logging.getLogger("plotaviz")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="plotaviz",
        description=(
            f"{__app_name__} — automatic data analytics and visualization. "
            "Run with no arguments to open the desktop app."
        ),
        epilog=(
            "Examples:\n"
            "  plotaviz\n"
            "  plotaviz data.csv\n"
            "  plotaviz --input data.csv --auto --export chart.png\n"
            "  plotaviz --input data.csv --auto --export-code plot.py --code-flavour plotly\n"
            "  plotaviz --input data.csv --describe\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "file", nargs="?", help="dataset or .pviz session to open in the GUI", default=None
    )
    parser.add_argument("-i", "--input", help="dataset to process headlessly")
    parser.add_argument(
        "--auto",
        action="store_true",
        help="pick the top-ranked chart automatically (implied by --export/--export-code)",
    )
    parser.add_argument(
        "-e", "--export", metavar="PATH", help="write the chart to PNG/SVG/PDF/HTML"
    )
    parser.add_argument(
        "--export-code", metavar="PATH", help="write a standalone Python script that reproduces it"
    )
    parser.add_argument(
        "--code-flavour",
        choices=("matplotlib", "plotly"),
        default="matplotlib",
        help="library the generated script uses (default: matplotlib)",
    )

    chart = parser.add_argument_group("chart selection")
    chart.add_argument("--chart", help="force a chart type instead of using the recommendation")
    chart.add_argument("--x", help="column for the x axis")
    chart.add_argument("--y", help="column for the y axis")
    chart.add_argument("--color", help="column to split series by")
    chart.add_argument("--agg", help="aggregation: sum, mean, median, min, max, count, nunique")

    cleaning = parser.add_argument_group("cleaning")
    cleaning.add_argument(
        "--missing",
        default="median",
        choices=("drop", "mean", "median", "mode", "ffill", "bfill", "zero"),
        help="missing-value strategy (default: median)",
    )
    cleaning.add_argument(
        "--no-outlier-flags", action="store_true", help="skip adding outlier flag columns"
    )
    cleaning.add_argument("--query", help='pandas query applied as a filter, e.g. "revenue > 1000"')

    output = parser.add_argument_group("output")
    output.add_argument("--dpi", type=int, default=300, help="export resolution (default: 300)")
    output.add_argument(
        "--size", default="10x6", help="export size in inches, WIDTHxHEIGHT (default: 10x6)"
    )
    output.add_argument(
        "--describe", action="store_true", help="print the dataset profile and exit"
    )
    output.add_argument(
        "--recommend", action="store_true", help="print the ranked chart recommendations and exit"
    )
    output.add_argument(
        "--json", action="store_true", help="machine-readable output for --describe/--recommend"
    )
    output.add_argument("-q", "--quiet", action="store_true", help="only print errors")
    output.add_argument("-v", "--verbose", action="store_true", help="print debug logging")
    parser.add_argument("--version", action="version", version=f"{__app_name__} {__version__}")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run PlotaViz.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        A process exit code: 0 on success, 1 on a handled error, 2 on bad usage.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    headless = bool(args.input or args.describe or args.recommend)
    if not headless:
        return _run_gui(args.file)
    return _run_cli(args, parser)


# ---------------------------------------------------------------------------- GUI


def _run_gui(initial_file: str | None) -> int:
    """Open the desktop app."""
    try:
        from .ui.main_window import launch
    except ImportError as exc:  # pragma: no cover - PySide6 is a hard dependency
        print(
            f"The {__app_name__} window needs PySide6, which is not installed.\n"
            f"Install it with: pip install PySide6\n\nUnderlying error: {exc}",
            file=sys.stderr,
        )
        return 1
    return launch(initial_file)


# ---------------------------------------------------------------------------- CLI


def _run_cli(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Run the headless pipeline."""
    from .core.analysis import Analysis
    from .core.errors import PlotaVizError
    from .core.preprocess import QueryFilter
    from .core.spec import ChartSpec

    source = args.input or args.file
    if not source:
        parser.error("--describe and --recommend need --input")
        return 2

    try:
        analysis = Analysis.from_file(
            source,
            missing_strategy=args.missing,
            flag_outliers=not args.no_outlier_flags,
        )

        if args.query:
            analysis.set_filters([QueryFilter(args.query)])

        if not args.quiet:
            print(analysis.load_result.summary(), file=sys.stderr)

        if args.describe:
            _print_profile(analysis, as_json=args.json)
            return 0

        if args.recommend:
            _print_recommendations(analysis, as_json=args.json)
            return 0

        # --- choose a chart
        if args.chart:
            spec = ChartSpec(
                chart=args.chart,
                x=args.x,
                y=args.y,
                color=args.color,
                agg=args.agg,
                why="Specified on the command line.",
                source="user",
            )
            analysis.choose(spec)
        elif args.x or args.y:
            # Partial mapping: rank charts, preferring the columns the user named.
            from .core.selector import ChartSelector

            prefer = [c for c in (args.x, args.y, args.color) if c]
            if analysis.profile is None:
                raise PlotaVizError("The dataset could not be profiled.")
            ranked = ChartSelector().recommend(analysis.profile, prefer_columns=prefer)
            analysis.choose(ranked[0])
        elif analysis.spec is None:
            raise PlotaVizError(
                "PlotaViz could not recommend a chart for this dataset.",
                hint="Pass --chart, --x, and --y to specify one.",
            )

        spec = analysis.spec
        assert spec is not None
        if not args.quiet:
            print(
                f"Chart: {spec.chart}  x={spec.x}  y={spec.y}  colour={spec.color}", file=sys.stderr
            )
            if spec.why:
                print(f"Why: {spec.why}", file=sys.stderr)

        wrote_something = False

        if args.export:
            width, height = _parse_size(args.size, parser)
            target = Path(args.export)
            if target.suffix.lower() == ".html":
                from .core.exporter import export_html

                written = export_html(analysis.df, spec, target)
            else:
                written = analysis.export_image(target, dpi=args.dpi, width=width, height=height)
            print(written)
            wrote_something = True

        if args.export_code:
            from .core import codegen

            written = codegen.write(
                args.export_code, analysis.generate_code(flavour=args.code_flavour)
            )
            print(written)
            wrote_something = True

        if not wrote_something:
            _print_recommendations(analysis, as_json=args.json)

        return 0

    except PlotaVizError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        logger.debug("Unhandled error", exc_info=True)
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _parse_size(text: str, parser: argparse.ArgumentParser) -> tuple[float, float]:
    """Parse a ``WIDTHxHEIGHT`` size string into inches."""
    try:
        width, _, height = text.lower().partition("x")
        return float(width), float(height)
    except ValueError:
        parser.error(f"--size expects WIDTHxHEIGHT in inches, e.g. 10x6 (got {text!r})")
        raise  # unreachable; parser.error exits


def _print_profile(analysis: object, *, as_json: bool) -> None:
    """Print the dataset profile."""
    profile = getattr(analysis, "profile", None)
    if profile is None:
        print("No profile is available.", file=sys.stderr)
        return

    if as_json:
        print(json.dumps(profile.schema_summary(), indent=2, default=str))
        return

    print(f"\n{profile.n_rows:,} rows × {profile.n_cols} columns")
    if profile.sampled:
        print(f"(profiled on a sample of {profile.n_rows:,} of {profile.total_rows:,} rows)")
    if profile.n_duplicate_rows:
        print(f"{profile.n_duplicate_rows:,} duplicate rows")

    header = f"\n{'column':<28} {'role':<12} {'unique':>10} {'missing':>9}  notes"
    print(header)
    print("-" * len(header))
    for prof in profile.columns.values():
        note = prof.note or ("identifier" if prof.is_identifier else "")
        print(
            f"{prof.name[:28]:<28} {prof.role:<12} {prof.n_unique:>10,} "
            f"{prof.pct_missing:>8.1f}%  {note}"
        )

    best = profile.strongest_correlation()
    if best:
        print(f"\nStrongest correlation: {best[0]} ↔ {best[1]}  (r = {best[2]:.2f})")


def _print_recommendations(analysis: object, *, as_json: bool) -> None:
    """Print the ranked chart recommendations."""
    recommendations = list(getattr(analysis, "recommendations", []))
    if not recommendations:
        print("No chart recommendations for this dataset.", file=sys.stderr)
        return

    if as_json:
        print(json.dumps([spec.to_dict() for spec in recommendations], indent=2, default=str))
        return

    print("\nRecommended charts, best first:\n")
    for i, spec in enumerate(recommendations, start=1):
        mapping = ", ".join(
            part
            for part in (
                f"x={spec.x}" if spec.x else "",
                f"y={spec.y}" if spec.y else "",
                f"colour={spec.color}" if spec.color else "",
                f"agg={spec.agg}" if spec.agg else "",
            )
            if part
        )
        print(f"{i}. {spec.chart}  ({spec.score:.0%})  {mapping}")
        if spec.why:
            print(f"   {spec.why}")
        print()


if __name__ == "__main__":
    sys.exit(main())
