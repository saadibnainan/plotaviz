"""PlotaViz — automatic data analytics and visualization.

The package is split into two halves that must not blur together:

``plotaviz.core``
    Fully headless. No Qt imports anywhere. Loading, cleaning, profiling, chart selection,
    plotting, code generation, and session persistence all live here, which is what makes the
    CLI mode and the unit tests cheap.

``plotaviz.ui``
    PySide6 widgets. Talks to ``core`` through plain data objects only.
"""

from __future__ import annotations

__version__ = "0.1.0"
__app_name__ = "PlotaViz"

__all__ = ["__app_name__", "__version__"]
