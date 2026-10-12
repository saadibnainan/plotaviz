"""ChartView tests: the Qt widget that hosts the chart.

CI runs these under xvfb on Linux; locally they also work with ``QT_QPA_PLATFORM=offscreen``.
"""

from __future__ import annotations

import pandas as pd

from plotaviz.core.spec import ChartSpec
from plotaviz.ui.interactive_view import ChartView


def test_static_rerender_keeps_a_single_toolbar(qtbot, categorical_df: pd.DataFrame) -> None:
    """Regression: every static re-render stacked another matplotlib toolbar above the chart."""
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT

    view = ChartView(prefer_static=True)
    qtbot.addWidget(view)
    spec = ChartSpec("bar", x="department", y="tenure", agg="mean")

    for _ in range(3):  # open, then two re-runs (type override, filter change...)
        view.render_spec(categorical_df, spec)

    toolbars = [
        view._layout.itemAt(i).widget()
        for i in range(view._layout.count())
        if isinstance(view._layout.itemAt(i).widget(), NavigationToolbar2QT)
    ]
    assert len(toolbars) == 1


def test_placeholder_removes_the_static_toolbar(qtbot, categorical_df: pd.DataFrame) -> None:
    view = ChartView(prefer_static=True)
    qtbot.addWidget(view)
    view.render_spec(categorical_df, ChartSpec("bar", x="department", y="tenure", agg="mean"))

    view.show_message("No data")

    assert view._toolbar is None
    assert view._layout.count() == 1
