"""Shared fixtures.

Every fixture builds data with a *known shape*, because the whole test suite is really about one
question: given a shape, does PlotaViz do the right thing with it. Random data would make the
selector tests meaningless.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded generator, so failures reproduce."""
    return np.random.default_rng(1234)


@pytest.fixture
def timeseries_df(rng: np.random.Generator) -> pd.DataFrame:
    """Datetime × numeric × category — the time-series path."""
    dates = pd.date_range("2025-01-01", periods=180, freq="D")
    frames = []
    for i, region in enumerate(("north", "south", "east")):
        frames.append(
            pd.DataFrame(
                {
                    "order_date": dates,
                    "region": region,
                    "revenue": 1000 * (i + 1)
                    + np.arange(len(dates)) * 3
                    + rng.normal(0, 40, len(dates)),
                    "units": rng.integers(5, 80, len(dates)),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def numeric_pair_df(rng: np.random.Generator) -> pd.DataFrame:
    """Two strongly correlated numeric columns plus a label — the scatter path."""
    x = rng.normal(50, 12, 300)
    return pd.DataFrame(
        {
            "width": np.round(x, 2),
            "height": np.round(x * 1.8 + rng.normal(0, 3, 300), 2),
            "grade": rng.choice(["a", "b", "c"], 300),
        }
    )


@pytest.fixture
def categorical_df(rng: np.random.Generator) -> pd.DataFrame:
    """Two categorical columns and a measure — the grouped-bar and heatmap path."""
    return pd.DataFrame(
        {
            "department": rng.choice(["eng", "sales", "support"], 200),
            "rating": rng.choice(["good", "ok", "bad"], 200),
            "tenure": np.round(rng.gamma(2, 2, 200), 2),
        }
    )


@pytest.fixture
def messy_df() -> pd.DataFrame:
    """Ugly column names, missing values, duplicates, outliers, and an ID column."""
    return pd.DataFrame(
        {
            "Order ID": [1, 2, 3, 4, 5, 5, 6, 7, 8, 9],
            " Total Revenue ": [10.0, 20.0, None, 40.0, 50.0, 50.0, 60.0, 70.0, 80.0, 5000.0],
            "customerName": ["a", "b", "c", "d", "e", "e", "f", "g", "h", "i"],
            "Signup-Date": [
                "2025-01-01",
                "2025-01-02",
                "2025-01-03",
                "2025-01-04",
                "2025-01-05",
                "2025-01-05",
                "2025-01-06",
                "2025-01-07",
                "2025-01-08",
                "2025-01-09",
            ],
            "is_active": ["yes", "no", "yes", "yes", "no", "no", "yes", "yes", "no", "yes"],
        }
    )


@pytest.fixture
def csv_path(tmp_path: Path, timeseries_df: pd.DataFrame) -> Path:
    """A CSV on disk, for loader and session tests."""
    target = tmp_path / "data.csv"
    timeseries_df.to_csv(target, index=False)
    return target


@pytest.fixture(
    params=["sales_timeseries.csv", "iris_like.csv", "survey_responses.csv", "sensor_readings.csv"]
)
def sample_file(request: pytest.FixtureRequest) -> Path:
    """Each shipped sample dataset in turn.

    Skips rather than fails when the samples have not been generated, so a fresh clone that has
    not run ``make samples`` still gets a green suite.
    """
    path = SAMPLES / str(request.param)
    if not path.exists():
        pytest.skip(f"{path.name} not generated; run `make samples`")
    return path


# ---------------------------------------------------------------------------- GUI helpers

#: Generous: CI runners are slow, and a load includes profiling and a first render.
LOAD_TIMEOUT_MS = 30_000


@pytest.fixture
def settings(tmp_path: Path) -> Any:
    """App settings stored under ``tmp_path``, with the static renderer forced on."""
    from PySide6.QtCore import QSettings

    QSettings.setPath(
        QSettings.Format.NativeFormat, QSettings.Scope.UserScope, str(tmp_path / "config")
    )
    store = QSettings("plotaviz", "plotaviz")
    store.clear()
    store.setValue("performance/prefer_static", True)
    store.sync()
    return store


@pytest.fixture
def window(qtbot: Any, settings: Any) -> Iterator[Any]:
    """A main window using the temporary settings."""
    from plotaviz.ui.main_window import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    yield win
    win.runner.cancel()
    win.runner.wait()


def wait_idle(qtbot: Any, win: Any) -> None:
    """Block until the worker thread has fully stopped and its callbacks have run."""
    qtbot.waitUntil(lambda: win.runner._thread is None, timeout=LOAD_TIMEOUT_MS)


def open_and_wait(qtbot: Any, win: Any, path: Path) -> None:
    """Open a dataset or session and wait for it to finish loading."""
    wait_idle(qtbot, win)
    win.open_path(path)
    wait_idle(qtbot, win)
    assert win.analysis is not None
