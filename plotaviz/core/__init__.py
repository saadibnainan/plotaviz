"""Headless core — everything PlotaViz can do without a GUI.

Nothing in this package imports Qt. That is enforced by convention, by the test suite, and by CI
running ``mypy`` over this directory specifically. It is what makes ``plotaviz --input data.csv
--auto --export chart.png`` nearly free, and what makes the whole engine unit-testable.
"""

from __future__ import annotations

from .analysis import Analysis
from .errors import (
    ExportError,
    LLMError,
    LoadError,
    PlotaVizError,
    PlotError,
    PreprocessError,
    ProfileError,
    SelectionError,
    SessionError,
    SpecError,
)
from .loader import LoadResult, load
from .preprocess import Pipeline, Step, default_pipeline
from .profiler import DatasetProfile, profile
from .selector import ChartSelector
from .session import Session
from .spec import CHART_TYPES, ChartSpec

__all__ = [
    "CHART_TYPES",
    "Analysis",
    "ChartSelector",
    "ChartSpec",
    "DatasetProfile",
    "ExportError",
    "LLMError",
    "LoadError",
    "LoadResult",
    "Pipeline",
    "PlotError",
    "PlotaVizError",
    "PreprocessError",
    "ProfileError",
    "SelectionError",
    "Session",
    "SessionError",
    "SpecError",
    "Step",
    "default_pipeline",
    "load",
    "profile",
]
