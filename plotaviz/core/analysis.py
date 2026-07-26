"""Orchestration facade: load → clean → profile → recommend → render.

The CLI and the GUI both need the same sequence, and duplicating it in two places is how the two
drift apart. :class:`Analysis` owns that sequence and nothing else — it holds state, calls the
single-responsibility modules around it, and stays free of Qt so the CLI can use it directly.

Everything is re-derivable: :meth:`Analysis.rerun` replays the pipeline from the raw frame, which
is what makes type overrides and filter changes cheap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from . import codegen, exporter, loader, plotter, selector
from .errors import PlotaVizError
from .preprocess import Pipeline, PipelineResult, Step, default_pipeline
from .profiler import DatasetProfile, profile
from .session import Session
from .spec import ChartSpec


@dataclass
class Analysis:
    """One dataset, its cleaning recipe, its profile, and its chart recommendations.

    Attributes:
        load_result: What :func:`plotaviz.core.loader.load` returned.
        pipeline: The active preprocessing recipe.
        type_overrides: User corrections to inferred column roles.
        result: Output of the last pipeline run.
        profile: Profile of the cleaned frame.
        recommendations: Ranked chart specs, best first.
        spec: The chart currently selected.
    """

    load_result: loader.LoadResult
    pipeline: Pipeline = field(default_factory=Pipeline)
    type_overrides: dict[str, str] = field(default_factory=dict)
    result: PipelineResult | None = None
    profile: DatasetProfile | None = None
    recommendations: list[ChartSpec] = field(default_factory=list)
    spec: ChartSpec | None = None

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        missing_strategy: str = "median",
        flag_outliers: bool = True,
        **load_kwargs: Any,
    ) -> Analysis:
        """Load a dataset and run the default cleaning pipeline against it.

        Args:
            path: Dataset to open.
            missing_strategy: Missing-value strategy for the default pipeline.
            flag_outliers: Whether to add outlier flag columns.
            **load_kwargs: Passed through to :func:`plotaviz.core.loader.load`.
        """
        load_result = loader.load(path, **load_kwargs)
        analysis = cls(load_result=load_result)

        # Profile the raw frame first so the pipeline knows which columns need coercing.
        raw_profile = profile(load_result.df, total_rows=load_result.total_rows)
        inferred = {name: prof.role for name, prof in raw_profile.columns.items()}

        analysis.pipeline = default_pipeline(
            _normalized_types(inferred),
            missing_strategy=missing_strategy,
            flag_outliers=flag_outliers,
        )
        analysis.rerun()
        return analysis

    # ------------------------------------------------------------------ pipeline

    @property
    def raw(self) -> pd.DataFrame:
        """The frame as loaded, before any cleaning."""
        return self.load_result.df

    @property
    def df(self) -> pd.DataFrame:
        """The cleaned, filtered frame — what gets charted."""
        return self.result.df if self.result else self.load_result.df

    def rerun(self) -> PipelineResult:
        """Replay the pipeline from the raw frame, then re-profile and re-recommend.

        This is the single path for every state change: a type override, a new filter, a changed
        missing-value strategy. Replaying is cheap because steps are pure.
        """
        self.result = self.pipeline.run(self.raw)
        self.profile = profile(
            self.result.df,
            overrides=self.type_overrides,
            total_rows=self.load_result.total_rows,
        )
        try:
            self.recommendations = selector.ChartSelector().recommend(self.profile)
        except PlotaVizError:
            self.recommendations = []
        if self.recommendations and (self.spec is None or self.spec.source == "rules"):
            self.spec = self.recommendations[0]
        return self.result

    def set_type_override(self, column: str, role: str) -> PipelineResult:
        """Correct one column's role and replay everything downstream."""
        self.type_overrides[column] = role
        for step in self.pipeline:
            if hasattr(step, "types"):
                step.types[column] = role  # type: ignore[attr-defined]
                break
        return self.rerun()

    def set_filters(self, filters: list[Step]) -> PipelineResult:
        """Replace the active filters and replay."""
        self.pipeline.replace_filters(filters)
        return self.rerun()

    # ------------------------------------------------------------------ charting

    def choose(self, spec: ChartSpec) -> ChartSpec:
        """Make ``spec`` the active chart after validating it against the real schema."""
        spec.validate(list(self.df.columns))
        self.spec = spec
        return spec

    def prepared(self, spec: ChartSpec | None = None) -> plotter.PreparedData:
        """Aggregate and downsample for a spec — shared by both renderers."""
        return plotter.prepare(self.df, self._require_spec(spec))

    def figure_plotly(self, spec: ChartSpec | None = None) -> Any:
        """Build the interactive Plotly figure."""
        return plotter.build_plotly(self.df, self._require_spec(spec))

    def figure_matplotlib(self, spec: ChartSpec | None = None, **kwargs: Any) -> Any:
        """Build the static matplotlib figure."""
        return plotter.build_matplotlib(self.df, self._require_spec(spec), **kwargs)

    # ------------------------------------------------------------------ output

    def export_image(
        self, path: str | Path, *, spec: ChartSpec | None = None, **kwargs: Any
    ) -> Path:
        """Write the chart to PNG/SVG/PDF."""
        options = exporter.ExportOptions(**kwargs) if kwargs else None
        return exporter.export_image(self.df, self._require_spec(spec), path, options=options)

    def generate_code(self, *, spec: ChartSpec | None = None, flavour: str = "matplotlib") -> str:
        """Emit a standalone script reproducing the current chart."""
        from .. import __version__

        return codegen.generate(
            self._require_spec(spec),
            self.pipeline,
            self.load_result.path,
            flavour=flavour,
            version=__version__,
        )

    def to_session(self, view: dict[str, Any] | None = None) -> Session:
        """Capture the current state as a savable session."""
        from .. import __version__

        return Session(
            source_path=str(self.load_result.path),
            source_hash=self.load_result.file_hash,
            type_overrides=dict(self.type_overrides),
            pipeline=self.pipeline,
            spec=self.spec,
            view=dict(view or {}),
            app_version=__version__,
        )

    @classmethod
    def from_session(cls, session: Session, **load_kwargs: Any) -> tuple[Analysis, str | None]:
        """Restore an analysis from a session file.

        Returns:
            ``(analysis, warning)`` where ``warning`` is ``None`` unless the source file moved or
            changed since the session was saved.
        """
        warning = session.check_source()
        load_result = loader.load(session.source_path, **load_kwargs)
        analysis = cls(
            load_result=load_result,
            pipeline=session.pipeline,
            type_overrides=dict(session.type_overrides),
        )
        analysis.rerun()
        if session.spec is not None:
            try:
                analysis.choose(session.spec)
            except PlotaVizError as exc:
                warning = (warning + "\n\n" if warning else "") + (
                    f"The saved chart could not be restored: {exc.message}"
                )
        return analysis, warning

    # ------------------------------------------------------------------ helpers

    def _require_spec(self, spec: ChartSpec | None) -> ChartSpec:
        """Return the given spec, or the active one, or raise a clear error."""
        chosen = spec or self.spec
        if chosen is None:
            raise PlotaVizError(
                "No chart has been chosen yet.",
                hint="Pick one of the recommendations, or describe what you want to see.",
            )
        return chosen


def _normalized_types(inferred: dict[str, str]) -> dict[str, str]:
    """Re-key inferred roles onto the snake_case names the pipeline will produce.

    ``NormalizeColumnNames`` runs before ``CoerceTypes``, so the type map has to speak in the
    post-rename vocabulary or it silently matches nothing.
    """
    from .preprocess import NormalizeColumnNames

    return {NormalizeColumnNames.normalize(name): role for name, role in inferred.items()}
