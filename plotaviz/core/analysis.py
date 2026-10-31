"""Orchestration facade: load → clean → profile → recommend → render.

The CLI and the GUI both need the same sequence, and duplicating it in two places is how the two
drift apart. :class:`Analysis` owns that sequence and nothing else — it holds state, calls the
single-responsibility modules around it, and stays free of Qt so the CLI can use it directly.

Everything is re-derivable: :meth:`Analysis.rerun` replays the pipeline from the raw frame, which
is what makes type overrides and filter changes cheap.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from . import codegen, exporter, loader, plotter, selector
from .errors import PlotaVizError, PreprocessError
from .preprocess import (
    MISSING_CHOICES,
    OUTLIER_ACTIONS,
    OUTLIER_METHODS,
    CoerceTypes,
    Deduplicate,
    FillMissing,
    FlagOutliers,
    NormalizeColumnNames,
    Pipeline,
    PipelineResult,
    Step,
    default_pipeline,
)
from .profiler import DatasetProfile, profile
from .session import Session
from .spec import MATRIX_CHARTS, ChartSpec

#: Undo depth. Each entry is a serialized step list, so this is cheap even for large data.
MAX_UNDO = 50

#: One undo entry: the serialized pipeline and the type overrides.
_Snapshot = tuple[list[dict[str, Any]], dict[str, str]]

#: The row-level cleaning steps :meth:`Analysis.set_cleaning` manages.
_CLEANING_STEPS = (Deduplicate, FillMissing, FlagOutliers)


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
        inferred_types: Roles inference assigned to each (normalized) column before any user
            override. Overrides are layered on top of this, so removing one restores inference.
        auto_spec: True while :attr:`spec` is the engine's own pick. :meth:`choose` clears it, and
            from then on a re-run (a filter, a type change) keeps the user's chart instead of
            jumping back to the top recommendation.
    """

    load_result: loader.LoadResult
    pipeline: Pipeline = field(default_factory=Pipeline)
    type_overrides: dict[str, str] = field(default_factory=dict)
    result: PipelineResult | None = None
    profile: DatasetProfile | None = None
    recommendations: list[ChartSpec] = field(default_factory=list)
    spec: ChartSpec | None = None
    inferred_types: dict[str, str] = field(default_factory=dict)
    auto_spec: bool = True
    load_options: dict[str, Any] = field(default_factory=dict)
    _undo: list[_Snapshot] = field(default_factory=list, repr=False)
    _redo: list[_Snapshot] = field(default_factory=list, repr=False)

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        missing_strategy: str = "keep",
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
        analysis = cls(load_result=load_result, load_options=dict(load_kwargs.get("options") or {}))

        # Profile the raw frame first so the pipeline knows which columns need coercing.
        analysis.inferred_types = _infer_types(load_result)

        analysis.pipeline = default_pipeline(
            dict(analysis.inferred_types),
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
        result = self.pipeline.run(self.raw)
        if result.df.empty and self.pipeline.filters:
            raise PreprocessError(
                "No rows match the current filters.",
                hint="Loosen or clear the filters to see data again.",
            )
        self.result = result
        self.profile = profile(
            self.result.df,
            overrides=self.type_overrides,
            total_rows=self.load_result.total_rows,
        )
        try:
            self.recommendations = selector.ChartSelector().recommend(self.profile)
        except PlotaVizError:
            self.recommendations = []
        self._reconcile_spec()
        return self.result

    def _reconcile_spec(self) -> None:
        """Decide what the active chart is after a replay.

        The engine's own pick follows the new ranking. A chart the user chose — from the list, the
        query bar, or a session — stays put as long as it still fits the data, and only falls back
        to the top recommendation when a column it needs has gone.
        """
        if self.spec is not None and not self.auto_spec:
            try:
                self.spec.validate(list(self.df.columns))
            except PlotaVizError:
                pass
            else:
                # A matrix chart's columns were pinned from the profile, not picked by the user;
                # follow type changes so a newly numeric column joins and a demoted one leaves.
                if self.spec.chart in MATRIX_CHARTS and self.profile is not None:
                    self.spec = self.spec.with_options(columns=self.profile.numeric)
                return
        self.spec = self.recommendations[0] if self.recommendations else None
        self.auto_spec = True

    def set_type_override(self, column: str, role: str) -> PipelineResult:
        """Correct one column's role and replay everything downstream."""
        return self.set_type_overrides({**self.type_overrides, column: role})

    def set_type_overrides(self, overrides: dict[str, str]) -> PipelineResult:
        """Replace every type correction at once and replay. Undoable.

        The coercion step is rebuilt from the inferred roles plus ``overrides``, so dropping an
        override — "Reset to detected" — really does undo its coercion instead of leaving the
        column converted.
        """

        def change() -> None:
            self.type_overrides = dict(overrides)
            types = {**self.inferred_types, **self.type_overrides}
            coerce = next((s for s in self.pipeline if isinstance(s, CoerceTypes)), None)
            if coerce is not None:
                coerce.types = types
            elif types:
                self.pipeline.steps.insert(_after_rename(self.pipeline), CoerceTypes(types))

        return self._transaction(change)

    def set_filters(self, filters: list[Step]) -> PipelineResult:
        """Replace the active filters and replay. Undoable.

        If the new filters fail — a typo in the query bar is the usual cause — everything is
        restored before the error propagates. Otherwise the broken step would stay in the pipeline
        and every later replay, and the saved session, would inherit it.
        """
        return self._transaction(lambda: self.pipeline.replace_filters(filters))

    def set_cleaning(
        self,
        *,
        missing: str = "keep",
        outliers: str = "off",
        outlier_method: str = "iqr",
        outlier_threshold: float = 1.5,
        dedupe: bool = True,
    ) -> PipelineResult:
        """Replace the row-level cleaning steps and replay. Undoable.

        Renaming, type coercion, filters, and any other step stay exactly where they are; only
        duplicate removal, missing-value handling, and outlier handling are swapped, and they are
        put back in their canonical place right after type coercion.

        Args:
            missing: One of :data:`~plotaviz.core.preprocess.MISSING_CHOICES`.
            outliers: One of :data:`~plotaviz.core.preprocess.OUTLIER_ACTIONS`.
            outlier_method: ``"iqr"`` or ``"zscore"``.
            outlier_threshold: IQR multiplier or z-score cutoff.
            dedupe: Remove fully duplicated rows.

        Raises:
            PreprocessError: If a choice is unknown, or the new pipeline fails.
        """
        if missing not in MISSING_CHOICES:
            raise PreprocessError(f"Unknown missing-value choice {missing!r}.")
        if outliers not in OUTLIER_ACTIONS:
            raise PreprocessError(f"Unknown outlier action {outliers!r}.")
        if outlier_method not in OUTLIER_METHODS:
            raise PreprocessError(f"Unknown outlier method {outlier_method!r}.")

        cleaning: list[Step] = []
        if dedupe:
            cleaning.append(Deduplicate())
        if missing != "keep":
            cleaning.append(FillMissing(strategy=missing))
        if outliers != "off":
            cleaning.append(
                FlagOutliers(
                    method=outlier_method, threshold=outlier_threshold, drop=outliers == "drop"
                )
            )

        def change() -> None:
            kept = [s for s in self.pipeline.steps if not isinstance(s, _CLEANING_STEPS)]
            at = 0
            for index, step in enumerate(kept):
                if isinstance(step, (NormalizeColumnNames, CoerceTypes)):
                    at = index + 1
            self.pipeline.steps = kept[:at] + cleaning + kept[at:]

        return self._transaction(change)

    def cleaning(self) -> dict[str, Any]:
        """The current row-level cleaning settings, in :meth:`set_cleaning`'s vocabulary."""
        fill = next((s for s in self.pipeline if isinstance(s, FillMissing)), None)
        flag = next((s for s in self.pipeline if isinstance(s, FlagOutliers)), None)
        return {
            "missing": fill.strategy if fill else "keep",
            "outliers": ("drop" if flag.drop else "flag") if flag else "off",
            "outlier_method": flag.method if flag else "iqr",
            "outlier_threshold": flag.threshold if flag else 1.5,
            "dedupe": any(isinstance(s, Deduplicate) for s in self.pipeline),
        }

    def remove_step(self, index: int) -> PipelineResult:
        """Delete one step from the pipeline and replay. Undoable.

        Raises:
            PreprocessError: For the column-renaming step — every other step, filter, chart, and
                type override refers to columns by the names it produces.
        """
        if not 0 <= index < len(self.pipeline.steps):
            raise PreprocessError(f"There is no step {index + 1} to remove.")
        if isinstance(self.pipeline.steps[index], NormalizeColumnNames):
            raise PreprocessError(
                "The column-renaming step cannot be removed.",
                hint="Every other step, filter, and chart refers to the names it produces.",
            )

        def change() -> None:
            removed = self.pipeline.remove(index)
            if isinstance(removed, CoerceTypes):
                self.type_overrides = {}

        return self._transaction(change)

    # ------------------------------------------------------------------ undo / redo

    @property
    def can_undo(self) -> bool:
        """Whether there is a change to undo."""
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        """Whether there is an undone change to redo."""
        return bool(self._redo)

    def undo(self) -> PipelineResult | None:
        """Revert the last pipeline change (filters, types, cleaning, removed step).

        Returns ``None`` when there is nothing to undo.
        """
        if not self._undo:
            return None
        target = self._undo.pop()
        self._redo.append(self._snapshot())
        try:
            return self._restore(target)
        except PlotaVizError:
            self._redo.pop()
            self._undo.append(target)
            raise

    def redo(self) -> PipelineResult | None:
        """Re-apply the last undone change. Returns ``None`` when there is nothing to redo."""
        if not self._redo:
            return None
        target = self._redo.pop()
        self._undo.append(self._snapshot())
        try:
            return self._restore(target)
        except PlotaVizError:
            self._undo.pop()
            self._redo.append(target)
            raise

    def _snapshot(self) -> _Snapshot:
        """The undoable state: the recipe and the type corrections, both serialized."""
        return (self.pipeline.to_list(), dict(self.type_overrides))

    def _restore(self, snapshot: _Snapshot) -> PipelineResult:
        """Replay a snapshot, putting the current state back if the replay fails."""
        steps, overrides = snapshot

        def change() -> None:
            self.pipeline = Pipeline.from_list(steps)
            self.type_overrides = dict(overrides)

        return self._replay_with_rollback(change)

    def _transaction(self, change: Callable[[], Any]) -> PipelineResult:
        """Apply an undoable change: checkpoint, change, replay — or roll everything back."""
        checkpoint = self._snapshot()
        result = self._replay_with_rollback(change)
        self._undo.append(checkpoint)
        del self._undo[:-MAX_UNDO]
        self._redo.clear()
        return result

    def _replay_with_rollback(self, change: Callable[[], Any]) -> PipelineResult:
        """Run ``change`` then :meth:`rerun`; on failure restore every piece of state."""
        saved = (
            self.pipeline,
            dict(self.type_overrides),
            self.result,
            self.profile,
            self.recommendations,
            self.spec,
            self.auto_spec,
        )
        steps_before = [s.to_dict() for s in self.pipeline.steps]
        try:
            change()
            return self.rerun()
        except PlotaVizError:
            (
                self.pipeline,
                self.type_overrides,
                self.result,
                self.profile,
                self.recommendations,
                self.spec,
                self.auto_spec,
            ) = saved
            # Steps are mutable objects; a change may have edited one in place (type coercion).
            self.pipeline = Pipeline.from_list(steps_before)
            raise

    # ------------------------------------------------------------------ charting

    def choose(self, spec: ChartSpec) -> ChartSpec:
        """Make ``spec`` the active chart after validating it against the real schema.

        Matrix charts get their column list pinned here, from the profile, so identifier columns
        stay out of the correlation matrix and the generated script plots the same columns.
        """
        spec.validate(list(self.df.columns))
        if spec.chart in MATRIX_CHARTS and "columns" not in spec.options and self.profile:
            spec = spec.with_options(columns=self.profile.numeric)
        self.spec = spec
        self.auto_spec = False
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
            read_options=self.load_result.read_options,
        )

    def to_session(self, view: dict[str, Any] | None = None) -> Session:
        """Capture the current state as a savable session."""
        from .. import __version__

        return Session(
            source_path=str(self.load_result.path),
            source_hash=self.load_result.file_hash,
            load_options=dict(self.load_options),
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
        if session.current_hash:
            load_kwargs.setdefault("known_hash", session.current_hash)
        if session.load_options:
            load_kwargs.setdefault("options", dict(session.load_options))
        load_result = loader.load(session.source_path, **load_kwargs)
        analysis = cls(
            load_result=load_result,
            pipeline=session.pipeline,
            type_overrides=dict(session.type_overrides),
            inferred_types=_infer_types(load_result),
            load_options=dict(session.load_options),
        )
        try:
            analysis.rerun()
        except PreprocessError as exc:
            # The data changed underneath the session and its filters no longer apply (or match
            # nothing). Open without them rather than refusing to open at all — the user cannot
            # loosen a filter in a session they cannot open.
            if not analysis.pipeline.filters:
                raise
            dropped = "; ".join(repr(f) for f in analysis.pipeline.filters)
            analysis.pipeline.replace_filters([])
            analysis.rerun()
            warning = (warning + "\n\n" if warning else "") + (
                f"The saved filters could not be applied to the current data ({exc.message}), "
                f"so the session was opened without them: {dropped}"
            )
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


def _infer_types(load_result: loader.LoadResult) -> dict[str, str]:
    """Inferred roles for the raw frame, keyed by the names the pipeline will produce."""
    raw_profile = profile(load_result.df, total_rows=load_result.total_rows)
    return _normalized_types({name: prof.role for name, prof in raw_profile.columns.items()})


def _after_rename(pipeline: Pipeline) -> int:
    """Index just after the column-renaming step, where type coercion belongs."""
    for index, step in enumerate(pipeline):
        if isinstance(step, NormalizeColumnNames):
            return index + 1
    return 0


def _normalized_types(inferred: dict[str, str]) -> dict[str, str]:
    """Re-key inferred roles onto the snake_case names the pipeline will produce.

    ``NormalizeColumnNames`` runs before ``CoerceTypes``, so the type map has to speak in the
    post-rename vocabulary or it silently matches nothing.
    """
    return {NormalizeColumnNames.normalize(name): role for name, role in inferred.items()}
