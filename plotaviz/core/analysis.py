"""Orchestration facade: load → clean → profile → recommend → render.

The CLI and the GUI both need the same sequence, and duplicating it in two places is how the two
drift apart. :class:`Analysis` owns that sequence and nothing else — it holds state, calls the
single-responsibility modules around it, and stays free of Qt so the CLI can use it directly.

Everything is re-derivable: :meth:`Analysis.rerun` replays the pipeline from the raw frame, which
is what makes type overrides and filter changes cheap.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from . import codegen, exporter, loader, plotter, selector
from .errors import PlotaVizError, PreprocessError, SpecError
from .preprocess import (
    MISSING_CHOICES,
    OUTLIER_ACTIONS,
    OUTLIER_METHODS,
    CoerceTypes,
    ColumnFilter,
    Deduplicate,
    FillMissing,
    FlagOutliers,
    JoinDataset,
    NormalizeColumnNames,
    Pipeline,
    PipelineResult,
    QueryFilter,
    Step,
    default_pipeline,
)
from .profiler import DatasetProfile, profile
from .session import Session
from .spec import MATRIX_CHARTS, ChartSpec

#: Largest cleaned frame kept in memory to make filter changes fast (bytes, shallow).
CLEAN_CACHE_MAX_BYTES = 1_500_000_000

#: Most charts a dashboard holds. Beyond this the tiles are too small to read.
MAX_TILES = 12

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
    dashboard: list[ChartSpec] = field(default_factory=list)
    #: A custom ``rules.yaml`` (``--rules``); ``None`` uses the shipped one.
    rules_path: str | None = None
    _clean_cache: tuple[str, PipelineResult] | None = field(default=None, repr=False)

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
        rules_path = load_kwargs.pop("rules_path", None)
        load_result = loader.load(path, **load_kwargs)
        analysis = cls(
            load_result=load_result,
            load_options=dict(load_kwargs.get("options") or {}),
            rules_path=str(rules_path) if rules_path else None,
        )

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
        result = self._run_pipeline()
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
            rules = selector.load_rules(self.rules_path) if self.rules_path else None
            self.recommendations = selector.ChartSelector(rules).recommend(self.profile)
        except PlotaVizError:
            self.recommendations = []
        self._reconcile_spec()
        return self.result

    def _run_pipeline(self) -> PipelineResult:
        """Run the pipeline, reusing the cleaned frame when only the filters changed.

        Cleaning (type coercion, deduplication, missing values, outliers) is the slow part on
        large data, and a filter edit does not change it. The cleaned frame is cached against the
        serialized cleaning steps, so a filter change replays only the filters. The result is
        identical to a full run; any change to a cleaning step or type misses the cache.
        """
        steps = self.pipeline.steps
        is_filter = [isinstance(s, (QueryFilter, ColumnFilter)) for s in steps]
        first_filter = is_filter.index(True) if True in is_filter else len(steps)
        if any(not f for f in is_filter[first_filter:]):
            # A cleaning step after a filter (possible in a hand-edited session): no shortcut.
            self._clean_cache = None
            return self.pipeline.run(self.raw)

        cleaning, filters = steps[:first_filter], steps[first_filter:]
        key = json.dumps([s.to_dict() for s in cleaning], sort_keys=True, default=str)
        if self._clean_cache is None or self._clean_cache[0] != key:
            self._clean_cache = None  # release the old frame before building the new one
            cleaned = Pipeline(cleaning).run(self.raw)
            # Caching keeps one more full frame alive while filters are applied. Above the
            # limit, memory matters more than filter speed: replay cleaning each time instead.
            if int(cleaned.df.memory_usage(deep=False).sum()) <= CLEAN_CACHE_MAX_BYTES:
                self._clean_cache = (key, cleaned)
        else:
            cleaned = self._clean_cache[1]
        if not filters:
            return cleaned
        filtered = Pipeline(filters).run(cleaned.df)
        return PipelineResult(
            filtered.df,
            [*cleaned.reports, *filtered.reports],
            cleaned.rows_before,
            filtered.rows_after,
        )

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

    def unfiltered_profile(self) -> DatasetProfile | None:
        """Profile of the cleaned data before filters — what filter widgets should span."""
        if not self.pipeline.filters:
            return self.profile
        cleaned = self._clean_cache[1].df if self._clean_cache else None
        if cleaned is None:
            steps = [
                s for s in self.pipeline.steps if not isinstance(s, (QueryFilter, ColumnFilter))
            ]
            cleaned = Pipeline(steps).run(self.raw).df
        return profile(cleaned, overrides=self.type_overrides)

    def join_base(self) -> pd.DataFrame:
        """The frame a new join would apply to: renamed, with earlier joins, before any
        coercion, cleaning, or filtering. The join dialog previews matches against this."""
        return Pipeline(self.pipeline.steps[: self._join_insert_index()]).run(self.raw).df

    def _join_insert_index(self) -> int:
        """Where a new join goes: after renaming and after every earlier join, so a second
        join can use columns the first one added."""
        index = _after_rename(self.pipeline)
        for position, step in enumerate(self.pipeline.steps):
            if isinstance(step, JoinDataset):
                index = max(index, position + 1)
        return index

    def _renames_columns(self) -> bool:
        rename = next((s for s in self.pipeline if isinstance(s, NormalizeColumnNames)), None)
        return bool(rename and rename.enabled)

    def join_dataset(
        self,
        path: str | Path,
        on: list[str],
        *,
        how: str = "left",
        options: dict[str, Any] | None = None,
    ) -> PipelineResult:
        """Bring in columns from another file, matched on ``on``. Undoable.

        The join runs after column renaming (and after earlier joins), before type coercion,
        so the joined columns are typed like the rest; their roles are inferred from the second
        file. Missing-value handling keeps to the main data's columns, so a left join's
        unmatched rows are never dropped or filled with invented lookup values.

        Raises:
            LoadError: If the second file cannot be read.
            PreprocessError: If a key is not in both datasets.
        """
        source = Path(path).expanduser().resolve()
        step = JoinDataset(
            str(source),
            list(on),
            how=how,
            read_options=options,
            normalize=self._renames_columns(),
        )
        other = step.other()  # read (and hash) it now, so a bad file fails before any change

        before = set(self.join_base().columns)
        new_types = {
            (f"{name}{step.suffix}" if name in before else name): prof.role
            for name, prof in profile(other).columns.items()
            if name not in on
        }
        main_columns = self._main_columns()

        def change() -> None:
            self.pipeline.steps.insert(self._join_insert_index(), step)
            for name, role in new_types.items():
                self.inferred_types.setdefault(name, role)
            coerce = next((s for s in self.pipeline if isinstance(s, CoerceTypes)), None)
            if coerce is not None:
                for name, role in new_types.items():
                    coerce.types.setdefault(name, role)
            _restrict_fill(self.pipeline, main_columns)

        return self._transaction(change)

    def _main_columns(self) -> list[str]:
        """The main dataset's own column names (after renaming), without joined columns."""
        rename = next((s for s in self.pipeline if isinstance(s, NormalizeColumnNames)), None)
        head = self.raw.head(0)
        return list(rename.apply(head)[0].columns if rename else head.columns)

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

        main_columns = self._main_columns()

        def change() -> None:
            kept = [s for s in self.pipeline.steps if not isinstance(s, _CLEANING_STEPS)]
            at = 0
            for index, step in enumerate(kept):
                if isinstance(step, (NormalizeColumnNames, CoerceTypes, JoinDataset)):
                    at = index + 1
            self.pipeline.steps = kept[:at] + cleaning + kept[at:]
            _restrict_fill(self.pipeline, main_columns)

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

    # ------------------------------------------------------------------ dashboard

    def pin(self, spec: ChartSpec | None = None) -> int:
        """Add a chart (the active one by default) to the dashboard; return its tile index.

        The tile is an independent copy, so editing the main chart afterwards does not change it.
        Pinning is deliberately outside the undo history: undoing a filter must not unpin charts.

        Raises:
            SpecError: If the dashboard is full or the chart does not fit the current data.
        """
        chosen = self._require_spec(spec)
        if len(self.dashboard) >= MAX_TILES:
            raise SpecError(
                f"The dashboard already has {MAX_TILES} charts.",
                hint="Remove one before pinning another — more than this stops being readable.",
            )
        chosen.validate(list(self.df.columns))
        self.dashboard.append(chosen.copy())
        return len(self.dashboard) - 1

    def unpin(self, index: int) -> ChartSpec:
        """Remove and return one tile."""
        self._check_tile(index)
        return self.dashboard.pop(index)

    def move_tile(self, index: int, to: int) -> None:
        """Move a tile to another position (clamped to the dashboard's bounds)."""
        self._check_tile(index)
        tile = self.dashboard.pop(index)
        self.dashboard.insert(max(0, min(to, len(self.dashboard))), tile)

    def replace_tile(self, index: int, spec: ChartSpec | None = None) -> None:
        """Swap a tile for another chart (the active one by default)."""
        self._check_tile(index)
        chosen = self._require_spec(spec)
        chosen.validate(list(self.df.columns))
        self.dashboard[index] = chosen.copy()

    def tile_status(self) -> list[str | None]:
        """Per tile: ``None`` if it can be drawn on the current data, else why not.

        A tile whose column was renamed away, retyped, or removed by a session restore is kept
        and flagged rather than silently dropped — the user decides whether to remove it.
        """
        columns = list(self.df.columns)
        statuses: list[str | None] = []
        for tile in self.dashboard:
            try:
                tile.validate(columns)
            except PlotaVizError as exc:
                statuses.append(exc.message)
            else:
                statuses.append(None)
        return statuses

    def _check_tile(self, index: int) -> None:
        if not 0 <= index < len(self.dashboard):
            raise SpecError(f"There is no dashboard tile {index + 1}.")

    # ------------------------------------------------------------------ charting

    def choose(self, spec: ChartSpec) -> ChartSpec:
        """Make ``spec`` the active chart after validating it against the real schema.

        Matrix charts get their column list pinned here, from the profile, so identifier columns
        stay out of the correlation matrix and the generated script plots the same columns.
        """
        spec.validate(list(self.df.columns))
        if any(spec is tile for tile in self.dashboard):
            # Never share an object with a dashboard tile: editing the main chart would
            # silently edit the tile too.
            spec = spec.copy()
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

    def generate_dashboard_code(self, *, flavour: str = "matplotlib") -> str:
        """Emit one standalone script reproducing every dashboard tile."""
        from .. import __version__

        return codegen.generate_dashboard(
            list(self.dashboard),
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
            dashboard=[tile.copy() for tile in self.dashboard],
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
        analysis.dashboard = [tile.copy() for tile in session.dashboard]
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


def _restrict_fill(pipeline: Pipeline, main_columns: list[str]) -> None:
    """With a join present, limit missing-value handling to the main data's columns.

    A left join leaves unmatched rows empty in the joined columns. ``drop`` would delete
    exactly those rows (turning the join into an inner join) and imputation would invent lookup
    values for them, so joined columns are left as they are.
    """
    if not any(isinstance(s, JoinDataset) for s in pipeline):
        return
    for step in pipeline:
        if isinstance(step, FillMissing) and step.columns is None:
            step.columns = list(main_columns)


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
