"""The preprocessing pipeline — an ordered, replayable list of steps.

This module is deliberately not a set of functions that mutate a dataframe. Every transformation
is a :class:`Step` object that can be serialized, replayed, reordered, removed, and rendered as
source code. One design buys four features:

* **Undo/redo** — drop the last step and re-run from the original frame.
* **Session replay** — a ``.pviz`` file stores the step list, not the cleaned data.
* **Code generation** — each step knows how to write itself as pandas source.
* **Cheap type overrides** — changing an inferred type re-runs a short list, not a bespoke path.

Steps never mutate their input. Each returns a new frame plus a :class:`StepReport` describing
what changed, and those reports are what the cleaning report in the UI is built from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, cast

import numpy as np
import pandas as pd

from .errors import PreprocessError
from .profiler import BOOLEAN, CATEGORICAL, DATETIME, NUMERIC, TEXT

#: Missing-value strategies accepted by :class:`FillMissing`.
FILL_STRATEGIES: tuple[str, ...] = ("drop", "mean", "median", "mode", "ffill", "bfill", "zero")

#: Outlier detection methods accepted by :class:`FlagOutliers`.
OUTLIER_METHODS: tuple[str, ...] = ("iqr", "zscore")

#: Column appended by :class:`FlagOutliers`. Flagging is non-destructive by default.
OUTLIER_FLAG_SUFFIX = "__outlier"


@dataclass
class StepReport:
    """What one step actually did.

    Attributes:
        step: The step's ``kind``.
        summary: One-line, user-facing description of the change.
        rows_before: Row count entering the step.
        rows_after: Row count leaving it.
        details: Structured extras (per-column counts, chosen fill values).
    """

    step: str
    summary: str
    rows_before: int = 0
    rows_after: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def rows_removed(self) -> int:
        """Rows dropped by this step (never negative)."""
        return max(0, self.rows_before - self.rows_after)


class Step:
    """Base class for a replayable preprocessing step.

    Subclasses implement :meth:`apply` and :meth:`to_code`, declare a unique :attr:`kind`, and
    are registered automatically for deserialization.
    """

    #: Unique identifier used in session files and the registry.
    kind: ClassVar[str] = "step"

    #: Populated by ``__init_subclass__``; maps ``kind`` to the class.
    registry: ClassVar[dict[str, type[Step]]] = {}

    #: Human-readable label for the UI's step list.
    label: str = "Step"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.kind != "step":
            Step.registry[cls.kind] = cls

    # ------------------------------------------------------------------ contract

    def apply(self, df: pd.DataFrame) -> tuple[pd.DataFrame, StepReport]:
        """Return a new frame plus a report. Must not mutate ``df``."""
        raise NotImplementedError

    def to_code(self) -> list[str]:
        """Return pandas source lines reproducing this step on a variable named ``df``."""
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """Serialize for a session file."""
        payload = {k: v for k, v in vars(self).items() if not k.startswith("_")}
        return {"kind": self.kind, **payload}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Step:
        """Rebuild a step from its serialized form.

        Raises:
            PreprocessError: If ``kind`` is missing or unknown (e.g. a newer session file).
        """
        data = dict(data)
        kind = data.pop("kind", None)
        if kind is None:
            raise PreprocessError("A preprocessing step is missing its 'kind' field.")
        cls = Step.registry.get(str(kind))
        if cls is None:
            raise PreprocessError(
                f"Unknown preprocessing step {kind!r}.",
                hint="This session may have been saved by a newer version of PlotaViz.",
            )
        try:
            return cls(**data)
        except TypeError as exc:
            raise PreprocessError(
                f"Preprocessing step {kind!r} has parameters this version does not understand.",
                hint=str(exc),
            ) from exc

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in vars(self).items() if not k.startswith("_"))
        return f"{type(self).__name__}({args})"


# ---------------------------------------------------------------------------- steps


class NormalizeColumnNames(Step):
    """Strip whitespace and convert column names to ``snake_case``.

    Runs first in the default pipeline so every later step, filter, and generated script can
    refer to columns by a predictable name.
    """

    kind = "normalize_columns"
    label = "Normalize column names"

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    @staticmethod
    def normalize(name: str) -> str:
        """Convert one column name to snake_case, preserving digits and collapsing separators."""
        text = str(name).strip()
        text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
        text = re.sub(r"[^0-9a-zA-Z]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_").lower()
        return text or "column"

    def apply(self, df: pd.DataFrame) -> tuple[pd.DataFrame, StepReport]:
        if not self.enabled:
            return df, StepReport(self.kind, "Column names left as they are.", len(df), len(df))

        mapping: dict[str, str] = {}
        used: set[str] = set()
        for col in df.columns:
            new = self.normalize(col)
            if new in used:  # de-duplicate collisions deterministically
                suffix = 2
                while f"{new}_{suffix}" in used:
                    suffix += 1
                new = f"{new}_{suffix}"
            used.add(new)
            mapping[str(col)] = new

        renamed = {k: v for k, v in mapping.items() if k != v}
        out = df.rename(columns=mapping)
        summary = (
            f"Renamed {len(renamed)} column(s) to snake_case."
            if renamed
            else "Column names were already clean."
        )
        return out, StepReport(self.kind, summary, len(df), len(out), {"renamed": renamed})

    def to_code(self) -> list[str]:
        if not self.enabled:
            return []
        return [
            "# Normalize column names to snake_case",
            "import re",
            "def _snake(name):",
            "    text = re.sub(r'([a-z0-9])([A-Z])', r'\\1_\\2', str(name).strip())",
            "    text = re.sub(r'[^0-9a-zA-Z]+', '_', text)",
            "    return re.sub(r'_+', '_', text).strip('_').lower() or 'column'",
            "df = df.rename(columns={c: _snake(c) for c in df.columns})",
        ]


class CoerceTypes(Step):
    """Apply column type decisions — inferred roles and any user overrides.

    Args:
        types: ``{column: role}`` where role is one of the profiler's roles. Columns not listed
            are left alone.
    """

    kind = "coerce_types"
    label = "Apply column types"

    def __init__(self, types: dict[str, str] | None = None) -> None:
        self.types = dict(types or {})

    def apply(self, df: pd.DataFrame) -> tuple[pd.DataFrame, StepReport]:
        out = df.copy()
        applied: dict[str, str] = {}
        failed: dict[str, str] = {}

        for col, role in self.types.items():
            if col not in out.columns:
                continue
            try:
                if role == DATETIME:
                    out[col] = pd.to_datetime(out[col], errors="coerce", format="mixed")
                elif role == NUMERIC:
                    out[col] = pd.to_numeric(out[col], errors="coerce")
                elif role == BOOLEAN:
                    out[col] = _to_boolean(out[col])
                elif role in (CATEGORICAL, TEXT):
                    out[col] = out[col].astype("object")
                else:
                    continue
            except Exception as exc:
                failed[col] = str(exc)
                continue
            applied[col] = role

        parts = []
        if applied:
            parts.append(f"Applied types to {len(applied)} column(s).")
        if failed:
            parts.append(f"{len(failed)} column(s) could not be converted and were left as-is.")
        summary = " ".join(parts) or "No type changes were needed."
        return out, StepReport(
            self.kind, summary, len(df), len(out), {"applied": applied, "failed": failed}
        )

    def to_code(self) -> list[str]:
        lines: list[str] = []
        for col, role in self.types.items():
            if role == DATETIME:
                lines.append(f"df[{col!r}] = pd.to_datetime(df[{col!r}], errors='coerce')")
            elif role == NUMERIC:
                lines.append(f"df[{col!r}] = pd.to_numeric(df[{col!r}], errors='coerce')")
            elif role == BOOLEAN:
                lines.append(f"df[{col!r}] = df[{col!r}].astype('boolean')")
        return ["# Column types", *lines] if lines else []


class FillMissing(Step):
    """Handle missing values with an explicit, reported strategy.

    Args:
        strategy: One of :data:`FILL_STRATEGIES`. ``"drop"`` removes rows with any null in the
            targeted columns; the rest impute.
        columns: Columns to act on. ``None`` means every column the strategy makes sense for.
    """

    kind = "fill_missing"
    label = "Handle missing values"

    def __init__(self, strategy: str = "median", columns: list[str] | None = None) -> None:
        if strategy not in FILL_STRATEGIES:
            raise PreprocessError(
                f"Unknown missing-value strategy {strategy!r}.",
                hint=f"Choose one of: {', '.join(FILL_STRATEGIES)}.",
            )
        self.strategy = strategy
        self.columns = list(columns) if columns else None

    def apply(self, df: pd.DataFrame) -> tuple[pd.DataFrame, StepReport]:
        out = df.copy()
        targets = [c for c in (self.columns or out.columns) if c in out.columns]
        before_missing = {c: int(out[c].isna().sum()) for c in targets}
        touched = {c: n for c, n in before_missing.items() if n > 0}

        if not touched:
            return out, StepReport(
                self.kind, "No missing values to handle.", len(df), len(out), {"filled": {}}
            )

        fills: dict[str, Any] = {}
        if self.strategy == "drop":
            out = out.dropna(subset=list(touched))
        else:
            for col in touched:
                series = out[col]
                value: Any = None
                if self.strategy == "mean" and pd.api.types.is_numeric_dtype(series):
                    value = series.mean()
                elif self.strategy == "median" and pd.api.types.is_numeric_dtype(series):
                    value = series.median()
                elif self.strategy == "zero" and pd.api.types.is_numeric_dtype(series):
                    value = 0
                elif self.strategy == "mode":
                    modes = series.mode(dropna=True)
                    value = modes.iloc[0] if not modes.empty else None
                elif self.strategy == "ffill":
                    out[col] = series.ffill()
                    fills[col] = "forward-filled"
                    continue
                elif self.strategy == "bfill":
                    out[col] = series.bfill()
                    fills[col] = "back-filled"
                    continue

                if value is None or (isinstance(value, float) and np.isnan(value)):
                    # A non-numeric column under a numeric strategy: leave it, and say so.
                    continue
                out[col] = series.fillna(value)
                fills[col] = value

        summary = (
            f"Dropped rows with missing values in {len(touched)} column(s)."
            if self.strategy == "drop"
            else f"Filled missing values in {len(fills)} column(s) using the "
            f"{self.strategy} strategy."
        )
        return out, StepReport(
            self.kind,
            summary,
            len(df),
            len(out),
            {"strategy": self.strategy, "missing_before": touched, "filled": _stringify(fills)},
        )

    def to_code(self) -> list[str]:
        cols = f"{self.columns!r}" if self.columns else "None"
        if self.strategy == "drop":
            subset = f"subset={cols}" if self.columns else ""
            return ["# Missing values: drop", f"df = df.dropna({subset})"]
        if self.strategy in {"ffill", "bfill"}:
            method = "ffill" if self.strategy == "ffill" else "bfill"
            return [f"# Missing values: {method}", f"df = df.{method}()"]
        if self.strategy == "zero":
            return [
                "# Missing values: fill numeric columns with 0",
                "_num = df.select_dtypes('number').columns",
                "df[_num] = df[_num].fillna(0)",
            ]
        if self.strategy == "mode":
            return [
                "# Missing values: fill with the most frequent value",
                "for _c in df.columns:",
                "    _modes = df[_c].mode(dropna=True)",
                "    if not _modes.empty:",
                "        df[_c] = df[_c].fillna(_modes.iloc[0])",
            ]
        return [
            f"# Missing values: fill numeric columns with the column {self.strategy}",
            "_num = df.select_dtypes('number').columns",
            f"df[_num] = df[_num].fillna(df[_num].{self.strategy}())",
        ]


class FlagOutliers(Step):
    """Flag outliers without removing them.

    Non-destructive by default: a boolean ``<column>__outlier`` companion column is added and the
    count is reported. Removing data silently is how people end up with charts that lie.

    Args:
        method: ``"iqr"`` (Tukey fences) or ``"zscore"``.
        threshold: IQR multiplier, or the absolute z-score cutoff.
        columns: Numeric columns to check. ``None`` checks all numeric columns.
        drop: When True, drop flagged rows instead of marking them.
    """

    kind = "flag_outliers"
    label = "Flag outliers"

    def __init__(
        self,
        method: str = "iqr",
        threshold: float = 1.5,
        columns: list[str] | None = None,
        drop: bool = False,
    ) -> None:
        if method not in OUTLIER_METHODS:
            raise PreprocessError(
                f"Unknown outlier method {method!r}.",
                hint=f"Choose one of: {', '.join(OUTLIER_METHODS)}.",
            )
        self.method = method
        self.threshold = float(threshold)
        self.columns = list(columns) if columns else None
        self.drop = bool(drop)

    def _mask(self, series: pd.Series) -> pd.Series:
        """Boolean mask of outliers in one numeric series.

        Both methods have a degenerate case: a column where most values are identical has an IQR
        (and a standard deviation) of zero, and the naive fence then flags nothing — including a
        value a thousand times larger than the rest, which is exactly the case a user cares about.
        When that happens both fall back to the **modified z-score** built on the median absolute
        deviation, with the conventional 3.5 cutoff.
        """
        values = pd.to_numeric(series, errors="coerce")

        if self.method == "iqr":
            q1, q3 = values.quantile(0.25), values.quantile(0.75)
            iqr = q3 - q1
            if pd.isna(iqr):
                return pd.Series(False, index=series.index)
            if iqr != 0:
                low, high = q1 - self.threshold * iqr, q3 + self.threshold * iqr
                return ((values < low) | (values > high)).fillna(False)
            return self._mad_mask(values)

        std = values.std()
        if pd.isna(std):
            return pd.Series(False, index=series.index)
        if std != 0:
            return (((values - values.mean()).abs() / std) > self.threshold).fillna(False)
        return self._mad_mask(values)

    @staticmethod
    def _mad_mask(values: pd.Series) -> pd.Series:
        """Modified z-score fallback for distributions with no usable spread."""
        median = values.median()
        deviation = (values - median).abs()
        mad = deviation.median()
        if pd.isna(mad):
            return pd.Series(False, index=values.index)
        if mad == 0:
            # Every value but a handful is identical; anything that differs is the anomaly.
            return (values != median).fillna(False)
        return ((0.6745 * deviation / mad) > 3.5).fillna(False)

    def apply(self, df: pd.DataFrame) -> tuple[pd.DataFrame, StepReport]:
        out = df.copy()
        targets = self.columns or list(out.select_dtypes(include="number").columns)
        targets = [c for c in targets if c in out.columns]

        counts: dict[str, int] = {}
        any_mask = pd.Series(False, index=out.index)
        for col in targets:
            mask = self._mask(out[col]).fillna(False)
            n = int(mask.sum())
            if n == 0:
                continue
            counts[col] = n
            any_mask |= mask
            if not self.drop:
                out[f"{col}{OUTLIER_FLAG_SUFFIX}"] = mask

        if self.drop and counts:
            out = out.loc[~any_mask]

        total = int(any_mask.sum())
        if not counts:
            summary = "No outliers detected."
        elif self.drop:
            summary = f"Removed {total:,} outlier row(s) across {len(counts)} column(s)."
        else:
            summary = (
                f"Flagged {total:,} outlier row(s) across {len(counts)} column(s). "
                "The rows are kept; flag columns were added."
            )
        return out, StepReport(
            self.kind,
            summary,
            len(df),
            len(out),
            {"method": self.method, "threshold": self.threshold, "per_column": counts},
        )

    def to_code(self) -> list[str]:
        cols = repr(self.columns) if self.columns else "list(df.select_dtypes('number').columns)"
        # The modified z-score fallback is emitted too, so the script flags the same rows the app
        # did on columns whose spread is zero.
        helper = [
            "def _mad_outliers(s):",
            '    """Modified z-score, used when a column has no usable spread."""',
            "    _med = s.median()",
            "    _dev = (s - _med).abs()",
            "    _mad = _dev.median()",
            "    if _mad == 0:",
            "        return (s != _med).fillna(False)",
            "    return ((0.6745 * _dev / _mad) > 3.5).fillna(False)",
            "",
        ]
        if self.method == "iqr":
            body = [
                f"for _c in {cols}:",
                "    _q1, _q3 = df[_c].quantile(0.25), df[_c].quantile(0.75)",
                "    _iqr = _q3 - _q1",
                "    if _iqr:",
                f"        _mask = ((df[_c] < _q1 - {self.threshold} * _iqr) | "
                f"(df[_c] > _q3 + {self.threshold} * _iqr)).fillna(False)",
                "    else:",
                "        _mask = _mad_outliers(df[_c])",
            ]
        else:
            body = [
                f"for _c in {cols}:",
                "    _std = df[_c].std()",
                "    if _std:",
                f"        _mask = (((df[_c] - df[_c].mean()).abs() / _std) > "
                f"{self.threshold}).fillna(False)",
                "    else:",
                "        _mask = _mad_outliers(df[_c])",
            ]
        # Only touch the frame when something was actually flagged — matching the app, which
        # does not add an all-False flag column for a column with no outliers.
        tail = (
            ["    if _mask.any():", "        df = df.loc[~_mask]"]
            if self.drop
            else ["    if _mask.any():", f"        df[_c + {OUTLIER_FLAG_SUFFIX!r}] = _mask"]
        )
        return [f"# Outliers ({self.method})", *helper, *body, *tail]


class Deduplicate(Step):
    """Remove duplicate rows and report how many went.

    Args:
        subset: Columns that define a duplicate. ``None`` means the whole row.
        keep: Which duplicate to keep — ``"first"``, ``"last"``, or ``False`` to drop all.
    """

    kind = "deduplicate"
    label = "Remove duplicate rows"

    def __init__(self, subset: list[str] | None = None, keep: str = "first") -> None:
        self.subset = list(subset) if subset else None
        self.keep = keep

    def apply(self, df: pd.DataFrame) -> tuple[pd.DataFrame, StepReport]:
        subset = [c for c in self.subset if c in df.columns] if self.subset else None
        keep = cast(Literal["first", "last", False], self.keep)
        out = df.drop_duplicates(subset=subset, keep=keep)
        removed = len(df) - len(out)
        summary = (
            f"Removed {removed:,} duplicate row(s)." if removed else "No duplicate rows found."
        )
        return out, StepReport(self.kind, summary, len(df), len(out), {"removed": removed})

    def to_code(self) -> list[str]:
        subset = f"subset={self.subset!r}, " if self.subset else ""
        return ["# Duplicates", f"df = df.drop_duplicates({subset}keep={self.keep!r})"]


class QueryFilter(Step):
    """Filter rows with a pandas ``query`` expression — the free-text filter bar.

    Args:
        expression: A pandas query string, e.g. ``"revenue > 1000 and region == 'EMEA'"``.
    """

    kind = "query_filter"
    label = "Query filter"

    def __init__(self, expression: str) -> None:
        self.expression = str(expression)

    def apply(self, df: pd.DataFrame) -> tuple[pd.DataFrame, StepReport]:
        expr = self.expression.strip()
        if not expr:
            return df, StepReport(self.kind, "Empty filter — nothing applied.", len(df), len(df))
        try:
            out = df.query(expr)
        except Exception as exc:
            raise PreprocessError(
                f"The filter {expr!r} could not be applied.",
                hint=(
                    "Use pandas query syntax, e.g. revenue > 1000 and region == 'EMEA'. "
                    f"Underlying error: {exc}"
                ),
            ) from exc
        return out, StepReport(
            self.kind,
            f"Filter {expr!r} kept {len(out):,} of {len(df):,} rows.",
            len(df),
            len(out),
            {"expression": expr},
        )

    def to_code(self) -> list[str]:
        return ["# Filter", f"df = df.query({self.expression!r})"]


class ColumnFilter(Step):
    """Filter one column with a structured predicate — the per-column filter widgets.

    Args:
        column: Column to filter on.
        op: ``"between"`` (numeric range or date range) or ``"in"`` (multi-select).
        value: ``[low, high]`` for ``between``, or a list of allowed values for ``in``.
    """

    kind = "column_filter"
    label = "Column filter"

    def __init__(self, column: str, op: str, value: Any) -> None:
        if op not in {"between", "in"}:
            raise PreprocessError(f"Unknown filter operation {op!r}.")
        self.column = str(column)
        self.op = op
        self.value = value

    def apply(self, df: pd.DataFrame) -> tuple[pd.DataFrame, StepReport]:
        if self.column not in df.columns:
            return df, StepReport(
                self.kind,
                f"Column {self.column!r} is not present; filter skipped.",
                len(df),
                len(df),
            )
        series = df[self.column]
        if self.op == "between":
            low, high = self.value
            if pd.api.types.is_datetime64_any_dtype(series):
                low, high = pd.to_datetime(low), pd.to_datetime(high)
            mask = series.between(low, high)
            described = f"{self.column} between {low} and {high}"
        else:
            allowed = list(self.value)
            mask = series.isin(allowed)
            shown = ", ".join(map(str, allowed[:4]))
            described = f"{self.column} in [{shown}{' …' if len(allowed) > 4 else ''}]"

        out = df.loc[mask.fillna(False)]
        return out, StepReport(
            self.kind,
            f"Filter {described} kept {len(out):,} of {len(df):,} rows.",
            len(df),
            len(out),
            {"column": self.column, "op": self.op, "value": _stringify(self.value)},
        )

    def to_code(self) -> list[str]:
        if self.op == "between":
            low, high = self.value
            return [
                "# Column filter",
                f"df = df[df[{self.column!r}].between({low!r}, {high!r})]",
            ]
        return ["# Column filter", f"df = df[df[{self.column!r}].isin({list(self.value)!r})]"]


# ---------------------------------------------------------------------------- pipeline


@dataclass
class PipelineResult:
    """The output of running a pipeline.

    Attributes:
        df: The cleaned frame.
        reports: One :class:`StepReport` per step, in execution order.
        rows_before: Rows in the input frame.
        rows_after: Rows in :attr:`df`.
    """

    df: pd.DataFrame
    reports: list[StepReport]
    rows_before: int
    rows_after: int

    @property
    def rows_removed(self) -> int:
        """Total rows removed across the whole pipeline."""
        return max(0, self.rows_before - self.rows_after)

    def cleaning_report(self) -> str:
        """Render the reports as the plain-text cleaning summary shown after loading."""
        lines = [
            f"Rows: {self.rows_before:,} → {self.rows_after:,}"
            + (f"  ({self.rows_removed:,} removed)" if self.rows_removed else ""),
            "",
        ]
        for i, report in enumerate(self.reports, start=1):
            lines.append(f"{i}. {report.summary}")
        return "\n".join(lines)


class Pipeline:
    """An ordered list of steps applied to a dataframe.

    The pipeline holds no data — it is a recipe. Re-running it against the original frame is how
    undo, type overrides, and filter changes all work.

    Args:
        steps: Initial steps, applied in order.
    """

    def __init__(self, steps: list[Step] | None = None) -> None:
        self.steps: list[Step] = list(steps or [])

    # ------------------------------------------------------------------ mutation

    def add(self, step: Step) -> Pipeline:
        """Append a step and return self, so calls chain."""
        self.steps.append(step)
        return self

    def remove(self, index: int) -> Step:
        """Remove and return the step at ``index``."""
        return self.steps.pop(index)

    def replace_filters(self, filters: list[Step]) -> Pipeline:
        """Swap every filter step for a new set, leaving cleaning steps untouched.

        Filters change on every widget interaction; cleaning does not. Keeping them separable is
        what makes live filtering cheap.
        """
        self.steps = [s for s in self.steps if not isinstance(s, (QueryFilter, ColumnFilter))]
        self.steps.extend(filters)
        return self

    @property
    def filters(self) -> list[Step]:
        """Just the filter steps, in order."""
        return [s for s in self.steps if isinstance(s, (QueryFilter, ColumnFilter))]

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)

    # ------------------------------------------------------------------ execution

    def run(self, df: pd.DataFrame) -> PipelineResult:
        """Apply every step in order to a copy of ``df``.

        Raises:
            PreprocessError: If a step fails. The message names the step so the user knows which
                one to fix.
        """
        rows_before = len(df)
        current = df
        reports: list[StepReport] = []
        for step in self.steps:
            try:
                current, report = step.apply(current)
            except PreprocessError:
                raise
            except Exception as exc:
                raise PreprocessError(f"The step {step.label!r} failed.", hint=str(exc)) from exc
            reports.append(report)
        return PipelineResult(current, reports, rows_before, len(current))

    # ------------------------------------------------------------------ serialization

    def to_list(self) -> list[dict[str, Any]]:
        """Serialize every step for a session file."""
        return [step.to_dict() for step in self.steps]

    @classmethod
    def from_list(cls, data: list[dict[str, Any]]) -> Pipeline:
        """Rebuild a pipeline from serialized steps."""
        return cls([Step.from_dict(item) for item in data])

    def to_code(self) -> list[str]:
        """Render the whole pipeline as pandas source lines."""
        lines: list[str] = []
        for step in self.steps:
            code = step.to_code()
            if code:
                lines.extend(code)
                lines.append("")
        return lines


def default_pipeline(
    types: dict[str, str] | None = None,
    *,
    missing_strategy: str = "median",
    flag_outliers: bool = True,
    deduplicate: bool = True,
) -> Pipeline:
    """Build the standard cleaning pipeline applied right after loading.

    The order matters: names first so everything downstream can address columns predictably,
    then types (which is what user overrides feed into), then row-level cleaning.

    Args:
        types: Column role decisions, including user overrides.
        missing_strategy: One of :data:`FILL_STRATEGIES`.
        flag_outliers: Whether to add outlier flag columns.
        deduplicate: Whether to drop duplicate rows.
    """
    pipeline = Pipeline([NormalizeColumnNames()])
    if types:
        pipeline.add(CoerceTypes(types))
    if deduplicate:
        pipeline.add(Deduplicate())
    pipeline.add(FillMissing(strategy=missing_strategy))
    if flag_outliers:
        pipeline.add(FlagOutliers())
    return pipeline


# ---------------------------------------------------------------------------- helpers


def _to_boolean(series: pd.Series) -> pd.Series:
    """Coerce common truthy/falsey tokens to a nullable boolean dtype."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")
    mapping = {
        "true": True,
        "yes": True,
        "y": True,
        "t": True,
        "1": True,
        "on": True,
        "false": False,
        "no": False,
        "n": False,
        "f": False,
        "0": False,
        "off": False,
    }
    return series.map(lambda v: mapping.get(str(v).strip().lower())).astype("boolean")


def _stringify(value: Any) -> Any:
    """Make report details JSON-safe."""
    if isinstance(value, dict):
        return {str(k): _stringify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_stringify(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)
