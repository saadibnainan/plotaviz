"""Column typing and dataset statistics.

The profiler answers the question the chart selector needs answered: *what shape is this data?*
It assigns every column a **role** — numeric, categorical, datetime, boolean, or high-cardinality
text — and attaches the statistics the scoring layer uses to rank charts (cardinality, missing
fraction, skew, correlation strength).

Roles are deliberately coarser than dtypes. An integer column of 30,000 unique customer IDs is
numerically typed but is not a *measure*, and charting it as one produces nonsense; the role
system exists to catch exactly that case.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .errors import ProfileError

# Roles ---------------------------------------------------------------------

NUMERIC = "numeric"
CATEGORICAL = "categorical"
DATETIME = "datetime"
BOOLEAN = "boolean"
TEXT = "text"  # high-cardinality free text — never charted directly

ROLES: tuple[str, ...] = (NUMERIC, CATEGORICAL, DATETIME, BOOLEAN, TEXT)

#: A column with at most this many distinct values is categorical even if it is numeric.
LOW_CARDINALITY_MAX = 20

#: Above this ratio of distinct values to rows, a string column is free text, not a category.
TEXT_CARDINALITY_RATIO = 0.5

#: A numeric column whose values are this unique is an identifier, not a measure.
ID_CARDINALITY_RATIO = 0.95

#: Rows sampled for profiling when the frame is bigger than this.
PROFILE_SAMPLE_ROWS = 100_000

#: Suffixes marking a column PlotaViz added itself. These are shown in the preview but are never
#: offered as chart axes — recommending a chart of your own outlier flags is noise.
DERIVED_SUFFIXES = ("__outlier",)

#: Column-name patterns that mark an identifier regardless of dtype.
_ID_NAME_PATTERN = re.compile(r"(^|_)(id|uuid|guid|key|code|index|no|num)$", re.IGNORECASE)

#: Column-name patterns that hint at a date, used to justify a parse attempt.
_DATE_NAME_PATTERN = re.compile(
    r"(date|time|timestamp|datetime|day|month|year|created|updated|_at$)", re.IGNORECASE
)

_BOOLEAN_TOKENS = (
    {"true", "false"},
    {"yes", "no"},
    {"y", "n"},
    {"t", "f"},
    {"0", "1"},
    {"on", "off"},
)


@dataclass
class ColumnProfile:
    """Everything known about one column.

    Attributes:
        name: Column name as it appears in the dataframe.
        dtype: The pandas dtype, as a string.
        role: One of :data:`ROLES`.
        n_missing: Count of nulls.
        pct_missing: Nulls as a percentage of rows, 0–100.
        n_unique: Distinct non-null values.
        cardinality_ratio: ``n_unique / n_rows``, the identifier detector's main signal.
        is_identifier: True when the column looks like a key rather than a measure.
        is_derived: True for columns PlotaViz itself added (outlier flags). They are shown in the
            preview table but never offered as chart axes.
        stats: Role-specific numbers — ``min``/``max``/``mean``/``median``/``std``/``skew`` for
            numeric columns, ``min``/``max``/``span_days`` for datetimes.
        top_values: Most frequent values with counts, for categorical and boolean columns.
        sample_values: A few example values, used in the LLM prompt and the UI.
        note: Human-readable remark, e.g. why a numeric column was treated as an identifier.
    """

    name: str
    dtype: str
    role: str
    n_missing: int = 0
    pct_missing: float = 0.0
    n_unique: int = 0
    cardinality_ratio: float = 0.0
    is_identifier: bool = False
    is_derived: bool = False
    stats: dict[str, float] = field(default_factory=dict)
    top_values: list[tuple[Any, int]] = field(default_factory=list)
    sample_values: list[Any] = field(default_factory=list)
    note: str = ""

    @property
    def is_low_cardinality(self) -> bool:
        """Whether this column is small enough to make a readable categorical axis."""
        return self.n_unique <= LOW_CARDINALITY_MAX

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view, used for session files and LLM prompts."""
        return {
            "name": self.name,
            "dtype": self.dtype,
            "role": self.role,
            "pct_missing": round(self.pct_missing, 2),
            "n_unique": self.n_unique,
            "is_identifier": self.is_identifier,
            "stats": {k: _json_safe(v) for k, v in self.stats.items()},
            "top_values": [[_json_safe(v), c] for v, c in self.top_values[:5]],
            "sample_values": [_json_safe(v) for v in self.sample_values[:5]],
        }


@dataclass
class DatasetProfile:
    """Aggregate view of a dataset: every column profile plus frame-level statistics.

    Attributes:
        n_rows: Rows profiled.
        n_cols: Column count.
        columns: Profiles keyed by column name, in dataframe order.
        n_duplicate_rows: Fully duplicated rows found.
        correlations: Pearson correlation matrix over numeric non-identifier columns, or ``None``
            when there are fewer than two of them.
        sampled: Whether profiling ran on a sample.
        total_rows: Rows in the source, which exceeds :attr:`n_rows` when sampled.
    """

    n_rows: int
    n_cols: int
    columns: dict[str, ColumnProfile]
    n_duplicate_rows: int = 0
    correlations: pd.DataFrame | None = None
    sampled: bool = False
    total_rows: int = 0

    # -------------------------------------------------------------- role queries

    def by_role(
        self, role: str, *, include_identifiers: bool = False, include_derived: bool = False
    ) -> list[str]:
        """Column names with the given role, in dataframe order.

        Identifiers and PlotaViz-generated columns are excluded by default — detecting them is
        pointless if they still end up on an axis.
        """
        return [
            name
            for name, prof in self.columns.items()
            if prof.role == role
            and (include_identifiers or not prof.is_identifier)
            and (include_derived or not prof.is_derived)
        ]

    @property
    def numeric(self) -> list[str]:
        """Numeric measure columns (identifiers excluded)."""
        return self.by_role(NUMERIC)

    @property
    def categorical(self) -> list[str]:
        """Categorical columns (identifiers excluded)."""
        return self.by_role(CATEGORICAL)

    @property
    def datetime(self) -> list[str]:
        """Datetime columns."""
        return self.by_role(DATETIME)

    @property
    def boolean(self) -> list[str]:
        """Boolean columns."""
        return self.by_role(BOOLEAN)

    @property
    def text(self) -> list[str]:
        """High-cardinality text columns."""
        return self.by_role(TEXT)

    def role_counts(self) -> dict[str, int]:
        """How many chartable columns exist per role — the selector's rule-matching input."""
        return {role: len(self.by_role(role)) for role in ROLES}

    def strongest_correlation(self) -> tuple[str, str, float] | None:
        """The most strongly correlated numeric pair, as ``(col_a, col_b, r)``.

        Returns ``None`` when there is no correlation matrix. The sign is preserved; the pair is
        chosen by absolute value.
        """
        if self.correlations is None or self.correlations.shape[0] < 2:
            return None
        corr = self.correlations
        best: tuple[str, str, float] | None = None
        cols = list(corr.columns)
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                value = as_float(corr.loc[a, b])
                if value is None:
                    continue
                if best is None or abs(value) > abs(best[2]):
                    best = (a, b, value)
        return best

    def schema_summary(self) -> dict[str, Any]:
        """Compact schema + statistics payload.

        This is exactly what gets sent to a language model — schema, summary statistics, and a
        handful of sample values. The dataset itself never leaves the machine.
        """
        return {
            "n_rows": self.total_rows or self.n_rows,
            "n_cols": self.n_cols,
            "sampled": self.sampled,
            "columns": [prof.to_dict() for prof in self.columns.values()],
            "strongest_correlation": self.strongest_correlation(),
        }


# ---------------------------------------------------------------------------- inference


def looks_like_datetime(
    series: pd.Series, *, name_hint: bool = True, threshold: float = 0.8
) -> bool:
    """Whether a string/object series parses cleanly as dates.

    Args:
        series: The column to test.
        name_hint: Whether the column name is allowed to lower the bar. Date-named columns get a
            parse attempt even when the first values are ambiguous.
        threshold: Fraction of non-null values that must parse for the answer to be yes.

    Pandas will happily coerce plain integers into 1970-era nanosecond timestamps, so numeric
    columns are rejected outright regardless of their name.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return False

    non_null = series.dropna()
    if non_null.empty:
        return False

    sample = non_null.head(1000).astype(str)
    if name_hint and _DATE_NAME_PATTERN.search(str(series.name or "")):
        threshold = min(threshold, 0.6)

    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return float(parsed.notna().mean()) >= threshold


def looks_like_boolean(series: pd.Series) -> bool:
    """Whether a two-valued column represents a boolean rather than a small category."""
    if pd.api.types.is_bool_dtype(series):
        return True
    non_null = series.dropna()
    if non_null.empty:
        return False
    values = {str(v).strip().lower() for v in non_null.unique()[:10]}
    return len(values) <= 2 and any(values <= tokens for tokens in _BOOLEAN_TOKENS)


def _is_key_like(values: pd.Series) -> bool:
    """Whether a near-unique integer column occupies a dense, contiguous range.

    Row identifiers and auto-increment keys sit in a narrow band — 1..n, or 1000..1500. Genuine
    measures that happen to be all-unique, like a city's population, are spread across orders of
    magnitude. Without this check, every unique-valued measure gets misfiled as an identifier and
    silently excluded from every chart.
    """
    numbers = pd.to_numeric(values, errors="coerce").dropna()
    if numbers.empty:
        return False
    span = float(numbers.max() - numbers.min()) + 1.0
    return span <= 2.0 * len(numbers)


def infer_role(series: pd.Series, *, n_rows: int | None = None) -> tuple[str, bool, str]:
    """Classify one column.

    Returns:
        ``(role, is_identifier, note)`` where ``note`` explains any non-obvious decision so the
        UI can show the user *why* their integer column is not being treated as a measure.
    """
    n_rows = n_rows or len(series)
    non_null = series.dropna()
    n_unique = int(non_null.nunique())
    ratio = n_unique / n_rows if n_rows else 0.0
    name = str(series.name or "")
    name_says_id = bool(_ID_NAME_PATTERN.search(name))

    if non_null.empty:
        return CATEGORICAL, False, "Column is entirely empty."

    if looks_like_boolean(series):
        return BOOLEAN, False, ""

    if pd.api.types.is_datetime64_any_dtype(series):
        return DATETIME, False, ""

    if pd.api.types.is_numeric_dtype(series):
        is_integral = (
            pd.api.types.is_integer_dtype(series) or (non_null.astype(float) % 1 == 0).all()
        )
        if name_says_id and is_integral:
            return NUMERIC, True, f"Treated {name!r} as an identifier because of its name."
        if is_integral and ratio >= ID_CARDINALITY_RATIO and n_rows > 20 and _is_key_like(non_null):
            return (
                NUMERIC,
                True,
                f"{name!r} is a nearly all-unique run of consecutive whole numbers, so it looks "
                "like a row identifier.",
            )
        if is_integral and n_unique <= LOW_CARDINALITY_MAX and n_unique <= 12:
            return (
                CATEGORICAL,
                False,
                f"{name!r} has only {n_unique} distinct whole numbers, so it reads as a category.",
            )
        return NUMERIC, False, ""

    # Object / string columns.
    if looks_like_datetime(series):
        return DATETIME, False, f"Parsed {name!r} from text into dates."

    if (name_says_id or ratio >= ID_CARDINALITY_RATIO) and ratio >= TEXT_CARDINALITY_RATIO:
        return (
            TEXT,
            True,
            f"{name!r} is nearly all-unique text, so it looks like an identifier.",
        )

    if ratio >= TEXT_CARDINALITY_RATIO and n_unique > LOW_CARDINALITY_MAX:
        return TEXT, False, f"{name!r} is high-cardinality free text ({n_unique:,} values)."

    return CATEGORICAL, False, ""


# ---------------------------------------------------------------------------- profiling


def profile(
    df: pd.DataFrame,
    *,
    overrides: dict[str, str] | None = None,
    sample_rows: int = PROFILE_SAMPLE_ROWS,
    total_rows: int | None = None,
) -> DatasetProfile:
    """Profile a dataframe.

    Args:
        df: The data to profile.
        overrides: User-supplied ``{column: role}`` corrections from the type override panel.
            These win over inference unconditionally — auto-inference gets IDs and dates wrong
            constantly, and the user is the authority.
        sample_rows: Profile on a random sample above this row count. Statistics stay
            representative and the UI stays responsive.
        total_rows: True source row count when ``df`` is already a sample.

    Returns:
        A :class:`DatasetProfile`.

    Raises:
        ProfileError: If the frame has no rows or no columns.
    """
    if df is None or df.empty:
        raise ProfileError("There is no data to profile.")
    if len(df.columns) == 0:
        raise ProfileError("The dataset has no columns.")

    overrides = overrides or {}
    sampled = len(df) > sample_rows
    frame = df.sample(sample_rows, random_state=0) if sampled else df
    n_rows = len(frame)

    columns: dict[str, ColumnProfile] = {}
    for name in df.columns:
        series = frame[name]
        role, is_id, note = infer_role(series, n_rows=n_rows)

        override = overrides.get(str(name))
        if override in ROLES:
            if override != role:
                note = f"Type set to {override} by you (inferred: {role})."
            role, is_id = override, False
            if override == DATETIME and not pd.api.types.is_datetime64_any_dtype(series):
                series = pd.to_datetime(series, errors="coerce", format="mixed")
            elif override == NUMERIC and not pd.api.types.is_numeric_dtype(series):
                series = pd.to_numeric(series, errors="coerce")

        n_missing = int(series.isna().sum())
        n_unique = int(series.dropna().nunique())
        prof = ColumnProfile(
            name=str(name),
            dtype=str(df[name].dtype),
            role=role,
            n_missing=n_missing,
            pct_missing=100.0 * n_missing / n_rows if n_rows else 0.0,
            n_unique=n_unique,
            cardinality_ratio=n_unique / n_rows if n_rows else 0.0,
            is_identifier=is_id,
            is_derived=str(name).endswith(DERIVED_SUFFIXES),
            note=note,
            sample_values=[_json_safe(v) for v in series.dropna().head(5).tolist()],
        )

        if role == NUMERIC:
            prof.stats = _numeric_stats(series)
        elif role == DATETIME:
            prof.stats = _datetime_stats(series)
        if role in (CATEGORICAL, BOOLEAN) or (role == TEXT and n_unique <= 50):
            counts = series.dropna().value_counts().head(10)
            prof.top_values = [(_json_safe(idx), int(cnt)) for idx, cnt in counts.items()]

        columns[str(name)] = prof

    correlations = _correlations(frame, columns)

    return DatasetProfile(
        n_rows=n_rows,
        n_cols=len(df.columns),
        columns=columns,
        n_duplicate_rows=int(frame.duplicated().sum()),
        correlations=correlations,
        sampled=sampled,
        total_rows=total_rows or len(df),
    )


def _numeric_stats(series: pd.Series) -> dict[str, float]:
    """Min/max/mean/median/std/skew for a numeric column, all as plain floats."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {}
    skew = as_float(values.skew()) or 0.0
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(values.std()) if len(values) > 1 else 0.0,
        "skew": 0.0 if np.isnan(skew) else skew,
        "zeros": float((values == 0).sum()),
    }


def _datetime_stats(series: pd.Series) -> dict[str, float]:
    """Span statistics for a datetime column, expressed in days."""
    values = pd.to_datetime(series, errors="coerce").dropna()
    if values.empty:
        return {}
    span = values.max() - values.min()
    return {
        "span_days": float(span.total_seconds() / 86400.0),
        "n_periods": float(values.nunique()),
    }


def _correlations(df: pd.DataFrame, columns: dict[str, ColumnProfile]) -> pd.DataFrame | None:
    """Pearson correlations across numeric measure columns, or ``None`` if fewer than two."""
    numeric = [
        name for name, prof in columns.items() if prof.role == NUMERIC and not prof.is_identifier
    ]
    if len(numeric) < 2:
        return None
    try:
        return df[numeric].apply(pd.to_numeric, errors="coerce").corr(numeric_only=True)
    except Exception:
        return None


def as_float(value: Any) -> float | None:
    """Coerce a pandas scalar to a plain float, or ``None`` if it is not numeric.

    Pandas scalars are typed as a broad union (timestamps, timedeltas, bytes …), so calling
    ``float()`` on one directly is both a type error and a real ``TypeError`` risk on odd data.
    """
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _json_safe(value: Any) -> Any:
    """Coerce numpy/pandas scalars into plain Python types for JSON serialization."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return str(value)
