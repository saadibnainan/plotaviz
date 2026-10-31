"""Dataset loading — CSV, TSV, Excel, JSON/NDJSON, and Parquet into a pandas DataFrame.

Small files go through pandas because it is convenient and its type inference is good. Files
above a configurable size threshold go through **polars + pyarrow**, which reads them
dramatically faster and lets us take a sample without materializing the whole frame. Either way
the caller gets a pandas DataFrame and a :class:`LoadResult` describing what actually happened,
including whether it is a sample — the UI has to be able to say "showing 200,000 of 8,400,000
rows" rather than quietly lying.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .errors import LoadError

#: Extensions recognised by :func:`load`, grouped by the reader used.
CSV_SUFFIXES = {".csv", ".tsv", ".txt", ".tab"}
EXCEL_SUFFIXES = {".xlsx", ".xls", ".xlsm"}
JSON_SUFFIXES = {".json", ".ndjson", ".jsonl"}
PARQUET_SUFFIXES = {".parquet", ".pq"}
SUPPORTED_SUFFIXES = CSV_SUFFIXES | EXCEL_SUFFIXES | JSON_SUFFIXES | PARQUET_SUFFIXES

#: Above this size, switch to the polars reader and profile on a sample.
DEFAULT_LARGE_FILE_MB: float = 500.0

#: How many rows to keep when a file is too large to hold comfortably in memory.
DEFAULT_SAMPLE_ROWS: int = 200_000

#: Encodings tried in order when the default UTF-8 read fails.
_ENCODING_FALLBACKS = ("utf-8-sig", "latin-1", "cp1252")

#: A decimal written with a comma, optionally with dot thousands grouping: ``1,5``, ``1.234,56``.
_COMMA_DECIMAL = re.compile(r"^\s*[+-]?(?:\d{1,3}(?:\.\d{3})+|\d+),\d+\s*$")

#: A comma number with exactly three digits after the comma and no dot grouping: ``1,250``.
_THREE_DECIMALS = re.compile(r"^\s*[+-]?\d{1,3},\d{3}\s*$")

#: A day-first dotted date: ``01.05.2024``.
_DOTTED_DATE = re.compile(r"^\s*\d{1,2}\.\d{1,2}\.\d{2,4}(?:\s.*)?$")

#: Dot used as a thousands separator inside a comma-decimal number.
_DOT_GROUPING = re.compile(r"\d\.\d{3}(?:,|$)")

#: Delimiters for which a comma cannot be the field separator, so it may be the decimal mark.
_DECIMAL_COMMA_DELIMITERS = {";", "\t", "|"}


@dataclass
class LoadResult:
    """A loaded dataset plus the provenance the rest of the app needs.

    Attributes:
        df: The data, always as a pandas DataFrame.
        path: Absolute path the data came from.
        engine: ``"pandas"`` or ``"polars"``.
        total_rows: Row count of the *source*, which exceeds ``len(df)`` when sampled.
        sampled: Whether ``df`` is a sample rather than the full dataset.
        file_hash: SHA-256 of the source file, used by sessions to detect that the data changed
            underneath a saved project.
        size_bytes: Source file size.
        notes: Human-readable remarks worth surfacing (fallback encoding used, sheet chosen).
        read_options: Reader keyword arguments that differ from the defaults (``sep``,
            ``encoding``, ``decimal``, ``thousands``). The code generator replays them so the
            exported script reads the file exactly as PlotaViz did.
    """

    df: pd.DataFrame
    path: Path
    engine: str = "pandas"
    total_rows: int = 0
    sampled: bool = False
    file_hash: str = ""
    size_bytes: int = 0
    notes: list[str] = field(default_factory=list)
    read_options: dict[str, Any] = field(default_factory=dict)

    @property
    def rows_loaded(self) -> int:
        """Number of rows actually present in :attr:`df`."""
        return len(self.df)

    def summary(self) -> str:
        """One-line description for a status bar."""
        cols = len(self.df.columns)
        if self.sampled:
            return (
                f"{self.path.name} — showing a sample of {self.rows_loaded:,} "
                f"of {self.total_rows:,} rows × {cols} columns ({self.engine})"
            )
        return f"{self.path.name} — {self.rows_loaded:,} rows × {cols} columns ({self.engine})"


def file_hash(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks so size does not matter."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sniff_delimiter(path: Path, encoding: str = "utf-8") -> str:
    """Guess a delimited file's separator from its first few KB.

    Falls back to the extension convention (tab for ``.tsv``/``.tab``, comma otherwise) when
    :mod:`csv`'s sniffer cannot decide, which it frequently cannot for single-column files.
    """
    default = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    try:
        with open(path, encoding=encoding, errors="replace") as handle:
            sample = handle.read(8192)
    except OSError:
        return default
    if not sample.strip():
        return default
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return default


def load(
    path: str | Path,
    *,
    large_file_mb: float = DEFAULT_LARGE_FILE_MB,
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
    sheet: str | int | None = None,
    force_engine: str | None = None,
    known_hash: str | None = None,
    options: dict[str, Any] | None = None,
) -> LoadResult:
    """Read a dataset from disk.

    Args:
        path: File to read. The extension decides the reader.
        large_file_mb: Above this size, use polars and sample. Configurable because "large"
            depends entirely on the machine.
        sample_rows: Rows to keep when sampling a large file.
        sheet: Excel sheet name or index. Defaults to the first sheet.
        force_engine: ``"pandas"`` or ``"polars"`` to override the size heuristic. Mostly for
            tests and for users who know their machine.
        known_hash: SHA-256 the caller already computed for this file (a session restore does),
            to skip hashing it again.
        options: Explicit reader settings from the import dialog, overriding detection:
            ``sep``, ``encoding``, ``decimal``, ``thousands``, ``header`` (row number, or
            ``None`` for no header row) for delimited text, and ``sheet`` for Excel.

    Returns:
        A :class:`LoadResult`.

    Raises:
        LoadError: If the file is missing, empty, of an unsupported type, or unreadable.
    """
    path = Path(path).expanduser().resolve()

    if not path.exists():
        raise LoadError(f"No such file: {path}", hint="Check the path and try again.")
    if path.is_dir():
        raise LoadError(f"{path.name} is a folder, not a data file.")

    size_bytes = path.stat().st_size
    if size_bytes == 0:
        raise LoadError(f"{path.name} is empty.")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise LoadError(
            f"PlotaViz cannot read {suffix or 'files without an extension'} yet.",
            hint="Supported formats: CSV, TSV, Excel (.xlsx/.xls), JSON, NDJSON, Parquet.",
        )

    is_large = (size_bytes / (1024 * 1024)) > large_file_mb
    use_polars = force_engine == "polars" or (force_engine is None and is_large)

    options = dict(options or {})
    if sheet is None and "sheet" in options:
        sheet = options["sheet"]

    if suffix in EXCEL_SUFFIXES:
        result = _load_excel(path, sheet=sheet)
    elif suffix in PARQUET_SUFFIXES:
        result = _load_parquet(path, use_polars=use_polars, sample_rows=sample_rows)
    elif suffix in JSON_SUFFIXES:
        result = _load_json(path, use_polars=use_polars, sample_rows=sample_rows)
    else:
        # Explicit options are a pandas feature set; the polars fast path honours only defaults.
        # A large file opened with options is still capped at `sample_rows`, through pandas.
        result = _load_csv(
            path,
            use_polars=use_polars and not options,
            sample_rows=sample_rows,
            options=options,
            limit_rows=sample_rows if (options and use_polars) else None,
        )

    if is_large and not result.sampled and suffix in EXCEL_SUFFIXES | {".json"}:
        result.notes.append(
            f"{path.name} is large ({size_bytes / 1048576:,.0f} MB) but this format cannot be "
            "sampled while reading, so it was loaded in full. Converting it to CSV or Parquet "
            "lets PlotaViz sample it."
        )

    if result.df.empty:
        raise LoadError(
            f"{path.name} parsed to zero rows.",
            hint="The file may have only a header row, or the wrong delimiter was detected.",
        )
    if len(result.df.columns) == 0:
        raise LoadError(f"{path.name} has no columns.")

    result.path = path
    result.size_bytes = size_bytes
    result.file_hash = known_hash or file_hash(path)
    if not result.total_rows:
        result.total_rows = len(result.df)
    return result


# ---------------------------------------------------------------------------- readers


def _load_csv(
    path: Path,
    *,
    use_polars: bool,
    sample_rows: int,
    options: dict[str, Any] | None = None,
    limit_rows: int | None = None,
) -> LoadResult:
    """Read a delimited text file, sniffing the separator and retrying on encoding errors.

    Anything in ``options`` (from the import dialog) wins over detection: a chosen delimiter is
    not sniffed, a chosen encoding is not retried, a chosen decimal mark switches off the
    decimal-comma heuristic.
    """
    options = dict(options or {})
    delimiter = str(options.get("sep") or sniff_delimiter(path))
    encodings = (
        (str(options["encoding"]),)
        if options.get("encoding")
        else (
            "utf-8",
            *_ENCODING_FALLBACKS,
        )
    )
    extra: dict[str, Any] = {
        key: options[key] for key in ("decimal", "thousands", "header") if key in options
    }
    # A large file opened with explicit options goes through pandas; sample it at random by
    # skipping lines, so the rows kept are spread over the whole file, not its first N. The
    # whole file is still tokenized — unavoidable for a uniform sample without polars — but
    # skipped lines are never materialized. (Measured: the per-line callback costs the same as
    # a chunked read, ~0.35 s per million lines.)
    row_kwargs: dict[str, Any] = {}
    sampled_total: int | None = None
    if limit_rows is not None:
        header = extra.get("header", 0)
        data_start = (header if isinstance(header, int) else -1) + 1
        data_lines = _count_lines(path) - data_start
        if data_lines > limit_rows:
            keep = set((_sample_positions(data_lines, limit_rows) + data_start).tolist())
            row_kwargs["skiprows"] = lambda i: i >= data_start and i not in keep
            sampled_total = data_lines

    if use_polars:
        try:
            import polars as pl

            lazy = pl.scan_csv(
                path,
                separator=delimiter,
                infer_schema_length=10_000,
                ignore_errors=True,
                try_parse_dates=True,
            )
            total = int(lazy.select(pl.len()).collect().item())
            if total > sample_rows:
                return LoadResult(
                    df=_polars_sample(lazy, total, sample_rows).to_pandas(),
                    path=path,
                    engine="polars",
                    total_rows=total,
                    sampled=True,
                    notes=[_sample_note(sample_rows, total)],
                )
            return LoadResult(
                df=lazy.collect().to_pandas(), path=path, engine="polars", total_rows=total
            )
        except ImportError:
            pass  # polars missing; fall through to pandas
        except Exception as exc:
            raise LoadError(
                f"Could not read {path.name} with the fast reader.",
                hint=f"Underlying error: {exc}",
            ) from exc

    notes: list[str] = []
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            df = pd.read_csv(
                path,
                sep=delimiter,
                encoding=encoding,
                low_memory=False,
                **row_kwargs,
                **extra,
            )
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        except pd.errors.EmptyDataError as exc:
            raise LoadError(f"{path.name} contains no parsable data.") from exc
        except pd.errors.ParserError as exc:
            raise LoadError(
                f"{path.name} is malformed and could not be parsed.",
                hint=f"Detected delimiter {delimiter!r}. Underlying error: {exc}",
            ) from exc
        used: dict[str, Any] = dict(extra)
        if delimiter != ",":
            used["sep"] = delimiter
        if encoding != "utf-8":
            if "encoding" not in options:
                notes.append(f"File is not UTF-8; read it as {encoding}.")
            used["encoding"] = encoding

        locale_options = {} if "decimal" in extra else _decimal_comma_options(df, delimiter)
        if locale_options.pop("_grouping_conflict", None):
            notes.append(
                "Some numbers use '.' as a thousands separator, but the file also has dotted "
                "dates, so those numbers were left as text. Set the thousands separator in "
                "File → Open with options if the dates can be sacrificed, or convert the column."
            )
        if locale_options:
            try:
                # The user's own options (header row, an explicit thousands mark) win.
                reread: dict[str, Any] = {**locale_options, **extra}
                df = pd.read_csv(
                    path,
                    sep=delimiter,
                    encoding=encoding,
                    low_memory=False,
                    **row_kwargs,
                    **reread,
                )
            except (pd.errors.ParserError, ValueError):
                pass  # keep the first read; the numbers stay text, which is visible and fixable
            else:
                used.update(reread)
                mark = "comma decimals" + (
                    " and dot thousands separators" if "thousands" in locale_options else ""
                )
                notes.append(f"Numbers use {mark} (European format); read them as numbers.")

        result = LoadResult(df=df, path=path, engine="pandas", notes=notes, read_options=used)
        if sampled_total is not None:
            result.total_rows = sampled_total
            result.sampled = True
            notes.append(_sample_note(len(df), sampled_total))
        return result

    raise LoadError(
        f"Could not decode {path.name} with any supported encoding.",
        hint=f"Tried UTF-8, {', '.join(_ENCODING_FALLBACKS)}. Underlying error: {last_error}",
    )


#: Seed for row sampling, so the same file always yields the same sample (and session replays,
#: exports, and recommendations stay stable).
SAMPLE_SEED = 0


def _sample_positions(total: int, size: int) -> Any:
    """``size`` distinct row positions out of ``total``, uniformly at random, in file order."""
    import numpy as np

    rng = np.random.default_rng(SAMPLE_SEED)
    return np.sort(rng.choice(total, size=size, replace=False))


def _polars_sample(lazy: Any, total: int, size: int) -> Any:
    """Collect a uniform random sample of ``size`` rows from a lazy polars frame.

    The old approach, ``head(size)``, kept only the start of the file — for a time-sorted file
    that is the earliest period, not a sample. Rows keep their original order.
    """
    import polars as pl

    positions = pl.Series(_sample_positions(total, size)).cast(pl.UInt32)
    return (
        lazy.with_row_index("__pv_row")
        .filter(pl.col("__pv_row").is_in(positions.implode()))
        .drop("__pv_row")
        .collect()
    )


def _sample_note(kept: int, total: int) -> str:
    return (
        f"Large file — loaded a random sample of {kept:,} of {total:,} rows (spread over the "
        "whole file, in file order). Statistics and charts are computed on this sample; "
        "exported code reads the full file."
    )


def _count_lines(path: Path, *, chunk_size: int = 1 << 20) -> int:
    """Count newline-terminated lines without decoding the file."""
    count = 0
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            count += chunk.count(b"\n")
    return count


def _decimal_comma_options(df: pd.DataFrame, delimiter: str) -> dict[str, str]:
    """Reader options for a file whose numbers use a comma as the decimal mark, or ``{}``.

    A semicolon-delimited CSV from a German or French locale writes ``1,5`` and ``1.234,56``.
    Read with the defaults, those columns arrive as text — and the date parser then claims them.
    This looks for text columns that are overwhelmingly comma-decimal numbers. It refuses when
    the file also contains dot-decimal floats, because one file cannot honour both conventions.
    """
    if delimiter not in _DECIMAL_COMMA_DELIMITERS:
        return {}

    candidates: list[pd.Series] = []
    dotted_dates = False
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_float_dtype(series):
            values = series.dropna()
            if not values.empty and bool(((values % 1) != 0).any()):
                return {}  # real dot decimals elsewhere: mixed conventions, leave it alone
            continue
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue
        sample = series.dropna().astype(str).head(1000)
        if sample.empty:
            continue
        if float(sample.str.match(_COMMA_DECIMAL).mean()) >= 0.9:
            candidates.append(sample)
        elif float(sample.str.match(_DOTTED_DATE).mean()) >= 0.5:
            dotted_dates = True

    if not candidates:
        return {}
    # "1,250" / "12,000" with exactly three digits after every comma is US thousands grouping as
    # often as it is a German decimal. Guessing wrong scales the column by 1000, so ambiguous
    # columns are left as text — visible, and fixable in File → Open with options.
    if all(bool(sample.str.match(_THREE_DECIMALS).all()) for sample in candidates):
        return {}
    options = {"decimal": ","}
    # pandas strips the thousands mark anywhere in a field, so with dotted dates in the file
    # ("01.05.2024") a "." thousands separator would turn every date into an integer.
    grouped = any(bool(sample.str.contains(_DOT_GROUPING).any()) for sample in candidates)
    if grouped and not dotted_dates:
        options["thousands"] = "."
    elif grouped:
        options["_grouping_conflict"] = "1"
    return options


def _load_excel(path: Path, *, sheet: str | int | None) -> LoadResult:
    """Read one sheet of an Excel workbook, defaulting to the first."""
    try:
        book = pd.ExcelFile(path)
    except ImportError as exc:
        package = "xlrd" if path.suffix.lower() == ".xls" else "openpyxl"
        raise LoadError(
            f"Reading {path.suffix.lower()} files needs the {package} package.",
            hint=f"Install it with: pip install {package}",
        ) from exc
    except Exception as exc:
        raise LoadError(f"Could not open {path.name}.", hint=str(exc)) from exc

    target = sheet if sheet is not None else 0
    if target == ALL_SHEETS:
        return _load_all_sheets(path, book)
    try:
        df = pd.read_excel(book, sheet_name=target)
    except Exception as exc:
        raise LoadError(
            f"Could not read sheet {target!r} from {path.name}.",
            hint=f"Sheets in this workbook: {', '.join(map(str, book.sheet_names))}",
        ) from exc

    notes = []
    if len(book.sheet_names) > 1:
        chosen = book.sheet_names[target] if isinstance(target, int) else target
        notes.append(f"Workbook has {len(book.sheet_names)} sheets; loaded {chosen!r}.")
    read_options = {"sheet_name": target} if target != 0 else {}
    return LoadResult(df=df, path=path, engine="pandas", notes=notes, read_options=read_options)


#: ``sheet`` value meaning "every sheet, stacked, with a ``sheet`` column".
ALL_SHEETS = "*"


def _load_all_sheets(path: Path, book: Any) -> LoadResult:
    """Stack every sheet of a workbook into one table, recording each row's sheet.

    For workbooks that split one table across sheets (one per month, per region…).
    """
    frames = pd.read_excel(book, sheet_name=None)
    df = pd.concat(frames, names=["sheet", None]).reset_index(level=0).reset_index(drop=True)
    return LoadResult(
        df=df,
        path=path,
        engine="pandas",
        notes=[f"Stacked {len(frames)} sheets into one table; the 'sheet' column says which."],
        read_options={"sheet_name": ALL_SHEETS},
    )


def _load_json(path: Path, *, use_polars: bool = False, sample_rows: int = 0) -> LoadResult:
    """Read JSON or newline-delimited JSON, flattening nested objects one level.

    Large NDJSON is sampled through polars like CSV. A plain JSON document has to be parsed whole
    before it can be sampled, so a large one is loaded in full and the user is told so.
    """
    suffix = path.suffix.lower()
    if use_polars and suffix in {".ndjson", ".jsonl"}:
        try:
            import polars as pl

            lazy = pl.scan_ndjson(path)
            total = int(lazy.select(pl.len()).collect().item())
            if total > sample_rows:
                return LoadResult(
                    df=_polars_sample(lazy, total, sample_rows).to_pandas(),
                    path=path,
                    engine="polars",
                    total_rows=total,
                    sampled=True,
                    notes=[_sample_note(sample_rows, total)],
                )
        except ImportError:
            pass
        except Exception:
            # polars is stricter than pandas about NDJSON (a field that is a number in early rows
            # and text later fails its schema inference). pandas read these files before, so it
            # still gets the chance to — in full, which is noted below.
            pass
    try:
        if suffix in {".ndjson", ".jsonl"}:
            df = pd.read_json(path, lines=True)
        else:
            with open(path, encoding="utf-8") as handle:
                payload: Any = json.load(handle)
            if isinstance(payload, dict):
                # A single object of arrays, or a wrapper like {"data": [...]}.
                for key in ("data", "records", "rows", "results", "items"):
                    if isinstance(payload.get(key), list):
                        payload = payload[key]
                        break
            if _is_object_of_arrays(payload):
                df = pd.DataFrame(payload)
            else:
                df = pd.json_normalize(payload)
    except ValueError as exc:
        raise LoadError(
            f"{path.name} is not valid JSON that maps onto a table.",
            hint=(
                "PlotaViz expects an array of objects, an object of arrays, or NDJSON. "
                f"Underlying error: {exc}"
            ),
        ) from exc
    except OSError as exc:
        raise LoadError(f"Could not read {path.name}.", hint=str(exc)) from exc

    return LoadResult(df=df, path=path, engine="pandas")


def _is_object_of_arrays(payload: Any) -> bool:
    """Whether ``payload`` is ``{"col": [...], ...}`` with equal-length, non-empty columns.

    ``json_normalize`` turns that shape into a single row of lists, so it is built directly.
    """
    if not isinstance(payload, dict) or not payload:
        return False
    values = list(payload.values())
    if not all(isinstance(v, list) for v in values):
        return False
    lengths = {len(v) for v in values}
    return len(lengths) == 1 and lengths != {0}


def _load_parquet(path: Path, *, use_polars: bool, sample_rows: int) -> LoadResult:
    """Read a Parquet file, sampling through polars when it is large."""
    if use_polars:
        try:
            import polars as pl

            lazy = pl.scan_parquet(path)
            total = int(lazy.select(pl.len()).collect().item())
            if total > sample_rows:
                return LoadResult(
                    df=_polars_sample(lazy, total, sample_rows).to_pandas(),
                    path=path,
                    engine="polars",
                    total_rows=total,
                    sampled=True,
                    notes=[_sample_note(sample_rows, total)],
                )
            return LoadResult(
                df=lazy.collect().to_pandas(), path=path, engine="polars", total_rows=total
            )
        except ImportError:
            pass
        except Exception as exc:
            raise LoadError(f"Could not read {path.name}.", hint=str(exc)) from exc

    try:
        df = pd.read_parquet(path)
    except ImportError as exc:
        raise LoadError(
            "Reading Parquet needs pyarrow.",
            hint="Install it with: pip install pyarrow",
        ) from exc
    except Exception as exc:
        raise LoadError(f"Could not read {path.name}.", hint=str(exc)) from exc

    return LoadResult(df=df, path=path, engine="pandas")


def read_code_for(path: str | Path, options: dict[str, Any] | None = None) -> str:
    """Return the pandas one-liner that reads ``path``, for the generated script.

    Keeping this next to the readers means generated code stays in step with how PlotaViz
    actually loads a file instead of drifting from it.

    Args:
        path: The dataset.
        options: :attr:`LoadResult.read_options` from the actual load. When given, the delimiter,
            encoding, and number format are replayed exactly instead of re-sniffed.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    literal = repr(str(path))
    if options and suffix in CSV_SUFFIXES:
        keys = ("sep", "encoding", "decimal", "thousands", "header")
        extra = "".join(f", {k}={options[k]!r}" for k in keys if k in options)
        return f"pd.read_csv({literal}{extra})"
    if suffix in EXCEL_SUFFIXES:
        if options and options.get("sheet_name") == ALL_SHEETS:
            return (
                f"pd.concat(pd.read_excel({literal}, sheet_name=None), names=['sheet', None])"
                ".reset_index(level=0).reset_index(drop=True)"
            )
        if options and "sheet_name" in options:
            return f"pd.read_excel({literal}, sheet_name={options['sheet_name']!r})"
        return f"pd.read_excel({literal})"
    if suffix in PARQUET_SUFFIXES:
        return f"pd.read_parquet({literal})"
    if suffix in {".ndjson", ".jsonl"}:
        return f"pd.read_json({literal}, lines=True)"
    if suffix == ".json":
        return f"pd.read_json({literal})"
    delimiter = sniff_delimiter(path) if path.exists() else ","
    if delimiter == ",":
        return f"pd.read_csv({literal})"
    return f"pd.read_csv({literal}, sep={delimiter!r})"
