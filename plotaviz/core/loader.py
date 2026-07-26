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
    """

    df: pd.DataFrame
    path: Path
    engine: str = "pandas"
    total_rows: int = 0
    sampled: bool = False
    file_hash: str = ""
    size_bytes: int = 0
    notes: list[str] = field(default_factory=list)

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

    if suffix in EXCEL_SUFFIXES:
        result = _load_excel(path, sheet=sheet)
    elif suffix in PARQUET_SUFFIXES:
        result = _load_parquet(path, use_polars=use_polars, sample_rows=sample_rows)
    elif suffix in JSON_SUFFIXES:
        result = _load_json(path)
    else:
        result = _load_csv(path, use_polars=use_polars, sample_rows=sample_rows)

    if result.df.empty:
        raise LoadError(
            f"{path.name} parsed to zero rows.",
            hint="The file may have only a header row, or the wrong delimiter was detected.",
        )
    if len(result.df.columns) == 0:
        raise LoadError(f"{path.name} has no columns.")

    result.path = path
    result.size_bytes = size_bytes
    result.file_hash = file_hash(path)
    if not result.total_rows:
        result.total_rows = len(result.df)
    return result


# ---------------------------------------------------------------------------- readers


def _load_csv(path: Path, *, use_polars: bool, sample_rows: int) -> LoadResult:
    """Read a delimited text file, sniffing the separator and retrying on encoding errors."""
    delimiter = sniff_delimiter(path)

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
                frame = lazy.head(sample_rows).collect()
                return LoadResult(
                    df=frame.to_pandas(),
                    path=path,
                    engine="polars",
                    total_rows=total,
                    sampled=True,
                    notes=[
                        f"Large file — loaded the first {sample_rows:,} of {total:,} rows. "
                        "Statistics and charts are computed on this sample."
                    ],
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
    for encoding in ("utf-8", *_ENCODING_FALLBACKS):
        try:
            df = pd.read_csv(path, sep=delimiter, encoding=encoding, low_memory=False)
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
        if encoding != "utf-8":
            notes.append(f"File is not UTF-8; read it as {encoding}.")
        return LoadResult(df=df, path=path, engine="pandas", notes=notes)

    raise LoadError(
        f"Could not decode {path.name} with any supported encoding.",
        hint=f"Tried UTF-8, {', '.join(_ENCODING_FALLBACKS)}. Underlying error: {last_error}",
    )


def _load_excel(path: Path, *, sheet: str | int | None) -> LoadResult:
    """Read one sheet of an Excel workbook, defaulting to the first."""
    try:
        book = pd.ExcelFile(path)
    except ImportError as exc:
        raise LoadError(
            "Reading Excel files needs the openpyxl package.",
            hint="Install it with: pip install openpyxl",
        ) from exc
    except Exception as exc:
        raise LoadError(f"Could not open {path.name}.", hint=str(exc)) from exc

    target = sheet if sheet is not None else 0
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
    return LoadResult(df=df, path=path, engine="pandas", notes=notes)


def _load_json(path: Path) -> LoadResult:
    """Read JSON or newline-delimited JSON, flattening nested objects one level."""
    suffix = path.suffix.lower()
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


def _load_parquet(path: Path, *, use_polars: bool, sample_rows: int) -> LoadResult:
    """Read a Parquet file, sampling through polars when it is large."""
    if use_polars:
        try:
            import polars as pl

            lazy = pl.scan_parquet(path)
            total = int(lazy.select(pl.len()).collect().item())
            if total > sample_rows:
                return LoadResult(
                    df=lazy.head(sample_rows).collect().to_pandas(),
                    path=path,
                    engine="polars",
                    total_rows=total,
                    sampled=True,
                    notes=[f"Large file — loaded the first {sample_rows:,} of {total:,} rows."],
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


def read_code_for(path: str | Path) -> str:
    """Return the pandas one-liner that reads ``path``, for the generated script.

    Keeping this next to the readers means generated code stays in step with how PlotaViz
    actually loads a file instead of drifting from it.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    literal = repr(str(path))
    if suffix in EXCEL_SUFFIXES:
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
