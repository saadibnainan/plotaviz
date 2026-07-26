"""Loader tests — formats, delimiters, encodings, and the errors users actually hit."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from plotaviz.core import loader
from plotaviz.core.errors import LoadError


def test_loads_csv(csv_path: Path) -> None:
    result = loader.load(csv_path)
    assert result.rows_loaded == 540
    assert list(result.df.columns) == ["order_date", "region", "revenue", "units"]
    assert result.engine == "pandas"
    assert not result.sampled
    assert len(result.file_hash) == 64


def test_hash_changes_when_file_changes(tmp_path: Path) -> None:
    target = tmp_path / "d.csv"
    target.write_text("a,b\n1,2\n")
    first = loader.load(target).file_hash

    target.write_text("a,b\n1,3\n")
    assert loader.load(target).file_hash != first


@pytest.mark.parametrize(
    ("delimiter", "suffix"),
    [(",", ".csv"), (";", ".csv"), ("\t", ".tsv"), ("|", ".csv")],
)
def test_sniffs_delimiters(tmp_path: Path, delimiter: str, suffix: str) -> None:
    target = tmp_path / f"data{suffix}"
    rows = [
        ["name", "score", "team"],
        ["ada", "9", "red"],
        ["bob", "7", "blue"],
        ["cy", "8", "red"],
    ]
    target.write_text("\n".join(delimiter.join(row) for row in rows))

    result = loader.load(target)
    assert list(result.df.columns) == ["name", "score", "team"]
    assert result.rows_loaded == 3


def test_falls_back_to_another_encoding(tmp_path: Path) -> None:
    target = tmp_path / "latin.csv"
    target.write_bytes("name,city\nJosé,Málaga\nRené,Nîmes\n".encode("latin-1"))

    result = loader.load(target)
    assert result.rows_loaded == 2
    assert any("UTF-8" in note or "utf" in note.lower() for note in result.notes)


def test_loads_json_array(tmp_path: Path) -> None:
    target = tmp_path / "d.json"
    target.write_text(json.dumps([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]))

    result = loader.load(target)
    assert result.rows_loaded == 2
    assert list(result.df.columns) == ["a", "b"]


def test_loads_json_wrapped_in_a_data_key(tmp_path: Path) -> None:
    target = tmp_path / "d.json"
    target.write_text(json.dumps({"meta": {"n": 2}, "data": [{"a": 1}, {"a": 2}]}))

    assert loader.load(target).rows_loaded == 2


def test_loads_ndjson(tmp_path: Path) -> None:
    target = tmp_path / "d.ndjson"
    target.write_text('{"a": 1}\n{"a": 2}\n{"a": 3}\n')

    assert loader.load(target).rows_loaded == 3


def test_loads_parquet(tmp_path: Path, numeric_pair_df: pd.DataFrame) -> None:
    target = tmp_path / "d.parquet"
    numeric_pair_df.to_parquet(target)

    result = loader.load(target)
    assert result.rows_loaded == len(numeric_pair_df)


def test_loads_excel(tmp_path: Path, numeric_pair_df: pd.DataFrame) -> None:
    target = tmp_path / "d.xlsx"
    numeric_pair_df.to_excel(target, index=False)

    result = loader.load(target)
    assert result.rows_loaded == len(numeric_pair_df)


def test_polars_engine_samples_large_files(tmp_path: Path) -> None:
    target = tmp_path / "big.csv"
    pd.DataFrame({"a": range(5_000), "b": range(5_000)}).to_csv(target, index=False)

    result = loader.load(target, force_engine="polars", sample_rows=1_000)
    assert result.engine == "polars"
    assert result.sampled
    assert result.rows_loaded == 1_000
    assert result.total_rows == 5_000
    assert "5,000" in result.summary()


def test_missing_file_raises_a_readable_error(tmp_path: Path) -> None:
    with pytest.raises(LoadError, match="No such file"):
        loader.load(tmp_path / "nope.csv")


def test_unsupported_extension_lists_what_is_supported(tmp_path: Path) -> None:
    target = tmp_path / "notes.docx"
    target.write_bytes(b"not a dataset")

    with pytest.raises(LoadError) as exc:
        loader.load(target)
    assert "CSV" in str(exc.value)


def test_empty_file_raises(tmp_path: Path) -> None:
    target = tmp_path / "empty.csv"
    target.write_text("")

    with pytest.raises(LoadError, match="empty"):
        loader.load(target)


def test_header_only_file_raises(tmp_path: Path) -> None:
    target = tmp_path / "header.csv"
    target.write_text("a,b,c\n")

    with pytest.raises(LoadError, match="zero rows"):
        loader.load(target)


def test_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(LoadError, match="folder"):
        loader.load(tmp_path)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("d.csv", "pd.read_csv"),
        ("d.parquet", "pd.read_parquet"),
        ("d.xlsx", "pd.read_excel"),
        ("d.ndjson", "pd.read_json"),
    ],
)
def test_read_code_matches_the_reader_used(name: str, expected: str) -> None:
    assert loader.read_code_for(name).startswith(expected)


def test_every_sample_dataset_loads(sample_file: Path) -> None:
    result = loader.load(sample_file)
    assert result.rows_loaded > 0
    assert len(result.df.columns) > 1
