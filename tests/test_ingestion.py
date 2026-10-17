from pathlib import Path

import pandas as pd
import pytest

from app.ingestion.validator import validate_file
from app.ingestion.loader import load_dataset


def test_validate_file_rejects_missing_file(tmp_path):
    missing_file = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        validate_file(str(missing_file))


def test_validate_file_rejects_unsupported_extension(tmp_path):
    file_path = tmp_path / "data.txt"
    file_path.write_text("sample data", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        validate_file(str(file_path))


def test_validate_file_rejects_oversized_file(tmp_path):
    file_path = tmp_path / "large.csv"

    # Create a sparse file larger than 1 MB.
    with file_path.open("wb") as file:
        file.truncate(1024 * 1024 + 1)

    with pytest.raises(ValueError, match="exceeds maximum size"):
        validate_file(str(file_path), max_size_mb=1)


def test_load_dataset_rejects_unsupported_extension(tmp_path):
    file_path = tmp_path / "data.txt"
    file_path.write_text("sample data", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        load_dataset(str(file_path))


def test_load_dataset_rejects_malformed_csv(tmp_path):
    file_path = tmp_path / "malformed.csv"
    file_path.write_bytes(b"\x00\x00\x00\x00")

    with pytest.raises((ValueError, UnicodeDecodeError)):
        load_dataset(str(file_path))


def test_validate_file_rejects_directory(tmp_path):
    directory = tmp_path / "directory.csv"
    directory.mkdir()

    with pytest.raises((ValueError, IsADirectoryError, PermissionError)):
        validate_file(str(directory))

def test_load_dataset_rejects_excessive_rows(tmp_path, monkeypatch):
    file_path = tmp_path / "rows.csv"
    file_path.write_text("value\n1\n", encoding="utf-8")

    dataframe = pd.DataFrame({"value": [1, 2, 3]})

    monkeypatch.setattr(
        "app.ingestion.loader.settings.max_dataset_rows",
        2,
    )
    monkeypatch.setattr(
        "app.ingestion.loader.pd.read_csv",
        lambda *args, **kwargs: dataframe,
    )

    with pytest.raises(ValueError, match="maximum of 2 rows"):
        load_dataset(str(file_path))


def test_load_dataset_rejects_excessive_columns(tmp_path, monkeypatch):
    file_path = tmp_path / "columns.csv"
    file_path.write_text("a,b,c\n1,2,3\n", encoding="utf-8")

    dataframe = pd.DataFrame({
        "a": [1],
        "b": [2],
        "c": [3],
    })

    monkeypatch.setattr(
        "app.ingestion.loader.settings.max_dataset_columns",
        2,
    )
    monkeypatch.setattr(
        "app.ingestion.loader.pd.read_csv",
        lambda *args, **kwargs: dataframe,
    )

    with pytest.raises(ValueError, match="maximum of 2 columns"):
        load_dataset(str(file_path))


def test_load_dataset_rejects_excessive_memory(tmp_path, monkeypatch):
    file_path = tmp_path / "memory.csv"
    file_path.write_text("value\n1\n", encoding="utf-8")

    dataframe = pd.DataFrame({
        "value": ["x" * 1024] * 100,
    })

    monkeypatch.setattr(
        "app.ingestion.loader.settings.max_dataset_memory_mb",
        0,
    )
    monkeypatch.setattr(
        "app.ingestion.loader.pd.read_csv",
        lambda *args, **kwargs: dataframe,
    )

    with pytest.raises(ValueError, match="memory limit"):
        load_dataset(str(file_path))