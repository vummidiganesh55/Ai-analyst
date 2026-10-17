import hashlib
from pathlib import Path

import pandas as pd
import pytest

import app.integrations.mcp_server as mcp_module


@pytest.fixture
def dataset_directory(tmp_path, monkeypatch):
    """Create an isolated approved dataset directory."""
    monkeypatch.setattr(
        mcp_module,
        "DATASET_DIRECTORY",
        tmp_path.resolve(),
    )
    return tmp_path


@pytest.fixture
def sample_csv(dataset_directory):
    """Create a small dataset for integration tests."""
    path = dataset_directory / "sales.csv"

    pd.DataFrame(
        {
            "region": ["South", "North", "South"],
            "revenue": [1000, 1500, 1200],
            "orders": [10, 15, 12],
        }
    ).to_csv(path, index=False)

    return path


def get_dataset_id(path: Path) -> str:
    """Generate the same dataset ID as the application."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"{path.stem}_{digest[:12]}"


def test_health_check_remains_available():
    result = mcp_module.health_check()

    assert result["status"] == "healthy"


def test_resolve_existing_dataset(dataset_directory, sample_csv):
    dataset_id = get_dataset_id(sample_csv)

    resolved = mcp_module.resolve_dataset_path(dataset_id)

    assert resolved == sample_csv.resolve()


def test_dataset_profile_returns_success(sample_csv):
    result = mcp_module.dataset_profile(get_dataset_id(sample_csv))

    assert result["status"] == "success"


def test_profile_contains_dataset_metadata(sample_csv):
    result = mcp_module.dataset_profile(get_dataset_id(sample_csv))

    assert result["dataset_id"] == get_dataset_id(sample_csv)
    assert result["dataset_name"] == "sales.csv"


def test_profile_contains_row_and_column_counts(sample_csv):
    result = mcp_module.dataset_profile(get_dataset_id(sample_csv))

    profile = result["profile"]

    assert profile["row_count"] == 3
    assert profile["column_count"] == 3


def test_profile_contains_column_names(sample_csv):
    result = mcp_module.dataset_profile(get_dataset_id(sample_csv))

    assert result["profile"]["columns"] == [
        "region",
        "revenue",
        "orders",
    ]


def test_profile_contains_numeric_columns(sample_csv):
    result = mcp_module.dataset_profile(get_dataset_id(sample_csv))

    assert set(result["profile"]["numeric_columns"]) == {
        "revenue",
        "orders",
    }


def test_profile_contains_categorical_columns(sample_csv):
    result = mcp_module.dataset_profile(get_dataset_id(sample_csv))

    assert "region" in result["profile"]["categorical_columns"]


def test_profile_contains_missing_value_counts(sample_csv):
    result = mcp_module.dataset_profile(get_dataset_id(sample_csv))

    assert result["profile"]["missing_values"]["revenue"] == 0


def test_profile_contains_duplicate_count(sample_csv):
    result = mcp_module.dataset_profile(get_dataset_id(sample_csv))

    assert result["profile"]["duplicate_rows"] == 0


@pytest.mark.parametrize(
    "dataset_id",
    [
        "",
        " ",
        "../sales.csv",
        "sales.csv",
        "../../secret_aabbccddeeff",
        "sales_ABCDEF123456",
    ],
)
def test_invalid_dataset_id_returns_error(dataset_id):
    result = mcp_module.dataset_profile(dataset_id)

    assert result["status"] == "error"


def test_nonexistent_dataset_returns_error(dataset_directory):
    result = mcp_module.dataset_profile("missing_aabbccddeeff")

    assert result["status"] == "error"
    assert "not found" in result["error"].lower()


def test_dataset_directory_must_exist(tmp_path, monkeypatch):
    missing_directory = tmp_path / "missing"

    monkeypatch.setattr(
        mcp_module,
        "DATASET_DIRECTORY",
        missing_directory,
    )

    with pytest.raises(FileNotFoundError):
        mcp_module.resolve_dataset_path("sales_aabbccddeeff")


def test_dataset_profile_handles_missing_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(
        mcp_module,
        "DATASET_DIRECTORY",
        tmp_path / "missing",
    )

    result = mcp_module.dataset_profile("sales_aabbccddeeff")

    assert result["status"] == "error"


def test_dataset_id_changes_when_file_content_changes(
    dataset_directory,
    sample_csv,
):
    original_id = get_dataset_id(sample_csv)

    sample_csv.write_text(
        "region,revenue\nSouth,9999\n",
        encoding="utf-8",
    )

    updated_id = get_dataset_id(sample_csv)

    assert original_id != updated_id


def test_unsupported_files_are_not_resolved(dataset_directory):
    path = dataset_directory / "notes.txt"
    path.write_text("Not a supported dataset.", encoding="utf-8")

    result = mcp_module.dataset_profile(
        "notes_aabbccddeeff"
    )

    assert result["status"] == "error"


def test_mcp_server_creation_registers_tools():
    server = mcp_module.create_mcp_server()

    assert server is not None