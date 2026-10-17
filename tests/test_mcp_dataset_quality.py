import hashlib

import pandas as pd

import app.integrations.mcp_server as mcp_module


def test_data_quality_check_returns_success(tmp_path, monkeypatch):
    # Use an isolated dataset directory.
    monkeypatch.setattr(
        mcp_module,
        "DATASET_DIRECTORY",
        tmp_path.resolve(),
    )

    # Create a sample dataset containing quality issues.
    dataset_path = tmp_path / "sales.csv"

    pd.DataFrame({
        "region": ["South", "North", "South"],
        "revenue": [1000, -500, 1000],
        "orders": [10, 15, 10],
    }).to_csv(dataset_path, index=False)

    # Generate the application-compatible dataset ID.
    digest = hashlib.sha256(
        dataset_path.read_bytes()
    ).hexdigest()

    dataset_id = f"sales_{digest[:12]}"

    # Execute the MCP tool.
    result = mcp_module.data_quality_check(dataset_id)

    # Verify the response.
    assert result["status"] == "success"
    assert result["dataset_id"] == dataset_id
    assert result["dataset_name"] == "sales.csv"

    quality = result["quality_report"]

    assert quality["missing_values"] == 0
    assert quality["duplicate_rows"] == 1
    assert quality["negative_values"]["revenue"] == 1
    assert quality["quality_score"] < 100


def test_data_quality_check_rejects_invalid_dataset():
    result = mcp_module.data_quality_check(
        "../unauthorized.csv"
    )

    assert result["status"] == "error"


def test_mcp_server_registers_data_quality_tool():
    server = mcp_module.create_mcp_server()

    assert server is not None