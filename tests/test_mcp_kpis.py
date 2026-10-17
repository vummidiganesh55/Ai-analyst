import hashlib

import pandas as pd
import pytest

import app.integrations.mcp_server as mcp_module


@pytest.fixture
def dataset_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(
        mcp_module,
        "DATASET_DIRECTORY",
        tmp_path.resolve(),
    )
    return tmp_path


@pytest.fixture
def sales_dataset(dataset_directory):
    path = dataset_directory / "sales.csv"

    pd.DataFrame(
        {
            "region": ["South", "North", "South"],
            "revenue": [1000, 1500, 1200],
            "cost": [600, 900, 700],
            "orders": [10, 15, 12],
            "customer_id": ["C001", "C002", "C001"],
        }
    ).to_csv(path, index=False)

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    dataset_id = f"{path.stem}_{digest[:12]}"

    return path, dataset_id


def test_calculate_kpis_returns_success(sales_dataset):
    _, dataset_id = sales_dataset

    result = mcp_module.calculate_kpis(dataset_id)

    assert result["status"] == "success"


def test_calculate_kpis_returns_dataset_metadata(sales_dataset):
    path, dataset_id = sales_dataset

    result = mcp_module.calculate_kpis(dataset_id)

    assert result["dataset_id"] == dataset_id
    assert result["dataset_name"] == path.name


def test_calculate_kpis_returns_seven_available_kpis(sales_dataset):
    _, dataset_id = sales_dataset

    result = mcp_module.calculate_kpis(dataset_id)

    assert result["kpi_count"] == 7
    assert len(result["kpis"]) == 7


def test_revenue_kpi(sales_dataset):
    _, dataset_id = sales_dataset

    result = mcp_module.calculate_kpis(dataset_id)

    revenue = next(
        item for item in result["kpis"]
        if item["name"] == "Revenue"
    )

    assert revenue["value"] == pytest.approx(3700.0)


def test_cost_kpi(sales_dataset):
    _, dataset_id = sales_dataset

    result = mcp_module.calculate_kpis(dataset_id)

    cost = next(
        item for item in result["kpis"]
        if item["name"] == "Cost"
    )

    assert cost["value"] == pytest.approx(2200.0)


def test_profit_kpi(sales_dataset):
    _, dataset_id = sales_dataset

    result = mcp_module.calculate_kpis(dataset_id)

    profit = next(
        item for item in result["kpis"]
        if item["name"] == "Profit"
    )

    assert profit["value"] == pytest.approx(1500.0)


def test_margin_kpi(sales_dataset):
    _, dataset_id = sales_dataset

    result = mcp_module.calculate_kpis(dataset_id)

    margin = next(
        item for item in result["kpis"]
        if item["name"] == "Margin"
    )

    assert margin["value"] == pytest.approx(
        (1500 / 3700) * 100
    )


def test_orders_kpi(sales_dataset):
    _, dataset_id = sales_dataset

    result = mcp_module.calculate_kpis(dataset_id)

    orders = next(
        item for item in result["kpis"]
        if item["name"] == "Orders"
    )

    assert orders["value"] == pytest.approx(37.0)


def test_customers_kpi_counts_unique_customers(sales_dataset):
    _, dataset_id = sales_dataset

    result = mcp_module.calculate_kpis(dataset_id)

    customers = next(
        item for item in result["kpis"]
        if item["name"] == "Customers"
    )

    assert customers["value"] == pytest.approx(2.0)


def test_average_order_value(sales_dataset):
    _, dataset_id = sales_dataset

    result = mcp_module.calculate_kpis(dataset_id)

    aov = next(
        item for item in result["kpis"]
        if item["name"] == "Average Order Value"
    )

    assert aov["value"] == pytest.approx(3700 / 37)


def test_invalid_dataset_id_returns_error():
    result = mcp_module.calculate_kpis("../secret.csv")

    assert result["status"] == "error"


def test_missing_dataset_returns_error(dataset_directory):
    result = mcp_module.calculate_kpis("missing_aabbccddeeff")

    assert result["status"] == "error"


def test_mcp_server_still_creates():
    server = mcp_module.create_mcp_server()

    assert server is not None