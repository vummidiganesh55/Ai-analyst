import pandas as pd
import pytest

from app.analytics.kpi import KPIDetectionEngine


@pytest.fixture
def sales_data():
    return pd.DataFrame(
        {
            "revenue": [1000, 2000, 3000],
            "cost": [600, 1200, 1800],
            "orders": [10, 20, 30],
            "customer_id": ["C1", "C2", "C1"],
        }
    )


def test_revenue(sales_data):
    engine = KPIDetectionEngine(sales_data)

    result = engine.revenue()

    assert result is not None
    assert result.name == "Revenue"
    assert result.value == 6000


def test_cost(sales_data):
    engine = KPIDetectionEngine(sales_data)

    result = engine.cost()

    assert result is not None
    assert result.value == 3600


def test_profit(sales_data):
    engine = KPIDetectionEngine(sales_data)

    result = engine.profit()

    assert result is not None
    assert result.value == 2400


def test_margin(sales_data):
    engine = KPIDetectionEngine(sales_data)

    result = engine.margin()

    assert result is not None
    assert result.value == pytest.approx(40.0)


def test_orders(sales_data):
    engine = KPIDetectionEngine(sales_data)

    result = engine.orders()

    assert result is not None
    assert result.value == 60


def test_customers(sales_data):
    engine = KPIDetectionEngine(sales_data)

    result = engine.customers()

    assert result is not None
    assert result.value == 2


def test_aov(sales_data):
    engine = KPIDetectionEngine(sales_data)

    result = engine.aov()

    assert result is not None
    assert result.value == pytest.approx(100.0)


def test_detect_all_kpis(sales_data):
    engine = KPIDetectionEngine(sales_data)

    results = engine.detect()

    names = [result.name for result in results]

    assert "Revenue" in names
    assert "Cost" in names
    assert "Profit" in names
    assert "Margin" in names
    assert "Orders" in names
    assert "Customers" in names
    assert "Average Order Value" in names


def test_missing_columns():
    df = pd.DataFrame(
        {
            "name": ["A", "B", "C"],
            "score": [10, 20, 30],
        }
    )

    engine = KPIDetectionEngine(df)

    assert engine.revenue() is None
    assert engine.cost() is None
    assert engine.profit() is None
    assert engine.margin() is None
    assert engine.orders() is None


def test_empty_dataframe():
    with pytest.raises(ValueError):
        KPIDetectionEngine(pd.DataFrame())


def test_invalid_dataframe_type():
    with pytest.raises(TypeError):
        KPIDetectionEngine([1, 2, 3])


def test_non_numeric_revenue():
    df = pd.DataFrame(
        {
            "revenue": ["100", "200", "300"],
        }
    )

    engine = KPIDetectionEngine(df)

    with pytest.raises(TypeError):
        engine.revenue()


def test_zero_revenue_margin():
    df = pd.DataFrame(
        {
            "revenue": [0, 0],
            "cost": [10, 20],
        }
    )

    engine = KPIDetectionEngine(df)

    result = engine.margin()

    assert result is not None
    assert result.value == 0.0


def test_to_dataframe(sales_data):
    engine = KPIDetectionEngine(sales_data)

    result = engine.to_dataframe()

    assert not result.empty
    assert "name" in result.columns
    assert "value" in result.columns
    assert "category" in result.columns


def test_to_dict(sales_data):
    engine = KPIDetectionEngine(sales_data)

    result = engine.to_dict()

    assert isinstance(result, list)
    assert len(result) > 0
    assert "name" in result[0]
    assert "value" in result[0]
    assert "unit" in result[0]