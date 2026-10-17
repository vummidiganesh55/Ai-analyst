import pandas as pd
import pytest

from app.analytics.statistics import StatisticsEngine


@pytest.fixture
def dataframe():
    return pd.DataFrame(
        {
            "revenue": [100, 200, 300, 400, 500],
            "cost": [60, 120, 180, 240, 300],
            "orders": [10, 20, 30, 40, 50],
            "profit": [40, 80, 120, 160, 200],
            "marketing_spend": [10, 20, 30, 40, 50],
        }
    )


@pytest.fixture
def engine():
    return StatisticsEngine()


def test_mean(engine, dataframe):
    assert engine.mean(dataframe, "revenue") == 300.0


def test_median(engine, dataframe):
    assert engine.median(dataframe, "revenue") == 300.0


def test_variance(engine, dataframe):
    assert engine.variance(dataframe, "revenue") == 25000.0


def test_standard_deviation(engine, dataframe):
    assert engine.standard_deviation(dataframe, "revenue") == pytest.approx(
        158.113883
    )


def test_minimum_and_maximum(engine, dataframe):
    assert engine.minimum(dataframe, "revenue") == 100.0
    assert engine.maximum(dataframe, "revenue") == 500.0


def test_percentile(engine, dataframe):
    assert engine.percentile(dataframe, "revenue", 50) == 300.0
    assert engine.percentile(dataframe, "revenue", 75) == 400.0


def test_correlation(engine, dataframe):
    result = engine.correlation(
        dataframe,
        "revenue",
        "orders",
    )

    assert result == pytest.approx(1.0)


def test_covariance(engine, dataframe):
    result = engine.covariance(
        dataframe,
        "revenue",
        "orders",
    )

    assert result == pytest.approx(2500.0)


def test_total(engine, dataframe):
    assert engine.total(dataframe, "revenue") == 1500.0


def test_revenue(engine, dataframe):
    assert engine.revenue(dataframe) == 1500.0


def test_cost(engine, dataframe):
    assert engine.cost(dataframe) == 900.0


def test_profit(engine, dataframe):
    assert engine.profit(dataframe) == 600.0


def test_margin(engine, dataframe):
    assert engine.margin(dataframe) == pytest.approx(40.0)


def test_aov(engine, dataframe):
    assert engine.aov(dataframe) == pytest.approx(10.0)


def test_growth_rate(engine):
    assert engine.growth_rate(100, 120) == pytest.approx(20.0)


def test_mom_growth(engine):
    assert engine.mom_growth(200, 250) == pytest.approx(25.0)


def test_yoy_growth(engine):
    assert engine.yoy_growth(1000, 800) == pytest.approx(-20.0)


def test_summary(engine, dataframe):
    result = engine.summary(dataframe, "revenue")

    assert result["count"] == 5
    assert result["mean"] == 300.0
    assert result["median"] == 300.0
    assert result["minimum"] == 100.0
    assert result["maximum"] == 500.0


def test_invalid_column(engine, dataframe):
    with pytest.raises(ValueError, match="Column not found"):
        engine.mean(dataframe, "unknown")


def test_invalid_percentile(engine, dataframe):
    with pytest.raises(ValueError, match="between 0 and 100"):
        engine.percentile(dataframe, "revenue", 101)


def test_zero_revenue_margin(engine):
    dataframe = pd.DataFrame(
        {
            "revenue": [0, 0],
            "cost": [0, 0],
        }
    )

    with pytest.raises(ValueError, match="revenue is zero"):
        engine.margin(dataframe)


def test_zero_orders_aov(engine):
    dataframe = pd.DataFrame(
        {
            "revenue": [100, 200],
            "orders": [0, 0],
        }
    )

    with pytest.raises(ValueError, match="orders are zero"):
        engine.aov(dataframe)


def test_zero_baseline_growth(engine):
    with pytest.raises(ValueError, match="zero baseline"):
        engine.growth_rate(0, 100)