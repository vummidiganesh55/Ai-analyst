import pandas as pd
import pytest

from app.analytics.visualization import (
    ChartSpecification,
    VisualizationEngine,
)


@pytest.fixture
def engine():
    return VisualizationEngine()


@pytest.fixture
def dataframe():
    return pd.DataFrame(
        {
            "date": pd.date_range(
                "2026-01-01",
                periods=5,
                freq="D",
            ),
            "region": [
                "South",
                "North",
                "East",
                "West",
                "South",
            ],
            "revenue": [
                100,
                200,
                150,
                120,
                250,
            ],
            "orders": [
                10,
                20,
                15,
                12,
                25,
            ],
            "is_anomaly": [
                False,
                False,
                True,
                False,
                False,
            ],
        }
    )


def test_line_chart(engine, dataframe):
    result = engine.line_chart(
        dataframe,
        "date",
        "revenue",
        "Revenue Trend",
    )

    assert isinstance(result, ChartSpecification)
    assert result.chart_type == "line"
    assert result.title == "Revenue Trend"
    assert len(result.data) == 5
    assert result.x_axis == "date"
    assert result.y_axis == "revenue"


def test_bar_chart(engine, dataframe):
    result = engine.bar_chart(
        dataframe,
        "region",
        "revenue",
    )

    assert result.chart_type == "bar"
    assert len(result.data) == 5


def test_histogram(engine, dataframe):
    result = engine.histogram(
        dataframe,
        "revenue",
        bins=5,
    )

    assert result.chart_type == "histogram"
    assert result.metadata["bins"] == 5
    assert len(result.data) == 5


def test_scatter_chart(engine, dataframe):
    result = engine.scatter_chart(
        dataframe,
        "orders",
        "revenue",
    )

    assert result.chart_type == "scatter"
    assert result.x_axis == "orders"
    assert result.y_axis == "revenue"
    assert len(result.data) == 5


def test_forecast_chart(engine):
    forecast = pd.DataFrame(
        {
            "ds": pd.date_range(
                "2026-01-01",
                periods=3,
                freq="D",
            ),
            "yhat": [100, 110, 120],
            "yhat_lower": [90, 100, 110],
            "yhat_upper": [110, 120, 130],
        }
    )

    result = engine.forecast_chart(
        forecast,
    )

    assert result.chart_type == "forecast"
    assert len(result.data) == 3
    assert result.metadata["lower_bound"] == "yhat_lower"
    assert result.metadata["upper_bound"] == "yhat_upper"


def test_anomaly_chart(engine, dataframe):
    result = engine.anomaly_chart(
        dataframe,
        "date",
        "revenue",
    )

    assert result.chart_type == "anomaly"
    assert result.metadata["anomaly_column"] == "is_anomaly"
    assert len(result.data) == 5


def test_kpi_card(engine):
    result = engine.kpi_card(
        name="Revenue",
        value=125000,
        unit="INR",
    )

    assert result.chart_type == "kpi"
    assert result.data[0]["name"] == "Revenue"
    assert result.data[0]["value"] == 125000
    assert result.data[0]["unit"] == "INR"


def test_create_line_chart(engine, dataframe):
    result = engine.create(
        "line",
        dataframe,
        x_column="date",
        y_column="revenue",
    )

    assert result.chart_type == "line"


def test_create_bar_chart(engine, dataframe):
    result = engine.create(
        "bar",
        dataframe,
        x_column="region",
        y_column="revenue",
    )

    assert result.chart_type == "bar"


def test_create_kpi(engine):
    result = engine.create(
        "kpi",
        pd.DataFrame(),
        name="Profit",
        value=50000,
        unit="INR",
    )

    assert result.chart_type == "kpi"


def test_to_dict(engine, dataframe):
    specification = engine.line_chart(
        dataframe,
        "date",
        "revenue",
    )

    result = engine.to_dict(specification)

    assert result["chart_type"] == "line"
    assert result["title"] == "Line Chart"
    assert len(result["data"]) == 5
    assert "metadata" in result


def test_missing_column(engine, dataframe):
    with pytest.raises(
        ValueError,
        match="Columns not found",
    ):
        engine.line_chart(
            dataframe,
            "unknown",
            "revenue",
        )


def test_invalid_chart_type(engine, dataframe):
    with pytest.raises(
        ValueError,
        match="Unsupported chart type",
    ):
        engine.create(
            "pie",
            dataframe,
            x_column="region",
            y_column="revenue",
        )


def test_invalid_histogram_bins(engine, dataframe):
    with pytest.raises(
        ValueError,
        match="bins must be greater than zero",
    ):
        engine.histogram(
            dataframe,
            "revenue",
            bins=0,
        )


def test_histogram_non_numeric_column(
    engine,
    dataframe,
):
    with pytest.raises(
        ValueError,
        match="contains no numeric values",
    ):
        engine.histogram(
            dataframe,
            "region",
        )
