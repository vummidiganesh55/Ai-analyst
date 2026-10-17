import numpy as np
import pandas as pd
import pytest

from app.analytics.forecasting import ForecastingEngine


@pytest.fixture
def engine():
    return ForecastingEngine()


@pytest.fixture
def time_series():
    dates = pd.date_range(
        "2025-01-01",
        periods=60,
        freq="D",
    )

    values = np.arange(100, 160)

    return pd.DataFrame(
        {
            "date": dates,
            "revenue": values,
        }
    )


def test_prepare_time_series(
    engine,
    time_series,
):
    result = engine.prepare_time_series(
        time_series,
        "date",
        "revenue",
    )

    assert list(result.columns) == ["ds", "y"]
    assert len(result) == 60
    assert result["y"].iloc[0] == 100


def test_prepare_time_series_aggregation(
    engine,
):
    dataframe = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-01",
                    "2025-01-02",
                ]
            ),
            "revenue": [
                100,
                50,
                200,
            ],
        }
    )

    result = engine.prepare_time_series(
        dataframe,
        "date",
        "revenue",
    )

    assert result["y"].iloc[0] == 150
    assert result["y"].iloc[1] == 200


def test_invalid_date_column(
    engine,
    time_series,
):
    with pytest.raises(
        ValueError,
        match="Date column not found",
    ):
        engine.prepare_time_series(
            time_series,
            "unknown",
            "revenue",
        )


def test_invalid_target_column(
    engine,
    time_series,
):
    with pytest.raises(
        ValueError,
        match="Target column not found",
    ):
        engine.prepare_time_series(
            time_series,
            "date",
            "unknown",
        )


def test_invalid_horizon(engine):
    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        engine.validate_horizon(0)


def test_negative_horizon(engine):
    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        engine.validate_horizon(-5)


def test_mae(engine):
    actual = np.array([100, 200, 300])
    predicted = np.array([110, 190, 310])

    assert engine.mae(
        actual,
        predicted,
    ) == pytest.approx(10.0)


def test_rmse(engine):
    actual = np.array([100, 200, 300])
    predicted = np.array([110, 190, 310])

    assert engine.rmse(
        actual,
        predicted,
    ) == pytest.approx(10.0)


def test_mape(engine):
    actual = np.array([100, 200])
    predicted = np.array([110, 180])

    assert engine.mape(
        actual,
        predicted,
    ) == pytest.approx(10.0)


def test_metric_length_mismatch(engine):
    actual = np.array([100, 200])
    predicted = np.array([100])

    with pytest.raises(
        ValueError,
        match="equal length",
    ):
        engine.mae(
            actual,
            predicted,
        )


def test_mape_all_zero(engine):
    actual = np.array([0, 0])
    predicted = np.array([1, 2])

    with pytest.raises(
        ValueError,
        match="all actual values are zero",
    ):
        engine.mape(
            actual,
            predicted,
        )


def test_evaluate(engine):
    actual = np.array([100, 200, 300])
    predicted = np.array([110, 190, 310])

    result = engine.evaluate(
        actual,
        predicted,
    )

    assert result.mae == pytest.approx(10.0)
    assert result.rmse == pytest.approx(10.0)
    assert result.mape == pytest.approx(
        6.111111,
    )


def test_forecast(
    engine,
    time_series,
):
    result = engine.forecast(
        time_series,
        "date",
        "revenue",
        horizon=7,
    )

    assert len(result.forecast) == 7

    assert {
        "ds",
        "yhat",
        "yhat_lower",
        "yhat_upper",
    }.issubset(result.forecast.columns)


def test_forecast_confidence_interval(
    engine,
    time_series,
):
    result = engine.forecast(
        time_series,
        "date",
        "revenue",
        horizon=7,
    )

    forecast = result.forecast

    assert (
        forecast["yhat_lower"]
        <= forecast["yhat"]
    ).all()

    assert (
        forecast["yhat"]
        <= forecast["yhat_upper"]
    ).all()


def test_forecast_to_records(
    engine,
    time_series,
):
    result = engine.forecast(
        time_series,
        "date",
        "revenue",
        horizon=5,
    )

    records = engine.forecast_to_records(result)

    assert len(records) == 5

    assert {
        "date",
        "forecast",
        "lower_bound",
        "upper_bound",
    }.issubset(records[0].keys())


def test_backtest(
    engine,
    time_series,
):
    result = engine.backtest(
        time_series,
        "date",
        "revenue",
        test_size=7,
    )

    assert result.mae >= 0
    assert result.rmse >= 0
    assert result.mape >= 0