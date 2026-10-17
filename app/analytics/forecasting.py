from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from prophet import Prophet


@dataclass
class ForecastResult:
    forecast: pd.DataFrame
    horizon: int
    frequency: str
    target_column: str
    date_column: str


@dataclass
class ForecastMetrics:
    mae: float
    rmse: float
    mape: float


class ForecastingEngine:
    """Deterministic time-series forecasting engine."""

    def prepare_time_series(
        self,
        dataframe: pd.DataFrame,
        date_column: str,
        target_column: str,
        frequency: str = "D",
    ) -> pd.DataFrame:

        if date_column not in dataframe.columns:
            raise ValueError(
                f"Date column not found: {date_column}"
            )

        if target_column not in dataframe.columns:
            raise ValueError(
                f"Target column not found: {target_column}"
            )

        data = dataframe[
            [date_column, target_column]
        ].copy()

        data[date_column] = pd.to_datetime(
            data[date_column],
            errors="coerce",
        )

        data[target_column] = pd.to_numeric(
            data[target_column],
            errors="coerce",
        )

        data = data.dropna()

        if data.empty:
            raise ValueError(
                "No valid observations available for forecasting."
            )

        data = (
            data.groupby(date_column, as_index=False)[target_column]
            .sum()
            .sort_values(date_column)
        )

        data = data.rename(
            columns={
                date_column: "ds",
                target_column: "y",
            }
        )

        data = (
            data.set_index("ds")
            .resample(frequency)["y"]
            .sum()
            .reset_index()
        )

        if len(data) < 2:
            raise ValueError(
                "At least two time-series observations are required."
            )

        return data

    def validate_horizon(
        self,
        horizon: int,
    ) -> None:

        if not isinstance(horizon, int):
            raise ValueError(
                "Forecast horizon must be an integer."
            )

        if horizon <= 0:
            raise ValueError(
                "Forecast horizon must be greater than zero."
            )

    def forecast(
        self,
        dataframe: pd.DataFrame,
        date_column: str,
        target_column: str,
        horizon: int = 7,
        frequency: str = "D",
    ) -> ForecastResult:

        self.validate_horizon(horizon)

        time_series = self.prepare_time_series(
            dataframe=dataframe,
            date_column=date_column,
            target_column=target_column,
            frequency=frequency,
        )

        if len(time_series) < 2:
            raise ValueError(
                "Insufficient observations for forecasting."
            )

        model = Prophet(
            interval_width=0.95,
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=False,
        )

        model.fit(time_series)

        future = model.make_future_dataframe(
            periods=horizon,
            freq=frequency,
        )

        forecast = model.predict(future)

        forecast = forecast[
            [
                "ds",
                "yhat",
                "yhat_lower",
                "yhat_upper",
            ]
        ].tail(horizon)

        return ForecastResult(
            forecast=forecast.reset_index(drop=True),
            horizon=horizon,
            frequency=frequency,
            target_column=target_column,
            date_column=date_column,
        )

    def mae(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
    ) -> float:

        actual = np.asarray(actual, dtype=float)
        predicted = np.asarray(predicted, dtype=float)

        if len(actual) != len(predicted):
            raise ValueError(
                "Actual and predicted arrays must have equal length."
            )

        return float(
            np.mean(np.abs(actual - predicted))
        )

    def rmse(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
    ) -> float:

        actual = np.asarray(actual, dtype=float)
        predicted = np.asarray(predicted, dtype=float)

        if len(actual) != len(predicted):
            raise ValueError(
                "Actual and predicted arrays must have equal length."
            )

        return float(
            np.sqrt(
                np.mean(
                    (actual - predicted) ** 2
                )
            )
        )

    def mape(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
    ) -> float:

        actual = np.asarray(actual, dtype=float)
        predicted = np.asarray(predicted, dtype=float)

        if len(actual) != len(predicted):
            raise ValueError(
                "Actual and predicted arrays must have equal length."
            )

        non_zero = actual != 0

        if not np.any(non_zero):
            raise ValueError(
                "MAPE cannot be calculated when all actual values are zero."
            )

        return float(
            np.mean(
                np.abs(
                    (
                        actual[non_zero]
                        - predicted[non_zero]
                    )
                    / actual[non_zero]
                )
            )
            * 100
        )

    def evaluate(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
    ) -> ForecastMetrics:

        return ForecastMetrics(
            mae=self.mae(actual, predicted),
            rmse=self.rmse(actual, predicted),
            mape=self.mape(actual, predicted),
        )

    def backtest(
        self,
        dataframe: pd.DataFrame,
        date_column: str,
        target_column: str,
        test_size: int = 7,
        frequency: str = "D",
    ) -> ForecastMetrics:

        time_series = self.prepare_time_series(
            dataframe=dataframe,
            date_column=date_column,
            target_column=target_column,
            frequency=frequency,
        )

        if test_size <= 0:
            raise ValueError(
                "Test size must be greater than zero."
            )

        if len(time_series) <= test_size:
            raise ValueError(
                "Not enough observations for backtesting."
            )

        train = time_series.iloc[:-test_size].copy()
        test = time_series.iloc[-test_size:].copy()

        model = Prophet(
            interval_width=0.95,
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=False,
        )

        model.fit(train)

        future = model.make_future_dataframe(
            periods=test_size,
            freq=frequency,
        )

        prediction = model.predict(future).tail(test_size)

        actual = test["y"].to_numpy()
        predicted = prediction["yhat"].to_numpy()

        return self.evaluate(
            actual,
            predicted,
        )

    def forecast_to_records(
        self,
        result: ForecastResult,
    ) -> list[dict[str, Any]]:

        records = []

        for row in result.forecast.itertuples(index=False):
            records.append(
                {
                    "date": row.ds,
                    "forecast": float(row.yhat),
                    "lower_bound": float(row.yhat_lower),
                    "upper_bound": float(row.yhat_upper),
                }
            )

        return records