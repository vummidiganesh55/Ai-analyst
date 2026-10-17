from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class StatisticsEngine:
    """Deterministic business and statistical analytics engine."""

    def _validate_column(
        self,
        dataframe: pd.DataFrame,
        column: str,
    ) -> pd.Series:
        if column not in dataframe.columns:
            raise ValueError(f"Column not found: {column}")

        series = pd.to_numeric(dataframe[column], errors="coerce").dropna()

        if series.empty:
            raise ValueError(
                f"Column '{column}' contains no valid numeric values."
            )

        return series

    def mean(
        self,
        dataframe: pd.DataFrame,
        column: str,
    ) -> float:
        series = self._validate_column(dataframe, column)
        return float(series.mean())

    def median(
        self,
        dataframe: pd.DataFrame,
        column: str,
    ) -> float:
        series = self._validate_column(dataframe, column)
        return float(series.median())

    def variance(
        self,
        dataframe: pd.DataFrame,
        column: str,
        sample: bool = True,
    ) -> float:
        series = self._validate_column(dataframe, column)

        ddof = 1 if sample else 0

        return float(series.var(ddof=ddof))

    def standard_deviation(
        self,
        dataframe: pd.DataFrame,
        column: str,
        sample: bool = True,
    ) -> float:
        series = self._validate_column(dataframe, column)

        ddof = 1 if sample else 0

        return float(series.std(ddof=ddof))

    def minimum(
        self,
        dataframe: pd.DataFrame,
        column: str,
    ) -> float:
        series = self._validate_column(dataframe, column)
        return float(series.min())

    def maximum(
        self,
        dataframe: pd.DataFrame,
        column: str,
    ) -> float:
        series = self._validate_column(dataframe, column)
        return float(series.max())

    def percentile(
        self,
        dataframe: pd.DataFrame,
        column: str,
        percentile: float,
    ) -> float:
        if not 0 <= percentile <= 100:
            raise ValueError("Percentile must be between 0 and 100.")

        series = self._validate_column(dataframe, column)

        return float(np.percentile(series, percentile))

    def correlation(
        self,
        dataframe: pd.DataFrame,
        column_x: str,
        column_y: str,
    ) -> float:
        x = self._validate_column(dataframe, column_x)
        y = self._validate_column(dataframe, column_y)

        aligned = pd.concat([x, y], axis=1).dropna()

        if len(aligned) < 2:
            raise ValueError(
                "At least two paired observations are required."
            )

        return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))

    def covariance(
        self,
        dataframe: pd.DataFrame,
        column_x: str,
        column_y: str,
        sample: bool = True,
    ) -> float:
        x = self._validate_column(dataframe, column_x)
        y = self._validate_column(dataframe, column_y)

        aligned = pd.concat([x, y], axis=1).dropna()

        if len(aligned) < 2:
            raise ValueError(
                "At least two paired observations are required."
            )

        ddof = 1 if sample else 0

        return float(
            aligned.iloc[:, 0].cov(
                aligned.iloc[:, 1],
                ddof=ddof,
            )
        )

    def total(
        self,
        dataframe: pd.DataFrame,
        column: str,
    ) -> float:
        series = self._validate_column(dataframe, column)
        return float(series.sum())

    def revenue(
        self,
        dataframe: pd.DataFrame,
        revenue_column: str = "revenue",
    ) -> float:
        return self.total(dataframe, revenue_column)

    def cost(
        self,
        dataframe: pd.DataFrame,
        cost_column: str = "cost",
    ) -> float:
        return self.total(dataframe, cost_column)

    def profit(
        self,
        dataframe: pd.DataFrame,
        revenue_column: str = "revenue",
        cost_column: str = "cost",
    ) -> float:
        revenue = self.revenue(dataframe, revenue_column)
        cost = self.cost(dataframe, cost_column)

        return float(revenue - cost)

    def margin(
        self,
        dataframe: pd.DataFrame,
        revenue_column: str = "revenue",
        cost_column: str = "cost",
    ) -> float:
        revenue = self.revenue(dataframe, revenue_column)
        profit = self.profit(
            dataframe,
            revenue_column,
            cost_column,
        )

        if revenue == 0:
            raise ValueError(
                "Cannot calculate margin when revenue is zero."
            )

        return float((profit / revenue) * 100)

    def aov(
        self,
        dataframe: pd.DataFrame,
        revenue_column: str = "revenue",
        orders_column: str = "orders",
    ) -> float:
        revenue = self.revenue(dataframe, revenue_column)
        orders = self.total(dataframe, orders_column)

        if orders == 0:
            raise ValueError(
                "Cannot calculate AOV when total orders are zero."
            )

        return float(revenue / orders)

    def growth_rate(
        self,
        previous_value: float,
        current_value: float,
    ) -> float:
        if previous_value == 0:
            raise ValueError(
                "Cannot calculate growth rate from a zero baseline."
            )

        return float(
            ((current_value - previous_value) / abs(previous_value))
            * 100
        )

    def mom_growth(
        self,
        previous_month_value: float,
        current_month_value: float,
    ) -> float:
        return self.growth_rate(
            previous_month_value,
            current_month_value,
        )

    def yoy_growth(
        self,
        previous_year_value: float,
        current_year_value: float,
    ) -> float:
        return self.growth_rate(
            previous_year_value,
            current_year_value,
        )

    def summary(
        self,
        dataframe: pd.DataFrame,
        column: str,
    ) -> dict[str, Any]:
        series = self._validate_column(dataframe, column)

        return {
            "column": column,
            "count": int(series.count()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "variance": float(series.var()),
            "standard_deviation": float(series.std()),
            "minimum": float(series.min()),
            "maximum": float(series.max()),
            "percentile_25": float(np.percentile(series, 25)),
            "percentile_50": float(np.percentile(series, 50)),
            "percentile_75": float(np.percentile(series, 75)),
        }