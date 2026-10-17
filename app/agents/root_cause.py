from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import math

import pandas as pd


@dataclass
class RootCauseResult:
    dimension: str
    segment: str
    previous_value: float
    current_value: float
    change: float
    contribution_pct: float | None
    direction: str
    evidence: dict[str, Any]


class RootCauseAnalyzer:
    """
    Deterministic root-cause contribution analysis.

    Compares a metric across two periods and ranks dimension segments
    by their contribution to the overall change.

    Results describe observed contributions, not proven causation.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        dimension_columns: list[str],
        metric_column: str,
        period_column: str,
        previous_period: Any,
        current_period: Any,
    ):
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame")

        if dataframe.empty:
            raise ValueError("dataframe cannot be empty")

        if not dimension_columns:
            raise ValueError("dimension_columns cannot be empty")

        required = set(
            dimension_columns
            + [metric_column, period_column]
        )

        missing = required.difference(dataframe.columns)

        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}"
            )

        if not pd.api.types.is_numeric_dtype(
            dataframe[metric_column]
        ):
            raise TypeError(
                f"Metric column '{metric_column}' must be numeric"
            )

        if previous_period == current_period:
            raise ValueError(
                "Previous and current periods must differ"
            )

        self.df = dataframe.copy()
        self.dimension_columns = dimension_columns
        self.metric_column = metric_column
        self.period_column = period_column
        self.previous_period = previous_period
        self.current_period = current_period

    @staticmethod
    def _safe_number(value: Any) -> float:
        number = float(value)

        if not math.isfinite(number):
            raise ValueError(
                "Metric values must be finite"
            )

        return number

    def _period_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        previous = self.df[
            self.df[self.period_column] == self.previous_period
        ]

        current = self.df[
            self.df[self.period_column] == self.current_period
        ]

        if previous.empty:
            raise ValueError("Previous period contains no data")

        if current.empty:
            raise ValueError("Current period contains no data")

        return previous, current

    def overall_change(self) -> dict[str, float]:
        previous, current = self._period_data()

        previous_total = self._safe_number(
            previous[self.metric_column].sum()
        )

        current_total = self._safe_number(
            current[self.metric_column].sum()
        )

        change = current_total - previous_total

        if previous_total == 0:
            change_pct = 0.0 if change == 0 else None
        else:
            change_pct = (
                change / abs(previous_total)
            ) * 100

        return {
            "previous_value": previous_total,
            "current_value": current_total,
            "change": change,
            "change_pct": change_pct,
        }

    def analyze_dimension(
        self,
        dimension: str,
    ) -> list[RootCauseResult]:
        if dimension not in self.dimension_columns:
            raise ValueError(
                f"Unsupported dimension: {dimension}"
            )

        previous, current = self._period_data()

        previous_grouped = (
            previous.groupby(
                dimension,
                dropna=False,
            )[self.metric_column]
            .sum()
        )

        current_grouped = (
            current.groupby(
                dimension,
                dropna=False,
            )[self.metric_column]
            .sum()
        )

        groups = previous_grouped.index.union(
            current_grouped.index
        )

        overall = self.overall_change()
        overall_change = overall["change"]

        results: list[RootCauseResult] = []

        for segment in groups:
            previous_value = self._safe_number(
                previous_grouped.get(segment, 0)
            )

            current_value = self._safe_number(
                current_grouped.get(segment, 0)
            )

            change = current_value - previous_value

            if overall_change == 0:
                contribution = None
            else:
                contribution = (
                    change / overall_change
                ) * 100

            if change > 0:
                direction = "increase"
            elif change < 0:
                direction = "decrease"
            else:
                direction = "unchanged"

            segment_name = (
                "<NULL>"
                if pd.isna(segment)
                else str(segment)
            )

            results.append(
                RootCauseResult(
                    dimension=dimension,
                    segment=segment_name,
                    previous_value=previous_value,
                    current_value=current_value,
                    change=change,
                    contribution_pct=contribution,
                    direction=direction,
                    evidence={
                        "metric": self.metric_column,
                        "previous_period": str(
                            self.previous_period
                        ),
                        "current_period": str(
                            self.current_period
                        ),
                        "interpretation": (
                            "Observed segment contribution; "
                            "not proof of causation."
                        ),
                    },
                )
            )

        results.sort(
            key=lambda item: abs(item.change),
            reverse=True,
        )

        return results

    def analyze(self) -> dict[str, Any]:
        overall = self.overall_change()

        dimensions = {
            dimension: [
                asdict(result)
                for result in self.analyze_dimension(dimension)
            ]
            for dimension in self.dimension_columns
        }

        return {
            "metric": self.metric_column,
            "overall": overall,
            "dimensions": dimensions,
            "interpretation": (
                "Results identify observed contributions to the metric "
                "change. They do not establish causal relationships."
            ),
        }