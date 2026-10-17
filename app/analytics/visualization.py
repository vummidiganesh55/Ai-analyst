from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class ChartSpecification:
    chart_type: str
    title: str
    data: list[dict[str, Any]]
    x_axis: str | None = None
    y_axis: str | None = None
    metadata: dict[str, Any] | None = None


class VisualizationEngine:
    """Creates frontend-ready chart specifications."""

    SUPPORTED_CHARTS = {
        "line",
        "bar",
        "histogram",
        "scatter",
        "forecast",
        "anomaly",
        "kpi",
    }

    def _validate_columns(
        self,
        dataframe: pd.DataFrame,
        columns: list[str],
    ) -> None:
        missing = [
            column
            for column in columns
            if column not in dataframe.columns
        ]

        if missing:
            raise ValueError(
                f"Columns not found: {', '.join(missing)}"
            )

    def _validate_chart_type(
        self,
        chart_type: str,
    ) -> None:
        if chart_type not in self.SUPPORTED_CHARTS:
            raise ValueError(
                f"Unsupported chart type: {chart_type}"
            )

    def line_chart(
        self,
        dataframe: pd.DataFrame,
        x_column: str,
        y_column: str,
        title: str = "Line Chart",
    ) -> ChartSpecification:
        self._validate_columns(
            dataframe,
            [x_column, y_column],
        )

        data = dataframe[
            [x_column, y_column]
        ].dropna()

        return ChartSpecification(
            chart_type="line",
            title=title,
            data=data.to_dict(orient="records"),
            x_axis=x_column,
            y_axis=y_column,
        )

    def bar_chart(
        self,
        dataframe: pd.DataFrame,
        x_column: str,
        y_column: str,
        title: str = "Bar Chart",
    ) -> ChartSpecification:
        self._validate_columns(
            dataframe,
            [x_column, y_column],
        )

        data = dataframe[
            [x_column, y_column]
        ].dropna()

        return ChartSpecification(
            chart_type="bar",
            title=title,
            data=data.to_dict(orient="records"),
            x_axis=x_column,
            y_axis=y_column,
        )

    def histogram(
        self,
        dataframe: pd.DataFrame,
        column: str,
        bins: int = 10,
        title: str = "Histogram",
    ) -> ChartSpecification:
        self._validate_columns(
            dataframe,
            [column],
        )

        if bins <= 0:
            raise ValueError(
                "Number of bins must be greater than zero."
            )

        values = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        ).dropna()

        if values.empty:
            raise ValueError(
                f"Column '{column}' contains no numeric values."
            )

        counts, edges = pd.cut(
            values,
            bins=bins,
            include_lowest=True,
            retbins=True,
        )

        frequency = (
            counts.value_counts(sort=False)
            .reset_index()
        )

        frequency.columns = [
            "bin",
            "count",
        ]

        frequency["bin"] = frequency["bin"].astype(str)

        return ChartSpecification(
            chart_type="histogram",
            title=title,
            data=frequency.to_dict(
                orient="records"
            ),
            x_axis="bin",
            y_axis="count",
            metadata={
                "bins": bins,
                "column": column,
            },
        )

    def scatter_chart(
        self,
        dataframe: pd.DataFrame,
        x_column: str,
        y_column: str,
        title: str = "Scatter Plot",
    ) -> ChartSpecification:
        self._validate_columns(
            dataframe,
            [x_column, y_column],
        )

        data = dataframe[
            [x_column, y_column]
        ].dropna()

        return ChartSpecification(
            chart_type="scatter",
            title=title,
            data=data.to_dict(orient="records"),
            x_axis=x_column,
            y_axis=y_column,
        )

    def forecast_chart(
        self,
        forecast: pd.DataFrame,
        date_column: str = "ds",
        forecast_column: str = "yhat",
        lower_column: str = "yhat_lower",
        upper_column: str = "yhat_upper",
        title: str = "Forecast",
    ) -> ChartSpecification:
        self._validate_columns(
            forecast,
            [
                date_column,
                forecast_column,
                lower_column,
                upper_column,
            ],
        )

        data = forecast[
            [
                date_column,
                forecast_column,
                lower_column,
                upper_column,
            ]
        ].dropna()

        return ChartSpecification(
            chart_type="forecast",
            title=title,
            data=data.to_dict(
                orient="records"
            ),
            x_axis=date_column,
            y_axis=forecast_column,
            metadata={
                "lower_bound": lower_column,
                "upper_bound": upper_column,
            },
        )

    def anomaly_chart(
        self,
        dataframe: pd.DataFrame,
        x_column: str,
        value_column: str,
        anomaly_column: str = "is_anomaly",
        title: str = "Anomaly Detection",
    ) -> ChartSpecification:
        self._validate_columns(
            dataframe,
            [
                x_column,
                value_column,
                anomaly_column,
            ],
        )

        data = dataframe[
            [
                x_column,
                value_column,
                anomaly_column,
            ]
        ].dropna()

        return ChartSpecification(
            chart_type="anomaly",
            title=title,
            data=data.to_dict(
                orient="records"
            ),
            x_axis=x_column,
            y_axis=value_column,
            metadata={
                "anomaly_column": anomaly_column,
            },
        )

    def kpi_card(
        self,
        name: str,
        value: float | int | str,
        unit: str | None = None,
        title: str | None = None,
    ) -> ChartSpecification:
        return ChartSpecification(
            chart_type="kpi",
            title=title or name,
            data=[
                {
                    "name": name,
                    "value": value,
                    "unit": unit,
                }
            ],
            metadata={
                "component": "kpi_card",
            },
        )

    def create(
        self,
        chart_type: str,
        dataframe: pd.DataFrame,
        **kwargs: Any,
    ) -> ChartSpecification:
        chart_type = chart_type.lower().strip()

        self._validate_chart_type(chart_type)

        builders = {
            "line": self.line_chart,
            "bar": self.bar_chart,
            "histogram": self.histogram,
            "scatter": self.scatter_chart,
            "forecast": self.forecast_chart,
            "anomaly": self.anomaly_chart,
        }

        if chart_type == "kpi":
            return self.kpi_card(**kwargs)

        return builders[chart_type](
            dataframe,
            **kwargs,
        )

    def to_dict(
        self,
        specification: ChartSpecification,
    ) -> dict[str, Any]:
        return {
            "chart_type": specification.chart_type,
            "title": specification.title,
            "data": specification.data,
            "x_axis": specification.x_axis,
            "y_axis": specification.y_axis,
            "metadata": specification.metadata or {},
        }