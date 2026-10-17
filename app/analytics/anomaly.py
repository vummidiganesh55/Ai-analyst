from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


@dataclass
class AnomalyResult:
    index: Any
    value: float
    method: str
    score: float
    severity: str
    is_anomaly: bool
    explanation: str


class AnomalyDetector:
    """Deterministic anomaly detection engine."""

    def _validate_column(
        self,
        dataframe: pd.DataFrame,
        column: str,
    ) -> pd.Series:
        if column not in dataframe.columns:
            raise ValueError(f"Column not found: {column}")

        series = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

        if series.dropna().empty:
            raise ValueError(
                f"Column '{column}' contains no valid numeric values."
            )

        return series

    def iqr_bounds(
        self,
        dataframe: pd.DataFrame,
        column: str,
    ) -> dict[str, float]:
        series = self._validate_column(
            dataframe,
            column,
        ).dropna()

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))

        iqr = q3 - q1

        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        return {
            "q1": q1,
            "q3": q3,
            "iqr": float(iqr),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
        }

    def detect_iqr(
        self,
        dataframe: pd.DataFrame,
        column: str,
    ) -> list[AnomalyResult]:
        series = self._validate_column(
            dataframe,
            column,
        )

        bounds = self.iqr_bounds(
            dataframe,
            column,
        )

        lower = bounds["lower_bound"]
        upper = bounds["upper_bound"]

        results: list[AnomalyResult] = []

        for index, value in series.items():
            if pd.isna(value):
                continue

            value = float(value)

            is_anomaly = value < lower or value > upper

            if not is_anomaly:
                continue

            distance = (
                lower - value
                if value < lower
                else value - upper
            )

            scale = max(bounds["iqr"], 1e-12)

            score = float(distance / scale)

            severity = self._severity(score)

            direction = (
                "below the expected range"
                if value < lower
                else "above the expected range"
            )

            explanation = (
                f"Value {value:g} is {direction} "
                f"based on the IQR bounds "
                f"[{lower:g}, {upper:g}]."
            )

            results.append(
                AnomalyResult(
                    index=index,
                    value=value,
                    method="iqr",
                    score=score,
                    severity=severity,
                    is_anomaly=True,
                    explanation=explanation,
                )
            )

        return results

    def detect_isolation_forest(
        self,
        dataframe: pd.DataFrame,
        column: str,
        contamination: float = 0.05,
        random_state: int = 42,
    ) -> list[AnomalyResult]:
        series = self._validate_column(
            dataframe,
            column,
        )

        valid = series.dropna()

        if len(valid) < 5:
            raise ValueError(
                "Isolation Forest requires at least 5 observations."
            )

        if not 0 < contamination < 0.5:
            raise ValueError(
                "Contamination must be between 0 and 0.5."
            )

        model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
        )

        values = valid.to_numpy().reshape(-1, 1)

        predictions = model.fit_predict(values)
        raw_scores = model.decision_function(values)

        results: list[AnomalyResult] = []

        for index, prediction, raw_score, value in zip(
            valid.index,
            predictions,
            raw_scores,
            valid.to_numpy(),
        ):
            is_anomaly = prediction == -1

            if not is_anomaly:
                continue

            score = float(max(0.0, -raw_score))

            severity = self._severity(
                score,
                isolation_forest=True,
            )

            explanation = (
                f"Value {float(value):g} was identified as anomalous "
                f"by Isolation Forest."
            )

            results.append(
                AnomalyResult(
                    index=index,
                    value=float(value),
                    method="isolation_forest",
                    score=score,
                    severity=severity,
                    is_anomaly=True,
                    explanation=explanation,
                )
            )

        return results

    def _severity(
        self,
        score: float,
        isolation_forest: bool = False,
    ) -> str:
        if isolation_forest:
            if score >= 0.20:
                return "high"

            if score >= 0.10:
                return "medium"

            return "low"

        if score >= 2.0:
            return "high"

        if score >= 1.0:
            return "medium"

        return "low"

    def detect(
        self,
        dataframe: pd.DataFrame,
        column: str,
        method: str = "iqr",
    ) -> list[AnomalyResult]:
        method = method.lower().strip()

        if method == "iqr":
            return self.detect_iqr(
                dataframe,
                column,
            )

        if method in {
            "isolation_forest",
            "isolation-forest",
        }:
            return self.detect_isolation_forest(
                dataframe,
                column,
            )

        raise ValueError(
            "Unsupported anomaly detection method: "
            f"{method}"
        )

    def to_dataframe(
        self,
        results: list[AnomalyResult],
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "index": result.index,
                    "value": result.value,
                    "method": result.method,
                    "score": result.score,
                    "severity": result.severity,
                    "is_anomaly": result.is_anomaly,
                    "explanation": result.explanation,
                }
                for result in results
            ]
        )