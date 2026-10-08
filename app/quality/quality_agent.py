import pandas as pd


class DataQualityAgent:

    def analyze(self, dataframe: pd.DataFrame) -> dict:

        # Missing values
        missing_count = int(
            dataframe.isna().sum().sum()
        )

        # Duplicate rows
        duplicate_count = int(
            dataframe.duplicated().sum()
        )

        # Numeric columns
        numeric_columns = [
            column
            for column in dataframe.columns
            if pd.api.types.is_numeric_dtype(
                dataframe[column].dtype
            )
        ]

        # Negative values
        negative_values = {}

        for column in numeric_columns:

            count = int(
                (dataframe[column] < 0).sum()
            )

            if count > 0:
                negative_values[column] = count

        # Total cells
        total_cells = (
            dataframe.shape[0]
            * dataframe.shape[1]
        )

        # Quality score
        if total_cells == 0:
            quality_score = 0.0
        else:
            total_issues = (
                missing_count
                + duplicate_count
                + sum(negative_values.values())
            )

            quality_score = max(
                0.0,
                100.0
                * (1 - total_issues / total_cells),
            )

        return {
            "missing_values": missing_count,
            "duplicate_rows": duplicate_count,
            "negative_values": negative_values,
            "quality_score": round(
                quality_score,
                2,
            ),
        }