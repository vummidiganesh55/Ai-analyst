import pandas as pd


class DataProfiler:

    def profile(self, dataframe: pd.DataFrame) -> dict:

        numeric_columns = []

        categorical_columns = []

        date_columns = []

        for column in dataframe.columns:

            series = dataframe[column]
            dtype = series.dtype

            # Numeric columns
            if pd.api.types.is_numeric_dtype(dtype):
                numeric_columns.append(column)

            # Date columns
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                date_columns.append(column)

            # Categorical / text columns
            elif (
                pd.api.types.is_object_dtype(dtype)
                or pd.api.types.is_string_dtype(dtype)
                or pd.api.types.is_categorical_dtype(dtype)
            ):
                categorical_columns.append(column)

        return {
            "row_count": len(dataframe),

            "column_count": len(
                dataframe.columns
            ),

            "columns": dataframe.columns.tolist(),

            "dtypes": {
                column: str(dtype)
                for column, dtype
                in dataframe.dtypes.items()
            },

            "numeric_columns": numeric_columns,

            "categorical_columns": categorical_columns,

            "date_columns": date_columns,

            "missing_values": {
                column: int(value)
                for column, value
                in dataframe.isna().sum().items()
            },

            "duplicate_rows": int(
                dataframe.duplicated().sum()
            ),
        }