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

            # Existing datetime columns
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                date_columns.append(column)

            # Detect date strings using explicit formats.
            elif (
                pd.api.types.is_object_dtype(dtype)
                or pd.api.types.is_string_dtype(dtype)
                or isinstance(dtype, pd.CategoricalDtype)
            ):
                non_null_values = series.dropna().astype(str).str.strip()

                date_formats = (
                    "%Y-%m-%d",
                    "%Y/%m/%d",
                    "%d-%m-%Y",
                    "%d/%m/%Y",
                    "%Y-%m-%d %H:%M:%S",
                    "%Y/%m/%d %H:%M:%S",
                )

                is_date_column = False

                if not non_null_values.empty:
                    for date_format in date_formats:
                        parsed_dates = pd.to_datetime(
                            non_null_values,
                            format=date_format,
                            errors="coerce",
                        )

                        if parsed_dates.notna().all():
                            is_date_column = True
                            break

                if is_date_column:
                    date_columns.append(column)
                else:
                    categorical_columns.append(column)

        return {
            "row_count": len(dataframe),
            "column_count": len(dataframe.columns),
            "columns": dataframe.columns.tolist(),
            "dtypes": {
                column: str(dtype)
                for column, dtype in dataframe.dtypes.items()
            },
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "date_columns": date_columns,
            "missing_values": {
                column: int(value)
                for column, value in dataframe.isna().sum().items()
            },
            "duplicate_rows": int(dataframe.duplicated().sum()),
        }