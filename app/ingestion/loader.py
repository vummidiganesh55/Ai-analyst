from pathlib import Path

import pandas as pd

from app.config.settings import settings


ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".parquet"}


def load_dataset(path: str) -> pd.DataFrame:
    file_path = Path(path)
    extension = file_path.suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {extension}")

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not file_path.is_file():
        raise ValueError("Path must point to a regular file.")

    if file_path.stat().st_size == 0:
        raise ValueError("File must not be empty.")

    if extension == ".csv":
        with file_path.open("rb") as file:
            sample = file.read(8192)

        if not sample or not sample.strip(b"\x00 \t\r\n"):
            raise ValueError("CSV file does not contain readable data.")

        dataframe = pd.read_csv(file_path)

    elif extension in {".xlsx", ".xls"}:
        dataframe = pd.read_excel(file_path)

    else:
        dataframe = pd.read_parquet(file_path)

    if dataframe.empty:
        raise ValueError("Dataset must contain at least one data row.")

    if len(dataframe.columns) == 0:
        raise ValueError("Dataset must contain at least one column.")

    if any(
        not isinstance(column, str) or not column.strip()
        for column in dataframe.columns
    ):
        raise ValueError("Dataset contains an empty or invalid column name.")

    if dataframe.columns.duplicated().any():
        raise ValueError("Dataset contains duplicate column names.")

    # Validate the decoded dataset against configured limits.
    if len(dataframe) > settings.max_dataset_rows:
        raise ValueError(
            f"Dataset exceeds the maximum of "
            f"{settings.max_dataset_rows} rows."
        )

    if len(dataframe.columns) > settings.max_dataset_columns:
        raise ValueError(
            f"Dataset exceeds the maximum of "
            f"{settings.max_dataset_columns} columns."
        )

    memory_limit_bytes = settings.max_dataset_memory_mb * 1024 * 1024
    dataframe_memory_bytes = int(
        dataframe.memory_usage(index=True, deep=True).sum()
    )

    if dataframe_memory_bytes > memory_limit_bytes:
        raise ValueError(
            f"Dataset exceeds the maximum decoded DataFrame memory "
            f"limit of {settings.max_dataset_memory_mb} MB."
        )

    return dataframe