from pathlib import Path

import pandas as pd


def load_dataset(path: str) -> pd.DataFrame:
    file_path = Path(path)

    extension = file_path.suffix.lower()

    if extension == ".csv":
        return pd.read_csv(file_path)

    if extension in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)

    if extension == ".parquet":
        return pd.read_parquet(file_path)

    raise ValueError(f"Unsupported file type: {extension}")