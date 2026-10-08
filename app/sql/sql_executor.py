import pandas as pd

from app.sql.sql_validator import validate_sql
from app.storage.duckdb_manager import DuckDBManager


class SQLExecutor:

    def __init__(
        self,
        database_path: str,
        max_rows: int = 10000,
    ):
        self.database = DuckDBManager(
            database_path
        )

        self.max_rows = max_rows

    def execute(
        self,
        query: str,
    ) -> pd.DataFrame:

        validated_query = validate_sql(query)

        result = self.database.execute_query(
            validated_query
        )

        if len(result) > self.max_rows:
            raise ValueError(
                f"Query returned {len(result)} rows. "
                f"Maximum allowed is {self.max_rows}."
            )

        return result