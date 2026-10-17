import pandas as pd

from app.config.settings import settings
from app.sql.sql_validator import validate_sql
from app.storage.duckdb_manager import DuckDBManager


class SQLExecutor:
    def __init__(
        self,
        database_path: str,
        max_rows: int | None = None,
    ):
        self.max_rows = (
            settings.max_result_rows
            if max_rows is None
            else max_rows
        )

        if self.max_rows < 1:
            raise ValueError("max_rows must be at least 1.")

        self.database = DuckDBManager(database_path)

    def execute(self, query: str) -> pd.DataFrame:
        validated_query = validate_sql(query)

        dataframe = self.database.execute_query(
            validated_query,
            max_rows=self.max_rows,
        )

        # Defense in depth: verify the returned DataFrame.
        if len(dataframe) > self.max_rows:
            raise ValueError(
                f"Query returned {len(dataframe)} rows. "
                f"Maximum allowed is {self.max_rows}."
            )

        return dataframe

    def read_table(self, table_name: str) -> pd.DataFrame:
        if not isinstance(table_name, str) or not table_name.strip():
            raise ValueError("A valid table name is required.")

        escaped_table_name = table_name.replace('"', '""')
        query = f'SELECT * FROM "{escaped_table_name}"'

        return self.execute(query)