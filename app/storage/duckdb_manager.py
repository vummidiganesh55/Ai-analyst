from pathlib import Path

import duckdb
import pandas as pd


class DuckDBManager:

    def __init__(self, database_path: str):
        Path(database_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.database_path = database_path

    def connection(self):
        return duckdb.connect(self.database_path)

    def register_dataframe(
        self,
        dataframe: pd.DataFrame,
        table_name: str,
    ) -> None:

        with self.connection() as conn:
            conn.register("source_dataframe", dataframe)

            conn.execute(
                f"""
                CREATE OR REPLACE TABLE "{table_name}" AS
                SELECT * FROM source_dataframe
                """
            )

    def execute_query(self, query: str) -> pd.DataFrame:

        with self.connection() as conn:
            return conn.execute(query).df()

    def get_tables(self) -> list[str]:

        with self.connection() as conn:
            result = conn.execute(
                "SHOW TABLES"
            ).fetchall()

        return [row[0] for row in result]