from pathlib import Path
import threading

import duckdb
import pandas as pd

from app.config.settings import settings


class QueryTimeoutError(TimeoutError):
    """Raised when a DuckDB query exceeds its execution deadline."""


class DuckDBManager:
    def __init__(
        self,
        database_path: str,
        memory_limit: str | None = None,
        threads: int | None = None,
        max_expression_depth: int | None = None,
        query_timeout_seconds: float | None = None,
    ):
        Path(database_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.database_path = database_path
        self.memory_limit = memory_limit or settings.duckdb_memory_limit
        self.threads = (
            settings.duckdb_threads if threads is None else threads
        )
        self.max_expression_depth = (
            settings.duckdb_max_expression_depth
            if max_expression_depth is None
            else max_expression_depth
        )
        self.query_timeout_seconds = (
            settings.query_timeout_seconds
            if query_timeout_seconds is None
            else query_timeout_seconds
        )

        if self.threads < 1:
            raise ValueError("threads must be at least 1.")

        if self.max_expression_depth < 1:
            raise ValueError(
                "max_expression_depth must be at least 1."
            )

        if self.query_timeout_seconds <= 0:
            raise ValueError(
                "query_timeout_seconds must be greater than 0."
            )

    def connection(self):
        conn = duckdb.connect(self.database_path)

        try:
            # Apply resource controls to every new connection.
            conn.execute("SET memory_limit = ?", [self.memory_limit])
            conn.execute("SET threads = ?", [self.threads])
            conn.execute(
                "SET max_expression_depth = ?",
                [self.max_expression_depth],
            )
            conn.execute("SET enable_external_access = false")
        except Exception:
            conn.close()
            raise

        return conn

    def register_dataframe(
        self,
        dataframe: pd.DataFrame,
        table_name: str,
    ) -> None:
        if not isinstance(table_name, str) or not table_name.strip():
            raise ValueError("A valid table name is required.")

        escaped_table_name = table_name.replace('"', '""')

        with self.connection() as conn:
            conn.register("source_dataframe", dataframe)
            conn.execute(
                f'''
                CREATE OR REPLACE TABLE "{escaped_table_name}" AS
                SELECT * FROM source_dataframe
                '''
            )

    def execute_query(
        self,
        query: str,
        max_rows: int | None = None,
    ) -> pd.DataFrame:
        if max_rows is not None and max_rows < 1:
            raise ValueError("max_rows must be at least 1.")

        conn = self.connection()

        timeout_reached = threading.Event()

        def interrupt_query() -> None:
            timeout_reached.set()

            try:
                conn.interrupt()
            except Exception:
                # The query or connection may already have finished.
                pass

        timer = threading.Timer(
            self.query_timeout_seconds,
            interrupt_query,
        )
        timer.daemon = True

        try:
            timer.start()

            cursor = conn.execute(query)

            if max_rows is None:
                return cursor.df()

            rows = cursor.fetchmany(max_rows + 1)

            if len(rows) > max_rows:
                raise ValueError(
                    f"Query returned more than {max_rows} rows. "
                    "Reduce the result size."
                )

            columns = [
                description[0]
                for description in cursor.description
            ]

            return pd.DataFrame(rows, columns=columns)

        except Exception as exc:
            if timeout_reached.is_set():
                raise QueryTimeoutError(
                    f"Query exceeded the execution timeout of "
                    f"{self.query_timeout_seconds} seconds."
                ) from exc

            raise

        finally:
            timer.cancel()
            conn.close()

    def get_tables(self) -> list[str]:
        with self.connection() as conn:
            result = conn.execute("SHOW TABLES").fetchall()

        return [row[0] for row in result]