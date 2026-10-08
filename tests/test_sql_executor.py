import pandas as pd
import pytest

from app.sql.sql_executor import SQLExecutor
from app.sql.sql_validator import SQLValidationError
from app.storage.duckdb_manager import DuckDBManager


def create_test_database(tmp_path):

    database_path = str(
        tmp_path / "test_executor.duckdb"
    )

    dataframe = pd.DataFrame({
        "region": [
            "South",
            "North",
            "East",
            "West",
        ],
        "revenue": [
            100000,
            80000,
            120000,
            90000,
        ],
    })

    database = DuckDBManager(
        database_path
    )

    database.register_dataframe(
        dataframe,
        "sales",
    )

    return database_path


def test_secure_sql_execution(tmp_path):

    database_path = create_test_database(
        tmp_path
    )

    executor = SQLExecutor(
        database_path=database_path
    )

    result = executor.execute(
        """
        SELECT
            region,
            SUM(revenue) AS total_revenue
        FROM sales
        GROUP BY region
        ORDER BY total_revenue DESC
        """
    )

    assert not result.empty

    assert "region" in result.columns

    assert "total_revenue" in result.columns


def test_write_query_never_reaches_database(tmp_path):

    database_path = create_test_database(
        tmp_path
    )

    executor = SQLExecutor(
        database_path=database_path
    )

    with pytest.raises(SQLValidationError):

        executor.execute(
            "DELETE FROM sales"
        )