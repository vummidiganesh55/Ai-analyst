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
@pytest.mark.parametrize(
    "query",
    [
        "",
        "   ",
        "INSERT INTO sales VALUES ('South', 100)",
        "UPDATE sales SET revenue = 0",
        "DROP TABLE sales",
        "ALTER TABLE sales ADD COLUMN test INTEGER",
        "SELECT * FROM sales; DELETE FROM sales",
        "SELEC revenue FROM sales",
    ],
)
def test_invalid_sql_is_rejected_before_execution(tmp_path, query):
    database_path = create_test_database(tmp_path)
    executor = SQLExecutor(database_path=database_path)

    with pytest.raises(SQLValidationError):
        executor.execute(query)

def test_query_within_row_limit_succeeds(tmp_path):
    database_path = create_test_database(tmp_path)

    executor = SQLExecutor(
        database_path=database_path,
        max_rows=4,
    )

    result = executor.execute("SELECT * FROM sales")

    assert len(result) == 4


def test_query_exactly_at_row_limit_succeeds(tmp_path):
    database_path = create_test_database(tmp_path)

    executor = SQLExecutor(
        database_path=database_path,
        max_rows=4,
    )

    result = executor.execute("SELECT * FROM sales LIMIT 4")

    assert len(result) == 4


def test_query_exceeding_row_limit_is_rejected(tmp_path):
    database_path = create_test_database(tmp_path)

    executor = SQLExecutor(
        database_path=database_path,
        max_rows=3,
    )

    with pytest.raises(ValueError, match="more than 3 rows"):
        executor.execute("SELECT * FROM sales")


@pytest.mark.parametrize("max_rows", [0, -1])
def test_invalid_row_limit_is_rejected(tmp_path, max_rows):
    database_path = create_test_database(tmp_path)

    with pytest.raises(ValueError, match="max_rows must be at least 1"):
        SQLExecutor(
            database_path=database_path,
            max_rows=max_rows,
        )


def test_sql_validation_occurs_before_database_execution(
    tmp_path,
    monkeypatch,
):
    database_path = create_test_database(tmp_path)

    executor = SQLExecutor(database_path=database_path)

    def fail_if_executed(*args, **kwargs):
        raise AssertionError(
            "Database execution must not occur for invalid SQL."
        )

    monkeypatch.setattr(
        executor.database,
        "execute_query",
        fail_if_executed,
    )

    with pytest.raises(SQLValidationError):
        executor.execute("DELETE FROM sales")