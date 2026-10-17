import time

import pytest

from app.storage.duckdb_manager import (
    DuckDBManager,
    QueryTimeoutError,
)


def test_long_query_is_interrupted_by_timeout(tmp_path):
    database_path = str(tmp_path / "timeout.duckdb")

    manager = DuckDBManager(
        database_path=database_path,
        query_timeout_seconds=0.2,
    )

    start_time = time.perf_counter()

    with pytest.raises(QueryTimeoutError, match="execution timeout"):
        manager.execute_query(
            """
            SELECT SUM(SIN(i))
            FROM range(1000000000) AS numbers(i)
            """
        )

    elapsed = time.perf_counter() - start_time

    # Allow scheduling and cleanup overhead.
    assert elapsed < 5.0


def test_fast_query_completes_before_timeout(tmp_path):
    database_path = str(tmp_path / "fast.duckdb")

    manager = DuckDBManager(
        database_path=database_path,
        query_timeout_seconds=5,
    )

    result = manager.execute_query("SELECT 1 AS value")

    assert result.to_dict(orient="records") == [{"value": 1}]


def test_non_positive_query_timeout_is_rejected(tmp_path):
    with pytest.raises(
        ValueError,
        match="query_timeout_seconds must be greater than 0",
    ):
        DuckDBManager(
            database_path=str(tmp_path / "invalid.duckdb"),
            query_timeout_seconds=0,
        )


def test_unrelated_duckdb_error_is_not_misreported_as_timeout(
    tmp_path,
):
    manager = DuckDBManager(
        database_path=str(tmp_path / "error.duckdb"),
        query_timeout_seconds=5,
    )

    with pytest.raises(Exception) as exc_info:
        manager.execute_query(
            "SELECT * FROM nonexistent_timeout_test_table"
        )

    assert not isinstance(exc_info.value, QueryTimeoutError)