import duckdb
import pytest

from app.storage.duckdb_manager import DuckDBManager


@pytest.fixture
def database_manager(tmp_path):
    database_path = str(tmp_path / "test.duckdb")

    return DuckDBManager(
        database_path=database_path,
        memory_limit="512MB",
        threads=2,
        max_expression_depth=500,
    )


def test_memory_limit_is_applied(database_manager):
    with database_manager.connection() as conn:
        result = conn.execute(
            """
            SELECT value
            FROM duckdb_settings()
            WHERE name = 'memory_limit'
            """
        ).fetchone()

    assert result is not None
    assert result[0] == "488.2 MiB"


def test_thread_limit_is_applied(database_manager):
    with database_manager.connection() as conn:
        result = conn.execute(
            """
            SELECT value
            FROM duckdb_settings()
            WHERE name = 'threads'
            """
        ).fetchone()

    assert result is not None
    assert int(result[0]) == 2


def test_expression_depth_limit_is_applied(database_manager):
    with database_manager.connection() as conn:
        result = conn.execute(
            """
            SELECT value
            FROM duckdb_settings()
            WHERE name = 'max_expression_depth'
            """
        ).fetchone()

    assert result is not None
    assert int(result[0]) == 500


def test_external_access_is_disabled(database_manager):
    with database_manager.connection() as conn:
        result = conn.execute(
            """
            SELECT value
            FROM duckdb_settings()
            WHERE name = 'enable_external_access'
            """
        ).fetchone()

    assert result is not None
    assert result[0].lower() == "false"


@pytest.mark.parametrize(
    "configuration",
    [
        {"threads": 0},
        {"max_expression_depth": 0},
    ],
)
def test_invalid_resource_limits_are_rejected(
    tmp_path,
    configuration,
):
    database_path = str(tmp_path / "invalid.duckdb")

    with pytest.raises(ValueError):
        DuckDBManager(
            database_path=database_path,
            **configuration,
        )