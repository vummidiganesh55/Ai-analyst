import pytest

from app.sql.sql_validator import (
    SQLValidationError,
    validate_sql,
)


@pytest.mark.parametrize(
    "query",
    [
        "SELECT * FROM read_csv_auto('sensitive.csv')",
        "SELECT * FROM read_parquet('sensitive.parquet')",
        "SELECT * FROM read_json_auto('sensitive.json')",
    ],
)
def test_external_file_reading_is_blocked(query):
    with pytest.raises(SQLValidationError):
        validate_sql(query)


def test_select_allowed():

    query = validate_sql(
        "SELECT region, SUM(revenue) "
        "FROM sales "
        "GROUP BY region"
    )

    assert query.startswith("SELECT")


def test_delete_blocked():

    with pytest.raises(SQLValidationError):
        validate_sql(
            "DELETE FROM sales"
        )


def test_drop_blocked():

    with pytest.raises(SQLValidationError):
        validate_sql(
            "DROP TABLE sales"
        )


def test_update_blocked():

    with pytest.raises(SQLValidationError):
        validate_sql(
            "UPDATE sales SET revenue = 0"
        )


def test_insert_blocked():

    with pytest.raises(SQLValidationError):
        validate_sql(
            "INSERT INTO sales VALUES (1)"
        )


def test_alter_blocked():

    with pytest.raises(SQLValidationError):
        validate_sql(
            "ALTER TABLE sales ADD COLUMN test INT"
        )


def test_multiple_statements_blocked():

    with pytest.raises(SQLValidationError):
        validate_sql(
            "SELECT * FROM sales; "
            "DROP TABLE sales"
        )


def test_empty_query_blocked():

    with pytest.raises(SQLValidationError):
        validate_sql("")

@pytest.mark.parametrize(
    "query",
    [
        # Multiple statements with different separators.
        "SELECT 1; SELECT 2",

        # Additional write operations.
        "CREATE TABLE unauthorized_table (id INTEGER)",
        "CREATE VIEW unauthorized_view AS SELECT 1",
        "COPY sales TO 'output.csv'",
        "EXPORT DATABASE 'export_directory'",

        # Attempts to access external files through table functions.
        "SELECT * FROM read_text('sensitive.txt')",
        "SELECT * FROM read_blob('sensitive.bin')",
        "SELECT * FROM parquet_scan('sensitive.parquet')",
        "SELECT * FROM glob('*')",
    ],
)
def test_additional_unsafe_sql_is_blocked(query):
    with pytest.raises(SQLValidationError):
        validate_sql(query)


@pytest.mark.parametrize(
    "query",
    [
        "SELECT 1",
        "SELECT region FROM sales",
        "SELECT SUM(revenue) FROM sales",
        "SELECT region, SUM(revenue) FROM sales GROUP BY region",
    ],
)
def test_valid_select_queries_are_allowed(query):
    validated_query = validate_sql(query)

    assert validated_query
    assert validated_query.upper().startswith("SELECT")