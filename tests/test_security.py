import pytest

from app.sql.sql_validator import (
    validate_sql,
    SQLValidationError,
)


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