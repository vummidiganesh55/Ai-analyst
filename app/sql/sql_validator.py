import sqlglot
from sqlglot import exp


class SQLValidationError(Exception):
    """Raised when SQL violates security policy."""


BLOCKED_EXPRESSIONS = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Alter,
    exp.Create,
    exp.Merge,
)


def validate_sql(query: str) -> str:
    """
    Validate SQL before execution.

    Security policy:
    - Query cannot be empty.
    - Exactly one statement.
    - SELECT queries only.
    - Write/DDL operations are blocked.
    """

    if not isinstance(query, str):
        raise SQLValidationError(
            "SQL query must be a string."
        )

    query = query.strip()

    if not query:
        raise SQLValidationError(
            "SQL query cannot be empty."
        )

    try:
        statements = sqlglot.parse(
            query,
            dialect="duckdb",
        )
    except Exception as exc:
        raise SQLValidationError(
            f"Invalid SQL syntax: {exc}"
        ) from exc

    if len(statements) != 1:
        raise SQLValidationError(
            "Only one SQL statement is allowed."
        )

    statement = statements[0]

    if not isinstance(statement, exp.Select):
        raise SQLValidationError(
            "Only SELECT queries are allowed."
        )

    for node in statement.walk():

        if isinstance(node, BLOCKED_EXPRESSIONS):
            raise SQLValidationError(
                f"Blocked SQL operation: "
                f"{type(node).__name__}"
            )

    return query