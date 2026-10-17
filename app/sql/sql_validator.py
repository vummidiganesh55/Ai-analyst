import sqlglot
from sqlglot import exp


class SQLValidationError(Exception):
    """Raised when SQL violates security policy."""


# SQL operations that must never be executed.
BLOCKED_EXPRESSIONS = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Alter,
    exp.Create,
    exp.Merge,
)


# Functions that can access external files or execute
# dynamically supplied SQL.
BLOCKED_FUNCTIONS = {
    "READ_CSV",
    "READ_CSV_AUTO",
    "READ_JSON",
    "READ_JSON_AUTO",
    "READ_NDJSON",
    "READ_PARQUET",
    "READ_TEXT",
    "READ_BLOB",
    "PARQUET_SCAN",
    "CSV_SCAN",
    "JSON_SCAN",
    "GLOB",
    "QUERY",
    "QUERY_TABLE",
}


def _get_function_name(node: exp.Func) -> str:
    """
    Extract the function name from a SQLGlot function node.

    SQLGlot represents some DuckDB functions as Anonymous
    nodes and others as dedicated expression classes.
    """

    if isinstance(node, exp.Anonymous):
        return str(node.args.get("this", "")).upper()

    return node.sql_name().upper()


def validate_sql(query: str) -> str:
    """
    Validate SQL before execution.

    Security policy:
    - Query must be a non-empty string.
    - Exactly one SQL statement is allowed.
    - Only SELECT statements are allowed.
    - Write and DDL operations are blocked.
    - Listed external-file and dynamic-query functions
      are blocked.
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
            "Invalid SQL syntax."
        ) from exc

    if len(statements) != 1 or statements[0] is None:
        raise SQLValidationError(
            "Only one SQL statement is allowed."
        )

    statement = statements[0]

    if not isinstance(statement, exp.Select):
        raise SQLValidationError(
            "Only SELECT queries are allowed."
        )

    for node in statement.walk():

        # Reject write and DDL operations.
        if isinstance(node, BLOCKED_EXPRESSIONS):
            raise SQLValidationError(
                f"Blocked SQL operation: "
                f"{type(node).__name__}"
            )

        # Reject external-file and dynamic-query functions.
        if isinstance(node, exp.Func):
            function_name = _get_function_name(node)

            if function_name in BLOCKED_FUNCTIONS:
                raise SQLValidationError(
                    f"Blocked SQL function: {function_name}"
                )

        # Check anonymous functions used as table sources.
        # Examples:
        # SELECT * FROM read_csv_auto('data.csv')
        # SELECT * FROM read_json_auto('data.json')
        elif isinstance(node, exp.Table):
            table_expression = node.this

            if isinstance(table_expression, exp.Func):
                function_name = _get_function_name(
                    table_expression
                )

                if function_name in BLOCKED_FUNCTIONS:
                    raise SQLValidationError(
                        f"Blocked SQL function: {function_name}"
                    )

    return query