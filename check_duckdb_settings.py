import duckdb

connection = duckdb.connect(":memory:")

print("DuckDB version:", duckdb.__version__)
print(
    connection.execute(
        """
        SELECT name, value
        FROM duckdb_settings()
        WHERE name IN (
            'memory_limit',
            'threads',
            'max_expression_depth',
            'enable_external_access'
        )
        ORDER BY name
        """
    ).fetchall()
)

connection.close()