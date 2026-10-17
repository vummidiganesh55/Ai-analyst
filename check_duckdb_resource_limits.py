import duckdb

connection = duckdb.connect(":memory:")

settings_to_test = {
    "memory_limit": "512MB",
    "threads": "2",
    "max_expression_depth": "500",
    "enable_external_access": "false",
}

for name, value in settings_to_test.items():
    try:
        connection.execute(
            f"SET {name} = ?",
            [value],
        )
        actual = connection.execute(
            "SELECT value FROM duckdb_settings() WHERE name = ?",
            [name],
        ).fetchone()

        print(f"{name}: supported; value={actual[0]}")
    except Exception as exc:
        print(f"{name}: FAILED; {exc}")

connection.close()