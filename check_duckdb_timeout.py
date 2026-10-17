import duckdb

print("DuckDB version:", duckdb.__version__)

connection = duckdb.connect(":memory:")

try:
    connection.execute("SET statement_timeout = '10s'")
    print("Statement timeout: supported")
except Exception as exc:
    print("Statement timeout: NOT supported")
    print("Reason:", exc)
finally:
    connection.close()