import threading
import time

import duckdb


def main():
    connection = duckdb.connect(":memory:")

    print("DuckDB version:", duckdb.__version__)

    if not callable(getattr(connection, "interrupt", None)):
        print("Query interruption API: NOT AVAILABLE")
        connection.close()
        return

    print("Query interruption API: AVAILABLE")

    timer = threading.Timer(
        0.2,
        connection.interrupt,
    )

    start_time = time.perf_counter()
    timer.start()

    try:
        connection.execute(
            """
            SELECT SUM(SIN(i))
            FROM range(1000000000) AS numbers(i)
            """
        ).fetchone()

        elapsed = time.perf_counter() - start_time
        print(f"Query completed without interruption in {elapsed:.2f}s")

    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        print("Query interruption result: ERROR RECEIVED")
        print("Exception:", type(exc).__name__)
        print("Details:", str(exc))
        print(f"Elapsed time: {elapsed:.2f}s")

    finally:
        timer.cancel()
        connection.close()


if __name__ == "__main__":
    main()