import pandas as pd

from app.profiling.profiler import DataProfiler


def test_profiler():
    dataframe = pd.DataFrame({
        "revenue": [100, 200, 300],
        "region": ["South", "North", "East"],
    })

    result = DataProfiler().profile(dataframe)

    assert result["row_count"] == 3
    assert result["column_count"] == 2
    assert "revenue" in result["numeric_columns"]
    assert "region" in result["categorical_columns"]


def test_profiler_detects_date_strings():
    dataframe = pd.DataFrame({
        "date": ["2026-01-01", "2026-01-02"],
        "revenue": [100, 200],
    })

    result = DataProfiler().profile(dataframe)

    assert "date" in result["date_columns"]
    assert "revenue" in result["numeric_columns"]


def test_profiler_counts_missing_values():
    dataframe = pd.DataFrame({
        "revenue": [100, None, 300],
        "region": ["South", "North", None],
    })

    result = DataProfiler().profile(dataframe)

    assert result["missing_values"]["revenue"] == 1
    assert result["missing_values"]["region"] == 1


def test_profiler_counts_duplicate_rows():
    dataframe = pd.DataFrame({
        "revenue": [100, 200, 200],
        "region": ["South", "North", "North"],
    })

    result = DataProfiler().profile(dataframe)

    assert result["duplicate_rows"] == 1