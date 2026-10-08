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