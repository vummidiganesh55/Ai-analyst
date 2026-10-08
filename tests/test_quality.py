import pandas as pd

from app.quality.quality_agent import DataQualityAgent


def test_quality():

    dataframe = pd.DataFrame({
        "revenue": [100, 200, None],
        "orders": [2, 3, 4],
    })

    result = DataQualityAgent().analyze(
        dataframe
    )

    assert result["missing_values"] == 1

    assert result["duplicate_rows"] == 0

    assert result["negative_values"] == {}

    assert result["quality_score"] < 100
def test_quality_detects_negative_values():

    dataframe = pd.DataFrame({
        "revenue": [100, -200, 300],
        "orders": [2, 3, 4],
    })

    result = DataQualityAgent().analyze(
        dataframe
    )

    assert result["negative_values"]["revenue"] == 1