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

def test_quality_handles_empty_dataframe():
    dataframe = pd.DataFrame()

    result = DataQualityAgent().analyze(dataframe)

    assert result["missing_values"] == 0
    assert result["duplicate_rows"] == 0
    assert result["negative_values"] == {}
    assert result["quality_score"] == 0.0


def test_quality_score_never_goes_below_zero():
    dataframe = pd.DataFrame({
        "revenue": [-1, -2, -3],
    })

    result = DataQualityAgent().analyze(dataframe)

    assert result["quality_score"] >= 0.0


def test_quality_detects_duplicate_rows():
    dataframe = pd.DataFrame({
        "revenue": [100, 100, 200],
        "region": ["South", "South", "North"],
    })

    result = DataQualityAgent().analyze(dataframe)

    assert result["duplicate_rows"] == 1