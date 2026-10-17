import pandas as pd
import pytest

from app.analytics.anomaly import AnomalyDetector


@pytest.fixture
def detector():
    return AnomalyDetector()


@pytest.fixture
def normal_data():
    return pd.DataFrame(
        {
            "revenue": [
                100,
                105,
                110,
                115,
                120,
                125,
                130,
                135,
                140,
                145,
            ]
        }
    )


@pytest.fixture
def anomaly_data():
    return pd.DataFrame(
        {
            "revenue": [
                100,
                105,
                110,
                115,
                120,
                125,
                130,
                135,
                140,
                1000,
            ]
        }
    )


def test_iqr_bounds(detector, normal_data):
    result = detector.iqr_bounds(
        normal_data,
        "revenue",
    )

    assert result["q1"] == pytest.approx(111.25)
    assert result["q3"] == pytest.approx(133.75)
    assert result["iqr"] == pytest.approx(22.5)


def test_iqr_detects_anomaly(detector, anomaly_data):
    results = detector.detect_iqr(
        anomaly_data,
        "revenue",
    )

    assert len(results) == 1
    assert results[0].value == 1000
    assert results[0].method == "iqr"
    assert results[0].is_anomaly is True


def test_iqr_anomaly_has_explanation(detector, anomaly_data):
    results = detector.detect_iqr(
        anomaly_data,
        "revenue",
    )

    assert results[0].explanation
    assert "IQR" in results[0].explanation


def test_iqr_anomaly_has_severity(detector, anomaly_data):
    results = detector.detect_iqr(
        anomaly_data,
        "revenue",
    )

    assert results[0].severity in {
        "low",
        "medium",
        "high",
    }


def test_iqr_no_anomaly(detector, normal_data):
    results = detector.detect_iqr(
        normal_data,
        "revenue",
    )

    assert results == []


def test_isolation_forest_detects_anomaly(
    detector,
    anomaly_data,
):
    results = detector.detect_isolation_forest(
        anomaly_data,
        "revenue",
        contamination=0.1,
    )

    assert len(results) >= 1
    assert any(
        result.value == 1000
        for result in results
    )


def test_isolation_forest_method(
    detector,
    anomaly_data,
):
    results = detector.detect(
        anomaly_data,
        "revenue",
        method="isolation_forest",
    )

    assert all(
        result.method == "isolation_forest"
        for result in results
    )


def test_detect_iqr_dispatch(
    detector,
    anomaly_data,
):
    results = detector.detect(
        anomaly_data,
        "revenue",
        method="iqr",
    )

    assert len(results) == 1
    assert results[0].method == "iqr"


def test_invalid_method(
    detector,
    anomaly_data,
):
    with pytest.raises(
        ValueError,
        match="Unsupported anomaly detection method",
    ):
        detector.detect(
            anomaly_data,
            "revenue",
            method="unknown",
        )


def test_invalid_column(
    detector,
    anomaly_data,
):
    with pytest.raises(
        ValueError,
        match="Column not found",
    ):
        detector.detect_iqr(
            anomaly_data,
            "unknown",
        )


def test_isolation_forest_small_dataset(detector):
    dataframe = pd.DataFrame(
        {
            "revenue": [100, 110, 120]
        }
    )

    with pytest.raises(
        ValueError,
        match="at least 5 observations",
    ):
        detector.detect_isolation_forest(
            dataframe,
            "revenue",
        )


def test_invalid_contamination(
    detector,
    anomaly_data,
):
    with pytest.raises(
        ValueError,
        match="between 0 and 0.5",
    ):
        detector.detect_isolation_forest(
            anomaly_data,
            "revenue",
            contamination=0.8,
        )


def test_to_dataframe(
    detector,
    anomaly_data,
):
    results = detector.detect_iqr(
        anomaly_data,
        "revenue",
    )

    result_df = detector.to_dataframe(results)

    assert len(result_df) == 1
    assert "value" in result_df.columns
    assert "method" in result_df.columns
    assert "score" in result_df.columns
    assert "severity" in result_df.columns
    assert "explanation" in result_df.columns