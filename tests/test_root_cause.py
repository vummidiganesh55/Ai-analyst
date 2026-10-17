import pandas as pd
import pytest

from app.agents.root_cause import RootCauseAnalyzer


@pytest.fixture
def sales_data():
    return pd.DataFrame(
        {
            "period": [
                "previous",
                "previous",
                "current",
                "current",
            ],
            "region": [
                "North",
                "South",
                "North",
                "South",
            ],
            "product": [
                "Laptop",
                "Phone",
                "Laptop",
                "Phone",
            ],
            "revenue": [
                100,
                200,
                70,
                230,
            ],
        }
    )


@pytest.fixture
def analyzer(sales_data):
    return RootCauseAnalyzer(
        dataframe=sales_data,
        dimension_columns=["region", "product"],
        metric_column="revenue",
        period_column="period",
        previous_period="previous",
        current_period="current",
    )


def test_initialization(analyzer):
    assert analyzer.metric_column == "revenue"


def test_overall_change(analyzer):
    result = analyzer.overall_change()

    assert result["previous_value"] == 300
    assert result["current_value"] == 300
    assert result["change"] == 0


def test_dimension_analysis(analyzer):
    results = analyzer.analyze_dimension("region")

    assert len(results) == 2
    assert all(
        result.dimension == "region"
        for result in results
    )


def test_segment_change(analyzer):
    results = analyzer.analyze_dimension("region")

    north = next(
        result
        for result in results
        if result.segment == "North"
    )

    assert north.previous_value == 100
    assert north.current_value == 70
    assert north.change == -30
    assert north.direction == "decrease"


def test_invalid_dimension(analyzer):
    with pytest.raises(ValueError):
        analyzer.analyze_dimension("customer")


def test_missing_columns():
    df = pd.DataFrame(
        {"region": ["North"]}
    )

    with pytest.raises(ValueError):
        RootCauseAnalyzer(
            dataframe=df,
            dimension_columns=["region"],
            metric_column="revenue",
            period_column="period",
            previous_period="previous",
            current_period="current",
        )


def test_empty_dataframe():
    with pytest.raises(ValueError):
        RootCauseAnalyzer(
            dataframe=pd.DataFrame(),
            dimension_columns=["region"],
            metric_column="revenue",
            period_column="period",
            previous_period="previous",
            current_period="current",
        )


def test_invalid_dataframe_type():
    with pytest.raises(TypeError):
        RootCauseAnalyzer(
            dataframe=[1, 2],
            dimension_columns=["region"],
            metric_column="revenue",
            period_column="period",
            previous_period="previous",
            current_period="current",
        )


def test_non_numeric_metric():
    df = pd.DataFrame(
        {
            "period": ["previous", "current"],
            "region": ["North", "North"],
            "revenue": ["100", "200"],
        }
    )

    with pytest.raises(TypeError):
        RootCauseAnalyzer(
            dataframe=df,
            dimension_columns=["region"],
            metric_column="revenue",
            period_column="period",
            previous_period="previous",
            current_period="current",
        )


def test_same_period_rejected(sales_data):
    with pytest.raises(ValueError):
        RootCauseAnalyzer(
            dataframe=sales_data,
            dimension_columns=["region"],
            metric_column="revenue",
            period_column="period",
            previous_period="previous",
            current_period="previous",
        )


def test_missing_previous_period(sales_data):
    analyzer = RootCauseAnalyzer(
        dataframe=sales_data,
        dimension_columns=["region"],
        metric_column="revenue",
        period_column="period",
        previous_period="missing",
        current_period="current",
    )

    with pytest.raises(ValueError):
        analyzer.analyze()


def test_missing_current_period(sales_data):
    analyzer = RootCauseAnalyzer(
        dataframe=sales_data,
        dimension_columns=["region"],
        metric_column="revenue",
        period_column="period",
        previous_period="previous",
        current_period="missing",
    )

    with pytest.raises(ValueError):
        analyzer.analyze()


def test_complete_analysis(analyzer):
    result = analyzer.analyze()

    assert result["metric"] == "revenue"
    assert "overall" in result
    assert "region" in result["dimensions"]
    assert "product" in result["dimensions"]


def test_revenue_decline_contribution():
    df = pd.DataFrame(
        {
            "period": [
                "previous",
                "previous",
                "current",
                "current",
            ],
            "region": [
                "North",
                "South",
                "North",
                "South",
            ],
            "revenue": [
                100,
                100,
                40,
                60,
            ],
        }
    )

    analyzer = RootCauseAnalyzer(
        dataframe=df,
        dimension_columns=["region"],
        metric_column="revenue",
        period_column="period",
        previous_period="previous",
        current_period="current",
    )

    results = analyzer.analyze_dimension("region")

    north = next(
        result
        for result in results
        if result.segment == "North"
    )

    assert north.change == -60
    assert north.contribution_pct == pytest.approx(60)


def test_zero_baseline_percentage():
    df = pd.DataFrame(
        {
            "period": ["previous", "current"],
            "region": ["North", "North"],
            "revenue": [0, 100],
        }
    )

    analyzer = RootCauseAnalyzer(
        dataframe=df,
        dimension_columns=["region"],
        metric_column="revenue",
        period_column="period",
        previous_period="previous",
        current_period="current",
    )

    result = analyzer.overall_change()

    assert result["change"] == 100
    assert result["change_pct"] is None


def test_results_are_ranked_by_absolute_change():
    df = pd.DataFrame(
        {
            "period": [
                "previous",
                "previous",
                "current",
                "current",
            ],
            "region": [
                "North",
                "South",
                "North",
                "South",
            ],
            "revenue": [
                100,
                100,
                10,
                80,
            ],
        }
    )

    analyzer = RootCauseAnalyzer(
        dataframe=df,
        dimension_columns=["region"],
        metric_column="revenue",
        period_column="period",
        previous_period="previous",
        current_period="current",
    )

    results = analyzer.analyze_dimension("region")

    assert abs(results[0].change) >= abs(results[1].change)


def test_evidence_is_included(analyzer):
    results = analyzer.analyze_dimension("region")

    assert "metric" in results[0].evidence
    assert "interpretation" in results[0].evidence


def test_dataframe_is_not_mutated(sales_data):
    original = sales_data.copy(deep=True)

    RootCauseAnalyzer(
        dataframe=sales_data,
        dimension_columns=["region"],
        metric_column="revenue",
        period_column="period",
        previous_period="previous",
        current_period="current",
    )

    pd.testing.assert_frame_equal(sales_data, original)