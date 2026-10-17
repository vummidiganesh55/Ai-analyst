import pytest

from app.agents.recommendation import (
    Recommendation,
    RecommendationAgent,
)


@pytest.fixture
def agent():
    return RecommendationAgent()


@pytest.fixture
def revenue_insight():
    return {
        "title": "Revenue declined",
        "observation": "Revenue decreased during the comparison period.",
        "business_impact": "Revenue performance requires investigation.",
        "category": "revenue",
        "confidence": "high",
        "evidence": {
            "previous_revenue": 100000,
            "current_revenue": 80000,
        },
    }


@pytest.fixture
def root_cause_result():
    return {
        "metric": "revenue",
        "overall": {
            "previous_value": 100000,
            "current_value": 80000,
            "change": -20000,
            "change_pct": -20.0,
        },
        "dimensions": {
            "region": [
                {
                    "segment": "South",
                    "change": -15000,
                    "contribution_pct": 75.0,
                },
                {
                    "segment": "North",
                    "change": -5000,
                    "contribution_pct": 25.0,
                },
            ]
        },
        "disclaimer": "Contribution does not prove causality.",
    }


def test_initialization(agent):
    assert agent.max_recommendations == 10


def test_custom_max_recommendations():
    agent = RecommendationAgent(max_recommendations=3)

    assert agent.max_recommendations == 3


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "3"])
def test_invalid_max_recommendations(value):
    with pytest.raises(ValueError):
        RecommendationAgent(max_recommendations=value)


def test_generate_revenue_recommendation(agent, revenue_insight):
    results = agent.generate(insights=[revenue_insight])

    assert len(results) >= 1
    assert any(item.category == "revenue" for item in results)


def test_recommendation_has_required_fields(agent, revenue_insight):
    results = agent.generate(insights=[revenue_insight])

    recommendation = results[0]

    assert recommendation.title
    assert recommendation.problem
    assert recommendation.action
    assert recommendation.rationale
    assert recommendation.priority in {
        "critical",
        "high",
        "medium",
        "low",
    }
    assert recommendation.expected_impact
    assert isinstance(recommendation.evidence, dict)


def test_source_evidence_is_preserved(agent, revenue_insight):
    results = agent.generate(insights=[revenue_insight])

    assert any(
        item.evidence["source_evidence"] == revenue_insight["evidence"]
        for item in results
    )


def test_profit_insight_generates_financial_recommendation(agent):
    insight = {
        "title": "Profit margin declined",
        "observation": "Profit margin is lower than the previous period.",
        "business_impact": "Profitability may be under pressure.",
        "category": "profit_margin",
        "evidence": {"margin_change": -5.0},
    }

    results = agent.generate(insights=[insight])

    assert any(item.category == "financial" for item in results)


def test_data_quality_insight_generates_quality_recommendation(agent):
    insight = {
        "title": "Missing values detected",
        "observation": "Several required values are missing.",
        "business_impact": "Analysis reliability may be affected.",
        "category": "data_quality",
        "evidence": {"missing_value_count": 12},
    }

    results = agent.generate(insights=[insight])

    assert any(item.category == "data_quality" for item in results)


def test_unknown_category_uses_general_recommendation(agent):
    insight = {
        "title": "Unusual business pattern",
        "observation": "An unusual pattern requires validation.",
        "business_impact": "Impact is not quantified.",
        "category": "unknown",
        "evidence": {},
    }

    results = agent.generate(insights=[insight])

    assert len(results) == 1
    assert results[0].category == "unknown"


def test_root_cause_decline_generates_high_priority_recommendation(
    agent,
    root_cause_result,
):
    results = agent.generate(root_cause=root_cause_result)

    assert len(results) == 1
    assert results[0].priority == "high"
    assert results[0].category == "root_cause"


def test_root_cause_increase_generates_medium_priority_recommendation(agent):
    root_cause = {
        "metric": "revenue",
        "overall": {
            "previous_value": 100000,
            "current_value": 120000,
            "change": 20000,
        },
    }

    results = agent.generate(root_cause=root_cause)

    assert len(results) == 1
    assert results[0].priority == "medium"


def test_zero_change_generates_low_priority_recommendation(agent):
    root_cause = {
        "metric": "orders",
        "overall": {
            "previous_value": 100,
            "current_value": 100,
            "change": 0,
        },
    }

    results = agent.generate(root_cause=root_cause)

    assert len(results) == 1
    assert results[0].priority == "low"


def test_missing_root_cause_overall_returns_no_recommendation(agent):
    results = agent.generate(
        root_cause={"metric": "revenue"}
    )

    assert results == []


def test_none_insights_returns_empty_list(agent):
    assert agent.generate() == []


def test_invalid_insights_type_raises_error(agent):
    with pytest.raises(TypeError):
        agent.generate(insights={"title": "invalid"})


def test_invalid_root_cause_type_raises_error(agent):
    with pytest.raises(TypeError):
        agent.generate(root_cause=["invalid"])


def test_none_insight_is_skipped(agent):
    results = agent.generate(insights=[None])

    assert results == []


def test_recommendations_are_sorted_by_priority(agent):
    insights = [
        {
            "title": "General observation",
            "observation": "A general issue was identified.",
            "category": "unknown",
        },
        {
            "title": "Revenue observation",
            "observation": "Revenue performance requires investigation.",
            "category": "revenue",
        },
    ]

    results = agent.generate(insights=insights)

    assert results[0].priority == "high"
    assert results[-1].priority == "medium"


def test_max_recommendations_limit_is_applied():
    agent = RecommendationAgent(max_recommendations=1)

    insights = [
        {
            "title": "Revenue issue",
            "observation": "Revenue changed.",
            "category": "revenue",
        },
        {
            "title": "Profit issue",
            "observation": "Profit changed.",
            "category": "profit",
        },
    ]

    results = agent.generate(insights=insights)

    assert len(results) == 1


def test_duplicate_recommendation_titles_are_removed(agent):
    insight = {
        "title": "Revenue declined",
        "observation": "Revenue decreased.",
        "category": "revenue",
    }

    results = agent.generate(insights=[insight, insight])

    titles = [item.title.casefold() for item in results]

    assert len(titles) == len(set(titles))


def test_recommendation_to_dict(agent, revenue_insight):
    results = agent.generate(insights=[revenue_insight])

    payload = results[0].to_dict()

    assert isinstance(payload, dict)
    assert payload["title"] == results[0].title
    assert "evidence" in payload


def test_generate_report_structure(agent, revenue_insight):
    report = agent.generate_report(insights=[revenue_insight])

    assert report["recommendation_count"] >= 1
    assert len(report["recommendations"]) == report["recommendation_count"]
    assert "limitations" in report


def test_no_unsupported_numeric_impact_claim(agent, revenue_insight):
    results = agent.generate(insights=[revenue_insight])

    for recommendation in results:
        assert "guaranteed" in recommendation.expected_impact.lower() or (
            "not guaranteed" in recommendation.expected_impact.lower()
            or "not established" in recommendation.expected_impact.lower()
            or "has not been established" in recommendation.expected_impact.lower()
            or "not quantified" in recommendation.expected_impact.lower()
        )


def test_dataclass_serialization():
    recommendation = Recommendation(
        title="Investigate",
        problem="Revenue declined.",
        action="Review regional revenue.",
        rationale="Find the affected segments.",
        priority="high",
        category="revenue",
        expected_impact="Identify improvement opportunities.",
        evidence={"change": -100},
    )

    result = recommendation.to_dict()

    assert result["priority"] == "high"
    assert result["evidence"]["change"] == -100


def test_invalid_numeric_values_are_not_used_as_root_cause_changes(agent):
    root_cause = {
        "metric": "revenue",
        "overall": {
            "previous_value": 100,
            "current_value": 120,
            "change": float("nan"),
        },
    }

    assert agent.generate(root_cause=root_cause) == []