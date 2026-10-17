import pytest

from app.agents.bi_insight import BIInsightAgent


@pytest.fixture
def agent():
    return BIInsightAgent()


def test_revenue_exceeds_cost(agent):
    insights = agent.generate(
        {
            "revenue": 1000,
            "cost": 600,
            "profit": 400,
        }
    )

    assert len(insights) == 1
    assert insights[0].title == "Revenue Exceeds Cost"
    assert insights[0].evidence["revenue"] == 1000


def test_cost_exceeds_revenue(agent):
    insights = agent.generate(
        {
            "revenue": 500,
            "cost": 700,
        }
    )

    assert insights[0].title == "Cost Exceeds Revenue"


def test_revenue_equals_cost(agent):
    insights = agent.generate(
        {
            "revenue": 500,
            "cost": 500,
        }
    )

    assert insights[0].title == "Revenue Equals Cost"


def test_negative_profit(agent):
    insights = agent.generate(
        {"profit": -100}
    )

    assert any(
        insight.title == "Negative Profit"
        for insight in insights
    )


def test_positive_margin(agent):
    insights = agent.generate(
        {"margin": 25.0}
    )

    assert insights[0].title == "Profit Margin Assessment"
    assert insights[0].evidence["margin"] == 25.0


def test_negative_margin(agent):
    insights = agent.generate(
        {"margin": -5.0}
    )

    assert "negative" in insights[0].observation.lower()


def test_zero_margin(agent):
    insights = agent.generate(
        {"margin": 0}
    )

    assert "zero" in insights[0].observation.lower()


def test_empty_results_rejected(agent):
    with pytest.raises(ValueError):
        agent.generate({})


def test_invalid_results_type(agent):
    with pytest.raises(TypeError):
        agent.generate([1, 2, 3])


def test_no_unsupported_insight(agent):
    insights = agent.generate(
        {"region": "South"}
    )

    assert insights == []


def test_boolean_is_not_numeric(agent):
    insights = agent.generate(
        {
            "revenue": True,
            "cost": 100,
        }
    )

    assert insights == []


def test_non_finite_numbers_are_ignored(agent):
    insights = agent.generate(
        {
            "revenue": float("nan"),
            "cost": float("inf"),
        }
    )

    assert insights == []


def test_json_compatible_output(agent):
    results = agent.to_dict(
        {
            "revenue": 1000,
            "cost": 600,
        }
    )

    assert isinstance(results, list)
    assert isinstance(results[0], dict)
    assert "title" in results[0]
    assert "evidence" in results[0]


def test_llm_requires_provider(agent):
    with pytest.raises(ValueError):
        agent.generate(
            {"revenue": 1000},
            use_llm=True,
        )


def test_llm_interpretation_preserves_evidence():
    class FakeProvider:
        def invoke(self, prompt, system_prompt=None):
            return "The supplied results show a positive revenue-cost spread."

    agent = BIInsightAgent(provider=FakeProvider())

    insights = agent.generate(
        {
            "revenue": 1000,
            "cost": 600,
        },
        use_llm=True,
    )

    assert len(insights) == 2
    assert insights[-1].category == "interpretation"
    assert insights[-1].confidence == "low"
    assert insights[-1].evidence["input_results"]["revenue"] == 1000


def test_empty_llm_response_rejected():
    class FakeProvider:
        def invoke(self, prompt, system_prompt=None):
            return "   "

    agent = BIInsightAgent(provider=FakeProvider())

    with pytest.raises(ValueError):
        agent.generate(
            {"revenue": 1000},
            use_llm=True,
        )