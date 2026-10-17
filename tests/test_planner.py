from unittest.mock import MagicMock

import pytest

from app.agents.planner import (
    ExecutionPlan,
    PlanStep,
    PlannerAgent,
)


class FakeProvider:
    def structured_invoke(
        self,
        schema,
        prompt,
        system_prompt=None,
    ):
        return ExecutionPlan(
            intent="revenue_analysis",
            reasoning="Revenue must be queried from the dataset.",
            steps=[
                PlanStep(
                    tool="sql_query",
                    purpose="Retrieve revenue data.",
                    input_description=prompt,
                )
            ],
            requires_sql=True,
        )


def test_plan_step():
    step = PlanStep(
        tool="sql_query",
        purpose="Query revenue.",
        input_description="Show total revenue.",
    )

    assert step.tool == "sql_query"
    assert step.purpose == "Query revenue."


def test_execution_plan():
    plan = ExecutionPlan(
        intent="revenue_analysis",
        reasoning="Use SQL.",
        steps=[],
        requires_sql=True,
    )

    assert plan.intent == "revenue_analysis"
    assert plan.requires_sql is True


def test_planner_initialization():
    provider = FakeProvider()

    planner = PlannerAgent(provider=provider)

    assert planner.provider is provider


def test_create_plan():
    provider = FakeProvider()

    planner = PlannerAgent(provider=provider)

    plan = planner.create_plan(
        "What is the total revenue?"
    )

    assert isinstance(plan, ExecutionPlan)
    assert plan.requires_sql is True
    assert len(plan.steps) == 1
    assert plan.steps[0].tool == "sql_query"


def test_empty_question():
    provider = FakeProvider()

    planner = PlannerAgent(provider=provider)

    with pytest.raises(ValueError):
        planner.create_plan("")


def test_whitespace_question():
    provider = FakeProvider()

    planner = PlannerAgent(provider=provider)

    with pytest.raises(ValueError):
        planner.create_plan("   ")


def test_invalid_question_type():
    provider = FakeProvider()

    planner = PlannerAgent(provider=provider)

    with pytest.raises(TypeError):
        planner.create_plan(None)


def test_fallback_revenue():
    planner = PlannerAgent(provider=FakeProvider())

    plan = planner.create_fallback_plan(
        "Show revenue by region."
    )

    assert plan.requires_sql is True
    assert any(
        step.tool == "sql_query"
        for step in plan.steps
    )


def test_fallback_anomaly():
    planner = PlannerAgent(provider=FakeProvider())

    plan = planner.create_fallback_plan(
        "Find revenue anomalies."
    )

    assert plan.requires_anomaly_detection is True
    assert any(
        step.tool == "anomaly_detection"
        for step in plan.steps
    )


def test_fallback_forecast():
    planner = PlannerAgent(provider=FakeProvider())

    plan = planner.create_fallback_plan(
        "Forecast revenue for next month."
    )

    assert plan.requires_forecast is True
    assert any(
        step.tool == "forecasting"
        for step in plan.steps
    )


def test_fallback_visualization():
    planner = PlannerAgent(provider=FakeProvider())

    plan = planner.create_fallback_plan(
        "Create a revenue chart."
    )

    assert plan.requires_visualization is True
    assert any(
        step.tool == "visualization"
        for step in plan.steps
    )


def test_fallback_kpi():
    planner = PlannerAgent(provider=FakeProvider())

    plan = planner.create_fallback_plan(
        "Show the main KPIs and margin."
    )

    assert plan.requires_kpi is True
    assert any(
        step.tool == "kpi_detection"
        for step in plan.steps
    )


def test_fallback_statistics():
    planner = PlannerAgent(provider=FakeProvider())

    plan = planner.create_fallback_plan(
        "Calculate the average revenue."
    )

    assert plan.requires_statistics is True
    assert any(
        step.tool == "statistics"
        for step in plan.steps
    )


def test_fallback_unknown_question():
    planner = PlannerAgent(provider=FakeProvider())

    plan = planner.create_fallback_plan(
        "Tell me something about the dataset."
    )

    assert plan.requires_sql is True
    assert len(plan.steps) >= 1


def test_fallback_combined_request():
    planner = PlannerAgent(provider=FakeProvider())

    plan = planner.create_fallback_plan(
        "Find revenue anomalies and forecast next month."
    )

    assert plan.requires_sql is True
    assert plan.requires_anomaly_detection is True
    assert plan.requires_forecast is True

def test_create_plan_falls_back_when_revenue_routes_to_kpi():
    class IncorrectRoutingProvider:
        def structured_invoke(
            self,
            schema,
            prompt,
            system_prompt=None,
        ):
            return ExecutionPlan(
                intent="kpi_detection",
                reasoning="Incorrectly routed revenue question.",
                steps=[
                    PlanStep(
                        tool="kpi_detection",
                        purpose="Identify total revenue KPI.",
                        input_description="Sales and revenue data.",
                    )
                ],
                requires_sql=False,
                requires_kpi=False,
            )

    planner = PlannerAgent(
        provider=IncorrectRoutingProvider()
    )

    plan = planner.create_plan(
        "What is the total revenue?"
    )

    assert any(
        step.tool == "sql_query"
        for step in plan.steps
    )
    assert plan.requires_sql is True


def test_create_plan_uses_fallback_when_provider_raises():
    class FailingProvider:
        def structured_invoke(
            self,
            schema,
            prompt,
            system_prompt=None,
        ):
            raise RuntimeError("Simulated LLM failure")

    planner = PlannerAgent(
        provider=FailingProvider()
    )

    plan = planner.create_plan(
        "What is the total revenue?"
    )

    assert plan.requires_sql is True
    assert any(
        step.tool == "sql_query"
        for step in plan.steps
    )


def test_fallback_average_revenue_selects_sql_and_statistics():
    planner = PlannerAgent(
        provider=FakeProvider()
    )

    plan = planner.create_fallback_plan(
        "Calculate the average revenue."
    )

    tools = [
        step.tool
        for step in plan.steps
    ]

    assert "sql_query" in tools
    assert "statistics" in tools
    assert plan.requires_sql is True
    assert plan.requires_statistics is True


def test_fallback_requirement_flags_match_selected_tools():
    planner = PlannerAgent(
        provider=FakeProvider()
    )

    plan = planner.create_fallback_plan(
        "Find revenue anomalies and forecast next month."
    )

    tools = {
        step.tool
        for step in plan.steps
    }

    assert plan.requires_sql == ("sql_query" in tools)
    assert plan.requires_anomaly_detection == (
        "anomaly_detection" in tools
    )
    assert plan.requires_forecast == (
        "forecasting" in tools
    )
    assert plan.requires_statistics == (
        "statistics" in tools
    )
    assert plan.requires_visualization == (
        "visualization" in tools
    )
    assert plan.requires_kpi == (
        "kpi_detection" in tools
    )

def test_revenue_by_region_fallback_uses_sql_without_kpi():
    planner = PlannerAgent(provider=FakeProvider())

    plan = planner.create_fallback_plan(
        "Show revenue by region"
    )

    tools = [
        step.tool
        for step in plan.steps
    ]

    assert tools == ["sql_query"]
    assert plan.requires_sql is True
    assert plan.requires_kpi is False

def test_create_plan_normalizes_revenue_by_region():
    class ExtraToolsProvider:
        def structured_invoke(
            self,
            schema,
            prompt,
            system_prompt=None,
        ):
            return ExecutionPlan(
                intent="kpi_detection",
                reasoning="Unnecessarily complex plan.",
                steps=[
                    PlanStep(
                        tool="sql_query",
                        purpose="Retrieve revenue.",
                        input_description=prompt,
                    ),
                    PlanStep(
                        tool="kpi_detection",
                        purpose="Identify revenue KPI.",
                        input_description=prompt,
                    ),
                    PlanStep(
                        tool="sql_query",
                        purpose="Group revenue by region.",
                        input_description=prompt,
                    ),
                ],
                requires_sql=True,
                requires_kpi=True,
            )

    planner = PlannerAgent(provider=ExtraToolsProvider())
    plan = planner.create_plan("Show revenue by region")

    assert [step.tool for step in plan.steps] == ["sql_query"]
    assert plan.requires_sql is True
    assert plan.requires_kpi is False

def test_fallback_unsupported_question_returns_empty_plan():
    planner = PlannerAgent(
        provider=FakeProvider()
    )

    plan = planner.create_fallback_plan(
        "Tell me a joke."
    )

    assert plan.steps == []
    assert plan.requires_sql is False
    assert plan.requires_statistics is False
    assert plan.requires_anomaly_detection is False
    assert plan.requires_forecast is False
    assert plan.requires_visualization is False
    assert plan.requires_kpi is False