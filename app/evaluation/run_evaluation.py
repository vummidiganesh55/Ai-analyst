from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.agents.planner import PlannerAgent
from app.config.settings import settings
from app.evaluation.evaluator import BIEvaluator
from app.graph.workflow import BIWorkflow
from app.sql.sql_executor import SQLExecutor


class FallbackProvider:
    """Force deterministic fallback planning during evaluation."""

    def structured_invoke(self, **kwargs: Any) -> Any:
        raise RuntimeError(
            "Use deterministic fallback planning for evaluation."
        )


def get_execution_result(
    state: dict[str, Any],
    tool_name: str,
) -> dict[str, Any]:
    """Retrieve a specific tool's execution result."""

    results = state.get("execution_results", [])

    for result in results:
        if result.get("tool") == tool_name:
            return result

    raise AssertionError(
        f"Expected execution result for tool '{tool_name}'."
    )


def require_success(result: dict[str, Any]) -> None:
    """Fail evaluation if the tool did not execute successfully."""

    if result.get("status") != "success":
        raise AssertionError(
            f"Tool execution failed: {result.get('error', result)}"
        )


def evaluate_total_revenue(
    workflow: BIWorkflow,
) -> float:
    state = workflow.invoke("What is the total revenue?")
    result = get_execution_result(state, "sql_query")
    require_success(result)

    data = result.get("data", [])

    if len(data) != 1:
        raise AssertionError(
            f"Expected one aggregate row; received {len(data)}."
        )

    if "total_revenue" not in data[0]:
        raise AssertionError(
            "The result does not contain total_revenue."
        )

    return float(data[0]["total_revenue"])


def evaluate_average_revenue(
    workflow: BIWorkflow,
) -> float:
    state = workflow.invoke("What is the average revenue?")
    result = get_execution_result(state, "sql_query")
    require_success(result)

    data = result.get("data", [])

    if len(data) != 1:
        raise AssertionError(
            f"Expected one aggregate row; received {len(data)}."
        )

    if "average_revenue" not in data[0]:
        raise AssertionError(
            "The result does not contain average_revenue."
        )

    return float(data[0]["average_revenue"])


def evaluate_revenue_by_region(
    workflow: BIWorkflow,
) -> float:
    state = workflow.invoke("Show revenue by region.")
    result = get_execution_result(state, "sql_query")
    require_success(result)

    data = result.get("data", [])

    actual = {
        row["region"]: float(row["total_revenue"])
        for row in data
    }

    expected = {
        "South": 410000.0,
        "North": 250000.0,
        "East": 230000.0,
        "West": 190000.0,
    }

    if actual != expected:
        raise AssertionError(
            f"Regional revenue mismatch. "
            f"Expected {expected}; received {actual}."
        )

    return {
    "actual_value": sum(actual.values()),
    "details": {
        "regional_revenue": actual,
    },
}


def evaluate_business_kpis(
    workflow: BIWorkflow,
) -> float:
    state = workflow.invoke(
        "Calculate business KPIs and margin."
    )

    result = get_execution_result(state, "kpi_detection")
    require_success(result)

    kpis = result.get("kpis", [])

    actual = {
        item["name"]: float(item["value"])
        for item in kpis
    }

    expected = {
        "Revenue": 1080000.0,
        "Cost": 765000.0,
        "Profit": 315000.0,
    }

    for name, expected_value in expected.items():
        if name not in actual:
            raise AssertionError(
                f"Expected KPI '{name}' was not returned."
            )

        if abs(actual[name] - expected_value) > 0.01:
            raise AssertionError(
                f"{name} mismatch: expected {expected_value}, "
                f"received {actual[name]}."
            )

    return actual["Profit"]


def evaluate_unsupported_question(
    workflow: BIWorkflow,
) -> float:
    state = workflow.invoke("Tell me a joke.")

    plan = state.get("plan")
    results = state.get("execution_results", [])

    if plan is None:
        raise AssertionError(
            "The workflow did not return an execution plan."
        )

    if plan.steps:
        raise AssertionError(
            f"Expected no analytical tools for an unsupported "
            f"question; received {[step.tool for step in plan.steps]}."
        )

    if results:
        raise AssertionError(
            f"Expected no tool executions; received {results}."
        )

    return 0.0


def main() -> None:
    evaluator = BIEvaluator()

    planner = PlannerAgent(provider=FallbackProvider())

    workflow = BIWorkflow(planner=planner)

    # Independent SQL executor for deterministic aggregate checks.
    sql_executor = SQLExecutor(
        database_path=settings.database_path,
        max_rows=settings.max_result_rows,
    )

    tables = sql_executor.database.get_tables()

    if "sales_data" not in tables:
        raise RuntimeError(
            "Evaluation requires the sales_data table."
        )

    cases = [
        {
            "case_id": "workflow_total_revenue",
            "category": "end_to_end_workflow",
            "operation": lambda: evaluate_total_revenue(workflow),
            "expected_value": 1080000.0,
        },
        {
            "case_id": "workflow_average_revenue",
            "category": "end_to_end_workflow",
            "operation": lambda: evaluate_average_revenue(workflow),
            "expected_value": 108000.0,
        },
        {
            "case_id": "workflow_revenue_by_region",
            "category": "end_to_end_workflow",
            "operation": lambda: evaluate_revenue_by_region(workflow),
            "expected_value": 1080000.0,
        },
        {
            "case_id": "workflow_business_kpis",
            "category": "end_to_end_workflow",
            "operation": lambda: evaluate_business_kpis(workflow),
            "expected_value": 315000.0,
        },
        {
            "case_id": "workflow_unsupported_question",
            "category": "error_handling",
            "operation": lambda: evaluate_unsupported_question(workflow),
            "expected_value": 0.0,
        },
    ]

    for case in cases:
        evaluator.evaluate_case(
            case_id=case["case_id"],
            category=case["category"],
            operation=case["operation"],
            expected_value=case["expected_value"],
            tolerance=0.01,
        )

    report = evaluator.report()

    output_path = Path("reports") / "evaluation_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print(f"\nEvaluation report saved to: {output_path}")


if __name__ == "__main__":
    main()