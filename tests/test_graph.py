from app.agents.planner import (
    ExecutionPlan,
    PlanStep,
)
from app.graph.state import BIState
from app.graph.workflow import BIWorkflow

import pandas as pd
class FakePlanner:
    def create_plan(
        self,
        question: str,
    ) -> ExecutionPlan:

        return ExecutionPlan(
            intent="revenue_analysis",
            reasoning="Revenue should be queried.",
            steps=[
                PlanStep(
                    tool="sql_query",
                    purpose="Retrieve revenue.",
                    input_description=question,
                )
            ],
            requires_sql=True,
        )


def test_state_can_store_question():
    state: BIState = {
        "question": "What is revenue?"
    }

    assert state["question"] == "What is revenue?"


def test_workflow_initialization():
    workflow = BIWorkflow(
        planner=FakePlanner()
    )

    assert workflow.graph is not None


def test_planner_node():
    workflow = BIWorkflow(
        planner=FakePlanner()
    )

    state: BIState = {
        "question": "What is revenue?",
        "iteration": 0,
    }

    result = workflow.planner_node(state)

    assert result["plan"] is not None
    assert result["iteration"] == 1


def test_execution_node():
    workflow = BIWorkflow(
        planner=FakePlanner()
    )

    state: BIState = {
        "question": "What is revenue?",
        "plan": FakePlanner().create_plan(
            "What is revenue?"
        ),
    }

    result = workflow.execution_node(state)

    assert "execution_results" in result
    assert len(result["execution_results"]) == 1
    assert result["execution_results"][0]["tool"] == "sql_query"


def test_evidence_node():
    workflow = BIWorkflow(
        planner=FakePlanner()
    )

    state: BIState = {
        "question": "What is revenue?",
        "execution_results": [
            {
                "tool": "sql_query",
                "purpose": "Retrieve revenue.",
                "status": "planned",
            }
        ],
    }

    result = workflow.evidence_node(state)

    assert "evidence" in result
    assert len(result["evidence"]) == 1
    assert result["evidence"][0]["tool"] == "sql_query"


def test_response_node():
    workflow = BIWorkflow(
        planner=FakePlanner()
    )

    state: BIState = {
        "question": "What is revenue?",
        "plan": FakePlanner().create_plan(
            "What is revenue?"
        ),
        "execution_results": [],
        "evidence": [],
    }

    result = workflow.response_node(state)

    assert "final_answer" in result
    assert "revenue" in result["final_answer"].lower()


def test_complete_workflow():
    workflow = BIWorkflow(
        planner=FakePlanner()
    )

    result = workflow.invoke(
        "What is the total revenue?"
    )

    assert result["question"] == (
        "What is the total revenue?"
    )

    assert result["plan"] is not None

    assert len(
        result["execution_results"]
    ) == 1

    assert len(
        result["evidence"]
    ) == 1

    assert result["final_answer"]


def test_empty_question():
    workflow = BIWorkflow(
        planner=FakePlanner()
    )

    try:
        workflow.invoke("")
        assert False
    except ValueError:
        assert True


def test_whitespace_question():
    workflow = BIWorkflow(
        planner=FakePlanner()
    )

    try:
        workflow.invoke("   ")
        assert False
    except ValueError:
        assert True


def test_invalid_question_type():
    workflow = BIWorkflow(
        planner=FakePlanner()
    )

    try:
        workflow.invoke(None)
        assert False
    except TypeError:
        assert True

def test_response_reports_execution_failure():
    workflow = BIWorkflow(planner=FakePlanner())

    state: BIState = {
        "question": "What is the customer lifetime value?",
        "plan": FakePlanner().create_plan(
            "What is the customer lifetime value?"
        ),
        "execution_results": [
            {
                "tool": "sql_query",
                "purpose": "Retrieve customer lifetime value.",
                "status": "error",
                "error": "Unable to generate SQL for the given question.",
                "error_type": "ValueError",
            }
        ],
        "evidence": [],
    }

    result = workflow.response_node(state)
    answer = result["final_answer"].lower()

    assert "execution errors" in answer
    assert "unable to generate sql" in answer
    assert "execution plan created successfully" not in answer


def test_response_reports_partial_execution():
    workflow = BIWorkflow(planner=FakePlanner())

    state: BIState = {
        "question": "Analyze revenue and customer lifetime value.",
        "plan": FakePlanner().create_plan(
            "Analyze revenue and customer lifetime value."
        ),
        "execution_results": [
            {
                "tool": "sql_query",
                "purpose": "Retrieve revenue.",
                "status": "success",
                "sql": "SELECT SUM(revenue) AS total_revenue FROM sales",
                "explanation": "Calculates total revenue.",
                "row_count": 1,
                "columns": ["total_revenue"],
                "data": [{"total_revenue": 1000}],
            },
            {
                "tool": "statistics",
                "purpose": "Analyze additional metrics.",
                "status": "error",
                "error": "Statistics execution failed.",
                "error_type": "RuntimeError",
            },
        ],
        "evidence": [],
    }

    result = workflow.response_node(state)
    answer = result["final_answer"].lower()

    assert "execution results" in answer
    assert "execution errors" in answer
    assert "total_revenue" in answer
    assert "statistics execution failed" in answer


def test_response_discloses_planned_tools():
    workflow = BIWorkflow(planner=FakePlanner())

    state: BIState = {
        "question": "Analyze revenue and detect anomalies.",
        "plan": FakePlanner().create_plan(
            "Analyze revenue and detect anomalies."
        ),
        "execution_results": [
            {
                "tool": "sql_query",
                "purpose": "Retrieve revenue.",
                "status": "success",
                "sql": "SELECT SUM(revenue) AS total_revenue FROM sales",
                "explanation": "Calculates total revenue.",
                "row_count": 1,
                "columns": ["total_revenue"],
                "data": [{"total_revenue": 1000}],
            },
            {
                "tool": "anomaly_detection",
                "purpose": "Detect unusual values.",
                "status": "planned",
            },
        ],
        "evidence": [],
    }

    result = workflow.response_node(state)
    answer = result["final_answer"].lower()

    assert "total_revenue" in answer
    assert "anomaly_detection" in answer
    assert "not executed" in answer

def test_kpi_detection_executes_and_records_results(monkeypatch):
    workflow = BIWorkflow(planner=FakePlanner())

    question = "Calculate business KPIs."

    dataframe = pd.DataFrame(
        {
            "revenue": [1000, 2000, 3000],
            "cost": [600, 1200, 1800],
            "orders": [10, 20, 30],
            "customer_id": ["C1", "C2", "C1"],
        }
    )

    plan = ExecutionPlan(
        intent="business_analytics",
        reasoning="Calculate business KPIs.",
        steps=[
            PlanStep(
                tool="kpi_detection",
                purpose="Calculate business KPIs.",
                input_description=question,
            )
        ],
        requires_sql=False,
        requires_kpi=True,
    )

    monkeypatch.setattr(
        workflow,
        "_select_table",
        lambda: "sales_data",
    )

    monkeypatch.setattr(
        workflow.sql_executor.database,
        "execute_query",
        lambda query, max_rows=None: dataframe.copy(),
    )

    state: BIState = {
        "question": question,
        "plan": plan,
    }

    result = workflow.execution_node(state)

    assert len(result["execution_results"]) == 1

    execution = result["execution_results"][0]

    assert execution["tool"] == "kpi_detection"
    assert execution["status"] == "success"
    assert execution["table_name"] == "sales_data"
    assert execution["row_count"] == 3

    revenue_kpi = next(
        kpi
        for kpi in execution["kpis"]
        if kpi["name"] == "Revenue"
    )

    assert revenue_kpi["value"] == 6000


def test_kpi_results_are_preserved_in_evidence(monkeypatch):
    workflow = BIWorkflow(planner=FakePlanner())

    question = "Calculate business KPIs."

    dataframe = pd.DataFrame(
        {
            "revenue": [1000, 2000, 3000],
            "cost": [600, 1200, 1800],
            "orders": [10, 20, 30],
            "customer_id": ["C1", "C2", "C1"],
        }
    )

    plan = ExecutionPlan(
        intent="business_analytics",
        reasoning="Calculate business KPIs.",
        steps=[
            PlanStep(
                tool="kpi_detection",
                purpose="Calculate business KPIs.",
                input_description=question,
            )
        ],
        requires_sql=False,
        requires_kpi=True,
    )

    monkeypatch.setattr(
        workflow,
        "_select_table",
        lambda: "sales_data",
    )

    monkeypatch.setattr(
        workflow.sql_executor.database,
        "execute_query",
        lambda query, max_rows=None: dataframe.copy(),
    )

    execution_state = workflow.execution_node(
        {
            "question": question,
            "plan": plan,
        }
    )

    evidence_state = workflow.evidence_node(execution_state)

    evidence = evidence_state["evidence"][0]

    assert evidence["tool"] == "kpi_detection"
    assert evidence["status"] == "success"
    assert evidence["table_name"] == "sales_data"
    assert evidence["data"] == (
        execution_state["execution_results"][0]["data"]
    )

    revenue_kpi = next(
        kpi
        for kpi in evidence["data"]
        if kpi["name"] == "Revenue"
    )

    assert revenue_kpi["value"] == 6000


def test_kpi_execution_failure_is_recorded(monkeypatch):
    workflow = BIWorkflow(planner=FakePlanner())

    question = "Calculate business KPIs."

    plan = ExecutionPlan(
        intent="business_analytics",
        reasoning="Calculate business KPIs.",
        steps=[
            PlanStep(
                tool="kpi_detection",
                purpose="Calculate business KPIs.",
                input_description=question,
            )
        ],
        requires_sql=False,
        requires_kpi=True,
    )

    monkeypatch.setattr(
        workflow,
        "_select_table",
        lambda: "sales_data",
    )

    def fail_query(query, max_rows=None):
        raise RuntimeError("Simulated database failure.")

    monkeypatch.setattr(
        workflow.sql_executor.database,
        "execute_query",
        fail_query,
    )

    result = workflow.execution_node(
        {
            "question": question,
            "plan": plan,
        }
    )

    execution = result["execution_results"][0]

    assert execution["tool"] == "kpi_detection"
    assert execution["status"] == "error"
    assert execution["error_type"] == "RuntimeError"
    assert "Simulated database failure" in execution["error"]

    evidence_state = workflow.evidence_node(result)
    evidence = evidence_state["evidence"][0]

    assert evidence["status"] == "error"
    assert evidence["error_type"] == "RuntimeError"

def test_statistics_executes_and_records_results(monkeypatch):
    workflow = BIWorkflow(planner=FakePlanner())

    question = "Calculate the average revenue."

    dataframe = pd.DataFrame(
        {
            "revenue": [100, 200, 300],
            "cost": [60, 120, 180],
            "orders": [10, 20, 30],
        }
    )

    plan = ExecutionPlan(
        intent="business_analytics",
        reasoning="Calculate average revenue.",
        steps=[
            PlanStep(
                tool="statistics",
                purpose="Calculate average revenue.",
                input_description=question,
            )
        ],
        requires_sql=False,
        requires_statistics=True,
    )

    monkeypatch.setattr(
        workflow,
        "_select_table",
        lambda: "sales_data",
    )

    monkeypatch.setattr(
        workflow.sql_executor.database,
        "execute_query",
        lambda query, max_rows=None: dataframe.copy(),
    )

    result = workflow.execution_node(
        {
            "question": question,
            "plan": plan,
        }
    )

    assert len(result["execution_results"]) == 1

    execution = result["execution_results"][0]

    assert execution["tool"] == "statistics"
    assert execution["status"] == "success"
    assert execution["table_name"] == "sales_data"

    assert execution["data"] == [
        {
            "metric": "mean",
            "column": "revenue",
            "value": 200.0,
        }
    ]


def test_visualization_executes_from_previous_results(monkeypatch):
    workflow = BIWorkflow(planner=FakePlanner())

    question = "Create a bar chart of revenue by region."

    plan = ExecutionPlan(
        intent="business_analytics",
        reasoning="Retrieve revenue data and visualize it.",
        steps=[
            PlanStep(
                tool="sql_query",
                purpose="Retrieve revenue data.",
                input_description=question,
            ),
            PlanStep(
                tool="visualization",
                purpose="Visualize revenue by region.",
                input_description=question,
            ),
        ],
        requires_sql=True,
    )

    source_data = [
        {"region": "South", "revenue": 100},
        {"region": "North", "revenue": 200},
        {"region": "East", "revenue": 150},
    ]

    monkeypatch.setattr(
        workflow,
        "_execute_sql_step",
        lambda question: {
            "status": "success",
            "sql": "SELECT region, revenue FROM sales_data",
            "table_name": "sales_data",
            "row_count": len(source_data),
            "columns": ["region", "revenue"],
            "data": source_data,
        },
    )

    result = workflow.execution_node(
        {
            "question": question,
            "plan": plan,
        }
    )

    executions = result["execution_results"]

    assert len(executions) == 2

    assert executions[0]["status"] == "success"
    assert executions[0]["tool"] == "sql_query"

    visualization = executions[1]

    assert visualization["tool"] == "visualization"
    assert visualization["status"] == "success"
    assert visualization["chart_type"] == "bar"
    assert visualization["chart"]["x_axis"] == "region"
    assert visualization["chart"]["y_axis"] == "revenue"
    assert len(visualization["chart"]["data"]) == 3

def test_anomaly_detection_executes_and_records_results(monkeypatch):
    workflow = BIWorkflow(planner=FakePlanner())

    question = "Detect anomalies in revenue."

    dataframe = pd.DataFrame(
        {
            "revenue": [
                100, 105, 110, 115, 120,
                125, 130, 135, 140, 1000,
            ]
        }
    )

    plan = ExecutionPlan(
        intent="business_analytics",
        reasoning="Detect unusual revenue values.",
        steps=[
            PlanStep(
                tool="anomaly_detection",
                purpose="Detect revenue anomalies.",
                input_description=question,
            )
        ],
        requires_sql=False,
    )

    monkeypatch.setattr(
        workflow,
        "_select_table",
        lambda: "sales_data",
    )

    monkeypatch.setattr(
        workflow.sql_executor,
        "read_table",
        lambda table_name: dataframe.copy(),
    )

    result = workflow.execution_node(
        {
            "question": question,
            "plan": plan,
        }
    )

    assert len(result["execution_results"]) == 1

    execution = result["execution_results"][0]

    assert execution["tool"] == "anomaly_detection"
    assert execution["status"] == "success"
    assert execution["table_name"] == "sales_data"
    assert execution["column"] == "revenue"
    assert execution["method"] == "iqr"
    assert execution["anomaly_count"] == 1
    assert execution["data"][0]["value"] == 1000
    assert execution["data"][0]["is_anomaly"] is True

def test_anomaly_detection_empty_dataset_records_error(monkeypatch):
    workflow = BIWorkflow(planner=FakePlanner())

    question = "Detect anomalies in revenue."

    plan = ExecutionPlan(
        intent="business_analytics",
        reasoning="Detect unusual revenue values.",
        steps=[
            PlanStep(
                tool="anomaly_detection",
                purpose="Detect revenue anomalies.",
                input_description=question,
            )
        ],
        requires_sql=False,
    )

    monkeypatch.setattr(
        workflow,
        "_select_table",
        lambda: "sales_data",
    )

    monkeypatch.setattr(
        workflow.sql_executor,
        "read_table",
        lambda table_name: pd.DataFrame(
            columns=["revenue"]
        ),
    )

    result = workflow.execution_node(
        {
            "question": question,
            "plan": plan,
        }
    )

    assert len(result["execution_results"]) == 1

    execution = result["execution_results"][0]

    assert execution["tool"] == "anomaly_detection"
    assert execution["status"] == "error"
    assert "contains no rows" in execution["error"]
    assert execution["error_type"] == "ValueError"

def test_visualization_records_error_when_sql_step_fails():
    workflow = BIWorkflow(planner=FakePlanner())

    question = "Create a bar chart of revenue by region."

    plan = ExecutionPlan(
        intent="business_analytics",
        reasoning="Retrieve revenue data and visualize it.",
        steps=[
            PlanStep(
                tool="sql_query",
                purpose="Retrieve revenue data.",
                input_description=question,
            ),
            PlanStep(
                tool="visualization",
                purpose="Visualize revenue by region.",
                input_description=question,
            ),
        ],
        requires_sql=True,
    )

    def failed_sql_step(question):
        raise ValueError("SQL execution failed.")

    from unittest.mock import patch

    with patch.object(
        workflow,
        "_execute_sql_step",
        side_effect=failed_sql_step,
    ):
        result = workflow.execution_node(
            {
                "question": question,
                "plan": plan,
            }
        )

    executions = result["execution_results"]

    assert len(executions) == 2

    assert executions[0]["tool"] == "sql_query"
    assert executions[0]["status"] == "error"
    assert "SQL execution failed" in executions[0]["error"]

    assert executions[1]["tool"] == "visualization"
    assert executions[1]["status"] == "error"
    assert "Visualization requires successful data-producing" in (
        executions[1]["error"]
    )

def test_visualization_records_error_when_upstream_data_is_empty():
    workflow = BIWorkflow(planner=FakePlanner())

    question = "Create a bar chart of revenue by region."

    plan = ExecutionPlan(
        intent="business_analytics",
        reasoning="Retrieve revenue data and visualize it.",
        steps=[
            PlanStep(
                tool="sql_query",
                purpose="Retrieve revenue data.",
                input_description=question,
            ),
            PlanStep(
                tool="visualization",
                purpose="Visualize revenue by region.",
                input_description=question,
            ),
        ],
        requires_sql=True,
    )

    def empty_sql_step(question):
        return {
            "status": "success",
            "sql": "SELECT region, revenue FROM sales_data",
            "table_name": "sales_data",
            "row_count": 0,
            "columns": ["region", "revenue"],
            "data": [],
        }

    from unittest.mock import patch

    with patch.object(
        workflow,
        "_execute_sql_step",
        side_effect=empty_sql_step,
    ):
        result = workflow.execution_node(
            {
                "question": question,
                "plan": plan,
            }
        )

    executions = result["execution_results"]

    assert len(executions) == 2

    assert executions[0]["tool"] == "sql_query"
    assert executions[0]["status"] == "success"
    assert executions[0]["data"] == []

    assert executions[1]["tool"] == "visualization"
    assert executions[1]["status"] == "error"
    assert "Visualization requires successful data-producing" in (
        executions[1]["error"]
    )

def test_statistics_execution_failure_is_recorded():
    from unittest.mock import patch

    workflow = BIWorkflow(planner=FakePlanner())

    question = "Calculate statistical metrics for revenue."

    plan = ExecutionPlan(
        intent="business_analytics",
        reasoning="Calculate statistical metrics.",
        steps=[
            PlanStep(
                tool="statistics",
                purpose="Calculate statistical metrics.",
                input_description=question,
            )
        ],
        requires_sql=False,
    )

    with patch.object(
        workflow,
        "_execute_statistics_step",
        side_effect=RuntimeError("Statistics calculation failed."),
    ):
        result = workflow.execution_node(
            {
                "question": question,
                "plan": plan,
            }
        )

    executions = result["execution_results"]

    assert len(executions) == 1

    execution = executions[0]

    assert execution["tool"] == "statistics"
    assert execution["status"] == "error"
    assert execution["error"] == "Statistics calculation failed."
    assert execution["error_type"] == "RuntimeError"