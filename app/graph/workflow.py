from __future__ import annotations

import re
from typing import Any
import pandas as pd

from app.analytics.anomaly import AnomalyDetector
from app.analytics.visualization import VisualizationEngine
from langgraph.graph import END, START, StateGraph

from app.agents.planner import PlannerAgent
from app.analytics.kpi import KPIDetectionEngine
from app.analytics.statistics import StatisticsEngine
from app.config.settings import settings
from app.graph.state import BIState
from app.sql.sql_agent import SQLAnalyst, SQLRequest
from app.sql.sql_executor import SQLExecutor


class BIWorkflow:
    """LangGraph orchestration layer for the Autonomous AI BI Platform.

    Responsibilities:
    - Generate a structured execution plan.
    - Execute supported SQL, statistics, and KPI tools.
    - Record actual execution results.
    - Preserve evidence and execution errors.
    - Produce a response grounded in tool results.

    Deterministic analytics engines perform numerical calculations.
    """

    def __init__(
        self,
        planner: PlannerAgent | None = None,
        sql_analyst: SQLAnalyst | None = None,
        sql_executor: SQLExecutor | None = None,
    ):
        self.planner = planner or PlannerAgent()
        self.sql_analyst = sql_analyst or SQLAnalyst()
        self.sql_executor = sql_executor or SQLExecutor(
            database_path=settings.database_path,
            max_rows=settings.max_result_rows,
        )
        self.graph = self._build_graph()

    # ------------------------------------------------------------------
    # Planner
    # ------------------------------------------------------------------

    def planner_node(
        self,
        state: BIState,
    ) -> BIState:
        """Convert the user's business question into an execution plan."""
        question = state["question"]
        plan = self.planner.create_plan(question)

        return {
            **state,
            "plan": plan,
            "iteration": state.get("iteration", 0) + 1,
        }

    # ------------------------------------------------------------------
    # Dataset selection and SQL execution
    # ------------------------------------------------------------------

    def _select_table(
        self,
        selected_table: str | None = None,
    ) -> str:
        """Select and validate a DuckDB table.

        Selection rules:
        1. Use the explicitly selected table when provided.
        2. Otherwise, prefer sales_data when available.
        3. Otherwise, use the only available table.
        4. Raise an error when selection is ambiguous.
        """
        tables = self.sql_executor.database.get_tables()

        if not tables:
            raise ValueError(
                "No tables are available in the DuckDB database."
            )

        if selected_table is not None:
            if not isinstance(selected_table, str) or not selected_table.strip():
                raise ValueError("The selected table name cannot be empty.")

            selected_table = selected_table.strip()

            if selected_table not in tables:
                raise ValueError(
                    f"Selected table '{selected_table}' does not exist. "
                    f"Available tables: {', '.join(tables)}"
                )

            table_name = selected_table
        elif "sales_data" in tables:
            table_name = "sales_data"
        elif len(tables) == 1:
            table_name = tables[0]
        else:
            raise ValueError(
                "Multiple tables are available. "
                "The correct table must be selected explicitly."
            )

        # Only support conventional SQL identifiers.
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_name):
            raise ValueError("The selected table name is not supported.")

        return table_name

    def _execute_sql_step(
        self,
        question: str,
        selected_table: str | None = None,
    ) -> dict[str, Any]:
        """Generate, validate, and execute one supported SQL request."""
        table_name = (
            self._select_table()
            if selected_table is None
            else self._select_table(selected_table)
        )
        request = SQLRequest(question=question, table_name=table_name)
        response = self.sql_analyst.generate_sql(request)
        dataframe = self.sql_executor.execute(response.sql)

        return {
            "status": "success",
            "sql": response.sql,
            "explanation": response.explanation,
            "table_name": table_name,
            "row_count": len(dataframe),
            "columns": dataframe.columns.tolist(),
            "data": dataframe.to_dict(orient="records"),
        }

    # ------------------------------------------------------------------
    # Statistics execution
    # ------------------------------------------------------------------

    def _execute_statistics_step(
        self,
        question: str,
        selected_table: str | None = None,
    ) -> dict[str, Any]:
        """Execute a supported deterministic statistical calculation."""
        table_name = (
            self._select_table()
            if selected_table is None
            else self._select_table(selected_table)
        )
        dataframe = self.sql_executor.read_table(table_name)

        if dataframe.empty:
            raise ValueError(f"Table '{table_name}' contains no rows.")

        question_lower = question.lower()
        columns = [str(column) for column in dataframe.columns]

        # Match column names mentioned by the user. This supports names
        # such as "revenue" and "total_revenue" when the full name appears.
        matched_columns = [
            column
            for column in columns
            if column.lower() in question_lower
        ]

        engine = StatisticsEngine()

        # Correlation and covariance require two explicitly identified columns.
        if "correlation" in question_lower or "correlate" in question_lower:
            if len(matched_columns) < 2:
                raise ValueError(
                    "Correlation requires two column names in the question."
                )

            column_x, column_y = matched_columns[:2]
            value = engine.correlation(dataframe, column_x, column_y)
            metric = "correlation"
            data = [{
                "metric": metric,
                "column_x": column_x,
                "column_y": column_y,
                "value": value,
            }]

        elif "covariance" in question_lower:
            if len(matched_columns) < 2:
                raise ValueError(
                    "Covariance requires two column names in the question."
                )

            column_x, column_y = matched_columns[:2]
            value = engine.covariance(dataframe, column_x, column_y)
            metric = "covariance"
            data = [{
                "metric": metric,
                "column_x": column_x,
                "column_y": column_y,
                "value": value,
            }]

        else:
            if not matched_columns:
                raise ValueError(
                    "Could not identify a column in the statistical question. "
                    f"Available columns: {', '.join(columns)}"
                )

            column = matched_columns[0]

            if "standard deviation" in question_lower or "stddev" in question_lower:
                metric = "standard_deviation"
                value = engine.standard_deviation(dataframe, column)
            elif "variance" in question_lower:
                metric = "variance"
                value = engine.variance(dataframe, column)
            elif "median" in question_lower:
                metric = "median"
                value = engine.median(dataframe, column)
            elif "minimum" in question_lower or re.search(r"\bmin\b", question_lower):
                metric = "minimum"
                value = engine.minimum(dataframe, column)
            elif "maximum" in question_lower or re.search(r"\bmax\b", question_lower):
                metric = "maximum"
                value = engine.maximum(dataframe, column)
            elif "percentile" in question_lower:
                percentile_match = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:st|nd|rd|th)?\s*percentile\b", question_lower)
                percentile = float(percentile_match.group(1)) if percentile_match else 50.0
                metric = f"percentile_{percentile:g}"
                value = engine.percentile(dataframe, column, percentile)
            elif "summary" in question_lower or "describe" in question_lower:
                metric = "summary"
                summary = engine.summary(dataframe, column)
                data = [{"metric": metric, **summary}]
                value = None
            elif "average" in question_lower or "mean" in question_lower:
                metric = "mean"
                value = engine.mean(dataframe, column)
            else:
                raise ValueError(
                    "Unsupported statistical operation. Supported operations "
                    "include average, mean, median, variance, standard deviation, "
                    "minimum, maximum, percentile, correlation, covariance, and summary."
                )

            if metric != "summary":
                data = [{
                    "metric": metric,
                    "column": column,
                    "value": value,
                }]

        return {
            "status": "success",
            "table_name": table_name,
            "row_count": len(dataframe),
            "columns": columns,
            "data": data,
            "explanation": f"Calculated {metric} from column data in '{table_name}'.",
        }
        # ------------------------------------------------------------------
    # Anomaly detection execution
    # ------------------------------------------------------------------

    def _execute_anomaly_step(
        self,
        question: str,
        selected_table: str | None = None,
    ) -> dict[str, Any]:
        """Detect anomalies in a numeric dataset column."""

        table_name = (
            self._select_table()
            if selected_table is None
            else self._select_table(selected_table)
        )

        dataframe = self.sql_executor.read_table(table_name)

        if dataframe.empty:
            raise ValueError(
                f"Table '{table_name}' contains no rows."
            )

        question_lower = question.lower()

        matched_columns = [
            str(column)
            for column in dataframe.columns
            if str(column).lower() in question_lower
        ]

        numeric_columns = [
            str(column)
            for column in dataframe.select_dtypes(
                include="number"
            ).columns
        ]

        if not matched_columns:
            if len(numeric_columns) == 1:
                column = numeric_columns[0]
            else:
                raise ValueError(
                    "Could not identify the numeric column for "
                    "anomaly detection. Specify a column name."
                )
        else:
            column = matched_columns[0]

        method = (
            "isolation_forest"
            if "isolation forest" in question_lower
            else "iqr"
        )

        detector = AnomalyDetector()

        anomalies = detector.detect(
            dataframe=dataframe,
            column=column,
            method=method,
        )

        anomaly_dataframe = detector.to_dataframe(
            anomalies
        )

        expected_columns = [
            "index",
            "value",
            "method",
            "score",
            "severity",
            "is_anomaly",
            "explanation",
        ]

        if anomaly_dataframe.empty:
            anomaly_dataframe = pd.DataFrame(
                columns=expected_columns
            )

        return {
            "status": "success",
            "table_name": table_name,
            "row_count": len(dataframe),
            "column": column,
            "method": method,
            "anomaly_count": len(anomalies),
            "columns": expected_columns,
            "data": anomaly_dataframe.to_dict(
                orient="records"
            ),
            "explanation": (
                f"Detected {len(anomalies)} anomalies in "
                f"'{column}' using {method}."
            ),
        }
    # ------------------------------------------------------------------
    # Execution node
    # ------------------------------------------------------------------

    def execution_node(
        self,
        state: BIState,
    ) -> BIState:
        """Execute supported tools and record their actual results."""
        plan = state.get("plan")

        if plan is None:
            return {
                **state,
                "error": "No execution plan was generated.",
            }

        results: list[dict[str, Any]] = []

        for step in plan.steps:
            result: dict[str, Any] = {
                "tool": step.tool,
                "purpose": step.purpose,
                "status": "planned",
            }

            if step.tool == "sql_query":
                try:
                    selected_table = state.get("selected_table")
                    if selected_table is None:
                        # Preserve compatibility with existing callers and test doubles.
                        sql_result = self._execute_sql_step(
                            question=state["question"],
                        )
                    else:
                        sql_result = self._execute_sql_step(
                            question=state["question"],
                            selected_table=selected_table,
                        )
                    result.update(sql_result)
                except Exception as exc:
                    result.update({
                        "status": "error",
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    })

            elif step.tool == "statistics":
                try:
                    selected_table = state.get("selected_table")
                    if selected_table is None:
                        statistics_result = self._execute_statistics_step(
                            question=state["question"],
                        )
                    else:
                        statistics_result = self._execute_statistics_step(
                            question=state["question"],
                            selected_table=selected_table,
                        )
                    result.update(statistics_result)
                except Exception as exc:
                    result.update({
                        "status": "error",
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    })

            elif step.tool == "kpi_detection":
                try:
                    selected_table = state.get("selected_table")
                    if selected_table is None:
                        table_name = self._select_table()
                    else:
                        table_name = self._select_table(
                            selected_table=selected_table
                        )
                    dataframe = self.sql_executor.read_table(table_name)
                    engine = KPIDetectionEngine(dataframe)
                    kpis = engine.to_dict()

                    result.update({
                        "status": "success",
                        "table_name": table_name,
                        "row_count": len(dataframe),
                        "columns": list(dataframe.columns),
                        "kpis": kpis,
                        "data": kpis,
                        "explanation": (
                            "Business KPIs calculated from the selected dataset."
                        ),
                    })
                except Exception as exc:
                    result.update({
                        "status": "error",
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    })
            elif step.tool == "anomaly_detection":
                try:
                    selected_table = state.get("selected_table")
                    if selected_table is None:
                        anomaly_result = self._execute_anomaly_step(
                            question=state["question"],
                        )
                    else:
                        anomaly_result = self._execute_anomaly_step(
                            question=state["question"],
                            selected_table=selected_table,
                        )
                    result.update(anomaly_result)

                except Exception as exc:
                    result.update({
                        "status": "error",
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    })

            elif step.tool == "visualization":
                try:
                    visualization_result = (
                        self._execute_visualization_step(
                            question=state["question"],
                            previous_results=results,
                        )
                    )
                    result.update(visualization_result)

                except Exception as exc:
                    result.update({
                        "status": "error",
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    })
            results.append(result)
            

        return {
            **state,
            "execution_results": results,
        }
        # ------------------------------------------------------------------
    # Visualization execution
    # ------------------------------------------------------------------

    def _execute_visualization_step(
        self,
        question: str,
        previous_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Create a chart from successfully executed tool results."""

        source_result = next(
            (
                result
                for result in reversed(previous_results)
                if result.get("status") == "success"
                and isinstance(result.get("data"), list)
                and result.get("data")
                and isinstance(result.get("columns"), list)
                and result.get("columns")
            ),
            None,
        )

        if source_result is None:
            raise ValueError(
                "Visualization requires successful data-producing "
                "results from an earlier execution step."
            )

        dataframe = pd.DataFrame(
            source_result["data"]
        )

        if dataframe.empty:
            raise ValueError(
                "Cannot visualize an empty result."
            )

        numeric_columns = [
            str(column)
            for column in dataframe.select_dtypes(
                include="number"
            ).columns
        ]

        categorical_columns = [
            str(column)
            for column in dataframe.columns
            if column not in numeric_columns
        ]

        question_lower = question.lower()
        engine = VisualizationEngine()

        # Infer the chart type from the request.
        if "histogram" in question_lower:
            chart_type = "histogram"
        elif "scatter" in question_lower:
            chart_type = "scatter"
        elif "line chart" in question_lower:
            chart_type = "line"
        else:
            chart_type = "bar"

        if chart_type == "histogram":
            requested_column = next(
                (
                    column
                    for column in numeric_columns
                    if column.lower() in question_lower
                ),
                None,
            )

            column = (
                requested_column
                or (
                    numeric_columns[0]
                    if numeric_columns
                    else None
                )
            )

            if column is None:
                raise ValueError(
                    "A histogram requires a numeric column."
                )

            specification = engine.create(
                "histogram",
                dataframe,
                column=column,
                title=f"Distribution of {column}",
            )

        else:
            if (
                chart_type == "scatter"
                and len(numeric_columns) >= 2
            ):
                x_column = numeric_columns[0]
                y_column = numeric_columns[1]
            else:
                if not categorical_columns:
                    raise ValueError(
                        "The selected chart requires a categorical "
                        "column and a numeric column."
                    )

                if not numeric_columns:
                    raise ValueError(
                        "The selected chart requires a numeric column."
                    )

                x_column = categorical_columns[0]

                requested_y = next(
                    (
                        column
                        for column in numeric_columns
                        if column.lower() in question_lower
                    ),
                    None,
                )

                y_column = (
                    requested_y
                    or numeric_columns[0]
                )

            title = (
                f"{y_column.replace('_', ' ').title()} "
                f"by {x_column.replace('_', ' ').title()}"
            )

            specification = engine.create(
                chart_type,
                dataframe,
                x_column=x_column,
                y_column=y_column,
                title=title,
            )

        chart = engine.to_dict(specification)

        return {
            "status": "success",
            "chart": chart,
            "chart_type": chart["chart_type"],
            "row_count": len(chart["data"]),
            "columns": list(dataframe.columns),
            "data": chart["data"],
            "explanation": (
                f"Created a {chart['chart_type']} chart: "
                f"{chart['title']}."
            ),
        }

    # ------------------------------------------------------------------
    # Evidence
    # ------------------------------------------------------------------

    def evidence_node(
        self,
        state: BIState,
    ) -> BIState:
        """Preserve execution results as evidence records."""
        execution_results = state.get("execution_results", [])
        evidence: list[dict[str, Any]] = []

        for result in execution_results:
            evidence_item = {
                "tool": result.get("tool"),
                "status": result.get("status"),
                "purpose": result.get("purpose"),
            }

            for key in (
                "sql",
                "explanation",
                "table_name",
                "row_count",
                "columns",
                "data",
                "kpis",
                "error",
                "error_type",
            ):
                if key in result:
                    evidence_item[key] = result[key]

            evidence.append(evidence_item)

        return {
            **state,
            "evidence": evidence,
        }

    # ------------------------------------------------------------------
    # Response generation
    # ------------------------------------------------------------------

    def response_node(
        self,
        state: BIState,
    ) -> BIState:
        """Generate a response from the plan and actual execution results."""
        question = state["question"]
        plan = state.get("plan")

        if plan is None:
            return {
                **state,
                "final_answer": "Unable to create an execution plan.",
            }

        tools = [step.tool for step in plan.steps]
        execution_results = state.get("execution_results", [])

        successful_results = [
            result for result in execution_results
            if result.get("status") == "success"
        ]
        failed_results = [
            result for result in execution_results
            if result.get("status") == "error"
        ]
        planned_results = [
            result for result in execution_results
            if result.get("status") == "planned"
        ]

        lines = [
            f"Question: {question}",
            "Execution plan created.",
            f"Tools: {', '.join(tools)}",
        ]

        if successful_results and not failed_results and not planned_results:
            lines.append("Execution status: completed successfully.")
        elif successful_results:
            lines.append("Execution status: partially completed.")
        elif failed_results:
            lines.append("Execution status: failed.")
        elif planned_results:
            lines.append("Execution status: not completed.")
        else:
            lines.append("Execution status: no execution results were recorded.")

        if successful_results:
            lines.extend(["", "Execution results:"])

            for result in successful_results:
                tool_name = result.get("tool", "unknown")
                lines.append(f"- Tool: {tool_name}")

                if "sql" in result:
                    lines.append(f"  SQL: {result['sql']}")
                if "explanation" in result:
                    lines.append(f"  Explanation: {result['explanation']}")
                if "row_count" in result:
                    lines.append(f"  Rows returned: {result['row_count']}")

                data = result.get("data")
                if isinstance(data, list) and data:
                    lines.append("  Result data:")
                    for row in data[:5]:
                        lines.append(f"    {row}")
                    if len(data) > 5:
                        lines.append(
                            f"    ... and {len(data) - 5} additional rows."
                        )
                elif "data" in result:
                    lines.append("  Query executed successfully; no rows were returned.")

        if failed_results:
            lines.extend(["", "Execution errors:"])
            for result in failed_results:
                lines.append(
                    f"- {result.get('tool', 'unknown')}: "
                    f"{result.get('error', 'Execution failed.')}"
                )

        if planned_results:
            lines.extend(["", "Not executed:"])
            for result in planned_results:
                lines.append(
                    f"- {result.get('tool', 'unknown')}: not executed."
                )

        return {
            **state,
            "final_answer": "\n".join(lines),
        }

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self):
        """Build the existing LangGraph execution workflow."""
        builder = StateGraph(BIState)

        builder.add_node("planner", self.planner_node)
        builder.add_node("execution", self.execution_node)
        builder.add_node("evidence", self.evidence_node)
        builder.add_node("response", self.response_node)

        builder.add_edge(START, "planner")
        builder.add_edge("planner", "execution")
        builder.add_edge("execution", "evidence")
        builder.add_edge("evidence", "response")
        builder.add_edge("response", END)

        return builder.compile()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def invoke(
        self,
        question: str,
        selected_table: str | None = None,
    ) -> BIState:
        """Execute the complete BI workflow for a business question."""
        if not isinstance(question, str):
            raise TypeError("question must be a string")

        question = question.strip()
        if not question:
            raise ValueError("question cannot be empty")

        initial_state: BIState = {
            "question": question,
            "execution_results": [],
            "evidence": [],
            "iteration": 0,
        }

        if selected_table is not None:
            if not isinstance(selected_table, str) or not selected_table.strip():
                raise ValueError("Selected table name cannot be empty.")

            # Validate before graph execution so invalid selections fail clearly.
            selected_table = selected_table.strip()
            self._select_table(selected_table)
            initial_state["selected_table"] = selected_table

        return self.graph.invoke(initial_state)
