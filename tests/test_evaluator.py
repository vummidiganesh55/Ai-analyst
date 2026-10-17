import pytest

from app.evaluation.evaluator import BIEvaluator


def test_correct_numerical_result():
    evaluator = BIEvaluator()

    result = evaluator.evaluate_case(
        case_id="revenue_001",
        category="numerical",
        operation=lambda: 1_080_000.0,
        expected_value=1_080_000.0,
    )

    assert result.execution_success
    assert result.passed
    assert result.actual_value == 1_080_000.0


def test_incorrect_numerical_result_fails():
    evaluator = BIEvaluator()

    result = evaluator.evaluate_case(
        case_id="revenue_002",
        category="numerical",
        operation=lambda: 900_000.0,
        expected_value=1_080_000.0,
    )

    assert result.execution_success
    assert not result.passed


def test_operation_failure_is_recorded():
    evaluator = BIEvaluator()

    def failing_operation():
        raise RuntimeError("Simulated failure")

    result = evaluator.evaluate_case(
        case_id="failure_001",
        category="execution",
        operation=failing_operation,
    )

    assert not result.execution_success
    assert not result.passed
    assert "Simulated failure" in result.error


def test_tool_routing_is_evaluated():
    evaluator = BIEvaluator()

    result = evaluator.evaluate_case(
        case_id="routing_001",
        category="routing",
        operation=lambda: {
            "actual_tool": "kpi_detection",
        },
        expected_tool="kpi_detection",
    )

    assert result.passed


def test_incorrect_tool_routing_fails():
    evaluator = BIEvaluator()

    result = evaluator.evaluate_case(
        case_id="routing_002",
        category="routing",
        operation=lambda: {
            "actual_tool": "sql_query",
        },
        expected_tool="kpi_detection",
    )

    assert not result.passed


def test_report_aggregates_metrics():
    evaluator = BIEvaluator()

    evaluator.evaluate_case(
        case_id="pass_001",
        category="numerical",
        operation=lambda: 100.0,
        expected_value=100.0,
    )

    evaluator.evaluate_case(
        case_id="fail_001",
        category="numerical",
        operation=lambda: 50.0,
        expected_value=100.0,
    )

    report = evaluator.report()

    assert report["total_cases"] == 2
    assert report["passed_cases"] == 1
    assert report["failed_cases"] == 1
    assert report["task_success_rate"] == 50.0
    assert report["execution_success_rate"] == 100.0
    assert report["error_rate"] == 0.0


def test_empty_report():
    report = BIEvaluator().report()

    assert report["total_cases"] == 0
    assert report["task_success_rate"] == 0.0


def test_numerical_tolerance():
    evaluator = BIEvaluator()

    result = evaluator.evaluate_case(
        case_id="tolerance_001",
        category="numerical",
        operation=lambda: 100.005,
        expected_value=100.0,
        tolerance=0.01,
    )

    assert result.passed