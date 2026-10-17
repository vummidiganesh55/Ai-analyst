from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable


@dataclass
class EvaluationCase:
    case_id: str
    category: str
    expected_tool: str | None = None
    expected_value: float | None = None
    actual_value: float | None = None
    tolerance: float = 0.01
    execution_success: bool = False
    passed: bool = False
    latency_seconds: float = 0.0
    error: str | None = None
    details: dict[str, Any] | None = None


class BIEvaluator:
    """Evaluate deterministic BI tasks and aggregate their metrics."""

    def __init__(self) -> None:
        self.results: list[EvaluationCase] = []

    def evaluate_case(
        self,
        case_id: str,
        category: str,
        operation: Callable[[], Any],
        expected_value: float | None = None,
        expected_tool: str | None = None,
        tolerance: float = 0.01,
    ) -> EvaluationCase:
        start = time.perf_counter()

        actual_value: float | None = None
        actual_tool: str | None = None
        details: dict[str, Any] | None = None
        error: str | None = None

        if tolerance < 0 or not math.isfinite(tolerance):
            raise ValueError("tolerance must be finite and non-negative.")

        if expected_value is not None and not math.isfinite(expected_value):
            raise ValueError("expected_value must be finite.")

        try:
            output = operation()

            if isinstance(output, dict):
                raw_value = output.get("actual_value")
                actual_tool = output.get("actual_tool")
                raw_details = output.get("details")

                if isinstance(raw_details, dict):
                    details = raw_details
            else:
                raw_value = output

            if raw_value is not None:
                actual_value = float(raw_value)

                if not math.isfinite(actual_value):
                    raise ValueError(
                        "Evaluation result must be a finite number."
                    )

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            actual_value = None

        latency = time.perf_counter() - start
        execution_success = error is None

        numerical_correct = True

        if expected_value is not None:
            numerical_correct = (
                actual_value is not None
                and abs(actual_value - expected_value) <= tolerance
            )

        routing_correct = (
            expected_tool is None
            or actual_tool == expected_tool
        )

        passed = (
            execution_success
            and numerical_correct
            and routing_correct
        )

        result = EvaluationCase(
            case_id=case_id,
            category=category,
            expected_tool=expected_tool,
            expected_value=expected_value,
            actual_value=actual_value,
            tolerance=tolerance,
            execution_success=execution_success,
            passed=passed,
            latency_seconds=round(latency, 6),
            error=error,
            details=details,
        )

        self.results.append(result)
        return result

    def report(self) -> dict[str, Any]:
        total = len(self.results)

        if total == 0:
            return {
                "total_cases": 0,
                "passed_cases": 0,
                "failed_cases": 0,
                "task_success_rate": 0.0,
                "execution_success_rate": 0.0,
                "error_rate": 0.0,
                "average_latency_seconds": 0.0,
                "results": [],
            }

        passed = sum(result.passed for result in self.results)
        execution_successes = sum(
            result.execution_success for result in self.results
        )

        failed = total - passed
        errors = total - execution_successes

        return {
            "total_cases": total,
            "passed_cases": passed,
            "failed_cases": failed,
            "task_success_rate": round(passed / total * 100, 2),
            "execution_success_rate": round(
                execution_successes / total * 100, 2
            ),
            "error_rate": round(errors / total * 100, 2),
            "average_latency_seconds": round(
                sum(
                    result.latency_seconds for result in self.results
                ) / total,
                6,
            ),
            "results": [
                asdict(result) for result in self.results
            ],
        }