from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class CorrectionResult:
    success: bool
    result: Any = None
    error: str | None = None
    attempts: int = 0
    corrected: bool = False


class SelfCorrectionEngine:
    """
    Bounded retry and self-correction engine.

    Responsibilities:
    - Execute a tool/function.
    - Detect execution failures.
    - Retry within a fixed limit.
    - Preserve the original input.
    - Prevent infinite retry loops.
    """

    def __init__(
        self,
        max_retries: int = 2,
    ):
        if max_retries < 0:
            raise ValueError(
                "max_retries must be greater than or equal to 0"
            )

        self.max_retries = max_retries

    def execute(
        self,
        function: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> CorrectionResult:
        """
        Execute a function with bounded retries.
        """

        attempts = 0
        last_error: str | None = None

        while attempts <= self.max_retries:
            attempts += 1

            try:
                result = function(*args, **kwargs)

                return CorrectionResult(
                    success=True,
                    result=result,
                    attempts=attempts,
                    corrected=attempts > 1,
                )

            except Exception as exc:
                last_error = str(exc)

        return CorrectionResult(
            success=False,
            error=last_error,
            attempts=attempts,
            corrected=attempts > 1,
        )

    def should_retry(
        self,
        attempts: int,
    ) -> bool:
        """
        Determine whether another retry is allowed.
        """

        return attempts <= self.max_retries

    def correct_error(
        self,
        error: str,
        original_input: Any,
    ) -> dict[str, Any]:
        """
        Produce a structured correction context.

        Actual LLM-based correction will be connected later.
        """

        return {
            "error": error,
            "original_input": original_input,
            "action": "retry",
            "attempt_limit": self.max_retries,
        }