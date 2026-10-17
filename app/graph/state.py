from __future__ import annotations

from typing import Any, TypedDict


class BIState(TypedDict, total=False):
    """
    Shared state passed between LangGraph nodes.
    """

    question: str

    selected_table: str

    plan: Any

    execution_results: list[dict[str, Any]]

    evidence: list[dict[str, Any]]

    final_answer: str

    error: str

    iteration: int