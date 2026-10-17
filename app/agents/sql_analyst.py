"""
Compatibility module for the NEXORA AI SQL Analyst.

The canonical implementation is:
    app.sql.sql_agent

This module re-exports the same classes so that existing imports and
tests continue to use the canonical implementation.
"""

from app.sql.sql_agent import (
    SQLAnalyst,
    SQLRequest,
    SQLResponse,
)

__all__ = [
    "SQLAnalyst",
    "SQLRequest",
    "SQLResponse",
]