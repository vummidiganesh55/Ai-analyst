"""
Unit tests for the NEXORA AI SQL Analyst.

These tests verify SQL generation through both supported import paths.
They test generated SQL without requiring a running API server.
"""

import pytest

from app.sql.sql_agent import (
    SQLAnalyst as CanonicalSQLAnalyst,
    SQLRequest as CanonicalSQLRequest,
    SQLResponse as CanonicalSQLResponse,
)

from app.agents.sql_analyst import (
    SQLAnalyst as AgentSQLAnalyst,
    SQLRequest as AgentSQLRequest,
    SQLResponse as AgentSQLResponse,
)


TABLE_NAME = "nexora_ai_sales_test_dataset"


# ---------------------------------------------------------
# Import compatibility
# ---------------------------------------------------------

def test_both_import_paths_use_same_sql_analyst():
    assert CanonicalSQLAnalyst is AgentSQLAnalyst


def test_both_import_paths_use_same_request_class():
    assert CanonicalSQLRequest is AgentSQLRequest


def test_both_import_paths_use_same_response_class():
    assert CanonicalSQLResponse is AgentSQLResponse


# ---------------------------------------------------------
# Total profit
# ---------------------------------------------------------

def test_generate_total_profit_sql():
    analyst = CanonicalSQLAnalyst()

    request = CanonicalSQLRequest(
        question="What is the total profit, calculated as revenue minus cost?",
        table_name=TABLE_NAME,
    )

    response = analyst.generate_sql(request)

    assert "SUM(revenue - cost)" in response.sql
    assert f'FROM "{TABLE_NAME}"' in response.sql
    assert "total_profit" in response.sql


# ---------------------------------------------------------
# Total revenue
# ---------------------------------------------------------

def test_generate_total_revenue_sql():
    analyst = CanonicalSQLAnalyst()

    request = CanonicalSQLRequest(
        question="What is the total revenue?",
        table_name=TABLE_NAME,
    )

    response = analyst.generate_sql(request)

    assert "SUM(revenue)" in response.sql
    assert "total_revenue" in response.sql
    assert f'FROM "{TABLE_NAME}"' in response.sql


# ---------------------------------------------------------
# Average revenue
# ---------------------------------------------------------

def test_generate_average_revenue_sql():
    analyst = CanonicalSQLAnalyst()

    request = CanonicalSQLRequest(
        question="What is the average revenue?",
        table_name=TABLE_NAME,
    )

    response = analyst.generate_sql(request)

    assert "AVG(revenue)" in response.sql
    assert "average_revenue" in response.sql
    assert f'FROM "{TABLE_NAME}"' in response.sql


# ---------------------------------------------------------
# Total cost
# ---------------------------------------------------------

def test_generate_total_cost_sql():
    analyst = CanonicalSQLAnalyst()

    request = CanonicalSQLRequest(
        question="What is the total cost?",
        table_name=TABLE_NAME,
    )

    response = analyst.generate_sql(request)

    assert "SUM(cost)" in response.sql
    assert "total_cost" in response.sql
    assert f'FROM "{TABLE_NAME}"' in response.sql


# ---------------------------------------------------------
# Revenue by region
# ---------------------------------------------------------

def test_generate_revenue_by_region_sql():
    analyst = CanonicalSQLAnalyst()

    request = CanonicalSQLRequest(
        question="Compare revenue by region.",
        table_name=TABLE_NAME,
    )

    response = analyst.generate_sql(request)

    assert "GROUP BY region" in response.sql
    assert "SUM(revenue)" in response.sql
    assert "total_revenue" in response.sql
    assert "ORDER BY total_revenue DESC" in response.sql
    assert f'FROM "{TABLE_NAME}"' in response.sql


# ---------------------------------------------------------
# Average revenue by region
# ---------------------------------------------------------

def test_generate_average_revenue_by_region_sql():
    analyst = CanonicalSQLAnalyst()

    request = CanonicalSQLRequest(
        question="What is the average revenue by region?",
        table_name=TABLE_NAME,
    )

    response = analyst.generate_sql(request)

    assert "AVG(revenue)" in response.sql
    assert "GROUP BY region" in response.sql
    assert "average_revenue" in response.sql
    assert f'FROM "{TABLE_NAME}"' in response.sql


# ---------------------------------------------------------
# Best-performing products
# ---------------------------------------------------------

def test_generate_best_performing_products_sql():
    analyst = CanonicalSQLAnalyst()

    request = CanonicalSQLRequest(
        question="Which products perform best?",
        table_name=TABLE_NAME,
    )

    response = analyst.generate_sql(request)

    assert "GROUP BY product" in response.sql
    assert "SUM(revenue)" in response.sql
    assert "total_revenue" in response.sql
    assert "SUM(cost)" in response.sql
    assert "total_cost" in response.sql
    assert "SUM(revenue - cost)" in response.sql
    assert "total_profit" in response.sql
    assert "profit_margin_pct" in response.sql
    assert "ORDER BY total_revenue DESC" in response.sql
    assert f'FROM "{TABLE_NAME}"' in response.sql


def test_best_products_question_variant():
    analyst = CanonicalSQLAnalyst()

    request = CanonicalSQLRequest(
        question="Show me the top products.",
        table_name=TABLE_NAME,
    )

    response = analyst.generate_sql(request)

    assert "GROUP BY product" in response.sql
    assert "ORDER BY total_revenue DESC" in response.sql


# ---------------------------------------------------------
# Product profitability
# ---------------------------------------------------------

def test_generate_product_profitability_sql():
    analyst = CanonicalSQLAnalyst()

    request = CanonicalSQLRequest(
        question="Show product profitability.",
        table_name=TABLE_NAME,
    )

    response = analyst.generate_sql(request)

    assert "GROUP BY product" in response.sql
    assert "total_revenue" in response.sql
    assert "total_cost" in response.sql
    assert "total_profit" in response.sql
    assert "profit_margin_pct" in response.sql
    assert "ORDER BY total_profit DESC" in response.sql


# ---------------------------------------------------------
# Table-name validation
# ---------------------------------------------------------

def test_reject_invalid_table_name():
    analyst = CanonicalSQLAnalyst()

    request = CanonicalSQLRequest(
        question="What is the total revenue?",
        table_name='sales"; DROP TABLE customers; --',
    )

    with pytest.raises(ValueError, match="Invalid table name"):
        analyst.generate_sql(request)


def test_reject_empty_table_name():
    analyst = CanonicalSQLAnalyst()

    request = CanonicalSQLRequest(
        question="What is the total revenue?",
        table_name="",
    )

    with pytest.raises(ValueError, match="Table name cannot be empty"):
        analyst.generate_sql(request)


# ---------------------------------------------------------
# Empty and unsupported questions
# ---------------------------------------------------------

def test_reject_empty_question():
    analyst = CanonicalSQLAnalyst()

    request = CanonicalSQLRequest(
        question="   ",
        table_name=TABLE_NAME,
    )

    with pytest.raises(ValueError, match="Question cannot be empty"):
        analyst.generate_sql(request)


def test_reject_unsupported_question():
    analyst = CanonicalSQLAnalyst()

    request = CanonicalSQLRequest(
        question="Predict next year's revenue using a neural network.",
        table_name=TABLE_NAME,
    )

    with pytest.raises(
        ValueError,
        match="Unable to generate SQL",
    ):
        analyst.generate_sql(request)


# ---------------------------------------------------------
# Response structure
# ---------------------------------------------------------

def test_sql_response_contains_sql_and_explanation():
    analyst = CanonicalSQLAnalyst()

    request = CanonicalSQLRequest(
        question="What is the total revenue?",
        table_name=TABLE_NAME,
    )

    response = analyst.generate_sql(request)

    assert isinstance(response, CanonicalSQLResponse)
    assert response.sql.strip()
    assert response.explanation.strip()