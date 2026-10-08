from app.sql.sql_agent import (
    SQLAnalyst,
    SQLRequest,
)


def test_total_revenue_sql():

    agent = SQLAnalyst()

    response = agent.generate_sql(
        SQLRequest(
            question="What is the total revenue?",
            table_name="sales",
        )
    )

    assert "SUM(revenue)" in response.sql
    assert "sales" in response.sql


def test_revenue_by_region_sql():

    agent = SQLAnalyst()

    response = agent.generate_sql(
        SQLRequest(
            question="Show revenue by region",
            table_name="sales",
        )
    )

    assert "GROUP BY region" in response.sql
    assert "SUM(revenue)" in response.sql