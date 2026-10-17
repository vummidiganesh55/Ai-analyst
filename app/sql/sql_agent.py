from dataclasses import dataclass
import re


@dataclass
class SQLRequest:
    question: str
    table_name: str


@dataclass
class SQLResponse:
    sql: str
    explanation: str


class SQLAnalyst:
    """
    Rule-based SQL generator for NEXORA AI.

    Generates SQL for supported business intelligence questions.
    The generated SQL is intended for DuckDB.
    """

    def generate_sql(self, request: SQLRequest) -> SQLResponse:
        question = " ".join(request.question.lower().strip().split())
        table = self._validate_table_name(request.table_name)

        if not question:
            raise ValueError("Question cannot be empty.")

        # --------------------------------------------------
        # 1. Best-performing products
        # --------------------------------------------------
        if self._matches(
            question,
            [
                "which products perform best",
                "which product performs best",
                "which products perform the best",
                "which product performs the best",
                "best performing products",
                "best-performing products",
                "top performing products",
                "top-performing products",
                "best performing product",
                "top performing product",
                "best products",
                "top products",
                "rank products by revenue",
                "rank products based on revenue",
                "highest revenue product",
                "highest revenue products",
            ],
        ):
            return SQLResponse(
                sql=f"""
SELECT
    product,
    SUM(revenue) AS total_revenue,
    SUM(cost) AS total_cost,
    SUM(revenue - cost) AS total_profit,
    ROUND(
        100.0 * SUM(revenue - cost)
        / NULLIF(SUM(revenue), 0),
        2
    ) AS profit_margin_pct
FROM "{table}"
GROUP BY product
ORDER BY total_revenue DESC
""".strip(),
                explanation=(
                    "Ranks products by total revenue and reports total "
                    "cost, total profit, and profit margin."
                ),
            )

        # --------------------------------------------------
        # 2. Revenue and profit by product
        # --------------------------------------------------
        if (
            "profit by product" in question
            or "product wise profit" in question
            or "product-wise profit" in question
            or "profit for each product" in question
            or "revenue and profit by product" in question
            or "product profitability" in question
        ):
            return SQLResponse(
                sql=f"""
SELECT
    product,
    SUM(revenue) AS total_revenue,
    SUM(cost) AS total_cost,
    SUM(revenue - cost) AS total_profit,
    ROUND(
        100.0 * SUM(revenue - cost)
        / NULLIF(SUM(revenue), 0),
        2
    ) AS profit_margin_pct
FROM "{table}"
GROUP BY product
ORDER BY total_profit DESC
""".strip(),
                explanation=(
                    "Calculates revenue, cost, profit, and profit margin "
                    "for each product, ordered by total profit."
                ),
            )

        # --------------------------------------------------
        # 3. Average revenue by region
        # --------------------------------------------------
        if (
            "average revenue by region" in question
            or "average revenue for each region" in question
            or "mean revenue by region" in question
            or "avg revenue by region" in question
        ):
            return SQLResponse(
                sql=f"""
SELECT
    region,
    AVG(revenue) AS average_revenue
FROM "{table}"
GROUP BY region
ORDER BY region
""".strip(),
                explanation="Calculates average revenue grouped by region.",
            )

        # --------------------------------------------------
        # 4. Revenue by region
        # --------------------------------------------------
        if (
            "revenue by region" in question
            or "compare revenue by region" in question
            or "compare the revenue by region" in question
            or "revenue for each region" in question
            or "revenue in each region" in question
            or "regional revenue" in question
            or "highest revenue region" in question
            or "region with the highest revenue" in question
            or "which region has the highest revenue" in question
            or "lowest revenue region" in question
            or "region with the lowest revenue" in question
        ):
            return SQLResponse(
                sql=f"""
SELECT
    region,
    SUM(revenue) AS total_revenue
FROM "{table}"
GROUP BY region
ORDER BY total_revenue DESC
""".strip(),
                explanation=(
                    "Calculates total revenue for each region, "
                    "ordered from highest to lowest."
                ),
            )

        # --------------------------------------------------
        # 5. Total profit
        # --------------------------------------------------
        if (
            "total profit" in question
            or "calculate profit" in question
            or "calculate the profit" in question
            or "profit calculated as revenue minus cost" in question
            or "revenue minus cost" in question
            or "sum of revenue minus cost" in question
            or "overall profit" in question
        ):
            return SQLResponse(
                sql=f"""
SELECT
    SUM(revenue - cost) AS total_profit
FROM "{table}"
""".strip(),
                explanation=(
                    "Calculates total profit as the sum of revenue minus "
                    "cost across all records."
                ),
            )

        # --------------------------------------------------
        # 6. Total cost
        # --------------------------------------------------
        if (
            "total cost" in question
            or "sum of cost" in question
            or "overall cost" in question
        ):
            return SQLResponse(
                sql=f"""
SELECT
    SUM(cost) AS total_cost
FROM "{table}"
""".strip(),
                explanation="Calculates total cost across all records.",
            )

        # --------------------------------------------------
        # 7. Total revenue
        # --------------------------------------------------
        if (
            "total revenue" in question
            or "sum of revenue" in question
            or "overall revenue" in question
        ):
            return SQLResponse(
                sql=f"""
SELECT
    SUM(revenue) AS total_revenue
FROM "{table}"
""".strip(),
                explanation="Calculates total revenue across all records.",
            )

        # --------------------------------------------------
        # 8. Average revenue
        # --------------------------------------------------
        if (
            "average revenue" in question
            or "mean revenue" in question
            or "avg revenue" in question
        ):
            return SQLResponse(
                sql=f"""
SELECT
    AVG(revenue) AS average_revenue
FROM "{table}"
""".strip(),
                explanation="Calculates average revenue across all records.",
            )

        # --------------------------------------------------
        # 9. Unsupported question
        # --------------------------------------------------
        raise ValueError(
            f"Unable to generate SQL for the given question: "
            f"{request.question}"
        )

    @staticmethod
    def _matches(question: str, phrases: list[str]) -> bool:
        """Return True when a supported phrase appears in the question."""
        return any(phrase in question for phrase in phrases)

    @staticmethod
    def _validate_table_name(table_name: str) -> str:
        """
        Validate a DuckDB table identifier before using it in SQL.

        Expected table names contain letters, digits, and underscores,
        and cannot begin with a digit.
        """
        if not table_name:
            raise ValueError("Table name cannot be empty.")

        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_name):
            raise ValueError(
                f"Invalid table name: {table_name!r}"
            )

        return table_name