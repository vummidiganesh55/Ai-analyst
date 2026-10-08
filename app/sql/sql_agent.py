from dataclasses import dataclass


@dataclass
class SQLRequest:
    question: str
    table_name: str


@dataclass
class SQLResponse:
    sql: str
    explanation: str


class SQLAnalyst:

    def generate_sql(
        self,
        request: SQLRequest,
    ) -> SQLResponse:

        question = request.question.lower()
        table = request.table_name

        if "total revenue" in question:
            sql = f"""
SELECT SUM(revenue) AS total_revenue
FROM "{table}"
""".strip()

            return SQLResponse(
                sql=sql,
                explanation="Calculates total revenue.",
            )

        if "average revenue" in question:
            sql = f"""
SELECT AVG(revenue) AS average_revenue
FROM "{table}"
""".strip()

            return SQLResponse(
                sql=sql,
                explanation="Calculates average revenue.",
            )

        if "revenue by region" in question:
            sql = f"""
SELECT
    region,
    SUM(revenue) AS total_revenue
FROM "{table}"
GROUP BY region
ORDER BY total_revenue DESC
""".strip()

            return SQLResponse(
                sql=sql,
                explanation="Calculates revenue grouped by region.",
            )

        raise ValueError(
            "Unable to generate SQL for the given question."
        )