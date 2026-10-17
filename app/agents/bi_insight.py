from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class BIInsight:
    """A structured, evidence-backed business insight."""

    title: str
    observation: str
    business_impact: str
    category: str
    confidence: str
    evidence: dict[str, Any]


class BIInsightAgent:
    """
    Generates business insights from existing analytical results.

    Numerical calculations remain the responsibility of deterministic
    analytics engines. This agent interprets supplied evidence.
    """

    VALID_CONFIDENCE = {"high", "medium", "low"}

    def __init__(self, provider: Any | None = None):
        self.provider = provider

    @staticmethod
    def _validate_results(
        results: dict[str, Any],
    ) -> None:
        if not isinstance(results, dict):
            raise TypeError("results must be a dictionary")

        if not results:
            raise ValueError("results cannot be empty")

    @staticmethod
    def _number(value: Any) -> float | None:
        """Return a finite numeric value, excluding booleans."""
        import math

        if isinstance(value, bool):
            return None

        if not isinstance(value, (int, float)):
            return None

        if not math.isfinite(value):
            return None

        return float(value)

    def _build_deterministic_insights(
        self,
        results: dict[str, Any],
    ) -> list[BIInsight]:
        """Generate insights only when the supplied evidence supports them."""

        insights: list[BIInsight] = []

        revenue = self._number(results.get("revenue"))
        cost = self._number(results.get("cost"))
        profit = self._number(results.get("profit"))
        margin = self._number(results.get("margin"))

        if revenue is not None and cost is not None:
            if revenue > cost:
                insights.append(
                    BIInsight(
                        title="Revenue Exceeds Cost",
                        observation=(
                            "The supplied results show revenue greater "
                            "than recorded cost."
                        ),
                        business_impact=(
                            "The supplied financial figures indicate "
                            "positive operating profitability before "
                            "any unrepresented expenses."
                        ),
                        category="financial",
                        confidence="high",
                        evidence={
                            "revenue": revenue,
                            "cost": cost,
                            "relationship": "revenue > cost",
                        },
                    )
                )

            elif revenue < cost:
                insights.append(
                    BIInsight(
                        title="Cost Exceeds Revenue",
                        observation=(
                            "The supplied results show recorded cost "
                            "greater than revenue."
                        ),
                        business_impact=(
                            "The supplied figures indicate a loss "
                            "before any unrepresented adjustments."
                        ),
                        category="financial",
                        confidence="high",
                        evidence={
                            "revenue": revenue,
                            "cost": cost,
                            "relationship": "revenue < cost",
                        },
                    )
                )

            else:
                insights.append(
                    BIInsight(
                        title="Revenue Equals Cost",
                        observation=(
                            "The supplied revenue and cost values are equal."
                        ),
                        business_impact=(
                            "These figures indicate a break-even result "
                            "before unrepresented adjustments."
                        ),
                        category="financial",
                        confidence="high",
                        evidence={
                            "revenue": revenue,
                            "cost": cost,
                        },
                    )
                )

        if profit is not None and profit < 0:
            insights.append(
                BIInsight(
                    title="Negative Profit",
                    observation=(
                        "The supplied profit result is negative."
                    ),
                    business_impact=(
                        "Investigate the cost structure, revenue drivers, "
                        "and reporting period."
                    ),
                    category="financial",
                    confidence="high",
                    evidence={"profit": profit},
                )
            )

        if margin is not None:
            if margin < 0:
                observation = (
                    "The supplied profit margin is negative."
                )
                impact = (
                    "Review revenue and costs to understand the negative "
                    "margin."
                )
            elif margin == 0:
                observation = (
                    "The supplied profit margin is zero."
                )
                impact = (
                    "Revenue and costs may be at break-even, depending "
                    "on the margin definition."
                )
            else:
                observation = (
                    "The supplied profit margin is positive."
                )
                impact = (
                    "The supplied margin indicates positive profitability "
                    "under the metric's definition."
                )

            insights.append(
                BIInsight(
                    title="Profit Margin Assessment",
                    observation=observation,
                    business_impact=impact,
                    category="financial",
                    confidence="high",
                    evidence={"margin": margin},
                )
            )

        return insights

    def generate(
        self,
        results: dict[str, Any],
        use_llm: bool = False,
    ) -> list[BIInsight]:
        """
        Generate insights from analytical results.

        Deterministic rules are always available. LLM interpretation
        can be enabled when a compatible provider is supplied.
        """

        self._validate_results(results)

        insights = self._build_deterministic_insights(results)

        if use_llm:
            if self.provider is None:
                raise ValueError(
                    "An LLM provider is required when use_llm=True"
                )

            # LLM interpretation is intentionally not allowed to
            # replace or modify deterministic numerical evidence.
            prompt = (
                "Interpret these analytical results for a business user. "
                "Use only the supplied facts. Do not invent values, "
                "calculate new metrics, or claim causation without evidence. "
                "Clearly distinguish observations from hypotheses.\n\n"
                f"Analytical results: {results}\n\n"
                f"Deterministic insights: "
                f"{[asdict(item) for item in insights]}"
            )

            response = self.provider.invoke(
                prompt=prompt,
                system_prompt=(
                    "You are an evidence-grounded business intelligence "
                    "analyst. Never fabricate data or numerical results."
                ),
            )

            # Preserve the original evidence-backed results and return
            # the interpretation separately as an explicitly labelled
            # narrative insight.
            if not isinstance(response, str) or not response.strip():
                raise ValueError(
                    "The LLM provider returned an empty response"
                )

            insights.append(
                BIInsight(
                    title="LLM Business Interpretation",
                    observation=response.strip(),
                    business_impact=(
                        "Review this interpretation against the supplied "
                        "analytical evidence before acting on it."
                    ),
                    category="interpretation",
                    confidence="low",
                    evidence={
                        "source": "llm_interpretation",
                        "input_results": results,
                    },
                )
            )

        return insights

    def to_dict(
        self,
        results: dict[str, Any],
        use_llm: bool = False,
    ) -> list[dict[str, Any]]:
        """Return JSON-compatible insight dictionaries."""

        insights = self.generate(
            results=results,
            use_llm=use_llm,
        )

        return [asdict(insight) for insight in insights]