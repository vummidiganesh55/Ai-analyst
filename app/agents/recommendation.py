"""
Recommendation Agent for the Autonomous AI Business Intelligence Platform.

Responsibilities:
- Convert BI insights and root-cause findings into recommendations.
- Assign priorities using deterministic rules.
- Preserve supporting evidence.
- Avoid unsupported financial claims.
- Produce structured, testable outputs.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class Recommendation:
    """A structured, evidence-grounded business recommendation."""

    title: str
    problem: str
    action: str
    rationale: str
    priority: str
    category: str
    expected_impact: str
    evidence: dict[str, Any] = field(default_factory=dict)
    confidence: str = "medium"

    def to_dict(self) -> dict[str, Any]:
        """Convert the recommendation into a JSON-compatible dictionary."""
        return asdict(self)


class RecommendationAgent:
    """
    Generate business recommendations from existing analytical results.

    This agent does not calculate financial metrics. It uses the supplied
    BI insights and root-cause analysis as evidence.
    """

    PRIORITY_LEVELS = {"critical", "high", "medium", "low"}

    def __init__(
        self,
        max_recommendations: int = 10,
    ) -> None:
        if (
            isinstance(max_recommendations, bool)
            or not isinstance(max_recommendations, int)
            or max_recommendations < 1
        ):
            raise ValueError("max_recommendations must be a positive integer.")

        self.max_recommendations = max_recommendations

    @staticmethod
    def _number(value: Any) -> Optional[float]:
        """Safely convert a numeric value, rejecting invalid or non-finite values."""
        import math

        if isinstance(value, bool):
            return None

        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None

        return number if math.isfinite(number) else None

    @staticmethod
    def _text(value: Any, default: str = "Not specified") -> str:
        """Convert a value into safe, readable text."""
        if value is None:
            return default

        text = str(value).strip()
        return text if text else default

    @staticmethod
    def _normalize_priority(priority: str) -> str:
        """Normalize priority values to the supported levels."""
        priority = priority.lower().strip()

        if priority in RecommendationAgent.PRIORITY_LEVELS:
            return priority

        return "medium"

    @staticmethod
    def _get_insight_value(insight: Any, key: str, default: Any = None) -> Any:
        """Read a field from either a dictionary or a dataclass-like object."""
        if isinstance(insight, dict):
            return insight.get(key, default)

        return getattr(insight, key, default)

    def _recommend_from_insight(
        self,
        insight: Any,
    ) -> list[Recommendation]:
        """Create recommendations from an existing BI insight."""
        title = self._text(
            self._get_insight_value(insight, "title"),
            "Business performance insight",
        )
        observation = self._text(
            self._get_insight_value(insight, "observation"),
            "A business performance observation requires investigation.",
        )
        impact = self._text(
            self._get_insight_value(insight, "business_impact"),
            "The financial impact has not been quantified.",
        )
        category = self._text(
            self._get_insight_value(insight, "category"),
            "general",
        ).lower()

        confidence = self._text(
            self._get_insight_value(insight, "confidence"),
            "medium",
        ).lower()

        evidence = self._get_insight_value(insight, "evidence", {})
        if not isinstance(evidence, dict):
            evidence = {"source_evidence": evidence}

        recommendations: list[Recommendation] = []

        if any(
            word in category
            for word in ("profit", "margin", "cost", "financial")
        ):
            recommendations.append(
                Recommendation(
                    title="Investigate profitability drivers",
                    problem=observation,
                    action=(
                        "Break down revenue, cost, and profit by product, "
                        "region, and time period. Verify the segments with "
                        "the largest adverse changes before taking action."
                    ),
                    rationale=(
                        "Segment-level analysis can identify where "
                        "profitability is deteriorating and which areas "
                        "require further investigation."
                    ),
                    priority="high",
                    category="financial",
                    expected_impact=(
                        "Identify potential opportunities to improve "
                        "profitability. The amount of improvement is "
                        "not established by the available evidence."
                    ),
                    evidence={
                        "insight_title": title,
                        "business_impact": impact,
                        "source_evidence": evidence,
                    },
                    confidence=confidence,
                )
            )

        if any(
            word in category
            for word in ("revenue", "sales", "growth", "performance")
        ):
            recommendations.append(
                Recommendation(
                    title="Investigate revenue performance by segment",
                    problem=observation,
                    action=(
                        "Compare revenue across products, regions, and "
                        "time periods. Validate significant increases or "
                        "declines against order volume and available "
                        "business records."
                    ),
                    rationale=(
                        "A segment-level comparison helps distinguish "
                        "broad performance changes from changes concentrated "
                        "in particular products or regions."
                    ),
                    priority="high",
                    category="revenue",
                    expected_impact=(
                        "Identify revenue growth opportunities or areas "
                        "requiring corrective action. No specific revenue "
                        "increase is guaranteed."
                    ),
                    evidence={
                        "insight_title": title,
                        "business_impact": impact,
                        "source_evidence": evidence,
                    },
                    confidence=confidence,
                )
            )

        if any(
            word in category
            for word in ("data_quality", "quality", "data_issue")
        ):
            recommendations.append(
                Recommendation(
                    title="Resolve data quality issues",
                    problem=observation,
                    action=(
                        "Inspect the affected records, correct verified "
                        "data errors at the source, and rerun the data "
                        "quality checks before relying on the analysis."
                    ),
                    rationale=(
                        "Incorrect, missing, or inconsistent data can "
                        "reduce the reliability of downstream analytics."
                    ),
                    priority="high",
                    category="data_quality",
                    expected_impact=(
                        "Improve confidence in downstream analysis after "
                        "the identified issues are verified and resolved."
                    ),
                    evidence={
                        "insight_title": title,
                        "business_impact": impact,
                        "source_evidence": evidence,
                    },
                    confidence=confidence,
                )
            )

        if not recommendations:
            recommendations.append(
                Recommendation(
                    title="Investigate the identified business issue",
                    problem=observation,
                    action=(
                        "Validate the observation against the underlying "
                        "data, compare relevant business segments, and "
                        "document the findings before making operational "
                        "changes."
                    ),
                    rationale=(
                        "Further validation helps establish whether the "
                        "observation represents a meaningful business "
                        "issue and what action is justified."
                    ),
                    priority="medium",
                    category=category,
                    expected_impact=(
                        "Improve understanding of the identified issue. "
                        "A measurable business outcome has not yet "
                        "been established."
                    ),
                    evidence={
                        "insight_title": title,
                        "business_impact": impact,
                        "source_evidence": evidence,
                    },
                    confidence=confidence,
                )
            )

        return recommendations

    def _recommend_from_root_cause(
        self,
        root_cause: dict[str, Any],
    ) -> list[Recommendation]:
        """Create recommendations from root-cause analysis results."""
        recommendations: list[Recommendation] = []

        metric = self._text(root_cause.get("metric"), "business metric")
        overall = root_cause.get("overall", {})

        if not isinstance(overall, dict):
            return recommendations

        previous = self._number(overall.get("previous_value"))
        current = self._number(overall.get("current_value"))
        change = self._number(overall.get("change"))

        if change is None:
            return recommendations

        if change < 0:
            direction = "declined"
            priority = "high"
            action = (
                "Inspect the segments with the largest negative "
                "contributions. Validate the underlying records and "
                "investigate relevant operational changes before "
                "choosing corrective actions."
            )
        elif change > 0:
            direction = "increased"
            priority = "medium"
            action = (
                "Identify the segments contributing most to the increase. "
                "Validate whether the change is sustainable and whether "
                "it is associated with acceptable cost, quality, and "
                "operational performance."
            )
        else:
            direction = "remained unchanged"
            priority = "low"
            action = (
                "Review segment-level changes to determine whether "
                "improvements in some areas are offset by declines "
                "in others."
            )

        recommendations.append(
            Recommendation(
                title=f"Investigate {metric} movement",
                problem=(
                    f"The {metric} {direction} between the selected "
                    "comparison periods."
                ),
                action=action,
                rationale=(
                    "The overall metric describes what changed. "
                    "Segment-level contributions help prioritize "
                    "which areas to investigate."
                ),
                priority=priority,
                category="root_cause",
                expected_impact=(
                    "Identify evidence-supported follow-up actions. "
                    "The analysis alone does not establish causality "
                    "or guarantee a particular financial outcome."
                ),
                evidence={
                    "metric": metric,
                    "previous_value": previous,
                    "current_value": current,
                    "change": change,
                    "root_cause_analysis": root_cause,
                },
                confidence="medium",
            )
        )

        return recommendations

    def generate(
        self,
        insights: Optional[list[Any]] = None,
        root_cause: Optional[dict[str, Any]] = None,
    ) -> list[Recommendation]:
        """
        Generate recommendations from BI insights and root-cause findings.

        Args:
            insights: BI insight dictionaries or BIInsight objects.
            root_cause: Root-cause analysis output dictionary.

        Returns:
            A prioritized list of structured recommendations.
        """
        if insights is None:
            insights = []

        if not isinstance(insights, list):
            raise TypeError("insights must be a list.")

        recommendations: list[Recommendation] = []

        for insight in insights:
            if insight is None:
                continue

            recommendations.extend(
                self._recommend_from_insight(insight)
            )

        if root_cause is not None:
            if not isinstance(root_cause, dict):
                raise TypeError("root_cause must be a dictionary.")

            recommendations.extend(
                self._recommend_from_root_cause(root_cause)
            )

        # Deduplicate by title while preserving the first occurrence.
        unique: dict[str, Recommendation] = {}

        for recommendation in recommendations:
            key = recommendation.title.casefold()
            if key not in unique:
                recommendation.priority = self._normalize_priority(
                    recommendation.priority
                )
                unique[key] = recommendation

        priority_rank = {
            "critical": 0,
            "high": 1,
            "medium": 2,
            "low": 3,
        }

        ordered = sorted(
            unique.values(),
            key=lambda item: priority_rank[item.priority],
        )

        return ordered[: self.max_recommendations]

    def generate_report(
        self,
        insights: Optional[list[Any]] = None,
        root_cause: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Generate a JSON-compatible recommendation report."""
        recommendations = self.generate(
            insights=insights,
            root_cause=root_cause,
        )

        return {
            "recommendation_count": len(recommendations),
            "recommendations": [
                recommendation.to_dict()
                for recommendation in recommendations
            ],
            "limitations": [
                "Recommendations are based only on supplied analytical evidence.",
                "Expected impacts are qualitative unless supported by measured evidence.",
                "Recommendations do not prove causality.",
                "Validate findings with business stakeholders before implementation.",
            ],
        }