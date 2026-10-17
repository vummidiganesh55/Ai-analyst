import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "sales_data_55bc5f65314e"


def print_result(name, result):
    print(f"\n{'=' * 65}")
    print(f"TOOL: {name}")
    print(f"{'=' * 65}")

    if result.structuredContent is not None:
        print(
            json.dumps(
                result.structuredContent,
                indent=2,
                default=str,
            )
        )
    else:
        for content in result.content:
            if hasattr(content, "text"):
                print(content.text)
            else:
                print(content)

    print("MCP tool error flag:", result.isError)


async def main():
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.integrations.mcp_server"],
        cwd=str(PROJECT_ROOT),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print("MCP CLIENT CONNECTED")

            tools_result = await session.list_tools()
            tool_names = {tool.name for tool in tools_result.tools}

            print("\nDiscovered tools:")
            for name in sorted(tool_names):
                print("-", name)

            expected = {
                "health_check",
                "dataset_profile",
                "calculate_kpis",
                "analyze_root_cause",
                "generate_recommendations",
                "search_business_documents",
            }

            missing = expected - tool_names

            if missing:
                raise AssertionError(
                    f"Missing MCP tools: {sorted(missing)}"
                )

            print("\nPASS: All six tools discovered.")

            # 1. Health check
            result = await session.call_tool(
                "health_check",
                arguments={},
            )
            print_result("health_check", result)

            # 2. Dataset profile
            result = await session.call_tool(
                "dataset_profile",
                arguments={"dataset_id": DATASET_ID},
            )
            print_result("dataset_profile", result)

            # 3. KPI calculation
            result = await session.call_tool(
                "calculate_kpis",
                arguments={"dataset_id": DATASET_ID},
            )
            print_result("calculate_kpis", result)

            # 4. Root-cause analysis
            # The current dataset has only January 1–10 records,
            # so it does not contain two distinct comparison periods.
            result = await session.call_tool(
                "analyze_root_cause",
                arguments={
                    "dataset_id": DATASET_ID,
                    "metric": "revenue",
                    "dimension": "region",
                    "period_column": "date",
                    "previous_period": "2026-01",
                    "current_period": "2026-02",
                },
            )
            print_result("analyze_root_cause", result)

            # 5. Recommendation generation
            result = await session.call_tool(
                "generate_recommendations",
                arguments={
                    "insights": [
                        {
                            "category": "financial",
                            "title": "Review gross profit margin",
                            "description": (
                                "Compare revenue, cost, and profit "
                                "before evaluating discount changes."
                            ),
                        }
                    ],
                    "max_recommendations": 5,
                },
            )
            print_result("generate_recommendations", result)

            # 6. RAG document search
            result = await session.call_tool(
                "search_business_documents",
                arguments={
                    "query": (
                        "How should managers evaluate discounts "
                        "and gross profit margin?"
                    ),
                    "top_k": 3,
                },
            )
            print_result("search_business_documents", result)

            print("\nMCP TOOL TEST RUN FINISHED.")


if __name__ == "__main__":
    asyncio.run(main())