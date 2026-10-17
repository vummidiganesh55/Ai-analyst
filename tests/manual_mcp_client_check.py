import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


PROJECT_ROOT = Path(__file__).resolve().parents[1]


async def main() -> None:
    server_parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.integrations.mcp_server"],
        cwd=str(PROJECT_ROOT),
        env=None,
    )

    print("Connecting to Autonomous BI MCP server...")

    async with stdio_client(server_parameters) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print("\nMCP connection established.")

            # Discover registered tools.
            tools_result = await session.list_tools()

            tool_names = [tool.name for tool in tools_result.tools]

            print("\nDiscovered MCP tools:")
            for name in tool_names:
                print(f"  - {name}")

            expected_tools = {
                "health_check",
                "dataset_profile",
                "calculate_kpis",
                "analyze_root_cause",
                "generate_recommendations",
                "search_business_documents",
            }

            missing_tools = expected_tools - set(tool_names)

            if missing_tools:
                raise AssertionError(
                    f"Missing MCP tools: {sorted(missing_tools)}"
                )

            print("\nPASS: All six expected tools are registered.")

            # Call the actual server's health-check tool.
            print("\nCalling health_check...")

            result = await session.call_tool(
                "health_check",
                arguments={},
            )

            if result.isError:
                raise RuntimeError(
                    f"Health check failed: {result.content}"
                )

            print("\nHealth-check response:")

            for content in result.content:
                if hasattr(content, "text"):
                    print(content.text)
                else:
                    print(content)

            print("\nPASS: health_check executed successfully.")
            print("\nMCP integration smoke test completed.")


if __name__ == "__main__":
    asyncio.run(main())