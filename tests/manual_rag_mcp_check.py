import asyncio
import json
import os
import sys
import time
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


PROJECT_ROOT = Path(__file__).resolve().parents[1]


async def main():
    print("Starting MCP RAG integration test...", flush=True)

    # Configure the MCP server subprocess.
    server_env = os.environ.copy()
    server_env["PYTHONUNBUFFERED"] = "1"
    server_env["HF_HUB_OFFLINE"] = "1"

    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-u", "-m", "app.integrations.mcp_server"],
        cwd=str(PROJECT_ROOT),
        env=server_env,
    )

    print("Launching MCP server...", flush=True)

    # Start the MCP server and establish communication.
    async with stdio_client(
        parameters,
        errlog=sys.stderr,
    ) as (read, write):

        async with ClientSession(read, write) as session:

            print("Initializing MCP session...", flush=True)

            await session.initialize()

            print("MCP client connected.", flush=True)

            # --------------------------------------------------
            # STEP 1: Discover MCP tools
            # --------------------------------------------------

            print("\nSTEP 1: Discovering MCP tools...", flush=True)

            tools_result = await session.list_tools()

            tool_names = [
                tool.name
                for tool in tools_result.tools
            ]

            print(
                "Discovered tools:",
                tool_names,
                flush=True,
            )

            required_tools = {
                "health_check",
                "search_business_documents",
            }

            missing_tools = required_tools - set(tool_names)

            if missing_tools:
                raise RuntimeError(
                    f"Required MCP tools are missing: "
                    f"{sorted(missing_tools)}"
                )

            # --------------------------------------------------
            # STEP 2: Test health_check
            # --------------------------------------------------

            print(
                "\nSTEP 2: Calling health_check...",
                flush=True,
            )

            health_started = time.perf_counter()

            try:
                health_result = await asyncio.wait_for(
                    session.call_tool(
                        "health_check",
                        arguments={},
                    ),
                    timeout=15,
                )

                health_elapsed = (
                    time.perf_counter() - health_started
                )

                print(
                    f"Health check completed in "
                    f"{health_elapsed:.2f} seconds.",
                    flush=True,
                )

                print(
                    "MCP error flag:",
                    health_result.isError,
                    flush=True,
                )

                for content in health_result.content:
                    if hasattr(content, "text"):
                        print(content.text, flush=True)

                if health_result.isError:
                    print(
                        "\nRESULT: health_check returned an error.",
                        flush=True,
                    )
                    return

                print(
                    "RESULT: health_check succeeded.",
                    flush=True,
                )

            except asyncio.TimeoutError:
                print(
                    "\nRESULT: health_check timed out after "
                    "15 seconds.",
                    flush=True,
                )
                return

            # --------------------------------------------------
            # STEP 3: Test business-document RAG through MCP
            # --------------------------------------------------

            print(
                "\nSTEP 3: Calling search_business_documents...",
                flush=True,
            )

            query = (
                "What does the sales performance policy "
                "say about revenue?"
            )

            rag_started = time.perf_counter()

            try:
                rag_result = await asyncio.wait_for(
                    session.call_tool(
                        "search_business_documents",
                        arguments={
                            "query": query,
                            "top_k": 3,
                        },
                    ),
                    timeout=45,
                )

                rag_elapsed = (
                    time.perf_counter() - rag_started
                )

                print(
                    f"\nRAG call completed in "
                    f"{rag_elapsed:.2f} seconds.",
                    flush=True,
                )

                print(
                    "MCP error flag:",
                    rag_result.isError,
                    flush=True,
                )

                # Print textual tool results.
                for content in rag_result.content:
                    if hasattr(content, "text"):
                        print(
                            "\nMCP tool response:",
                            content.text,
                            flush=True,
                        )

                # Print structured content when provided.
                structured_content = getattr(
                    rag_result,
                    "structuredContent",
                    None,
                )

                if structured_content is not None:
                    print(
                        "\nStructured MCP response:",
                        flush=True,
                    )

                    print(
                        json.dumps(
                            structured_content,
                            indent=2,
                            default=str,
                        ),
                        flush=True,
                    )

                if rag_result.isError:
                    print(
                        "\nRESULT: MCP returned a RAG tool error.",
                        flush=True,
                    )
                else:
                    print(
                        "\nRESULT: MCP RAG tool executed successfully.",
                        flush=True,
                    )

            except asyncio.TimeoutError:
                print(
                    "\nRESULT: RAG tool timed out after "
                    "45 seconds.",
                    flush=True,
                )

            except Exception as exc:
                print(
                    "\nRESULT: Unexpected MCP RAG error.",
                    flush=True,
                )

                print(
                    f"Error type: {type(exc).__name__}",
                    flush=True,
                )

                print(
                    f"Error details: {exc}",
                    flush=True,
                )

    print(
        "\nMCP RAG integration test finished.",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())