import pytest

from app.registry.tool_registry import ToolRegistry


def test_register_and_get_tool():

    registry = ToolRegistry()

    registry.register(
        name="profile_data",
        description="Profile a dataset.",
        function=lambda: None,
        risk_level="low",
        timeout_seconds=10,
    )

    tool = registry.get("profile_data")

    assert tool.name == "profile_data"
    assert tool.risk_level == "low"
    assert tool.timeout_seconds == 10


def test_unknown_tool():

    registry = ToolRegistry()

    with pytest.raises(KeyError):

        registry.get("unknown_tool")


def test_duplicate_tool():

    registry = ToolRegistry()

    registry.register(
        name="execute_sql",
        description="Execute validated SQL.",
        function=lambda: None,
    )

    with pytest.raises(ValueError):

        registry.register(
            name="execute_sql",
            description="Duplicate tool.",
            function=lambda: None,
        )


def test_list_tools():

    registry = ToolRegistry()

    registry.register(
        name="profile_data",
        description="Profile data.",
        function=lambda: None,
    )

    registry.register(
        name="execute_sql",
        description="Execute SQL.",
        function=lambda: None,
        risk_level="medium",
    )

    tools = registry.list_tools()

    assert len(tools) == 2
    assert registry.has_tool("profile_data")
    assert registry.has_tool("execute_sql")