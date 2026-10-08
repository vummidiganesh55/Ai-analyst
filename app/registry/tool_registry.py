from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ToolDefinition:
    name: str
    description: str
    function: Callable[..., Any] | None
    risk_level: str
    timeout_seconds: int


class ToolRegistry:

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str,
        description: str,
        function: Callable[..., Any] | None,
        risk_level: str = "low",
        timeout_seconds: int = 10,
    ) -> None:

        if name in self._tools:
            raise ValueError(
                f"Tool already registered: {name}"
            )

        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            function=function,
            risk_level=risk_level,
            timeout_seconds=timeout_seconds,
        )

    def get(self, name: str) -> ToolDefinition:

        if name not in self._tools:
            raise KeyError(
                f"Unknown tool: {name}"
            )

        return self._tools[name]

    def list_tools(self) -> list[ToolDefinition]:

        return list(self._tools.values())

    def has_tool(self, name: str) -> bool:

        return name in self._tools