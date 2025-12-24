"""Base classes for the tool system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class ConfirmationRequired(Exception):
    """Raised when a tool requires user confirmation before execution."""

    def __init__(self, tool_name: str, message: str, details: str = ""):
        self.tool_name = tool_name
        self.message = message
        self.details = details
        super().__init__(message)


@dataclass
class ToolResult:
    """Result of a tool execution."""
    output: str
    is_error: bool = False


class Tool(ABC):
    """Abstract base class for tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name used in function calls."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what the tool does."""
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON Schema for tool parameters."""
        ...

    @property
    def is_destructive(self) -> bool:
        """Whether this tool can modify files or run commands.

        Override in subclasses for tools that modify state.
        """
        return False

    def check_confirmation(self, skip_confirmation: bool = False, **kwargs: Any) -> None:
        """Check if this operation requires confirmation.

        Args:
            skip_confirmation: If True, skip the confirmation check
            **kwargs: Tool arguments for context in confirmation message

        Raises:
            ConfirmationRequired: If user confirmation is needed
        """
        pass  # Default: no confirmation needed

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with given parameters."""
        ...

    def to_schema(self) -> dict[str, Any]:
        """Convert tool to JSON schema for LLM."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    """Registry of available tools."""

    def __init__(self, skip_confirmation: bool = False) -> None:
        self._tools: dict[str, Tool] = {}
        self.skip_confirmation = skip_confirmation

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        """List all registered tools."""
        return list(self._tools.values())

    def get_schemas(self) -> list[dict[str, Any]]:
        """Get JSON schemas for all tools."""
        return [tool.to_schema() for tool in self._tools.values()]

    async def execute(self, name: str, **kwargs: Any) -> ToolResult:
        """Execute a tool by name.

        Args:
            name: Tool name to execute
            **kwargs: Arguments to pass to the tool

        Returns:
            ToolResult with output

        Raises:
            ConfirmationRequired: If tool needs confirmation and skip_confirmation is False
        """
        tool = self.get(name)
        if tool is None:
            return ToolResult(
                output=f"Unknown tool: {name}",
                is_error=True,
            )
        try:
            # Check for confirmation (may raise ConfirmationRequired)
            tool.check_confirmation(
                skip_confirmation=self.skip_confirmation,
                **kwargs,
            )
            return await tool.execute(**kwargs)
        except ConfirmationRequired:
            raise  # Re-raise confirmation requests
        except Exception as e:
            return ToolResult(
                output=f"Tool execution error: {e}",
                is_error=True,
            )


def create_default_registry() -> ToolRegistry:
    """Create a registry with default tools."""
    from .file_tools import ReadTool, WriteTool, EditTool, GlobTool
    from .shell_tools import BashTool
    from .search_tools import GrepTool

    registry = ToolRegistry()
    registry.register(ReadTool())
    registry.register(WriteTool())
    registry.register(EditTool())
    registry.register(GlobTool())
    registry.register(BashTool())
    registry.register(GrepTool())

    return registry
