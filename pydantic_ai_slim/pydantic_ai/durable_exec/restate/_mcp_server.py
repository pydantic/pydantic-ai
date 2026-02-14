from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from typing_extensions import Self

from pydantic_ai import ToolDefinition
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool
from pydantic_ai.toolsets.wrapper import WrapperToolset

from ._restate_types import Context, RunOptions
from ._toolset import (
    CONTEXT_RUN_SERDE,
    RestateContextRunResult,
    run_get_tools_step,
    run_tool_call_step,
)

if TYPE_CHECKING:
    from pydantic_ai.mcp import MCPServer, ToolResult
else:
    MCPServer = Any
    ToolResult = Any


class RestateMCPServer(WrapperToolset[Any]):
    """A wrapper for [`MCPServer`][pydantic_ai.mcp.MCPServer] that integrates with Restate."""

    def __init__(self, wrapped: MCPServer, context: Context):
        super().__init__(wrapped)
        self._context = context
        self._call_options = RunOptions[RestateContextRunResult](serde=CONTEXT_RUN_SERDE)

    @property
    def _mcp_server(self) -> MCPServer:
        return cast(MCPServer, self.wrapped)

    @property
    def id(self) -> str | None:  # pragma: no cover
        return self.wrapped.id

    async def __aenter__(self) -> Self:
        """No-op: MCP server connections must be opened inside `ctx.run_typed()` for durability."""
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        """No-op: MCP server connections must be opened inside `ctx.run_typed()` for durability."""
        return None

    def visit_and_replace(
        self, visitor: Callable[[AbstractToolset[Any]], AbstractToolset[Any]]
    ) -> AbstractToolset[Any]:
        return self

    async def get_tools(self, ctx: RunContext[Any]) -> dict[str, ToolsetTool[Any]]:
        async def get_tools_action() -> dict[str, ToolDefinition]:
            async with self._mcp_server:
                res = await self._mcp_server.get_tools(ctx)
                # ToolsetTool is not serializable as it holds a SchemaValidator
                # (which is also the same for every MCP tool so unnecessary to pass along the wire every time),
                # so we return ToolDefinitions and wrap them in ToolsetTool outside durable steps.
                return {name: tool.tool_def for name, tool in res.items()}

        tool_defs = await run_get_tools_step(self._context, 'get mcp tools', get_tools_action)

        return {name: self.tool_for_tool_def(tool_def) for name, tool_def in tool_defs.items()}

    def tool_for_tool_def(self, tool_def: ToolDefinition) -> ToolsetTool[Any]:
        # Wrap the underlying tool so `tool.toolset` points at the Restate wrapper.
        tool = self._mcp_server.tool_for_tool_def(tool_def)
        return ToolsetTool(
            toolset=self,
            tool_def=tool.tool_def,
            max_retries=tool.max_retries,
            args_validator=tool.args_validator,
        )

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[Any],
        tool: ToolsetTool[Any],
    ) -> ToolResult:
        async def call_tool_action() -> ToolResult:
            async with self._mcp_server:
                return await self._mcp_server.call_tool(name, tool_args, ctx, tool)

        return await run_tool_call_step(self._context, f'Calling mcp tool {name}', call_tool_action, self._call_options)
