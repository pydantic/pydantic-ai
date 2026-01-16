from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from restate import Context, RunOptions

from pydantic_ai import ToolDefinition
from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.mcp import MCPServer, ToolResult
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool
from pydantic_ai.toolsets.wrapper import WrapperToolset

from ._serde import PydanticTypeAdapter


@dataclass
class RestateMCPGetToolsContextRunResult:
    """A simple wrapper for tool results to be used with Restate's `ctx.run_typed()`."""

    output: dict[str, ToolDefinition]


MCP_GET_TOOLS_SERDE = PydanticTypeAdapter(RestateMCPGetToolsContextRunResult)


@dataclass
class RestateMCPToolRunResult:
    """A simple wrapper for tool results to be used with Restate's `ctx.run_typed()`."""

    output: ToolResult


MCP_RUN_SERDE = PydanticTypeAdapter(RestateMCPToolRunResult)


class RestateMCPServer(WrapperToolset[AgentDepsT]):
    """A wrapper for [`MCPServer`][pydantic_ai.mcp.MCPServer] that integrates with Restate."""

    def __init__(self, wrapped: MCPServer, context: Context):
        super().__init__(wrapped)
        self._wrapped = wrapped
        self._context = context

    def visit_and_replace(
        self, visitor: Callable[[AbstractToolset[AgentDepsT]], AbstractToolset[AgentDepsT]]
    ) -> AbstractToolset[AgentDepsT]:
        return visitor(self)

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        async def get_tools_in_context() -> RestateMCPGetToolsContextRunResult:
            res = await self._wrapped.get_tools(ctx)
            # ToolsetTool is not serializable as it holds a SchemaValidator
            # (which is also the same for every MCP tool so unnecessary to pass along the wire every time),
            # so we just return the ToolDefinitions and wrap them in ToolsetTool outside of ctx.run_typed().
            return RestateMCPGetToolsContextRunResult(output={name: tool.tool_def for name, tool in res.items()})

        options = RunOptions(serde=MCP_GET_TOOLS_SERDE)
        tool_defs = await self._context.run_typed('get mcp tools', get_tools_in_context, options)

        return {name: self.tool_for_tool_def(tool_def) for name, tool_def in tool_defs.output.items()}

    def tool_for_tool_def(self, tool_def: ToolDefinition) -> ToolsetTool[AgentDepsT]:
        assert isinstance(self.wrapped, MCPServer)
        return self.wrapped.tool_for_tool_def(tool_def)

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> ToolResult:
        async def call_tool_in_context() -> RestateMCPToolRunResult:
            res = await self._wrapped.call_tool(name, tool_args, ctx, tool)
            return RestateMCPToolRunResult(output=res)

        options = RunOptions(serde=MCP_RUN_SERDE)
        res = await self._context.run_typed(f'Calling mcp tool {name}', call_tool_in_context, options)

        return res.output

