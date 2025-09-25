from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic

from restate import Context, RunOptions

from pydantic_ai import ToolDefinition
from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.mcp import MCPServer, ToolResult
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool
from pydantic_ai.toolsets.wrapper import WrapperToolset

from ._serde import PydanticTypeAdapter


@dataclass
class RestateContextRunResult:
    """A simple wrapper for tool results to be used with Restate's run_typed."""

    output: Any


@dataclass
class RestateMCPGetToolsContextRunResult:
    """A simple wrapper for tool results to be used with Restate's run_typed."""

    output: dict[str, ToolDefinition]


@dataclass
class RestateMCPToolRunResult(Generic[AgentDepsT]):
    """A simple wrapper for tool results to be used with Restate's run_typed."""

    output: ToolResult


class RestateContextRunToolSet(WrapperToolset[AgentDepsT]):
    """A toolset that automatically wraps tool calls with restate's `ctx.run_typed()`."""

    def __init__(self, wrapped: AbstractToolset[AgentDepsT], context: Context):
        super().__init__(wrapped)
        self._context = context
        self.options = RunOptions[RestateContextRunResult](serde=PydanticTypeAdapter(RestateContextRunResult))

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        async def action() -> RestateContextRunResult:
            output = await self.wrapped.call_tool(name, tool_args, ctx, tool)
            return RestateContextRunResult(output=output)

        res = await self._context.run_typed(f'Calling {name}', action, self.options)
        return res.output

    def visit_and_replace(
        self, visitor: Callable[[AbstractToolset[AgentDepsT]], AbstractToolset[AgentDepsT]]
    ) -> AbstractToolset[AgentDepsT]:
        return visitor(self)


class RestateMCPServer(WrapperToolset[AgentDepsT]):
    """A wrapper for MCPServer that integrates with restate."""

    def __init__(self, wrapped: MCPServer, context: Context):
        super().__init__(wrapped)
        self._wrapped = wrapped
        self._context = context
        self._mcp_tool_run_serde = PydanticTypeAdapter(RestateMCPToolRunResult[AgentDepsT])
        self._mcp_get_tools_serde = PydanticTypeAdapter(RestateMCPGetToolsContextRunResult)

    def visit_and_replace(
        self, visitor: Callable[[AbstractToolset[AgentDepsT]], AbstractToolset[AgentDepsT]]
    ) -> AbstractToolset[AgentDepsT]:
        return visitor(self)

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        async def get_tools_in_context() -> RestateMCPGetToolsContextRunResult:
            res = await self._wrapped.get_tools(ctx)
            # ToolsetTool is not serializable as it holds a SchemaValidator (which is also the same for every MCP tool so unnecessary to pass along the wire every time),
            # so we just return the ToolDefinitions and wrap them in ToolsetTool outside of the activity.
            return RestateMCPGetToolsContextRunResult(output={name: tool.tool_def for name, tool in res.items()})

        options = RunOptions(serde=self._mcp_get_tools_serde)

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
        async def call_tool_in_context() -> RestateMCPToolRunResult[AgentDepsT]:
            res = await self._wrapped.call_tool(name, tool_args, ctx, tool)
            return RestateMCPToolRunResult(output=res)

        options = RunOptions(serde=self._mcp_tool_run_serde)
        res = await self._context.run_typed(f'Calling mcp tool {name}', call_tool_in_context, options)

        return res.output
