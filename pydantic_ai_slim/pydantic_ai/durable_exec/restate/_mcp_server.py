from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic.errors import PydanticUserError
from typing_extensions import Self

from pydantic_ai import ToolDefinition
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, ModelRetry, UserError
from pydantic_ai.mcp import MCPServer, ToolResult
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool
from pydantic_ai.toolsets.wrapper import WrapperToolset

from ._restate_types import Context, RunOptions, TerminalError
from ._serde import PydanticTypeAdapter


@dataclass
class RestateMCPGetToolsContextRunResult:
    """A simple wrapper for tool results to be used with Restate's `ctx.run_typed()`."""

    output: dict[str, ToolDefinition]


MCP_GET_TOOLS_SERDE = PydanticTypeAdapter(RestateMCPGetToolsContextRunResult)


@dataclass
class RestateMCPToolRunResult:
    """A simple wrapper for tool results to be used with Restate's `ctx.run_typed()`."""

    kind: Literal['output', 'call_deferred', 'approval_required', 'model_retry']
    output: ToolResult | None
    error: str | None = None
    metadata: dict[str, Any] | None = None


MCP_RUN_SERDE = PydanticTypeAdapter(RestateMCPToolRunResult)


class RestateMCPServer(WrapperToolset[AgentDepsT]):
    """A wrapper for [`MCPServer`][pydantic_ai.mcp.MCPServer] that integrates with Restate."""

    def __init__(self, wrapped: MCPServer, context: Context):
        super().__init__(wrapped)
        self._wrapped = wrapped
        self._context = context

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

        options = RunOptions[RestateMCPGetToolsContextRunResult](serde=MCP_GET_TOOLS_SERDE)
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
            try:
                res = await self._wrapped.call_tool(name, tool_args, ctx, tool)
                return RestateMCPToolRunResult(kind='output', output=res, error=None)
            except ModelRetry as e:
                return RestateMCPToolRunResult(kind='model_retry', output=None, error=e.message)
            except CallDeferred as e:
                return RestateMCPToolRunResult(kind='call_deferred', output=None, metadata=e.metadata)
            except ApprovalRequired as e:
                return RestateMCPToolRunResult(kind='approval_required', output=None, metadata=e.metadata)
            except (UserError, PydanticUserError) as e:
                raise TerminalError(str(e)) from e

        options = RunOptions[RestateMCPToolRunResult](serde=MCP_RUN_SERDE)
        res = await self._context.run_typed(f'Calling mcp tool {name}', call_tool_in_context, options)

        if res.kind == 'call_deferred':
            raise CallDeferred(metadata=res.metadata)
        elif res.kind == 'approval_required':
            raise ApprovalRequired(metadata=res.metadata)
        elif res.kind == 'model_retry':
            assert res.error is not None
            raise ModelRetry(res.error)
        else:
            assert res.kind == 'output'
            assert res.output is not None
            return res.output

