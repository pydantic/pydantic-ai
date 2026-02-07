from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeAlias, cast

from pydantic.errors import PydanticUserError
from typing_extensions import Protocol, Self

from pydantic_ai import ToolDefinition
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, ModelRetry, UserError
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool
from pydantic_ai.toolsets.wrapper import WrapperToolset

from ._restate_types import Context, RunOptions, TerminalError
from ._serde import PydanticTypeAdapter
from ._toolset import CONTEXT_RUN_SERDE, RestateContextRunResult, unwrap_context_run_result

ToolResult: TypeAlias = Any


class _MCPServer(Protocol):
    @property
    def id(self) -> str | None: ...

    async def get_tools(self, ctx: RunContext[Any]) -> dict[str, ToolsetTool[Any]]: ...

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[Any],
        tool: ToolsetTool[Any],
    ) -> ToolResult: ...

    def tool_for_tool_def(self, tool_def: ToolDefinition) -> ToolsetTool[Any]: ...


@dataclass
class RestateMCPGetToolsContextRunResult:
    """A simple wrapper for tool results to be used with Restate's `ctx.run_typed()`."""

    output: dict[str, ToolDefinition]


MCP_GET_TOOLS_SERDE = PydanticTypeAdapter(RestateMCPGetToolsContextRunResult)


class RestateMCPServer(WrapperToolset[Any]):
    """A wrapper for [`MCPServer`][pydantic_ai.mcp.MCPServer] that integrates with Restate."""

    def __init__(self, wrapped: AbstractToolset[Any], context: Context):
        super().__init__(wrapped)
        self._wrapped = cast(_MCPServer, wrapped)
        self._context = context
        self._call_options = RunOptions[RestateContextRunResult](serde=CONTEXT_RUN_SERDE)

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
        async def get_tools_in_context() -> RestateMCPGetToolsContextRunResult:
            res = await self._wrapped.get_tools(ctx)
            # ToolsetTool is not serializable as it holds a SchemaValidator
            # (which is also the same for every MCP tool so unnecessary to pass along the wire every time),
            # so we just return the ToolDefinitions and wrap them in ToolsetTool outside of ctx.run_typed().
            return RestateMCPGetToolsContextRunResult(output={name: tool.tool_def for name, tool in res.items()})

        options = RunOptions[RestateMCPGetToolsContextRunResult](serde=MCP_GET_TOOLS_SERDE)
        tool_defs = await self._context.run_typed('get mcp tools', get_tools_in_context, options)

        return {name: self.tool_for_tool_def(tool_def) for name, tool_def in tool_defs.output.items()}

    def tool_for_tool_def(self, tool_def: ToolDefinition) -> ToolsetTool[Any]:
        return self._wrapped.tool_for_tool_def(tool_def)

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[Any],
        tool: ToolsetTool[Any],
    ) -> ToolResult:
        async def call_tool_in_context() -> RestateContextRunResult:
            try:
                res = await self._wrapped.call_tool(name, tool_args, ctx, tool)
                return RestateContextRunResult(kind='output', output=res, error=None)
            except ModelRetry as e:
                return RestateContextRunResult(kind='model_retry', output=None, error=e.message)
            except CallDeferred as e:
                return RestateContextRunResult(kind='call_deferred', output=None, metadata=e.metadata)
            except ApprovalRequired as e:
                return RestateContextRunResult(kind='approval_required', output=None, metadata=e.metadata)
            except (UserError, PydanticUserError) as e:
                raise TerminalError(str(e)) from e

        res = await self._context.run_typed(f'Calling mcp tool {name}', call_tool_in_context, self._call_options)
        return unwrap_context_run_result(res)
