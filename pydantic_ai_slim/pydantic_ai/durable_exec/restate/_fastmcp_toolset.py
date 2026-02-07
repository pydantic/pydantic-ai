from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic.errors import PydanticUserError
from typing_extensions import Self

from pydantic_ai import ToolDefinition
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, ModelRetry, UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool
from pydantic_ai.toolsets.fastmcp import FastMCPToolset
from pydantic_ai.toolsets.wrapper import WrapperToolset

from ._restate_types import Context, RunOptions, TerminalError
from ._serde import PydanticTypeAdapter
from ._toolset import CONTEXT_RUN_SERDE, RestateContextRunResult, unwrap_context_run_result


@dataclass
class RestateFastMCPGetToolsContextRunResult:
    """A simple wrapper for tool results to be used with Restate's `ctx.run_typed()`."""

    output: dict[str, ToolDefinition]


FAST_MCP_GET_TOOLS_SERDE = PydanticTypeAdapter(RestateFastMCPGetToolsContextRunResult)


class RestateFastMCPToolset(WrapperToolset[AgentDepsT]):
    """A wrapper for [`FastMCPToolset`][pydantic_ai.toolsets.fastmcp.FastMCPToolset] that integrates with Restate."""

    def __init__(self, wrapped: FastMCPToolset[AgentDepsT], context: Context):
        super().__init__(wrapped)
        self._wrapped = wrapped
        self._context = context
        self._options = RunOptions[RestateContextRunResult](serde=CONTEXT_RUN_SERDE)

    @property
    def id(self) -> str | None:  # pragma: no cover
        return self._wrapped.id

    async def __aenter__(self) -> Self:
        """No-op: FastMCP connections must be opened inside `ctx.run_typed()` for durability."""
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        """No-op: FastMCP connections must be opened inside `ctx.run_typed()` for durability."""
        return None

    def visit_and_replace(
        self, visitor: Callable[[AbstractToolset[AgentDepsT]], AbstractToolset[AgentDepsT]]
    ) -> AbstractToolset[AgentDepsT]:
        return self

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        async def get_tools_in_context() -> RestateFastMCPGetToolsContextRunResult:
            res = await self._wrapped.get_tools(ctx)
            # ToolsetTool is not serializable as it holds a SchemaValidator
            # so we just return ToolDefinitions and reconstruct ToolsetTool outside of ctx.run_typed().
            return RestateFastMCPGetToolsContextRunResult(output={name: tool.tool_def for name, tool in res.items()})

        options = RunOptions[RestateFastMCPGetToolsContextRunResult](serde=FAST_MCP_GET_TOOLS_SERDE)
        tool_defs = await self._context.run_typed('get fastmcp tools', get_tools_in_context, options)
        return {name: self.tool_for_tool_def(tool_def) for name, tool_def in tool_defs.output.items()}

    def tool_for_tool_def(self, tool_def: ToolDefinition) -> ToolsetTool[AgentDepsT]:
        return self._wrapped.tool_for_tool_def(tool_def)

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> Any:
        async def call_tool_in_context() -> RestateContextRunResult:
            try:
                output = await self._wrapped.call_tool(name, tool_args, ctx, tool)
                return RestateContextRunResult(kind='output', output=output, error=None)
            except ModelRetry as e:
                return RestateContextRunResult(kind='model_retry', output=None, error=e.message)
            except CallDeferred as e:
                return RestateContextRunResult(kind='call_deferred', output=None, error=None, metadata=e.metadata)
            except ApprovalRequired as e:
                return RestateContextRunResult(kind='approval_required', output=None, error=None, metadata=e.metadata)
            except (UserError, PydanticUserError) as e:
                raise TerminalError(str(e)) from e

        res = await self._context.run_typed(f'Calling fastmcp tool {name}', call_tool_in_context, self._options)
        return unwrap_context_run_result(res)
