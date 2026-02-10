from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from typing_extensions import Self

from pydantic_ai import ToolDefinition
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool
from pydantic_ai.toolsets.fastmcp import FastMCPToolset
from pydantic_ai.toolsets.wrapper import WrapperToolset

from ._restate_types import Context, RunOptions
from ._serde import PydanticTypeAdapter
from ._toolset import CONTEXT_RUN_SERDE, RestateContextRunResult, unwrap_context_run_result, wrap_tool_call_result


@dataclass
class RestateFastMCPGetToolsContextRunResult:
    """A simple wrapper for tool results to be used with Restate's `ctx.run_typed()`."""

    output: dict[str, ToolDefinition]


FAST_MCP_GET_TOOLS_SERDE = PydanticTypeAdapter(RestateFastMCPGetToolsContextRunResult)


class RestateFastMCPToolset(WrapperToolset[AgentDepsT]):
    """A wrapper for [`FastMCPToolset`][pydantic_ai.toolsets.fastmcp.FastMCPToolset] that integrates with Restate."""

    def __init__(self, wrapped: FastMCPToolset[AgentDepsT], context: Context):
        super().__init__(wrapped)
        self._context = context
        self._options = RunOptions[RestateContextRunResult](serde=CONTEXT_RUN_SERDE)

    @property
    def _fastmcp_toolset(self) -> FastMCPToolset[AgentDepsT]:
        return cast(FastMCPToolset[AgentDepsT], self.wrapped)

    @property
    def id(self) -> str | None:  # pragma: no cover
        return self._fastmcp_toolset.id

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
            async with self._fastmcp_toolset:
                res = await self._fastmcp_toolset.get_tools(ctx)
            # ToolsetTool is not serializable as it holds a SchemaValidator
            # so we just return ToolDefinitions and reconstruct ToolsetTool outside of ctx.run_typed().
            return RestateFastMCPGetToolsContextRunResult(output={name: tool.tool_def for name, tool in res.items()})

        options = RunOptions[RestateFastMCPGetToolsContextRunResult](serde=FAST_MCP_GET_TOOLS_SERDE)
        tool_defs = await self._context.run_typed('get fastmcp tools', get_tools_in_context, options)
        return {name: self.tool_for_tool_def(tool_def) for name, tool_def in tool_defs.output.items()}

    def tool_for_tool_def(self, tool_def: ToolDefinition) -> ToolsetTool[AgentDepsT]:
        # Wrap the underlying tool so `tool.toolset` points at the Restate wrapper.
        tool = self._fastmcp_toolset.tool_for_tool_def(tool_def)
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
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> Any:
        async def call_tool_action() -> Any:
            async with self._fastmcp_toolset:
                return await self._fastmcp_toolset.call_tool(name, tool_args, ctx, tool)

        async def call_tool_in_context() -> RestateContextRunResult:
            return await wrap_tool_call_result(call_tool_action)

        res = await self._context.run_typed(f'Calling fastmcp tool {name}', call_tool_in_context, self._options)
        return unwrap_context_run_result(res)
