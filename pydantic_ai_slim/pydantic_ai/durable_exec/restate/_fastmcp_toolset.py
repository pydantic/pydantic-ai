from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from typing_extensions import Self

from pydantic_ai import ToolDefinition
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool
from pydantic_ai.toolsets.fastmcp import FastMCPToolset
from pydantic_ai.toolsets.wrapper import WrapperToolset

from ._restate_types import Context, RunOptions
from ._toolset import (
    CONTEXT_RUN_SERDE,
    RestateContextRunResult,
    run_get_tools_step,
    run_tool_call_step,
)


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
        # Restate-wrapped toolsets are sealed after wrapping and should not be replaced.
        return self

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        async def get_tools_action() -> dict[str, ToolDefinition]:
            async with self._fastmcp_toolset:
                res = await self._fastmcp_toolset.get_tools(ctx)
                # ToolsetTool is not serializable as it holds a SchemaValidator,
                # so we return ToolDefinitions and reconstruct ToolsetTool outside durable steps.
                return {name: tool.tool_def for name, tool in res.items()}

        tool_defs = await run_get_tools_step(self._context, 'get fastmcp tools', get_tools_action)
        return {name: self.tool_for_tool_def(tool_def) for name, tool_def in tool_defs.items()}

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

        return await run_tool_call_step(self._context, f'Calling fastmcp tool {name}', call_tool_action, self._options)
