from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from dbos import DBOS
from typing_extensions import Self

from pydantic_ai import AbstractToolset, ToolsetTool, WrapperToolset
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition

from ._utils import StepConfig

if TYPE_CHECKING:
    from pydantic_ai.mcp import ToolResult


class DBOSMCPToolset(WrapperToolset[AgentDepsT], ABC):
    """A wrapper for MCP toolset that integrates with DBOS, turning call_tool and get_tools to DBOS steps."""

    def __init__(
        self,
        wrapped: AbstractToolset[AgentDepsT],
        *,
        step_name_prefix: str,
        step_config: StepConfig,
    ):
        super().__init__(wrapped)
        self._step_config = step_config or {}
        self._step_name_prefix = step_name_prefix
        id_suffix = f'__{wrapped.id}' if wrapped.id else ''
        self._name = f'{step_name_prefix}__mcp_server{id_suffix}'

        # Wrap get_tools in a DBOS step.
        @DBOS.step(
            name=f'{self._name}.get_tools',
            **self._step_config,
        )
        async def wrapped_get_tools_step(
            ctx: RunContext[AgentDepsT],
        ) -> dict[str, ToolDefinition]:
            # Need to return a serializable dict, so we cannot return ToolsetTool directly.
            tools = await super(DBOSMCPToolset, self).get_tools(ctx)
            # ToolsetTool is not serializable as it holds a SchemaValidator (which is also the same for every MCP tool so unnecessary to pass along the wire every time),
            # so we just return the ToolDefinitions and wrap them in ToolsetTool outside of the activity.
            return {name: tool.tool_def for name, tool in tools.items()}

        self._dbos_wrapped_get_tools_step = wrapped_get_tools_step

        # Wrap call_tool in a DBOS step.
        @DBOS.step(
            name=f'{self._name}.call_tool',
            **self._step_config,
        )
        async def wrapped_call_tool_step(
            name: str,
            tool_args: dict[str, Any],
            ctx: RunContext[AgentDepsT],
            tool: ToolsetTool[AgentDepsT],
        ) -> ToolResult:
            return await super(DBOSMCPToolset, self).call_tool(name, tool_args, ctx, tool)

        self._dbos_wrapped_call_tool_step = wrapped_call_tool_step

    @abstractmethod
    def tool_for_tool_def(self, tool_def: ToolDefinition) -> ToolsetTool[AgentDepsT]:
        raise NotImplementedError

    @property
    def id(self) -> str | None:
        return self.wrapped.id

    async def __aenter__(self) -> Self:
        # The wrapped MCP toolset enters itself around listing and calling tools
        # so we don't need to enter it here (nor could we because we're not inside a DBOS step).
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        return None

    def visit_and_replace(
        self, visitor: Callable[[AbstractToolset[AgentDepsT]], AbstractToolset[AgentDepsT]]
    ) -> AbstractToolset[AgentDepsT]:
        # DBOS-ified toolsets cannot be swapped out after the fact.
        return self

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        tool_defs = await self._dbos_wrapped_get_tools_step(ctx)
        return {name: self.tool_for_tool_def(tool_def) for name, tool_def in tool_defs.items()}

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> ToolResult:
        return await self._dbos_wrapped_call_tool_step(name, tool_args, ctx, tool)
