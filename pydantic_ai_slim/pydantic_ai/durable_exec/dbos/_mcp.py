from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from dbos import DBOS
from typing_extensions import Self

from pydantic_ai import AbstractToolset, ToolsetTool, WrapperToolset
from pydantic_ai.messages import InstructionPart
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

        # Wrap get_instructions in a DBOS step.
        @DBOS.step(
            name=f'{self._name}.get_instructions',
            **self._step_config,
        )
        async def wrapped_get_instructions_step(
            ctx: RunContext[AgentDepsT],
        ) -> str | InstructionPart | Sequence[str | InstructionPart] | None:
            async with self.wrapped:
                return await super(DBOSMCPToolset, self).get_instructions(ctx)

        self._dbos_wrapped_get_instructions_step = wrapped_get_instructions_step

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

    async def get_instructions(
        self, ctx: RunContext[AgentDepsT]
    ) -> str | InstructionPart | Sequence[str | InstructionPart] | None:
        # Try locally first (fast path: returns None when disabled or returns cached instructions).
        result = await super().get_instructions(ctx)
        if result is not None:
            return result
        # If instructions are enabled but the server isn't initialized locally, fetch via step.
        _mcp_types: tuple[type, ...] = ()
        try:
            from pydantic_ai.mcp import MCPServer

            _mcp_types += (MCPServer,)
        except ImportError:
            pass
        try:
            from pydantic_ai.toolsets.fastmcp import FastMCPToolset

            _mcp_types += (FastMCPToolset,)
        except ImportError:
            pass
        if _mcp_types and isinstance(self.wrapped, _mcp_types) and self.wrapped.include_instructions:  # type: ignore[union-attr]
            return await self._dbos_wrapped_get_instructions_step(ctx)
        return None

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> ToolResult:
        return await self._dbos_wrapped_call_tool_step(name, tool_args, ctx, tool)
