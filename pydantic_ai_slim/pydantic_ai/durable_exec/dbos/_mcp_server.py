from __future__ import annotations

from abc import ABC
from typing import Any, Callable

from dbos import DBOS, DBOSConfiguredInstance
from typing_extensions import Self

from pydantic_ai.mcp import MCPServer, ToolResult
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool
from pydantic_ai.toolsets.wrapper import WrapperToolset

from ._utils import StepConfig


class DBOSMCPServer(WrapperToolset[AgentDepsT], ABC, DBOSConfiguredInstance):
    """A wrapper for MCPServer that integrates with DBOS, turning call_tool and get_tools to DBOS steps."""

    def __init__(
        self,
        wrapped: MCPServer,
        *,
        step_name_prefix: str,
        step_config: StepConfig,
    ):
        super().__init__(wrapped)
        self._step_config = step_config or {}
        self._step_name_prefix = step_name_prefix
        self._name = f'{step_name_prefix}__mcp_server__{wrapped.id}'

        DBOSConfiguredInstance.__init__(self, self._name)

    @property
    def id(self) -> str:
        # An error is raised in `DBOS` if no `id` is set.
        assert self.wrapped.id is not None
        return self.wrapped.id

    async def __aenter__(self) -> Self:
        # The wrapped MCPServer enters itself around listing and calling tools
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
        @DBOS.step(
            name=f'{self._name}.get_tools',
            **self._step_config,
        )
        async def wrapped_get_tools_step(
            ctx: RunContext[AgentDepsT],
        ) -> dict[str, ToolsetTool[AgentDepsT]]:
            return await super(DBOSMCPServer, self).get_tools(ctx)

        return await wrapped_get_tools_step(ctx)

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> ToolResult:
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
            return await super(DBOSMCPServer, self).call_tool(name, tool_args, ctx, tool)

        return await wrapped_call_tool_step(name, tool_args, ctx, tool)
