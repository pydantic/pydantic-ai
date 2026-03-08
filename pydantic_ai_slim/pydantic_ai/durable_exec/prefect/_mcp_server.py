from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any

from prefect import task
from typing_extensions import Self

from pydantic_ai import ToolsetTool
from pydantic_ai.tools import AgentDepsT, RunContext

from ._toolset import PrefectWrapperToolset
from ._types import TaskConfig, default_task_config

if TYPE_CHECKING:
    from pydantic_ai.mcp import MCPServer, ToolResult


class PrefectMCPServer(PrefectWrapperToolset[AgentDepsT], ABC):
    """A wrapper for MCPServer that integrates with Prefect, turning call_tool and get_tools into Prefect tasks."""

    def __init__(
        self,
        wrapped: MCPServer,
        *,
        task_config: TaskConfig,
    ):
        super().__init__(wrapped)
        self._task_config = default_task_config | (task_config or {})
        self._mcp_id = wrapped.id

        @task
        async def _call_tool_task(
            tool_name: str,
            tool_args: dict[str, Any],
            ctx: RunContext[AgentDepsT],
            tool: ToolsetTool[AgentDepsT],
        ) -> ToolResult:
            return await super(PrefectMCPServer, self).call_tool(tool_name, tool_args, ctx, tool)

        self._call_tool_task = _call_tool_task

    async def __aenter__(self) -> Self:
        await self.wrapped.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        return await self.wrapped.__aexit__(*args)

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> ToolResult:
        """Call an MCP tool, wrapped as a Prefect task with a descriptive name."""
        return await self._call_tool_task.with_options(name=f'Call MCP Tool: {name}', **self._task_config)(
            name, tool_args, ctx, tool
        )
