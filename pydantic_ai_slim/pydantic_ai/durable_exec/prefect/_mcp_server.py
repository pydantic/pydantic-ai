from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any

from prefect import get_run_logger, task
from typing_extensions import Self

from pydantic_ai import ToolsetTool
from pydantic_ai.tools import AgentDepsT, RunContext

from ._logging import format_tool_args, format_tool_result
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
        log_tool_calls: bool = False,
    ):
        super().__init__(wrapped)
        self._task_config = default_task_config | (task_config or {})
        self._mcp_id = wrapped.id
        self._log_tool_calls = log_tool_calls

        @task
        async def _call_tool_task(
            tool_name: str,
            tool_args: dict[str, Any],
            ctx: RunContext[AgentDepsT],
            tool: ToolsetTool[AgentDepsT],
        ) -> ToolResult:
            if self._log_tool_calls:
                logger = get_run_logger()
                args_str = format_tool_args(tool_args)
                logger.info(f'Calling MCP tool: {tool_name} with args: {args_str}')

            result = await super(PrefectMCPServer, self).call_tool(tool_name, tool_args, ctx, tool)

            if self._log_tool_calls:
                logger = get_run_logger()
                result_str = format_tool_result(result)
                logger.info(f'MCP tool {tool_name} returned: {result_str}')

            return result

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
