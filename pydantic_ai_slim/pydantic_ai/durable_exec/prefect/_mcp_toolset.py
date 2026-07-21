from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from prefect import task
from prefect.context import FlowRunContext

from pydantic_ai import ToolsetTool
from pydantic_ai.durable_exec._toolset import DurableMCPToolset
from pydantic_ai.tools import AgentDepsT, RunContext

from ._types import TaskConfig, default_task_config

if TYPE_CHECKING:
    from pydantic_ai.mcp import MCPToolset, ToolResult


class PrefectMCPToolset(DurableMCPToolset[AgentDepsT]):
    """A wrapper for `MCPToolset` that runs tool calls as Prefect tasks inside flows."""

    def __init__(
        self,
        wrapped: MCPToolset[AgentDepsT],
        *,
        task_config: TaskConfig,
        _in_durable_context: Callable[[], bool] | None = None,
    ):
        base_config = default_task_config | (task_config or {})

        @task
        async def call_tool_task(
            tool_name: str,
            tool_args: dict[str, Any],
            ctx: RunContext[AgentDepsT],
            tool: ToolsetTool[AgentDepsT],
        ) -> ToolResult:
            return await wrapped.call_tool(tool_name, tool_args, ctx, tool)

        async def call_tool_operation(
            name: str,
            tool_args: dict[str, Any],
            ctx: RunContext[AgentDepsT],
            tool: ToolsetTool[AgentDepsT],
            config: Mapping[str, Any],
        ) -> ToolResult:
            return await call_tool_task.with_options(name=f'Call MCP Tool: {name}', **base_config)(
                name, tool_args, ctx, tool
            )

        super().__init__(
            wrapped,
            in_durable_context=_in_durable_context or (lambda: True),
            get_tools_operation=None,
            get_instructions_operation=None,
            call_tool_operation=call_tool_operation,
            resolve_tool_config=lambda tool, name: {},
            lifecycle='enter-always',
            durable_config=base_config,
        )


def prefectify_mcp_toolset(
    wrapped: MCPToolset[AgentDepsT], *, task_config: TaskConfig
) -> PrefectMCPToolset[AgentDepsT]:
    return PrefectMCPToolset(
        wrapped, task_config=task_config, _in_durable_context=lambda: FlowRunContext.get() is not None
    )
