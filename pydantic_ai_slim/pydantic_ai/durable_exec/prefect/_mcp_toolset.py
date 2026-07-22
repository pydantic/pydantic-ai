from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from prefect import task
from prefect.context import FlowRunContext
from typing_extensions import deprecated

from pydantic_ai import ToolsetTool
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.durable_exec._toolset import CallToolOperation, DurableMCPToolset
from pydantic_ai.tools import AgentDepsT, RunContext

from ._types import TaskConfig, default_task_config

if TYPE_CHECKING:
    from pydantic_ai.mcp import MCPToolset, ToolResult


def _call_tool_operation(wrapped: MCPToolset[AgentDepsT], base_config: TaskConfig) -> CallToolOperation:
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

    return call_tool_operation


@deprecated(
    "`PrefectMCPToolset` is deprecated alongside `PrefectAgent`. Use the `PrefectDurability` capability, which wraps the agent's toolsets in Prefect tasks automatically.",
    category=PydanticAIDeprecationWarning,
)
class PrefectMCPToolset(DurableMCPToolset[AgentDepsT]):
    """A wrapper for `MCPToolset` that runs tool calls as Prefect tasks inside flows."""

    def __init__(
        self,
        wrapped: MCPToolset[AgentDepsT],
        *,
        task_config: TaskConfig,
    ):
        base_config = default_task_config | (task_config or {})

        super().__init__(
            wrapped,
            in_durable_context=lambda: True,
            get_tools_operation=None,
            get_instructions_operation=None,
            call_tool_operation=_call_tool_operation(wrapped, base_config),
            resolve_tool_config=lambda tool, name: {},
            lifecycle='enter-always',
            durable_config=base_config,
        )


def prefectify_mcp_toolset(
    wrapped: MCPToolset[AgentDepsT], *, task_config: TaskConfig
) -> DurableMCPToolset[AgentDepsT]:
    base_config = default_task_config | (task_config or {})
    return DurableMCPToolset(
        wrapped,
        in_durable_context=lambda: FlowRunContext.get() is not None,
        get_tools_operation=None,
        get_instructions_operation=None,
        call_tool_operation=_call_tool_operation(wrapped, base_config),
        resolve_tool_config=lambda tool, name: {},
        lifecycle='enter-always',
        durable_config=base_config,
    )
