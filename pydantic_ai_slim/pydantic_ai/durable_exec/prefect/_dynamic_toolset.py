from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from prefect import task
from prefect.context import FlowRunContext

from pydantic_ai import ToolsetTool
from pydantic_ai.durable_exec._toolset import (
    DurableDynamicToolset,
    DynamicToolsResult,
    call_dynamic_tool,
    get_dynamic_tools,
)
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets._dynamic import DynamicToolset

from ._toolset import resolve_tool_task_config
from ._types import TaskConfig, default_task_config


def prefectify_dynamic_toolset(
    wrapped: DynamicToolset[AgentDepsT],
    *,
    task_config: TaskConfig,
    tool_task_config: dict[str, TaskConfig | None],
) -> DurableDynamicToolset[AgentDepsT]:
    base_config = default_task_config | (task_config or {})

    async def get_tools_operation(ctx: RunContext[AgentDepsT]) -> DynamicToolsResult:
        # Runs in flow code, like static Prefect MCP `get_tools`: flow retries re-execute
        # resolution anyway, so only tool *calls* get task retry/caching semantics.
        return await get_dynamic_tools(wrapped, ctx)

    @task
    async def call_tool_task(tool_name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT]) -> Any:
        return await call_dynamic_tool(wrapped, tool_name, tool_args, ctx)

    async def call_tool_operation(
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
        config: Mapping[str, Any],
    ) -> Any:
        merged_config = cast('TaskConfig', base_config | dict(config))
        return await call_tool_task.with_options(name=f'Call Tool: {name}', **merged_config)(name, tool_args, ctx)

    return DurableDynamicToolset(
        wrapped,
        # Prefect tasks do NOT degrade outside a flow (the full task engine runs, with
        # retries and cache lookups), so gate on an active flow run like the other
        # Prefect toolset factories.
        in_durable_context=lambda: FlowRunContext.get() is not None,
        get_tools_operation=get_tools_operation,
        call_tool_operation=call_tool_operation,
        resolve_tool_config=lambda tool, name: resolve_tool_task_config(tool, name, tool_task_config),
        lifecycle='enter-never',
        durable_config=base_config,
    )
