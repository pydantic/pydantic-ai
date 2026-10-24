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
    unwrap_recorded_tool_call_result,
    validate_dynamic_tool_args,
    wrap_tool_call_result,
)
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets._dynamic import DynamicToolset

from ._toolset import guard_task_enqueue, resolve_tool_task_config, with_non_retryable_errors
from ._types import TaskConfig, default_task_config


def prefectify_dynamic_toolset(
    wrapped: DynamicToolset[AgentDepsT],
    *,
    task_config: TaskConfig,
    tool_task_config: dict[str, TaskConfig | None],
) -> DurableDynamicToolset[AgentDepsT]:
    base_config = default_task_config | (task_config or {})

    @task
    async def get_tools_task(toolset_id: str | None, ctx: RunContext[AgentDepsT]) -> DynamicToolsResult:
        # Forks the cache key so toolsets sharing this task's source don't collide.
        del toolset_id
        return await get_dynamic_tools(wrapped, ctx)

    async def get_tools_operation(ctx: RunContext[AgentDepsT]) -> DynamicToolsResult:
        task_config = with_non_retryable_errors(base_config)
        return await get_tools_task.with_options(name=f'Discover Tools: {wrapped.id}', **task_config)(wrapped.id, ctx)

    @task
    async def call_tool_task(tool_name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT]) -> Any:
        task_ctx = guard_task_enqueue(ctx)
        return await wrap_tool_call_result(call_dynamic_tool(wrapped, tool_name, tool_args, task_ctx))

    @task
    async def validate_args_task(tool_name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT]) -> Any:
        task_ctx = guard_task_enqueue(ctx)
        return await wrap_tool_call_result(validate_dynamic_tool_args(wrapped, tool_name, tool_args, task_ctx))

    async def call_tool_operation(
        name: str,
        tool_args: dict[str, Any],
        *,
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
        config: Mapping[str, Any],
    ) -> Any:
        merged_config = with_non_retryable_errors(cast('TaskConfig', base_config | dict(config)))
        result = await call_tool_task.with_options(name=f'Call Tool: {name}', **merged_config)(name, tool_args, ctx)
        # A persisted cache entry written before this task wrapped control-flow exceptions (still
        # reachable under a custom `cache_policy` that omits `TASK_SOURCE`) holds the raw result.
        return unwrap_recorded_tool_call_result(result)

    async def validate_args_operation(
        name: str,
        tool_args: dict[str, Any],
        *,
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
        config: Mapping[str, Any],
    ) -> None:
        merged_config = with_non_retryable_errors(cast('TaskConfig', base_config | dict(config)))
        result = await validate_args_task.with_options(name=f'Validate Tool Args: {name}', **merged_config)(
            name, tool_args, ctx
        )
        unwrap_recorded_tool_call_result(result)

    return DurableDynamicToolset(
        wrapped,
        # Prefect tasks do NOT degrade outside a flow (the full task engine runs, with
        # retries and cache lookups), so gate on an active flow run like the other
        # Prefect toolset factories.
        in_durable_context=lambda: FlowRunContext.get() is not None,
        get_tools_operation=get_tools_operation,
        call_tool_operation=call_tool_operation,
        validate_args_operation=validate_args_operation,
        resolve_tool_config=lambda tool, name: resolve_tool_task_config(tool, name, tool_task_config),
        lifecycle='enter-never',
        durable_config=base_config,
    )
