from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, cast

from prefect import task
from prefect.context import FlowRunContext
from typing_extensions import deprecated

from pydantic_ai import FunctionToolset, ToolsetTool
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.durable_exec._toolset import (
    CallToolOperation,
    DurableFunctionToolset,
    unwrap_recorded_tool_call_result,
    wrap_tool_call_result,
)
from pydantic_ai.tools import AgentDepsT, RunContext

from ._toolset import enqueue_guard, resolve_tool_task_config, with_non_retryable_errors
from ._types import TaskConfig, default_task_config


def _call_tool_operation(wrapped: FunctionToolset[AgentDepsT], base_config: TaskConfig) -> CallToolOperation:
    @task
    async def call_tool_task(
        tool_name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> Any:
        task_ctx = replace(ctx, pending_messages=enqueue_guard())
        return await wrap_tool_call_result(wrapped.call_tool(tool_name, tool_args, task_ctx, tool))

    async def call_tool_operation(
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
        config: Mapping[str, Any],
    ) -> Any:
        merged_config = with_non_retryable_errors(cast('TaskConfig', base_config | dict(config)))
        result = await call_tool_task.with_options(name=f'Call Tool: {name}', **merged_config)(
            name, tool_args, ctx, tool
        )
        # A persisted cache entry written before this task wrapped control-flow exceptions (still
        # reachable under a custom `cache_policy` that omits `TASK_SOURCE`) holds the raw result.
        return unwrap_recorded_tool_call_result(result)

    return call_tool_operation


@deprecated(
    "`PrefectFunctionToolset` is deprecated alongside `PrefectAgent`. Use the `PrefectDurability` capability, which wraps the agent's toolsets in Prefect tasks automatically.",
    category=PydanticAIDeprecationWarning,
)
class PrefectFunctionToolset(DurableFunctionToolset[AgentDepsT]):
    """A wrapper for `FunctionToolset` that runs tool calls as Prefect tasks inside flows."""

    def __init__(
        self,
        wrapped: FunctionToolset[AgentDepsT],
        *,
        task_config: TaskConfig,
        tool_task_config: dict[str, TaskConfig | None],
    ):
        base_config = default_task_config | (task_config or {})

        super().__init__(
            wrapped,
            in_durable_context=lambda: True,
            call_tool_operation=_call_tool_operation(wrapped, base_config),
            resolve_tool_config=lambda tool, name: resolve_tool_task_config(tool, name, tool_task_config),
            lifecycle='enter-always',
            durable_config=base_config,
        )


def prefectify_function_toolset(
    wrapped: FunctionToolset[AgentDepsT],
    *,
    task_config: TaskConfig,
    tool_task_config: dict[str, TaskConfig | None],
) -> DurableFunctionToolset[AgentDepsT]:
    base_config = default_task_config | (task_config or {})
    return DurableFunctionToolset(
        wrapped,
        in_durable_context=lambda: FlowRunContext.get() is not None,
        call_tool_operation=_call_tool_operation(wrapped, base_config),
        resolve_tool_config=lambda tool, name: resolve_tool_task_config(tool, name, tool_task_config),
        lifecycle='enter-always',
        durable_config=base_config,
    )
