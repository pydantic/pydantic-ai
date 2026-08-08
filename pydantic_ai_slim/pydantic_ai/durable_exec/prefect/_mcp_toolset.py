from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal, cast

from prefect import task
from prefect.context import FlowRunContext
from typing_extensions import deprecated

from pydantic_ai import ToolsetTool
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.durable_exec._toolset import (
    CallToolOperation,
    DurableMCPToolset,
    Instructions,
    unwrap_recorded_tool_call_result,
    wrap_tool_call_result,
)
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition

from ._toolset import guard_task_enqueue, resolve_tool_task_config, with_non_retryable_errors
from ._types import TaskConfig, default_task_config

if TYPE_CHECKING:
    from pydantic_ai.mcp import MCPToolset, ToolResult


def _discovery_operations(
    wrapped: MCPToolset[AgentDepsT], base_config: TaskConfig
) -> tuple[
    Callable[[RunContext[AgentDepsT]], Awaitable[dict[str, ToolDefinition]]],
    Callable[[RunContext[AgentDepsT]], Awaitable[Instructions]],
]:
    @task
    async def get_tools_task(
        operation: Literal['get_tools'], toolset_id: str | None, ctx: RunContext[AgentDepsT]
    ) -> dict[str, ToolDefinition]:
        # Forks the cache key so discovery operations sharing other inputs don't collide.
        del operation
        # Forks the cache key so toolsets sharing this task's source don't collide.
        del toolset_id
        return {name: tool.tool_def for name, tool in (await wrapped.get_tools(ctx)).items()}

    @task
    async def get_instructions_task(
        operation: Literal['get_instructions'], toolset_id: str | None, ctx: RunContext[AgentDepsT]
    ) -> Instructions:
        # Forks the cache key so discovery operations sharing other inputs don't collide.
        del operation
        # Forks the cache key so toolsets sharing this task's source don't collide.
        del toolset_id
        return await wrapped.get_instructions(ctx)

    async def get_tools_operation(ctx: RunContext[AgentDepsT]) -> dict[str, ToolDefinition]:
        task_config = with_non_retryable_errors(base_config)
        return await get_tools_task.with_options(name=f'Get MCP Tools: {wrapped.id}', **task_config)(
            'get_tools', wrapped.id, ctx
        )

    async def get_instructions_operation(ctx: RunContext[AgentDepsT]) -> Instructions:
        task_config = with_non_retryable_errors(base_config)
        return await get_instructions_task.with_options(name=f'Get MCP Instructions: {wrapped.id}', **task_config)(
            'get_instructions', wrapped.id, ctx
        )

    return get_tools_operation, get_instructions_operation


def _call_tool_operation(wrapped: MCPToolset[AgentDepsT], base_config: TaskConfig) -> CallToolOperation:
    @task
    async def call_tool_task(
        tool_name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> Any:
        # The context is guarded because a `process_tool_call=` hook receives it and could enqueue.
        task_ctx = guard_task_enqueue(ctx)
        return await wrap_tool_call_result(wrapped.call_tool(tool_name, tool_args, task_ctx, tool))

    async def call_tool_operation(
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
        config: Mapping[str, Any],
    ) -> ToolResult:
        task_config = with_non_retryable_errors(cast('TaskConfig', base_config | dict(config)))
        result = await call_tool_task.with_options(name=f'Call MCP Tool: {name}', **task_config)(
            name, tool_args, ctx, tool
        )
        # A persisted cache entry written before this task wrapped control-flow exceptions (still
        # reachable under a custom `cache_policy` that omits `TASK_SOURCE`) holds the raw result.
        return unwrap_recorded_tool_call_result(result)

    return call_tool_operation


# TODO(v3): remove `PrefectMCPToolset` alongside `PrefectAgent`.
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
        get_tools_operation, get_instructions_operation = _discovery_operations(wrapped, base_config)

        super().__init__(
            wrapped,
            in_durable_context=lambda: True,
            get_tools_operation=get_tools_operation,
            get_instructions_operation=get_instructions_operation,
            call_tool_operation=_call_tool_operation(wrapped, base_config),
            # The deprecated wrapper never read per-tool metadata; leave its behavior frozen.
            resolve_tool_config=lambda tool, name: {},
            lifecycle='enter-always',
            durable_config=base_config,
        )


def prefectify_mcp_toolset(
    wrapped: MCPToolset[AgentDepsT], *, task_config: TaskConfig
) -> DurableMCPToolset[AgentDepsT]:
    base_config = default_task_config | (task_config or {})
    get_tools_operation, get_instructions_operation = _discovery_operations(wrapped, base_config)
    return DurableMCPToolset(
        wrapped,
        in_durable_context=lambda: FlowRunContext.get() is not None,
        get_tools_operation=get_tools_operation,
        get_instructions_operation=get_instructions_operation,
        call_tool_operation=_call_tool_operation(wrapped, base_config),
        # Per-tool config on MCP tools works the same as on function and dynamic tools: unlike
        # Temporal, a Prefect flow can do I/O itself, so `False` runs the call inline in flow code
        # rather than being rejected.
        resolve_tool_config=lambda tool, name: resolve_tool_task_config(tool, name, {}),
        lifecycle='enter-always',
        durable_config=base_config,
    )
