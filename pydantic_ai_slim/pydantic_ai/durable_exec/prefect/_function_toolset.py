from __future__ import annotations

from collections.abc import Mapping
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
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets.function import FunctionToolsetTool

from ._toolset import guard_task_enqueue, resolve_tool_task_config, with_non_retryable_errors
from ._types import TaskConfig, default_task_config


def _registered_tool_name(wrapped: FunctionToolset[AgentDepsT], tool: FunctionToolsetTool[AgentDepsT]) -> str:
    """Map a prepared `FunctionToolsetTool` back to its registry key in `wrapped.tools`.

    `prepare` may rename the tool, so the call name is not always the registry key. Prefer matching
    the bound `FunctionSchema` on `call_func` (accessing `.call` creates a new bound method each
    time, so identity on `call_func` itself is unreliable) rather than re-running `get_tools` /
    `prepare_tool_def`.
    """
    schema = getattr(tool.call_func, '__self__', None)
    if schema is not None:
        for name, registered in wrapped.tools.items():
            if registered.function_schema is schema:
                return name
    return tool.tool_def.name


def _function_tool_from_prepared(
    wrapped: FunctionToolset[AgentDepsT],
    *,
    registered_name: str,
    tool_def: ToolDefinition,
    max_retries: int,
) -> FunctionToolsetTool[AgentDepsT]:
    """Rebuild a `FunctionToolsetTool` from the already-prepared definition + registered callable.

    Uses the prepared `tool_def` as-is so `prepare_tool_def` is not invoked again inside the task.
    """
    try:
        registered = wrapped.tools[registered_name]
    except KeyError as exc:  # pragma: no cover
        raise UserError(
            f'Tool {registered_name!r} not found in toolset {wrapped.id!r}. '
            'Removing or renaming tools during an agent run is not supported with Prefect.'
        ) from exc
    return FunctionToolsetTool(
        toolset=wrapped,
        tool_def=tool_def,
        max_retries=max_retries,
        args_validator=registered.function_schema.validator,
        args_validator_func=registered.args_validator,
        call_func=registered.function_schema.call,
        is_async=registered.function_schema.is_async,
        timeout=tool_def.timeout,
    )


def _call_tool_operation(wrapped: FunctionToolset[AgentDepsT], base_config: TaskConfig) -> CallToolOperation:
    # Do not pass the live `ToolsetTool` into the Prefect task. Its `args_validator` is not
    # JSON-serializable, so Prefect's input hash falls back to cloudpickle, which is sensitive to
    # object identity / memo sharing. On a flow retry the model-request task often cache-hits and
    # deserializes a fresh `tool_name` string while `tool_def.name` is a different object with the
    # same value — identical logical inputs, different pickle bytes, cache miss, duplicated side
    # effects (#6907).
    #
    # Pass the already-prepared JSON-native `tool_def` (plus registry key / max_retries) and rebuild
    # the callable wrapper inside the task. Do not call `get_tools` here: that would re-run every
    # tool's `prepare_tool_def` (side effects / non-determinism) for a call that was already prepared
    # when the model request was built.
    @task
    async def call_tool_task(
        tool_name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool_def: ToolDefinition,
        max_retries: int,
        registered_name: str,
    ) -> Any:
        task_ctx = guard_task_enqueue(ctx)
        tool = _function_tool_from_prepared(
            wrapped,
            registered_name=registered_name,
            tool_def=tool_def,
            max_retries=max_retries,
        )
        return await wrap_tool_call_result(wrapped.call_tool(tool_name, tool_args, task_ctx, tool))

    async def call_tool_operation(
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
        config: Mapping[str, Any],
    ) -> Any:
        assert isinstance(tool, FunctionToolsetTool)
        merged_config = with_non_retryable_errors(cast('TaskConfig', base_config | dict(config)))
        result = await call_tool_task.with_options(name=f'Call Tool: {name}', **merged_config)(
            name,
            tool_args,
            ctx,
            tool.tool_def,
            tool.max_retries,
            _registered_tool_name(wrapped, tool),
        )
        # A persisted cache entry written before this task wrapped control-flow exceptions (still
        # reachable under a custom `cache_policy` that omits `TASK_SOURCE`) holds the raw result.
        return unwrap_recorded_tool_call_result(result)

    return call_tool_operation


# TODO(v3): remove `PrefectFunctionToolset` alongside `PrefectAgent`.
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
