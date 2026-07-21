from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast

from prefect import task
from prefect.context import FlowRunContext

from pydantic_ai import FunctionToolset, ToolsetTool
from pydantic_ai.durable_exec._toolset import DurableFunctionToolset
from pydantic_ai.tools import AgentDepsT, RunContext

from ._toolset import resolve_tool_task_config
from ._types import TaskConfig, default_task_config


class PrefectFunctionToolset(DurableFunctionToolset[AgentDepsT]):
    """A wrapper for `FunctionToolset` that runs tool calls as Prefect tasks inside flows."""

    def __init__(
        self,
        wrapped: FunctionToolset[AgentDepsT],
        *,
        task_config: TaskConfig,
        tool_task_config: dict[str, TaskConfig | None],
        _in_durable_context: Callable[[], bool] | None = None,
    ):
        base_config = default_task_config | (task_config or {})

        @task
        async def call_tool_task(
            tool_name: str,
            tool_args: dict[str, Any],
            ctx: RunContext[AgentDepsT],
            tool: ToolsetTool[AgentDepsT],
        ) -> Any:
            return await wrapped.call_tool(tool_name, tool_args, ctx, tool)

        async def call_tool_operation(
            name: str,
            tool_args: dict[str, Any],
            ctx: RunContext[AgentDepsT],
            tool: ToolsetTool[AgentDepsT],
            config: Mapping[str, Any],
        ) -> Any:
            merged_config = cast('TaskConfig', base_config | dict(config))
            return await call_tool_task.with_options(name=f'Call Tool: {name}', **merged_config)(
                name, tool_args, ctx, tool
            )

        super().__init__(
            wrapped,
            in_durable_context=_in_durable_context or (lambda: True),
            call_tool_operation=call_tool_operation,
            resolve_tool_config=lambda tool, name: resolve_tool_task_config(tool, name, tool_task_config),
            lifecycle='enter-always',
            durable_config=base_config,
        )


def prefectify_function_toolset(
    wrapped: FunctionToolset[AgentDepsT],
    *,
    task_config: TaskConfig,
    tool_task_config: dict[str, TaskConfig | None],
) -> PrefectFunctionToolset[AgentDepsT]:
    return PrefectFunctionToolset(
        wrapped,
        task_config=task_config,
        tool_task_config=tool_task_config,
        _in_durable_context=lambda: FlowRunContext.get() is not None,
    )
