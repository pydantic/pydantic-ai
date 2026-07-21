from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from prefect import task

from pydantic_ai import FunctionToolset, ToolsetTool
from pydantic_ai.durable_exec._toolset import DurableFunctionToolset
from pydantic_ai.tools import AgentDepsT, RunContext

from ._toolset import resolve_tool_task_config
from ._types import TaskConfig, default_task_config


def prefectify_function_toolset(
    wrapped: FunctionToolset[AgentDepsT],
    *,
    task_config: TaskConfig,
    tool_task_config: dict[str, TaskConfig | None],
) -> DurableFunctionToolset[AgentDepsT]:
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
        return await call_tool_task.with_options(name=f'Call Tool: {name}', **merged_config)(name, tool_args, ctx, tool)

    return DurableFunctionToolset(
        wrapped,
        # Prefect tasks degrade gracefully to plain calls outside a flow, so the durable
        # path is always taken — matching the previous Prefect wrapper.
        in_durable_context=lambda: True,
        call_tool_operation=call_tool_operation,
        resolve_tool_config=lambda tool, name: resolve_tool_task_config(tool, name, tool_task_config),
        lifecycle='enter-always',
        durable_config=base_config,
    )


PrefectFunctionToolset = DurableFunctionToolset
