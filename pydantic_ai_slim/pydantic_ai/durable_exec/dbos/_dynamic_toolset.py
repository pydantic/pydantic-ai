from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dbos import DBOS

from pydantic_ai import ToolsetTool
from pydantic_ai.durable_exec._toolset import (
    CallToolResult,
    DurableDynamicToolset,
    DynamicToolsResult,
    call_dynamic_tool,
    get_dynamic_tools,
    unwrap_recorded_tool_call_result,
    wrap_tool_call_result,
)
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets._dynamic import DynamicToolset

from ._utils import StepConfig, guard_enqueue_in_workflow


def dbosify_dynamic_toolset(
    wrapped: DynamicToolset[AgentDepsT], *, step_name_prefix: str, step_config: StepConfig
) -> DurableDynamicToolset[AgentDepsT]:
    name = f'{step_name_prefix}__dynamic_toolset__{wrapped.id}'

    @DBOS.step(name=f'{name}.get_tools', **(step_config or {}))
    async def get_tools_step(ctx: RunContext[AgentDepsT]) -> DynamicToolsResult:
        return await get_dynamic_tools(wrapped, ctx)

    @DBOS.step(name=f'{name}.call_tool', **(step_config or {}))
    async def call_tool_step(tool_name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT]) -> CallToolResult:
        # DBOS has no selective non-retryable-exception support, so control-flow
        # exceptions must cross the step boundary as successful values.
        return await wrap_tool_call_result(
            call_dynamic_tool(wrapped, tool_name, tool_args, guard_enqueue_in_workflow(ctx))
        )

    async def call_tool_operation(
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
        config: Mapping[str, Any],
    ) -> Any:
        # A recovering workflow may replay outputs this step recorded before it wrapped
        # control-flow exceptions as values; those recordings are the raw tool result.
        return unwrap_recorded_tool_call_result(await call_tool_step(name, tool_args, ctx))

    return DurableDynamicToolset(
        wrapped,
        # DBOS steps degrade gracefully to plain calls outside a workflow.
        in_durable_context=lambda: True,
        get_tools_operation=get_tools_step,
        call_tool_operation=call_tool_operation,
        # DBOS takes no per-tool config; tool metadata is ignored.
        resolve_tool_config=lambda tool, name: {},
        lifecycle='enter-never',
        durable_config=step_config,
    )
