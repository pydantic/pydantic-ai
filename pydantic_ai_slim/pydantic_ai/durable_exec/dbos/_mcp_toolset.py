from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dbos import DBOS

from pydantic_ai import ToolsetTool
from pydantic_ai.durable_exec._toolset import (
    CallToolResult,
    DurableMCPToolset,
    unwrap_recorded_tool_call_result,
    wrap_tool_call_result,
)
from pydantic_ai.mcp import MCPToolset, ToolResult
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition

from ._utils import StepConfig, guard_enqueue_in_workflow


def dbosify_mcp_toolset(
    wrapped: MCPToolset[AgentDepsT], *, step_name_prefix: str, step_config: StepConfig
) -> DurableMCPToolset[AgentDepsT]:
    id_suffix = f'__{wrapped.id}' if wrapped.id else ''
    name = f'{step_name_prefix}__mcp_server{id_suffix}'

    @DBOS.step(name=f'{name}.get_tools', **(step_config or {}))
    async def get_tools_step(ctx: RunContext[AgentDepsT]) -> dict[str, ToolDefinition]:
        return {tool_name: tool.tool_def for tool_name, tool in (await wrapped.get_tools(ctx)).items()}

    @DBOS.step(name=f'{name}.get_instructions', **(step_config or {}))
    async def get_instructions_step(ctx: RunContext[AgentDepsT]):
        async with wrapped:
            return await wrapped.get_instructions(ctx)

    @DBOS.step(name=f'{name}.call_tool', **(step_config or {}))
    async def call_tool_step(
        tool_name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> CallToolResult:
        # The context is guarded because a `process_tool_call=` hook receives it and could enqueue.
        # DBOS has no selective non-retryable-exception support, so control-flow
        # exceptions must cross the step boundary as successful values.
        return await wrap_tool_call_result(
            wrapped.call_tool(tool_name, tool_args, guard_enqueue_in_workflow(ctx), tool)
        )

    async def call_tool_operation(
        tool_name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
        config: Mapping[str, Any],
    ) -> ToolResult:
        # A recovering workflow may replay outputs this step recorded before it wrapped
        # control-flow exceptions as values; those recordings are the raw tool result.
        return unwrap_recorded_tool_call_result(await call_tool_step(tool_name, tool_args, ctx, tool))

    return DurableMCPToolset(
        wrapped,
        # DBOS steps degrade gracefully to plain calls outside a workflow, so the durable
        # path is always taken — matching the previous DBOS wrapper, which never gated on
        # workflow state (outside a workflow, the step fallback still enters the server
        # around `get_instructions`).
        in_durable_context=lambda: True,
        get_tools_operation=get_tools_step,
        get_instructions_operation=get_instructions_step,
        call_tool_operation=call_tool_operation,
        # DBOS takes no per-tool config; tool metadata is ignored, as before.
        resolve_tool_config=lambda tool, name: {},
        lifecycle='enter-never',
        durable_config=step_config,
    )


DBOSMCPToolset = DurableMCPToolset
