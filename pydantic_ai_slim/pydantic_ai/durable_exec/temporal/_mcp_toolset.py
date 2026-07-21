from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

from temporalio import activity, workflow
from temporalio.workflow import ActivityConfig

from pydantic_ai import ToolsetTool
from pydantic_ai.durable_exec._toolset import (
    CallToolResult,
    DurableMCPToolset,
    ToolConfig,
    unwrap_tool_call_result,
    wrap_tool_call_result,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import InstructionPart
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition

from ._run_context import TemporalRunContext, deserialize_run_context
from ._toolset import CallToolParams, GetToolsParams, resolve_tool_activity_config

if TYPE_CHECKING:
    from pydantic_ai.agent.abstract import AbstractAgent


def temporalize_mcp_toolset(
    toolset: MCPToolset[AgentDepsT],
    *,
    activity_name_prefix: str,
    activity_config: ActivityConfig,
    tool_activity_config: dict[str, ActivityConfig | Literal[False]],
    deps_type: type[AgentDepsT],
    run_context_type: type[TemporalRunContext[AgentDepsT]] = TemporalRunContext[AgentDepsT],
    agent: AbstractAgent[AgentDepsT, Any] | None = None,
) -> DurableMCPToolset[AgentDepsT]:
    for tool_name, config in tool_activity_config.items():
        if config is False:
            raise UserError(
                f'Temporal activity config for MCP tool {tool_name!r} has been explicitly set to `False` (activity disabled), '
                'but MCP tools require the use of IO and so cannot be run outside of an activity.'
            )

    async def get_tools_activity(params: GetToolsParams, deps: AgentDepsT) -> dict[str, ToolDefinition]:
        ctx = deserialize_run_context(run_context_type, params.serialized_run_context, deps=deps, agent=agent)
        return {name: tool.tool_def for name, tool in (await toolset.get_tools(ctx)).items()}

    async def get_instructions_activity(
        params: GetToolsParams, deps: AgentDepsT
    ) -> str | InstructionPart | Sequence[str | InstructionPart] | None:
        ctx = deserialize_run_context(run_context_type, params.serialized_run_context, deps=deps, agent=agent)
        async with toolset:
            return await toolset.get_instructions(ctx)

    async def call_tool_activity(params: CallToolParams, deps: AgentDepsT) -> CallToolResult:
        ctx = deserialize_run_context(run_context_type, params.serialized_run_context, deps=deps, agent=agent)
        assert isinstance(params.tool_def, ToolDefinition)
        return await wrap_tool_call_result(
            toolset.call_tool(params.name, params.tool_args, ctx, toolset.tool_for_tool_def(params.tool_def))
        )

    for activity_func in (get_tools_activity, get_instructions_activity, call_tool_activity):
        activity_func.__annotations__['deps'] = deps_type
    get_tools_activity_def = activity.defn(name=f'{activity_name_prefix}__mcp_server__{toolset.id}__get_tools')(
        get_tools_activity
    )
    get_instructions_activity_def = activity.defn(
        name=f'{activity_name_prefix}__mcp_server__{toolset.id}__get_instructions'
    )(get_instructions_activity)
    call_tool_activity_def = activity.defn(name=f'{activity_name_prefix}__mcp_server__{toolset.id}__call_tool')(
        call_tool_activity
    )

    def resolve_tool_config(tool: ToolsetTool[Any] | None, name: str) -> ToolConfig:
        config = resolve_tool_activity_config(tool, name, tool_activity_config)
        if (
            config is False
        ):  # pragma: no cover — the constructor-dict path raises above; metadata is the only route here
            raise UserError(
                f'Temporal activity config for MCP tool {name!r} has been explicitly set to `False` (activity disabled), '
                'but MCP tools require the use of IO and so cannot be run outside of an activity.'
            )
        return config

    async def get_tools_operation(ctx: RunContext[AgentDepsT]) -> dict[str, ToolDefinition]:
        config: ActivityConfig = {'summary': f'get tools: {toolset.id}', **activity_config}
        return await workflow.execute_activity(
            activity=get_tools_activity_def,
            args=[GetToolsParams(serialized_run_context=run_context_type.serialize_run_context(ctx)), ctx.deps],
            **config,
        )

    async def get_instructions_operation(
        ctx: RunContext[AgentDepsT],
    ) -> str | InstructionPart | Sequence[str | InstructionPart] | None:
        config: ActivityConfig = {'summary': f'get instructions: {toolset.id}', **activity_config}
        return await workflow.execute_activity(
            activity=get_instructions_activity_def,
            args=[GetToolsParams(serialized_run_context=run_context_type.serialize_run_context(ctx)), ctx.deps],
            **config,
        )

    async def call_tool_operation(
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
        config: Mapping[str, Any],
    ) -> Any:
        merged_config = cast(
            'ActivityConfig',
            {'summary': f'call tool: {toolset.id}:{name}', **activity_config, **config},
        )
        result = await workflow.execute_activity(
            activity=call_tool_activity_def,
            args=[
                CallToolParams(
                    name=name,
                    tool_args=tool_args,
                    serialized_run_context=run_context_type.serialize_run_context(ctx),
                    tool_def=tool.tool_def,
                ),
                ctx.deps,
            ],
            **merged_config,
        )
        return unwrap_tool_call_result(result)

    return DurableMCPToolset(
        toolset,
        in_durable_context=workflow.in_workflow,
        get_tools_operation=get_tools_operation,
        get_instructions_operation=get_instructions_operation,
        call_tool_operation=call_tool_operation,
        resolve_tool_config=resolve_tool_config,
        lifecycle='enter-outside-durable',
        durable_registrations=[get_instructions_activity_def, get_tools_activity_def, call_tool_activity_def],
        durable_config=activity_config,
    )


TemporalMCPToolset = DurableMCPToolset
