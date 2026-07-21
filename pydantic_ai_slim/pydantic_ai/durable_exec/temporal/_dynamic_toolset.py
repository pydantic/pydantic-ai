from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, cast

from temporalio import activity, workflow
from temporalio.workflow import ActivityConfig

from pydantic_ai import ToolsetTool
from pydantic_ai.durable_exec._toolset import (
    CallToolResult,
    DurableDynamicToolset,
    DynamicToolsResult,
    ToolConfig,
    call_dynamic_tool,
    get_dynamic_tools,
    unwrap_tool_call_result,
    wrap_tool_call_result,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets._dynamic import DynamicToolset

from ._run_context import TemporalRunContext, deserialize_run_context
from ._toolset import (
    CallToolParams,
    GetToolsParams,
    resolve_tool_activity_config,
)

if TYPE_CHECKING:
    from pydantic_ai.agent.abstract import AbstractAgent


def temporalize_dynamic_toolset(
    toolset: DynamicToolset[AgentDepsT],
    *,
    activity_name_prefix: str,
    activity_config: ActivityConfig,
    tool_activity_config: dict[str, ActivityConfig | Literal[False]],
    deps_type: type[AgentDepsT],
    run_context_type: type[TemporalRunContext[AgentDepsT]] = TemporalRunContext[AgentDepsT],
    agent: AbstractAgent[AgentDepsT, Any] | None = None,
) -> DurableDynamicToolset[AgentDepsT]:
    """Temporalize a dynamic toolset.

    Registers static `get_tools`/`call_tool` activities at worker start time; the actual
    toolset resolution happens inside the activities, where I/O is allowed.
    """

    async def get_tools_activity(params: GetToolsParams, deps: AgentDepsT) -> DynamicToolsResult:
        ctx = deserialize_run_context(run_context_type, params.serialized_run_context, deps=deps, agent=agent)
        return await get_dynamic_tools(toolset, ctx)

    get_tools_activity.__annotations__['deps'] = deps_type
    registered_get_tools = activity.defn(name=f'{activity_name_prefix}__dynamic_toolset__{toolset.id}__get_tools')(
        get_tools_activity
    )

    async def call_tool_activity(params: CallToolParams, deps: AgentDepsT) -> CallToolResult:
        ctx = deserialize_run_context(run_context_type, params.serialized_run_context, deps=deps, agent=agent)
        return await wrap_tool_call_result(call_dynamic_tool(toolset, params.name, params.tool_args, ctx))

    call_tool_activity.__annotations__['deps'] = deps_type
    registered_call_tool = activity.defn(name=f'{activity_name_prefix}__dynamic_toolset__{toolset.id}__call_tool')(
        call_tool_activity
    )

    async def get_tools_operation(ctx: RunContext[AgentDepsT]) -> DynamicToolsResult:
        config: ActivityConfig = {'summary': f'get tools: {toolset.id}', **activity_config}
        return await workflow.execute_activity(
            activity=registered_get_tools,
            args=[
                GetToolsParams(serialized_run_context=run_context_type.serialize_run_context(ctx)),
                ctx.deps,
            ],
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
            {
                'summary': f'call tool: {toolset.id}:{name}',
                **activity_config,
                **config,
            },
        )
        result = await workflow.execute_activity(
            activity=registered_call_tool,
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

    def resolve_tool_config(tool: ToolsetTool[Any] | None, name: str) -> ToolConfig:
        config = resolve_tool_activity_config(tool, name, tool_activity_config)
        if config is False:
            raise UserError(
                f'Temporal activity config for dynamic toolset tool {name!r} has been explicitly set to `False` '
                '(activity disabled), but dynamic-toolset tools cannot run inside the workflow: resolving the '
                'toolset and calling the tool may perform I/O. Remove the opt-out, or move the tool to a static '
                '`FunctionToolset` (async tools there may opt out of activities).'
            )
        return config

    return DurableDynamicToolset(
        toolset,
        in_durable_context=workflow.in_workflow,
        get_tools_operation=get_tools_operation,
        call_tool_operation=call_tool_operation,
        resolve_tool_config=resolve_tool_config,
        # Resolution and lifecycle happen inside the activities (or, outside a workflow,
        # on the resolved toolset that `for_run` hands the run); the construction-time
        # factory itself has nothing to enter.
        lifecycle='enter-never',
        durable_registrations=[registered_get_tools, registered_call_tool],
        durable_config=activity_config,
    )


TemporalDynamicToolset = DurableDynamicToolset
