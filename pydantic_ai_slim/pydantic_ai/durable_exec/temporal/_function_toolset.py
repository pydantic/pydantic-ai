from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, cast

from temporalio import activity, workflow
from temporalio.workflow import ActivityConfig

from pydantic_ai import FunctionToolset, ToolsetTool
from pydantic_ai.durable_exec._toolset import (
    CallToolResult,
    DurableFunctionToolset,
    ToolConfig,
    unwrap_tool_call_result,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets.function import FunctionToolsetTool

from ._activity_execution import execute_activity
from ._run_context import TemporalRunContext, deserialize_run_context
from ._toolset import (
    CallToolParams,
    call_tool_in_activity,
    heartbeating,
    resolve_tool_activity_config,
    tool_result_payload_errors,
)

if TYPE_CHECKING:
    from pydantic_ai.agent.abstract import AbstractAgent

_TOOL_INSTANCE_DEFAULTS_PATCH = 'pydantic_ai:function_toolset_defaults_on_tool_instances'
"""`workflow.patched()` ID marking histories in which `FunctionToolset` applied its `sequential` and `defer_loading` to `Tool` instances."""


def gate_tool_instance_defaults(
    toolset: FunctionToolset[AgentDepsT], tools: dict[str, ToolsetTool[AgentDepsT]]
) -> dict[str, ToolsetTool[AgentDepsT]]:
    """Drop `toolset`'s `sequential` and `defer_loading` from its `Tool` instances in histories recorded before it set them.

    A sequential or deferred tool changes which activities the workflow schedules, so those histories
    replay only without them.
    """
    made_sequential = toolset._tools_made_sequential  # pyright: ignore[reportPrivateUsage]
    made_deferred = toolset._tools_made_deferred  # pyright: ignore[reportPrivateUsage]
    unpatched: dict[str, ToolsetTool[AgentDepsT]] = {}
    for name, tool in tools.items():
        assert isinstance(tool, FunctionToolsetTool)
        tool_def = replace(
            tool.tool_def,
            sequential=tool.tool_def.sequential and tool.original_name not in made_sequential,
            defer_loading=tool.tool_def.defer_loading and tool.original_name not in made_deferred,
        )
        if tool_def != tool.tool_def:
            unpatched[name] = replace(tool, tool_def=tool_def)
    if not unpatched or workflow.patched(_TOOL_INSTANCE_DEFAULTS_PATCH):
        return tools
    return {**tools, **unpatched}


def temporalize_function_toolset(
    toolset: FunctionToolset[AgentDepsT],
    *,
    activity_name_prefix: str,
    activity_config: ActivityConfig,
    tool_activity_config: dict[str, ActivityConfig | Literal[False]],
    deps_type: type[AgentDepsT],
    run_context_type: type[TemporalRunContext[AgentDepsT]] = TemporalRunContext[AgentDepsT],
    agent: AbstractAgent[AgentDepsT, Any] | None = None,
) -> DurableFunctionToolset[AgentDepsT]:
    async def call_tool_activity(params: CallToolParams, deps: AgentDepsT) -> CallToolResult:
        async with heartbeating():
            ctx = deserialize_run_context(run_context_type, params.serialized_run_context, deps=deps, agent=agent)
            try:
                if params.tool_def is not None:
                    # Rebuild the tool from the definition the workflow prepared, so a tool's `prepare`
                    # function isn't run a second time here against the activity's limited run context.
                    tool = toolset.tool_for_tool_def(params.tool_def, ctx=ctx, original_name=params.original_name)
                else:
                    # Only reachable for an activity scheduled by a worker predating `tool_def` on these
                    # params; re-prepare so in-flight executions still complete across the upgrade.
                    tool = (await toolset.get_tools(ctx))[params.name]
            except KeyError as exc:
                raise UserError(
                    f'Tool {params.name!r} not found in toolset {toolset.id!r}. '
                    'Removing or renaming tools during an agent run is not supported with Temporal.'
                ) from exc
            return await call_tool_in_activity(toolset, params.name, params.tool_args, ctx, tool)

    call_tool_activity.__annotations__['deps'] = deps_type
    registered_activity = activity.defn(name=f'{activity_name_prefix}__toolset__{toolset.id}__call_tool')(
        call_tool_activity
    )

    def resolve_tool_config(tool: ToolsetTool[Any] | None, name: str) -> ToolConfig:
        config = resolve_tool_activity_config(tool, name, tool_activity_config)
        if config is False:
            assert isinstance(tool, FunctionToolsetTool)
            if not tool.is_async:
                raise UserError(
                    f'Temporal activity config for tool {name!r} has been explicitly set to `False` (activity disabled), '
                    'but non-async tools are run in threads which are not supported outside of an activity. Make the tool function async instead.'
                )
        return config

    async def call_tool_operation(
        name: str,
        tool_args: dict[str, Any],
        *,
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
        with tool_result_payload_errors(name):
            result = await execute_activity(
                activity=registered_activity,
                args=[
                    CallToolParams(
                        name=name,
                        tool_args=tool_args,
                        serialized_run_context=run_context_type.serialize_run_context(ctx),
                        tool_def=tool.tool_def,
                        # A `prepare` function can expose a tool under a different name than the toolset
                        # holds it under; the activity needs the latter to find the function to call.
                        original_name=tool.original_name if isinstance(tool, FunctionToolsetTool) else None,
                    ),
                    ctx.deps,
                ],
                **merged_config,
            )
        return unwrap_tool_call_result(result)

    return DurableFunctionToolset(
        toolset,
        in_durable_context=workflow.in_workflow,
        tools_in_durable_context=partial(gate_tool_instance_defaults, toolset),
        call_tool_operation=call_tool_operation,
        resolve_tool_config=resolve_tool_config,
        lifecycle='enter-outside-durable',
        durable_registrations=[registered_activity],
        durable_config=activity_config,
    )


TemporalFunctionToolset = DurableFunctionToolset
