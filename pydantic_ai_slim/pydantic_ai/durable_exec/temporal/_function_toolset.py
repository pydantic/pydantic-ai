from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, cast

from temporalio import activity, workflow
from temporalio.workflow import ActivityConfig

from pydantic_ai import FunctionToolset, ToolsetTool
from pydantic_ai.durable_exec._toolset import CallToolResult, DurableFunctionToolset, unwrap_tool_call_result
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext

from ._run_context import TemporalRunContext, deserialize_run_context
from ._toolset import CallToolParams, call_tool_in_activity, resolve_tool_activity_config

if TYPE_CHECKING:
    from pydantic_ai.agent.abstract import AbstractAgent


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
        ctx = deserialize_run_context(run_context_type, params.serialized_run_context, deps=deps, agent=agent)
        try:
            tool = (await toolset.get_tools(ctx))[params.name]
        except KeyError as exc:  # pragma: no cover
            raise UserError(
                f'Tool {params.name!r} not found in toolset {toolset.id!r}. '
                'Removing or renaming tools during an agent run is not supported with Temporal.'
            ) from exc
        return await call_tool_in_activity(toolset, params.name, params.tool_args, ctx, tool)

    call_tool_activity.__annotations__['deps'] = deps_type
    registered_activity = activity.defn(name=f'{activity_name_prefix}__toolset__{toolset.id}__call_tool')(
        call_tool_activity
    )

    async def call_tool_segment(
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
            activity=registered_activity,
            args=[
                CallToolParams(
                    name=name,
                    tool_args=tool_args,
                    serialized_run_context=run_context_type.serialize_run_context(ctx),
                    tool_def=None,
                ),
                ctx.deps,
            ],
            **merged_config,
        )
        return unwrap_tool_call_result(result)

    return DurableFunctionToolset(
        toolset,
        in_durable_context=workflow.in_workflow,
        call_tool_segment=call_tool_segment,
        resolve_tool_config=lambda tool, name: resolve_tool_activity_config(tool, name, tool_activity_config),
        inline_requires_async=True,
        lifecycle='enter-outside-durable',
        durable_registrations=[registered_activity],
        durable_config=activity_config,
    )


TemporalFunctionToolset = DurableFunctionToolset
