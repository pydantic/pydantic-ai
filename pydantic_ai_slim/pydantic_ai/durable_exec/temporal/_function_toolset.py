from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal, cast

from temporalio import activity, workflow
from temporalio.workflow import ActivityConfig

from pydantic_ai import FunctionToolset, ToolsetTool
from pydantic_ai.durable_exec._toolset import (
    CallToolResult,
    DurableFunctionToolset,
    ToolConfig,
    unwrap_tool_call_result,
    validate_tool_args,
    validation_context_from_agent,
    wrap_tool_call_result,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets.function import FunctionToolsetTool

from ._activity_execution import execute_activity
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
    # `RunContext.validation_context` holds an arbitrary user object that can't cross the activity
    # boundary, so it's rebuilt inside the activity from the agent's spec.
    validation_context = validation_context_from_agent(agent)

    def tool_in_activity(tools: dict[str, ToolsetTool[AgentDepsT]], name: str) -> ToolsetTool[AgentDepsT]:
        try:
            return tools[name]
        except KeyError as exc:  # pragma: no cover
            raise UserError(
                f'Tool {name!r} not found in toolset {toolset.id!r}. '
                'Removing or renaming tools during an agent run is not supported with Temporal.'
            ) from exc

    async def call_tool_activity(params: CallToolParams, deps: AgentDepsT) -> CallToolResult:
        ctx = deserialize_run_context(run_context_type, params.serialized_run_context, deps=deps, agent=agent)
        tool = tool_in_activity(await toolset.get_tools(ctx), params.name)
        return await call_tool_in_activity(
            toolset, params.name, params.tool_args, ctx, tool, validation_context=validation_context
        )

    call_tool_activity.__annotations__['deps'] = deps_type
    registered_activity = activity.defn(name=f'{activity_name_prefix}__toolset__{toolset.id}__call_tool')(
        call_tool_activity
    )

    async def validate_args_activity(params: CallToolParams, deps: AgentDepsT) -> CallToolResult:
        ctx = deserialize_run_context(run_context_type, params.serialized_run_context, deps=deps, agent=agent)
        tool = tool_in_activity(await toolset.get_tools(ctx), params.name)
        return await wrap_tool_call_result(
            validate_tool_args(tool, params.tool_args, ctx, validation_context=validation_context)
        )

    validate_args_activity.__annotations__['deps'] = deps_type
    registered_validate_args_activity = activity.defn(
        name=f'{activity_name_prefix}__toolset__{toolset.id}__validate_args'
    )(validate_args_activity)

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

    async def run_tool_activity(
        registered: Callable[..., Any],
        summary: str,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        config: Mapping[str, Any],
    ) -> Any:
        merged_config = cast('ActivityConfig', {'summary': summary, **activity_config, **config})
        result = await execute_activity(
            activity=registered,
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

    async def call_tool_operation(
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
        config: Mapping[str, Any],
    ) -> Any:
        return await run_tool_activity(
            registered_activity, f'call tool: {toolset.id}:{name}', name, tool_args, ctx, config
        )

    async def validate_args_operation(
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
        config: Mapping[str, Any],
    ) -> None:
        await run_tool_activity(
            registered_validate_args_activity,
            f'validate tool args: {toolset.id}:{name}',
            name,
            tool_args,
            ctx,
            config,
        )

    return DurableFunctionToolset(
        toolset,
        in_durable_context=workflow.in_workflow,
        call_tool_operation=call_tool_operation,
        validate_args_operation=validate_args_operation,
        resolve_tool_config=resolve_tool_config,
        lifecycle='enter-outside-durable',
        durable_registrations=[registered_activity, registered_validate_args_activity],
        durable_config=activity_config,
    )


TemporalFunctionToolset = DurableFunctionToolset
