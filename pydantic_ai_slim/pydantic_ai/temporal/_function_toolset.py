from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pydantic import ConfigDict, with_config
from temporalio import activity, workflow

from pydantic_ai.toolsets.function import FunctionToolset

from .._run_context import RunContext
from ..toolsets import ToolsetTool
from ._settings import TemporalSettings


@dataclass
@with_config(ConfigDict(arbitrary_types_allowed=True))
class _CallToolParams:
    name: str
    tool_args: dict[str, Any]
    serialized_run_context: Any


def temporalize_function_toolset(
    toolset: FunctionToolset,
    settings: TemporalSettings | None = None,
) -> list[Callable[..., Any]]:
    """Temporalize a function toolset.

    Args:
        toolset: The function toolset to temporalize.
        settings: The temporal settings to use.
    """
    if activities := getattr(toolset, '__temporal_activities', None):
        return activities

    id = toolset.id
    if not id:
        raise ValueError(
            "A function toolset needs to have an ID in order to be used in a durable execution environment like Temporal. The ID will be used to identify the toolset's activities within the workflow."
        )

    settings = settings or TemporalSettings()

    original_call_tool = toolset.call_tool

    @activity.defn(name=f'function_toolset__{id}__call_tool')
    async def call_tool_activity(params: _CallToolParams) -> Any:
        name = params.name
        ctx = settings.for_tool(id, name).deserialize_run_context(params.serialized_run_context)
        tool = (await toolset.get_tools(ctx))[name]
        return await original_call_tool(name, params.tool_args, ctx, tool)

    async def call_tool(name: str, tool_args: dict[str, Any], ctx: RunContext, tool: ToolsetTool) -> Any:
        tool_settings = settings.for_tool(id, name)
        serialized_run_context = tool_settings.serialize_run_context(ctx)
        return await workflow.execute_activity(  # pyright: ignore[reportUnknownMemberType]
            activity=call_tool_activity,
            arg=_CallToolParams(name=name, tool_args=tool_args, serialized_run_context=serialized_run_context),
            **tool_settings.execute_activity_options,
        )

    toolset.call_tool = call_tool

    activities = [call_tool_activity]
    setattr(toolset, '__temporal_activities', activities)
    return activities
