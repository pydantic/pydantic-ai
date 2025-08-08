from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from pydantic import ConfigDict, with_config
from temporalio import activity, workflow
from temporalio.workflow import ActivityConfig

from pydantic_ai._run_context import RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.toolsets import FunctionToolset, ToolsetTool
from pydantic_ai.toolsets.function import FunctionToolsetTool

from ._run_context import TemporalRunContext
from ._toolset import TemporalWrapperToolset


@dataclass
@with_config(ConfigDict(arbitrary_types_allowed=True))
class _CallToolParams:
    name: str
    tool_args: dict[str, Any]
    serialized_run_context: Any


class TemporalFunctionToolset(TemporalWrapperToolset):
    def __init__(
        self,
        toolset: FunctionToolset,
        *,
        activity_name_prefix: str,
        activity_config: ActivityConfig,
        tool_activity_config: dict[str, ActivityConfig | Literal[False]],
        run_context_type: type[TemporalRunContext] = TemporalRunContext,
    ):
        super().__init__(toolset)
        self.activity_config = activity_config
        self.tool_activity_config = tool_activity_config
        self.run_context_type = run_context_type

        @activity.defn(name=f'{activity_name_prefix}__toolset__{self.id}__call_tool')
        async def call_tool_activity(params: _CallToolParams) -> Any:
            name = params.name
            ctx = self.run_context_type.deserialize_run_context(params.serialized_run_context)
            try:
                tool = (await toolset.get_tools(ctx))[name]
            except KeyError as e:
                raise UserError(
                    f'Tool {name!r} not found in toolset {self.id!r}. '
                    'Removing or renaming tools during an agent run is not supported with Temporal.'
                ) from e

            return await self.wrapped.call_tool(name, params.tool_args, ctx, tool)

        self.call_tool_activity = call_tool_activity

    @property
    def wrapped_function_toolset(self) -> FunctionToolset:
        assert isinstance(self.wrapped, FunctionToolset)
        return self.wrapped

    @property
    def temporal_activities(self) -> list[Callable[..., Any]]:
        return [self.call_tool_activity]

    async def call_tool(self, name: str, tool_args: dict[str, Any], ctx: RunContext, tool: ToolsetTool) -> Any:
        if not workflow.in_workflow():
            return await super().call_tool(name, tool_args, ctx, tool)

        tool_activity_config = self.tool_activity_config.get(name, {})
        if tool_activity_config is False:
            assert isinstance(tool, FunctionToolsetTool)
            if not tool.is_async:
                raise UserError(
                    f'Temporal activity config for tool {name!r} has been explicitly set to `False` (activity disabled), '
                    'but non-async tools are run in threads which are not supported outside of an activity. Make the tool function async instead.'
                )
            return await super().call_tool(name, tool_args, ctx, tool)

        tool_activity_config = self.activity_config | tool_activity_config
        serialized_run_context = self.run_context_type.serialize_run_context(ctx)
        return await workflow.execute_activity(  # pyright: ignore[reportUnknownMemberType]
            activity=self.call_tool_activity,
            arg=_CallToolParams(name=name, tool_args=tool_args, serialized_run_context=serialized_run_context),
            **tool_activity_config,
        )
