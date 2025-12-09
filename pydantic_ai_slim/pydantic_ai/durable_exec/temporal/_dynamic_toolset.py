from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import ConfigDict, with_config
from temporalio import activity, workflow
from temporalio.workflow import ActivityConfig
from typing_extensions import Self

from pydantic_ai import ToolsetTool
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets._dynamic import DynamicToolset
from pydantic_ai.toolsets.external import TOOL_SCHEMA_VALIDATOR

from ._run_context import TemporalRunContext
from ._toolset import (
    CallToolParams,
    CallToolResult,
    TemporalWrapperToolset,
)


@dataclass
@with_config(ConfigDict(arbitrary_types_allowed=True))
class _GetToolsParams:
    serialized_run_context: Any


class TemporalDynamicToolset(TemporalWrapperToolset[AgentDepsT]):
    """Temporal wrapper for DynamicToolset.

    This provides static activities (get_tools, call_tool) that are registered at worker start time,
    while the actual toolset selection happens dynamically inside the activities where I/O is allowed.
    """

    def __init__(
        self,
        toolset: DynamicToolset[AgentDepsT],
        *,
        activity_name_prefix: str,
        activity_config: ActivityConfig,
        tool_activity_config: dict[str, ActivityConfig | Literal[False]],
        deps_type: type[AgentDepsT],
        run_context_type: type[TemporalRunContext[AgentDepsT]] = TemporalRunContext[AgentDepsT],
    ):
        super().__init__(toolset)
        self.activity_config = activity_config
        self.tool_activity_config = tool_activity_config
        self.run_context_type = run_context_type

        async def get_tools_activity(params: _GetToolsParams, deps: AgentDepsT) -> dict[str, ToolDefinition]:
            """Activity that calls the dynamic function and returns tool definitions."""
            ctx = self.run_context_type.deserialize_run_context(params.serialized_run_context, deps=deps)

            async with self.wrapped:
                tools = await self.wrapped.get_tools(ctx)
                return {name: tool.tool_def for name, tool in tools.items()}

        get_tools_activity.__annotations__['deps'] = deps_type

        self.get_tools_activity = activity.defn(name=f'{activity_name_prefix}__dynamic_toolset__{self.id}__get_tools')(
            get_tools_activity
        )

        async def call_tool_activity(params: CallToolParams, deps: AgentDepsT) -> CallToolResult:
            """Activity that instantiates the dynamic toolset and calls the tool."""
            ctx = self.run_context_type.deserialize_run_context(params.serialized_run_context, deps=deps)

            async with self.wrapped:
                tools = await self.wrapped.get_tools(ctx)
                tool = tools.get(params.name)
                if tool is None:
                    from pydantic_ai.exceptions import UserError

                    raise UserError(
                        f'Tool {params.name!r} not found in dynamic toolset {self.id!r}. '
                        'The dynamic toolset function may have returned a different toolset than expected.'
                    )

                args_dict = tool.args_validator.validate_python(params.tool_args)
                return await self._wrap_call_tool_result(self.wrapped.call_tool(params.name, args_dict, ctx, tool))

        call_tool_activity.__annotations__['deps'] = deps_type

        self.call_tool_activity = activity.defn(name=f'{activity_name_prefix}__dynamic_toolset__{self.id}__call_tool')(
            call_tool_activity
        )

    @property
    def temporal_activities(self) -> list[Callable[..., Any]]:
        return [self.get_tools_activity, self.call_tool_activity]

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        return None

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        if not workflow.in_workflow():
            return await super().get_tools(ctx)

        serialized_run_context = self.run_context_type.serialize_run_context(ctx)
        tool_defs = await workflow.execute_activity(
            activity=self.get_tools_activity,
            args=[
                _GetToolsParams(serialized_run_context=serialized_run_context),
                ctx.deps,
            ],
            **self.activity_config,
        )
        return {name: self._tool_for_tool_def(tool_def) for name, tool_def in tool_defs.items()}

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> Any:
        if not workflow.in_workflow():
            return await super().call_tool(name, tool_args, ctx, tool)

        tool_activity_config = self.tool_activity_config.get(name)
        if tool_activity_config is False:
            return await super().call_tool(name, tool_args, ctx, tool)

        merged_config = self.activity_config | (tool_activity_config or {})
        serialized_run_context = self.run_context_type.serialize_run_context(ctx)
        return self._unwrap_call_tool_result(
            await workflow.execute_activity(
                activity=self.call_tool_activity,
                args=[
                    CallToolParams(
                        name=name,
                        tool_args=tool_args,
                        serialized_run_context=serialized_run_context,
                        tool_def=tool.tool_def,
                    ),
                    ctx.deps,
                ],
                **merged_config,
            )
        )

    def _tool_for_tool_def(self, tool_def: ToolDefinition) -> ToolsetTool[AgentDepsT]:
        """Create a ToolsetTool from a ToolDefinition for use outside activities."""
        return ToolsetTool(
            toolset=self,
            tool_def=tool_def,
            max_retries=1,
            args_validator=TOOL_SCHEMA_VALIDATOR,
        )
