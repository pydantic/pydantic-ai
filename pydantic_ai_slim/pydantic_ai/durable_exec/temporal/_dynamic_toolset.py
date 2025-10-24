from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import ConfigDict, with_config
from pydantic_core import SchemaValidator
from pydantic_core.core_schema import any_schema, dict_schema, str_schema
from temporalio import activity, workflow
from temporalio.workflow import ActivityConfig
from typing_extensions import Self, assert_never

from pydantic_ai import ToolsetTool
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, ModelRetry, UserError
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets._dynamic import DynamicToolset

from ._function_toolset import (
    _ApprovalRequired,  # pyright: ignore[reportPrivateUsage]
    _CallDeferred,  # pyright: ignore[reportPrivateUsage]
    _CallToolResult,  # pyright: ignore[reportPrivateUsage]
    _ModelRetry,  # pyright: ignore[reportPrivateUsage]
    _ToolReturn,  # pyright: ignore[reportPrivateUsage]
)
from ._run_context import TemporalRunContext
from ._toolset import TemporalWrapperToolset

# Generic validator for tool args (similar to MCPServer's TOOL_SCHEMA_VALIDATOR)
_TOOL_ARGS_VALIDATOR = SchemaValidator(dict_schema(str_schema(), any_schema()))


@dataclass
@with_config(ConfigDict(arbitrary_types_allowed=True))
class _GetToolsParams:
    serialized_run_context: Any


@dataclass
@with_config(ConfigDict(arbitrary_types_allowed=True))
class _CallToolParams:
    name: str
    tool_args: dict[str, Any]
    serialized_run_context: Any
    tool_def: ToolDefinition


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
        toolset_id: str,
    ):
        super().__init__(toolset)
        self.activity_config = activity_config

        # Store a typed reference to the dynamic toolset for type safety
        self._dynamic_toolset = toolset

        # Filter out False values and raise error since dynamic toolsets require I/O
        self.tool_activity_config: dict[str, ActivityConfig] = {}
        for tool_name, tool_config in tool_activity_config.items():
            if tool_config is False:
                raise UserError(
                    f'Temporal activity config for dynamic toolset tool {tool_name!r} has been explicitly set to `False` (activity disabled), '
                    'but dynamic toolsets require I/O to instantiate their tools and so cannot be run outside of an activity.'
                )
            self.tool_activity_config[tool_name] = tool_config

        self.run_context_type = run_context_type
        self._toolset_id = toolset_id

        async def get_tools_activity(params: _GetToolsParams, deps: AgentDepsT) -> dict[str, ToolDefinition]:
            """Activity that calls the dynamic function and returns tool definitions."""
            resp = {}
            ctx = self.run_context_type.deserialize_run_context(params.serialized_run_context, deps=deps)

            # Call the user's dynamic function to get the toolset
            dynamic_toolset = toolset.toolset_func(ctx)
            if inspect.isawaitable(dynamic_toolset):
                dynamic_toolset = await dynamic_toolset

            if dynamic_toolset:
                # Enter the toolset, get tools, then exit
                await dynamic_toolset.__aenter__()
                try:
                    tools = await dynamic_toolset.get_tools(ctx)
                    # Return just the tool definitions (ToolsetTool is not serializable)
                    resp = {name: tool.tool_def for name, tool in tools.items()}
                finally:
                    await dynamic_toolset.__aexit__(None, None, None)

            return resp

        # Set type hint explicitly so that Temporal can take care of serialization and deserialization
        get_tools_activity.__annotations__['deps'] = deps_type

        self.get_tools_activity = activity.defn(
            name=f'{activity_name_prefix}__dynamic_toolset__{self._toolset_id}__get_tools'
        )(get_tools_activity)

        async def call_tool_activity(params: _CallToolParams, deps: AgentDepsT) -> _CallToolResult:
            """Activity that calls the dynamic function and executes the tool."""
            ctx = self.run_context_type.deserialize_run_context(params.serialized_run_context, deps=deps)

            # Call the user's dynamic function to get the toolset
            dynamic_toolset = toolset.toolset_func(ctx)
            if inspect.isawaitable(dynamic_toolset):
                dynamic_toolset = await dynamic_toolset

            if dynamic_toolset is None:
                raise UserError(f'Dynamic toolset function returned None, but tool {params.name!r} was called')

            # Enter the toolset, call the tool, then exit
            await dynamic_toolset.__aenter__()
            try:
                # Get all tools to find the one we need
                tools = await dynamic_toolset.get_tools(ctx)
                try:
                    tool = tools[params.name]
                except KeyError as e:
                    raise UserError(
                        f'Tool {params.name!r} not found in dynamic toolset. '
                        'The dynamic function must return consistent tools for a given context.'
                    ) from e

                # Validate args and call the tool
                args_dict = tool.args_validator.validate_python(params.tool_args)
                try:
                    result = await dynamic_toolset.call_tool(params.name, args_dict, ctx, tool)
                    return _ToolReturn(result=result)
                except ApprovalRequired:
                    return _ApprovalRequired()
                except CallDeferred:
                    return _CallDeferred()
                except ModelRetry as e:
                    return _ModelRetry(message=e.message)
            finally:
                await dynamic_toolset.__aexit__(None, None, None)

        # Set type hint explicitly so that Temporal can take care of serialization and deserialization
        call_tool_activity.__annotations__['deps'] = deps_type

        self.call_tool_activity = activity.defn(
            name=f'{activity_name_prefix}__dynamic_toolset__{self._toolset_id}__call_tool'
        )(call_tool_activity)

    def tool_for_tool_def(self, tool_def: ToolDefinition) -> ToolsetTool[AgentDepsT]:
        """Create a ToolsetTool from a ToolDefinition."""
        return ToolsetTool(
            toolset=self,
            tool_def=tool_def,
            max_retries=1,  # Default, will be overridden by tool manager
            args_validator=_TOOL_ARGS_VALIDATOR,
        )

    @property
    def id(self) -> str:
        """Return the fixed ID for this dynamic toolset."""
        return self._toolset_id

    @property
    def temporal_activities(self) -> list[Callable[..., Any]]:
        return [self.get_tools_activity, self.call_tool_activity]

    async def __aenter__(self) -> Self:
        # The dynamic toolset enters its wrapped toolsets inside activities
        # so we don't need to enter anything here
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        return None

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        if not workflow.in_workflow():
            # Don't use super().get_tools() because DynamicToolset has lifecycle issues.
            # Instead, call the dynamic function and manage lifecycle properly in the same scope.
            toolset = self._dynamic_toolset.toolset_func(ctx)
            if inspect.isawaitable(toolset):
                toolset = await toolset

            if toolset is None:
                return {}

            # Enter and exit in the same async context to avoid lifecycle violations
            async with toolset:
                tools = await toolset.get_tools(ctx)
                # Wrap the tools with our toolset reference
                return {
                    name: ToolsetTool(
                        toolset=self,
                        tool_def=tool.tool_def,
                        max_retries=tool.max_retries,
                        args_validator=tool.args_validator,
                    )
                    for name, tool in tools.items()
                }

        serialized_run_context = self.run_context_type.serialize_run_context(ctx)
        tool_defs = await workflow.execute_activity(  # pyright: ignore[reportUnknownMemberType]
            activity=self.get_tools_activity,
            args=[
                _GetToolsParams(serialized_run_context=serialized_run_context),
                ctx.deps,
            ],
            **self.activity_config,
        )
        return {name: self.tool_for_tool_def(tool_def) for name, tool_def in tool_defs.items()}

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> Any:
        if not workflow.in_workflow():
            # Don't use super().call_tool() because DynamicToolset has lifecycle issues.
            # Instead, call the dynamic function and manage lifecycle properly in the same scope.
            toolset = self._dynamic_toolset.toolset_func(ctx)
            if inspect.isawaitable(toolset):
                toolset = await toolset

            if toolset is None:
                raise UserError(f'Dynamic toolset function returned None, but tool {name!r} was called')

            # Enter and exit in the same async context to avoid lifecycle violations
            async with toolset:
                tools = await toolset.get_tools(ctx)
                try:
                    actual_tool = tools[name]
                except KeyError as e:
                    raise UserError(
                        f'Tool {name!r} not found in dynamic toolset. '
                        'The dynamic function must return consistent tools for a given context.'
                    ) from e
                return await toolset.call_tool(name, tool_args, ctx, actual_tool)

        tool_activity_config = self.activity_config | self.tool_activity_config.get(name, {})
        serialized_run_context = self.run_context_type.serialize_run_context(ctx)
        result = await workflow.execute_activity(  # pyright: ignore[reportUnknownMemberType]
            activity=self.call_tool_activity,
            args=[
                _CallToolParams(
                    name=name,
                    tool_args=tool_args,
                    serialized_run_context=serialized_run_context,
                    tool_def=tool.tool_def,
                ),
                ctx.deps,
            ],
            **tool_activity_config,
        )

        # Handle the result (same pattern as TemporalFunctionToolset)
        if isinstance(result, _ApprovalRequired):
            raise ApprovalRequired()
        elif isinstance(result, _CallDeferred):
            raise CallDeferred()
        elif isinstance(result, _ModelRetry):
            raise ModelRetry(result.message)
        elif isinstance(result, _ToolReturn):
            return result.result
        else:
            assert_never(result)
