from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from pydantic import ConfigDict, with_config
from temporalio import activity, workflow
from temporalio.workflow import ActivityConfig
from typing_extensions import Self

from pydantic_ai.exceptions import UserError
from pydantic_ai.mcp import MCPServer, ToolResult
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets.abstract import ToolsetTool

from ._run_context import TemporalRunContext
from ._toolset import TemporalWrapperToolset


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


class TemporalMCPServer(TemporalWrapperToolset[AgentDepsT]):
    def __init__(
        self,
        server: MCPServer,
        *,
        activity_name_prefix: str,
        activity_config: ActivityConfig,
        tool_activity_config: dict[str, ActivityConfig | Literal[False]],
        deps_type: type[AgentDepsT],
        run_context_type: type[TemporalRunContext[AgentDepsT]] = TemporalRunContext[AgentDepsT],
    ):
        super().__init__(server)
        self.activity_config = activity_config

        self.tool_activity_config: dict[str, ActivityConfig] = {}
        for tool_name, tool_config in tool_activity_config.items():
            if tool_config is False:
                raise UserError(
                    f'Temporal activity config for MCP tool {tool_name!r} has been explicitly set to `False` (activity disabled), '
                    'but MCP tools require the use of IO and so cannot be run outside of an activity.'
                )
            self.tool_activity_config[tool_name] = tool_config

        self.run_context_type = run_context_type

        async def get_tools_activity(params: _GetToolsParams, deps: AgentDepsT) -> dict[str, ToolDefinition]:
            run_context = self.run_context_type.deserialize_run_context(params.serialized_run_context, deps=deps)
            tools = await self.wrapped.get_tools(run_context)
            # ToolsetTool is not serializable as it holds a SchemaValidator (which is also the same for every MCP tool so unnecessary to pass along the wire every time),
            # so we just return the ToolDefinitions and wrap them in ToolsetTool outside of the activity.
            return {name: tool.tool_def for name, tool in tools.items()}

        # Set type hint explicitly so that Temporal can take care of serialization and deserialization
        get_tools_activity.__annotations__['deps'] = deps_type

        self.get_tools_activity = activity.defn(name=f'{activity_name_prefix}__mcp_server__{self.id}__get_tools')(
            get_tools_activity
        )

        async def call_tool_activity(params: _CallToolParams, deps: AgentDepsT) -> ToolResult:
            run_context = self.run_context_type.deserialize_run_context(params.serialized_run_context, deps=deps)
            return await self.wrapped.call_tool(
                params.name,
                params.tool_args,
                run_context,
                self.tool_for_tool_def(params.tool_def),
            )

        # Set type hint explicitly so that Temporal can take care of serialization and deserialization
        call_tool_activity.__annotations__['deps'] = deps_type

        self.call_tool_activity = activity.defn(name=f'{activity_name_prefix}__mcp_server__{self.id}__call_tool')(
            call_tool_activity
        )

    def tool_for_tool_def(self, tool_def: ToolDefinition) -> ToolsetTool[AgentDepsT]:
        assert isinstance(self.wrapped, MCPServer)
        return self.wrapped.tool_for_tool_def(tool_def)

    @property
    def temporal_activities(self) -> list[Callable[..., Any]]:
        return [self.get_tools_activity, self.call_tool_activity]

    async def __aenter__(self) -> Self:
        # The wrapped MCPServer enters itself around listing and calling tools
        # so we don't need to enter it here (nor could we because we're not inside a Temporal activity).
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        return None

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        if not workflow.in_workflow():
            return await super().get_tools(ctx)

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
    ) -> ToolResult:
        if not workflow.in_workflow():
            return await super().call_tool(name, tool_args, ctx, tool)

        tool_activity_config = self.activity_config | self.tool_activity_config.get(name, {})
        serialized_run_context = self.run_context_type.serialize_run_context(ctx)
        return await workflow.execute_activity(  # pyright: ignore[reportUnknownMemberType]
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
