from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal

from temporalio import activity, workflow
from temporalio.workflow import ActivityConfig

from pydantic_ai import ToolsetTool
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import InstructionPart
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset

from ._run_context import TemporalRunContext, deserialize_run_context

if TYPE_CHECKING:
    from pydantic_ai.agent.abstract import AbstractAgent
from ._toolset import (
    CallToolParams,
    CallToolResult,
    GetToolsParams,
    TemporalWrapperToolset,
)


class TemporalMCPToolset(TemporalWrapperToolset[AgentDepsT], ABC):
    def __init__(
        self,
        toolset: AbstractToolset[AgentDepsT],
        *,
        activity_name_prefix: str,
        activity_config: ActivityConfig,
        tool_activity_config: dict[str, ActivityConfig | Literal[False]],
        deps_type: type[AgentDepsT],
        run_context_type: type[TemporalRunContext[AgentDepsT]] = TemporalRunContext[AgentDepsT],
        agent: AbstractAgent[AgentDepsT, Any] | None = None,
    ):
        super().__init__(toolset)
        self._agent = agent
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

        async def get_tools_activity(params: GetToolsParams, deps: AgentDepsT) -> dict[str, ToolDefinition]:
            run_context = deserialize_run_context(
                self.run_context_type, params.serialized_run_context, deps=deps, agent=self._agent
            )
            tools = await self.wrapped.get_tools(run_context)
            # ToolsetTool is not serializable as it holds a SchemaValidator (which is also the same for every MCP tool so unnecessary to pass along the wire every time),
            # so we just return the ToolDefinitions and wrap them in ToolsetTool outside of the activity.
            return {name: tool.tool_def for name, tool in tools.items()}

        # Set type hint explicitly so that Temporal can take care of serialization and deserialization
        get_tools_activity.__annotations__['deps'] = deps_type

        self.get_tools_activity = activity.defn(name=f'{activity_name_prefix}__mcp_server__{self.id}__get_tools')(
            get_tools_activity
        )

        async def get_instructions_activity(
            params: GetToolsParams, deps: AgentDepsT
        ) -> str | InstructionPart | Sequence[str | InstructionPart] | None:
            run_context = deserialize_run_context(
                self.run_context_type, params.serialized_run_context, deps=deps, agent=self._agent
            )
            async with self.wrapped:
                return await self.wrapped.get_instructions(run_context)

        # Set type hint explicitly so that Temporal can take care of serialization and deserialization
        get_instructions_activity.__annotations__['deps'] = deps_type

        self.get_instructions_activity = activity.defn(
            name=f'{activity_name_prefix}__mcp_server__{self.id}__get_instructions'
        )(get_instructions_activity)

        async def call_tool_activity(params: CallToolParams, deps: AgentDepsT) -> CallToolResult:
            run_context = deserialize_run_context(
                self.run_context_type, params.serialized_run_context, deps=deps, agent=self._agent
            )
            assert isinstance(params.tool_def, ToolDefinition)
            return await self._wrap_call_tool_result(
                self.wrapped.call_tool(
                    params.name,
                    params.tool_args,
                    run_context,
                    self.tool_for_tool_def(params.tool_def),
                )
            )

        # Set type hint explicitly so that Temporal can take care of serialization and deserialization
        call_tool_activity.__annotations__['deps'] = deps_type

        self.call_tool_activity = activity.defn(name=f'{activity_name_prefix}__mcp_server__{self.id}__call_tool')(
            call_tool_activity
        )

    @abstractmethod
    def tool_for_tool_def(self, tool_def: ToolDefinition) -> ToolsetTool[AgentDepsT]:
        raise NotImplementedError

    @property
    def temporal_activities(self) -> list[Callable[..., Any]]:
        return [self.get_instructions_activity, self.get_tools_activity, self.call_tool_activity]

    async def get_instructions(
        self, ctx: RunContext[AgentDepsT]
    ) -> str | InstructionPart | Sequence[str | InstructionPart] | None:
        if not workflow.in_workflow():  # pragma: no cover
            return await super().get_instructions(ctx)

        # If instructions are enabled, fetch via activity (the server isn't initialized locally in workflows).
        _mcp_types: tuple[type, ...] = ()
        try:
            from pydantic_ai.mcp import MCPServer

            _mcp_types += (MCPServer,)
        except ImportError:
            pass
        try:
            from pydantic_ai.toolsets.fastmcp import FastMCPToolset

            _mcp_types += (FastMCPToolset,)
        except ImportError:
            pass
        if _mcp_types and isinstance(self.wrapped, _mcp_types) and self.wrapped.include_instructions:  # type: ignore[union-attr]
            serialized_run_context = self.run_context_type.serialize_run_context(ctx)
            activity_config: ActivityConfig = {'summary': f'get instructions: {self.id}', **self.activity_config}
            return await workflow.execute_activity(
                activity=self.get_instructions_activity,
                args=[
                    GetToolsParams(serialized_run_context=serialized_run_context),
                    ctx.deps,
                ],
                **activity_config,
            )
        return None

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        if not workflow.in_workflow():  # pragma: no cover
            return await super().get_tools(ctx)

        serialized_run_context = self.run_context_type.serialize_run_context(ctx)
        activity_config: ActivityConfig = {'summary': f'get tools: {self.id}', **self.activity_config}
        tool_defs = await workflow.execute_activity(
            activity=self.get_tools_activity,
            args=[
                GetToolsParams(serialized_run_context=serialized_run_context),
                ctx.deps,
            ],
            **activity_config,
        )
        return {name: self.tool_for_tool_def(tool_def) for name, tool_def in tool_defs.items()}

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> CallToolResult:
        if not workflow.in_workflow():  # pragma: no cover
            return await super().call_tool(name, tool_args, ctx, tool)

        activity_config: ActivityConfig = {
            'summary': f'call tool: {self.id}:{name}',
            **self.activity_config,
            **self.tool_activity_config.get(name, {}),
        }
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
                **activity_config,
            )
        )
