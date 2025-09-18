from abc import ABC
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from hatchet_sdk import DurableContext, Hatchet
from hatchet_sdk.runnables.workflow import Standalone
from pydantic import BaseModel, ConfigDict

from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets.abstract import (
    ToolDefinition,
    ToolsetTool,
)

from ._run_context import HatchetRunContext, SerializedHatchetRunContext
from ._toolset import HatchetWrapperToolset
from ._utils import TaskConfig

if TYPE_CHECKING:
    from pydantic_ai.mcp import MCPServer, ToolResult

T = TypeVar('T')


class GetToolsInput(BaseModel, Generic[AgentDepsT]):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    serialized_run_context: SerializedHatchetRunContext
    deps: AgentDepsT


class CallToolInput(BaseModel, Generic[AgentDepsT]):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    tool_args: dict[str, Any]
    tool_def: ToolDefinition

    serialized_run_context: SerializedHatchetRunContext
    deps: AgentDepsT


class CallToolOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    result: 'ToolResult'


class HatchetMCPServer(HatchetWrapperToolset[AgentDepsT], ABC):
    """A wrapper for MCPServer that integrates with Hatchet, turning call_tool and get_tools to Hatchet tasks."""

    def __init__(
        self,
        wrapped: 'MCPServer',
        *,
        hatchet: Hatchet,
        task_name_prefix: str,
        task_config: TaskConfig,
        deps_type: type[AgentDepsT],
        run_context_type: type[HatchetRunContext[AgentDepsT]] = HatchetRunContext[AgentDepsT],
    ):
        super().__init__(wrapped)
        self._task_config = task_config
        self._task_name_prefix = task_name_prefix
        self._hatchet = hatchet
        id_suffix = f'__{wrapped.id}' if wrapped.id else ''
        self._name = f'{task_name_prefix}__mcp_server{id_suffix}'
        self.run_context_type: type[HatchetRunContext[AgentDepsT]] = run_context_type

        @hatchet.durable_task(
            name=f'{self._name}.get_tools',
            description=self._task_config.description,
            input_validator=GetToolsInput[AgentDepsT],
            version=self._task_config.version,
            sticky=self._task_config.sticky,
            default_priority=self._task_config.default_priority,
            concurrency=self._task_config.concurrency,
            schedule_timeout=self._task_config.schedule_timeout,
            execution_timeout=self._task_config.execution_timeout,
            retries=self._task_config.retries,
            rate_limits=self._task_config.rate_limits,
            desired_worker_labels=self._task_config.desired_worker_labels,
            backoff_factor=self._task_config.backoff_factor,
            backoff_max_seconds=self._task_config.backoff_max_seconds,
            default_filters=self._task_config.default_filters,
        )
        async def wrapped_get_tools_task(
            input: GetToolsInput[AgentDepsT],
            _ctx: DurableContext,
        ) -> dict[str, ToolDefinition]:
            run_context = self.run_context_type.deserialize_run_context(input.serialized_run_context, deps=input.deps)

            # ToolsetTool is not serializable as it holds a SchemaValidator (which is also the same for every MCP tool so unnecessary to pass along the wire every time),
            # so we just return the ToolDefinitions and wrap them in ToolsetTool outside of the activity.
            tools = await super(HatchetMCPServer, self).get_tools(run_context)

            return {name: tool.tool_def for name, tool in tools.items()}

        self.hatchet_wrapped_get_tools_task = wrapped_get_tools_task

        @hatchet.durable_task(
            name=f'{self._name}.call_tool',
            description=self._task_config.description,
            input_validator=CallToolInput[AgentDepsT],
            version=self._task_config.version,
            sticky=self._task_config.sticky,
            default_priority=self._task_config.default_priority,
            concurrency=self._task_config.concurrency,
            schedule_timeout=self._task_config.schedule_timeout,
            execution_timeout=self._task_config.execution_timeout,
            retries=self._task_config.retries,
            rate_limits=self._task_config.rate_limits,
            desired_worker_labels=self._task_config.desired_worker_labels,
            backoff_factor=self._task_config.backoff_factor,
            backoff_max_seconds=self._task_config.backoff_max_seconds,
            default_filters=self._task_config.default_filters,
        )
        async def wrapped_call_tool_task(
            input: CallToolInput[AgentDepsT],
            _ctx: DurableContext,
        ) -> CallToolOutput[AgentDepsT]:
            run_context = self.run_context_type.deserialize_run_context(input.serialized_run_context, deps=input.deps)
            tool = self.tool_for_tool_def(input.tool_def)

            result = await super(HatchetMCPServer, self).call_tool(input.name, input.tool_args, run_context, tool)

            return CallToolOutput[AgentDepsT](result=result)

        self.hatchet_wrapped_call_tool_task = wrapped_call_tool_task

    @property
    def hatchet_tasks(self) -> list[Standalone[Any, Any]]:
        """Return the list of Hatchet tasks for this toolset."""
        return [
            self.hatchet_wrapped_get_tools_task,
            self.hatchet_wrapped_call_tool_task,
        ]

    def tool_for_tool_def(self, tool_def: ToolDefinition) -> ToolsetTool[AgentDepsT]:
        assert isinstance(self.wrapped, MCPServer)
        return self.wrapped.tool_for_tool_def(tool_def)

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        serialized_run_context = self.run_context_type.serialize_run_context(ctx)
        tool_defs = await self.hatchet_wrapped_get_tools_task.aio_run(
            GetToolsInput(
                serialized_run_context=serialized_run_context,
                deps=ctx.deps,
            )
        )

        return {name: self.tool_for_tool_def(tool_def) for name, tool_def in tool_defs.items()}

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> 'ToolResult':
        serialized_run_context = self.run_context_type.serialize_run_context(ctx)

        wrapped_tool_output = await self.hatchet_wrapped_call_tool_task.aio_run(
            CallToolInput(
                name=name,
                tool_args=tool_args,
                tool_def=tool.tool_def,
                serialized_run_context=serialized_run_context,
                deps=ctx.deps,
            )
        )

        return wrapped_tool_output.result
