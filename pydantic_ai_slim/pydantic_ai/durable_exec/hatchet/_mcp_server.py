from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from hatchet_sdk import DurableContext, Hatchet
from pydantic import BaseModel, ConfigDict

from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets.abstract import ToolsetTool
from pydantic_ai.toolsets.wrapper import WrapperToolset

from ._utils import TaskConfig

if TYPE_CHECKING:
    from pydantic_ai.mcp import MCPServer, ToolResult

T = TypeVar('T')


class GetToolsInput(BaseModel, Generic[AgentDepsT]):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    ctx: RunContext[AgentDepsT]


class CallToolInput(BaseModel, Generic[AgentDepsT]):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    tool_args: dict[str, Any]
    ctx: RunContext[AgentDepsT]
    tool: ToolsetTool[AgentDepsT]


class CallToolOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    result: ToolResult


class HatchetMCPServer(WrapperToolset[AgentDepsT], ABC):
    """A wrapper for MCPServer that integrates with Hatchet, turning call_tool and get_tools to Hatchet tasks."""

    def __init__(self, wrapped: MCPServer, *, hatchet: Hatchet, task_name_prefix: str, task_config: TaskConfig):
        super().__init__(wrapped)
        self._task_config = task_config
        self._task_name_prefix = task_name_prefix
        self._hatchet = hatchet
        id_suffix = f'__{wrapped.id}' if wrapped.id else ''
        self._name = f'{task_name_prefix}__mcp_server{id_suffix}'

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
            ctx: DurableContext,
        ) -> dict[str, ToolsetTool[AgentDepsT]]:
            return await super(HatchetMCPServer, self).get_tools(input.ctx)

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
            result = await super(HatchetMCPServer, self).call_tool(input.name, input.tool_args, input.ctx, input.tool)

            return CallToolOutput[AgentDepsT](result=result)

        self.hatchet_wrapped_call_tool_task = wrapped_call_tool_task

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        return await self.hatchet_wrapped_get_tools_task.aio_run(GetToolsInput(ctx=ctx))

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> ToolResult:
        wrapped_tool_output = await self.hatchet_wrapped_call_tool_task.aio_run(
            CallToolInput(
                name=name,
                tool_args=tool_args,
                ctx=ctx,
                tool=tool,
            )
        )

        return wrapped_tool_output.result
