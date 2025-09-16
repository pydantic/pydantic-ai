from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from hatchet_sdk import DurableContext, Hatchet
from pydantic import BaseModel

from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets.abstract import ToolsetTool
from pydantic_ai.toolsets.wrapper import WrapperToolset

from ._utils import TaskConfig

if TYPE_CHECKING:
    from pydantic_ai.mcp import MCPServer, ToolResult

T = TypeVar('T')


class GetToolsInput(BaseModel, Generic[AgentDepsT]):
    ctx: RunContext[AgentDepsT]


class CallToolInput(BaseModel, Generic[AgentDepsT]):
    name: str
    tool_args: dict[str, Any]
    ctx: RunContext[AgentDepsT]
    tool: ToolsetTool[AgentDepsT]


class CallToolOutput(BaseModel):
    result: ToolResult


class HatchetMCPServer(WrapperToolset[AgentDepsT], ABC):
    """A wrapper for MCPServer that integrates with Hatchet, turning call_tool and get_tools to Hatchet tasks."""

    def __init__(self, wrapped: MCPServer, *, task_config: TaskConfig, hatchet: Hatchet):
        super().__init__(wrapped)
        self.task_config = task_config
        self.hatchet = hatchet

        @hatchet.durable_task(
            name=f'{self.task_config.name}.get_tools',
            description=self.task_config.description,
            input_validator=GetToolsInput[AgentDepsT],
            on_events=self.task_config.on_events,
            on_crons=self.task_config.on_crons,
            version=self.task_config.version,
            sticky=self.task_config.sticky,
            default_priority=self.task_config.default_priority,
            concurrency=self.task_config.concurrency,
            schedule_timeout=self.task_config.schedule_timeout,
            execution_timeout=self.task_config.execution_timeout,
            retries=self.task_config.retries,
            rate_limits=self.task_config.rate_limits,
            desired_worker_labels=self.task_config.desired_worker_labels,
            backoff_factor=self.task_config.backoff_factor,
            backoff_max_seconds=self.task_config.backoff_max_seconds,
            default_filters=self.task_config.default_filters,
        )
        async def wrapped_get_tools_task(
            input: GetToolsInput[AgentDepsT],
            ctx: DurableContext,
        ) -> dict[str, ToolsetTool[AgentDepsT]]:
            return await super(HatchetMCPServer, self).get_tools(input.ctx)

        self._hatchet_wrapped_get_tools_task = wrapped_get_tools_task

        @hatchet.durable_task(
            name=f'{self.task_config.name}.get_tools',
            description=self.task_config.description,
            input_validator=CallToolInput[AgentDepsT],
            on_events=self.task_config.on_events,
            on_crons=self.task_config.on_crons,
            version=self.task_config.version,
            sticky=self.task_config.sticky,
            default_priority=self.task_config.default_priority,
            concurrency=self.task_config.concurrency,
            schedule_timeout=self.task_config.schedule_timeout,
            execution_timeout=self.task_config.execution_timeout,
            retries=self.task_config.retries,
            rate_limits=self.task_config.rate_limits,
            desired_worker_labels=self.task_config.desired_worker_labels,
            backoff_factor=self.task_config.backoff_factor,
            backoff_max_seconds=self.task_config.backoff_max_seconds,
            default_filters=self.task_config.default_filters,
        )
        async def wrapped_call_tool_task(
            input: CallToolInput[AgentDepsT],
            _ctx: DurableContext,
        ) -> CallToolOutput[AgentDepsT]:
            result = await super(HatchetMCPServer, self).call_tool(input.name, input.tool_args, input.ctx, input.tool)

            return CallToolOutput[AgentDepsT](result=result)

        self._hatchet_wrapped_call_tool_task = wrapped_call_tool_task

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        return await self._hatchet_wrapped_get_tools_task.aio_run(GetToolsInput(ctx=ctx))

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> ToolResult:
        wrapped_tool_output = await self._hatchet_wrapped_call_tool_task.aio_run(
            CallToolInput(
                name=name,
                tool_args=tool_args,
                ctx=ctx,
                tool=tool,
            )
        )

        return wrapped_tool_output.result
