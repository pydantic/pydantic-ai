from __future__ import annotations

from typing import Any, Generic

from hatchet_sdk import Context, Hatchet
from hatchet_sdk.runnables.workflow import Standalone
from pydantic import BaseModel, ConfigDict

from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import FunctionToolset, ToolsetTool

from ._toolset import HatchetWrapperToolset
from ._utils import TaskConfig


class CallToolInput(BaseModel, Generic[AgentDepsT]):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    tool_args: dict[str, Any]
    ctx: RunContext[AgentDepsT]


class HatchetFunctionToolset(HatchetWrapperToolset[AgentDepsT]):
    """A wrapper for FunctionToolset that integrates with Hatchet, turning tool calls into Hatchet tasks."""

    def __init__(
        self, wrapped: FunctionToolset[AgentDepsT], *, hatchet: Hatchet, task_name_prefix: str, task_config: TaskConfig
    ):
        super().__init__(wrapped)
        self._task_config = task_config
        self._task_name_prefix = task_name_prefix
        self._hatchet = hatchet
        self._tool_tasks: dict[str, Standalone[Any, Any]] = {}

        for tool_name, tool in wrapped.tools.items():
            task_name = f'{task_name_prefix}__function_tool__{tool_name}'

            def make_tool_task(current_tool_name: str, current_tool: Any):
                @hatchet.task(
                    name=task_name,
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
                async def tool_task(
                    input: CallToolInput[AgentDepsT],
                    _ctx: Context,
                ) -> Any:
                    return await super(HatchetFunctionToolset, self).call_tool(
                        current_tool_name, input.tool_args, input.ctx, current_tool
                    )

                return tool_task

            self._tool_tasks[tool_name] = make_tool_task(tool_name, tool)

    @property
    def hatchet_tasks(self) -> list[Standalone[Any, Any]]:
        """Return the list of Hatchet tasks for this toolset."""
        return list(self._tool_tasks.values())

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> Any:
        if name not in self._tool_tasks:
            raise UserError(
                f'Tool {name!r} not found in toolset {self.id!r}. '
                'Removing or renaming tools during an agent run is not supported with Hatchet.'
            )

        tool_task = self._tool_tasks[name]
        return await tool_task.aio_run(
            CallToolInput(
                name=name,
                tool_args=tool_args,
                ctx=ctx,
            )
        )
