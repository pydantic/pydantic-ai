from typing import Any

from hatchet_sdk import Context, Hatchet
from hatchet_sdk.runnables.workflow import Standalone
from pydantic import BaseModel, ConfigDict

from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import FunctionToolset, ToolsetTool

from ._mcp_server import CallToolInput
from ._run_context import HatchetRunContext
from ._toolset import HatchetWrapperToolset
from ._utils import TaskConfig


class ToolOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    result: Any


class HatchetFunctionToolset(HatchetWrapperToolset[AgentDepsT]):
    """A wrapper for FunctionToolset that integrates with Hatchet, turning tool calls into Hatchet tasks."""

    def __init__(
        self,
        wrapped: FunctionToolset[AgentDepsT],
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
        self._tool_tasks: dict[str, Standalone[CallToolInput[AgentDepsT], ToolOutput]] = {}
        self.run_context_type = run_context_type

        for tool_name in wrapped.tools.keys():
            task_name = f'{task_name_prefix}__function_tool__{tool_name}'

            def make_tool_task(current_tool_name: str):
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
                ) -> ToolOutput:
                    run_context = self.run_context_type.deserialize_run_context(
                        input.serialized_run_context, deps=input.deps, hatchet_context=_ctx
                    )
                    tool = (await wrapped.get_tools(run_context))[current_tool_name]

                    result = await super(HatchetFunctionToolset, self).call_tool(
                        current_tool_name, input.tool_args, run_context, tool
                    )

                    return ToolOutput(result=result)

                return tool_task

            self._tool_tasks[tool_name] = make_tool_task(tool_name)

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

        tool_task: Standalone[CallToolInput[AgentDepsT], ToolOutput] = self._tool_tasks[name]
        serialized_run_context = self.run_context_type.serialize_run_context(ctx)

        output = await tool_task.aio_run(
            CallToolInput(
                name=name,
                tool_args=tool_args,
                tool_def=tool.tool_def,
                serialized_run_context=serialized_run_context,
                deps=ctx.deps,
            )
        )

        return output.result
