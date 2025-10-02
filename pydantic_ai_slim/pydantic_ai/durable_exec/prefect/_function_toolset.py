from __future__ import annotations

from typing import TYPE_CHECKING, Any

from prefect import get_run_logger, task
from prefect.context import FlowRunContext

from pydantic_ai import FunctionToolset, ToolsetTool, WrapperToolset
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, ModelRetry, UserError
from pydantic_ai.tools import AgentDepsT, RunContext

from ._run_context import SerializableRunContext
from ._types import TaskConfig, default_task_config

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic_ai import AbstractToolset


class PrefectFunctionToolset(WrapperToolset[AgentDepsT]):
    """A wrapper for FunctionToolset that integrates with Prefect, turning tool calls into Prefect tasks."""

    def __init__(
        self,
        wrapped: FunctionToolset[AgentDepsT],
        *,
        task_config: TaskConfig,
        tool_task_config: dict[str, TaskConfig | None],
    ):
        super().__init__(wrapped)
        self._task_config = default_task_config | (task_config or {})
        self._tool_task_config = tool_task_config or {}

    def visit_and_replace(
        self, visitor: Callable[[AbstractToolset[AgentDepsT]], AbstractToolset[AgentDepsT]]
    ) -> AbstractToolset[AgentDepsT]:
        # Prefect-ified toolsets cannot be swapped out after the fact.
        return self

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> Any:
        """Call a tool, wrapped as a Prefect task with a descriptive name."""
        # If not in a flow, just call the tool directly
        if FlowRunContext.get() is None:
            return await super().call_tool(name, tool_args, ctx, tool)

        # Check if this specific tool has custom config or is disabled
        tool_specific_config = self._tool_task_config.get(name, default_task_config)
        if tool_specific_config is None:
            # None means this tool should not be wrapped as a task
            return await super().call_tool(name, tool_args, ctx, tool)

        # Merge tool-specific config with default config
        merged_config = self._task_config | tool_specific_config

        # Wrap ctx in SerializableRunContext for proper cache key serialization
        serializable_ctx = SerializableRunContext.wrap(ctx)

        @task(
            name=f'Call Tool: {name}',
            **merged_config,
        )
        async def call_tool_task(
            tool_name: str,
            args: dict[str, Any],
            serializable_ctx: SerializableRunContext,
        ) -> Any:
            logger = get_run_logger()
            logger.info(f'Calling tool: {tool_name}')

            # Unwrap to get the original RunContext
            unwrapped_ctx = serializable_ctx.unwrap()

            # Get the tool (need to fetch it again inside the task)
            try:
                tools = await self.wrapped.get_tools(unwrapped_ctx)
                tool_instance = tools[tool_name]
            except KeyError as e:  # pragma: no cover
                raise UserError(
                    f'Tool {tool_name!r} not found in toolset {self.id!r}. '
                    'Removing or renaming tools during an agent run is not supported with Prefect.'
                ) from e

            # Call the tool
            result = await super(PrefectFunctionToolset, self).call_tool(tool_name, args, unwrapped_ctx, tool_instance)
            logger.info(f'Tool call completed: {tool_name}')
            return result

        try:
            return await call_tool_task(name, tool_args, serializable_ctx)
        except (ApprovalRequired, CallDeferred, ModelRetry):
            # Re-raise these exceptions as they're part of the agent control flow
            raise
