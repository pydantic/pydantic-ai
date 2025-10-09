from __future__ import annotations

from typing import TYPE_CHECKING, Any

from prefect import task

from pydantic_ai import FunctionToolset, ToolsetTool, WrapperToolset
from pydantic_ai.tools import AgentDepsT, RunContext

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

        @task
        async def _call_tool_task(
            tool_name: str,
            tool_args: dict[str, Any],
            ctx: RunContext[AgentDepsT],
            tool: ToolsetTool[AgentDepsT],
        ) -> Any:
            return await super(PrefectFunctionToolset, self).call_tool(tool_name, tool_args, ctx, tool)

        self._call_tool_task = _call_tool_task

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
        # Check if this specific tool has custom config or is disabled
        tool_specific_config = self._tool_task_config.get(name, default_task_config)
        if tool_specific_config is None:
            # None means this tool should not be wrapped as a task
            return await super().call_tool(name, tool_args, ctx, tool)

        # Merge tool-specific config with default config
        merged_config = self._task_config | tool_specific_config

        return await self._call_tool_task.with_options(name=f'Call Tool: {name}', **merged_config)(
            name, tool_args, ctx, tool
        )
