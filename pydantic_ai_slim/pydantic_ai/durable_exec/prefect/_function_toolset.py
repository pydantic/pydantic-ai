from __future__ import annotations

from typing import Any

from prefect import task

from pydantic_ai import FunctionToolset, ToolsetTool
from pydantic_ai.tools import AgentDepsT, RunContext

from ._toolset import PrefectWrapperToolset, resolve_tool_task_config
from ._types import TaskConfig, default_task_config


class PrefectFunctionToolset(PrefectWrapperToolset[AgentDepsT]):
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

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> Any:
        """Call a tool, wrapped as a Prefect task with a descriptive name."""
        # Per-tool config comes from `metadata={'prefect': ...}` first, then the deprecated
        # `PrefectAgent`'s by-name dict. `False` disables task wrapping for this tool.
        tool_task_config = resolve_tool_task_config(tool, name, self._tool_task_config)
        if tool_task_config is False:
            return await super().call_tool(name, tool_args, ctx, tool)

        merged_config = self._task_config | tool_task_config

        return await self._call_tool_task.with_options(name=f'Call Tool: {name}', **merged_config)(
            name, tool_args, ctx, tool
        )
