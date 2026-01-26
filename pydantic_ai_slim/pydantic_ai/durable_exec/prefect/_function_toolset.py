from __future__ import annotations

from typing import Any

from prefect import get_run_logger, task

from pydantic_ai import FunctionToolset, ToolsetTool
from pydantic_ai.tools import AgentDepsT, RunContext

from ._logging import format_tool_args, format_tool_result
from ._toolset import PrefectWrapperToolset
from ._types import TaskConfig, default_task_config


class PrefectFunctionToolset(PrefectWrapperToolset[AgentDepsT]):
    """A wrapper for FunctionToolset that integrates with Prefect, turning tool calls into Prefect tasks."""

    def __init__(
        self,
        wrapped: FunctionToolset[AgentDepsT],
        *,
        task_config: TaskConfig,
        tool_task_config: dict[str, TaskConfig | None],
        log_tool_calls: bool = False,
    ):
        super().__init__(wrapped)
        self._task_config = default_task_config | (task_config or {})
        self._tool_task_config = tool_task_config or {}
        self._log_tool_calls = log_tool_calls

        @task
        async def _call_tool_task(
            tool_name: str,
            tool_args: dict[str, Any],
            ctx: RunContext[AgentDepsT],
            tool: ToolsetTool[AgentDepsT],
        ) -> Any:
            if self._log_tool_calls:
                logger = get_run_logger()
                args_str = format_tool_args(tool_args)
                logger.info(f'Calling tool: {tool_name} with args: {args_str}')

            result = await super(PrefectFunctionToolset, self).call_tool(tool_name, tool_args, ctx, tool)

            if self._log_tool_calls:
                logger = get_run_logger()
                result_str = format_tool_result(result)
                logger.info(f'Tool {tool_name} returned: {result_str}')

            return result

        self._call_tool_task = _call_tool_task

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
