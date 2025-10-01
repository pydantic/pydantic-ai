from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from prefect import get_run_logger, task
from typing_extensions import Self

from pydantic_ai import AbstractToolset, ToolsetTool, WrapperToolset
from pydantic_ai.tools import AgentDepsT, RunContext

from ._run_context import SerializableRunContext
from ._types import TaskConfig

if TYPE_CHECKING:
    from pydantic_ai.mcp import MCPServer, ToolResult


class PrefectMCPServer(WrapperToolset[AgentDepsT], ABC):
    """A wrapper for MCPServer that integrates with Prefect, turning call_tool and get_tools into Prefect tasks."""

    def __init__(
        self,
        wrapped: MCPServer,
        *,
        task_config: TaskConfig,
    ):
        super().__init__(wrapped)
        self._task_config = task_config or {}
        self._mcp_id = wrapped.id

    @property
    def id(self) -> str | None:
        return self.wrapped.id

    async def __aenter__(self) -> Self:
        # The wrapped MCPServer enters itself around listing and calling tools
        # so we don't need to enter it here (nor could we because we're not inside a Prefect task).
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        return None

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
    ) -> ToolResult:
        """Call an MCP tool, wrapped as a Prefect task with a descriptive name."""
        # Wrap ctx in SerializableRunContext for proper cache key serialization
        serializable_ctx = SerializableRunContext.wrap(ctx)

        @task(
            name=f'Call MCP Tool: {name}',
            **self._task_config,
        )
        async def call_tool_task(
            tool_name: str,
            args: dict[str, Any],
            serializable_ctx: SerializableRunContext,
        ) -> ToolResult:
            logger = get_run_logger()
            logger.info(f'Calling MCP tool: {tool_name}')

            # Unwrap to get the original RunContext
            # Note: We don't include 'tool' parameter as it contains non-serializable objects
            unwrapped_ctx = serializable_ctx.unwrap()
            result = await super(PrefectMCPServer, self).call_tool(tool_name, args, unwrapped_ctx, tool)
            logger.info(f'MCP tool call completed: {tool_name}')
            return result

        return await call_tool_task(name, tool_args, serializable_ctx)
