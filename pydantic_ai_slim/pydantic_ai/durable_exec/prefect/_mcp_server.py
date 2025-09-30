from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from prefect import task
from typing_extensions import Self

from pydantic_ai import AbstractToolset, ToolsetTool, WrapperToolset
from pydantic_ai.tools import AgentDepsT, RunContext

from ._run_context import SerializableRunContext
from ._utils import TaskConfig

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

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        """Get tools from the MCP server, wrapped as a Prefect task."""
        mcp_label = f' ({self._mcp_id})' if self._mcp_id else ''

        # Wrap ctx in SerializableRunContext for proper cache key serialization
        serializable_ctx = SerializableRunContext.wrap(ctx)

        @task(
            name=f'Get MCP Tools{mcp_label}',
            **self._task_config,
        )
        async def get_tools_task(serializable_ctx: SerializableRunContext) -> dict[str, ToolsetTool[AgentDepsT]]:
            # Unwrap to get the original RunContext
            unwrapped_ctx = serializable_ctx.unwrap()
            return await super(PrefectMCPServer, self).get_tools(unwrapped_ctx)

        return await get_tools_task(serializable_ctx)

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
            name=f'Call Tool: {name}',
            **self._task_config,
        )
        async def call_tool_task(
            tool_name: str,
            args: dict[str, Any],
            serializable_ctx: SerializableRunContext,
        ) -> ToolResult:
            # Unwrap to get the original RunContext
            # Note: We don't include 'tool' parameter as it contains non-serializable objects
            unwrapped_ctx = serializable_ctx.unwrap()
            return await super(PrefectMCPServer, self).call_tool(tool_name, args, unwrapped_ctx, tool)

        return await call_tool_task(name, tool_args, serializable_ctx)
