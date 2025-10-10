from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic_ai import AbstractToolset, FunctionToolset, WrapperToolset
from pydantic_ai.tools import AgentDepsT

from ._types import TaskConfig

if TYPE_CHECKING:
    pass


class PrefectWrapperToolset(WrapperToolset[AgentDepsT], ABC):
    """Base class for Prefect-wrapped toolsets."""

    @property
    def id(self) -> str | None:
        # Prefect toolsets should have IDs for better task naming
        return self.wrapped.id

    def visit_and_replace(
        self, visitor: Callable[[AbstractToolset[AgentDepsT]], AbstractToolset[AgentDepsT]]
    ) -> AbstractToolset[AgentDepsT]:
        # Prefect-ified toolsets cannot be swapped out after the fact.
        return self


def prefectify_toolset(
    toolset: AbstractToolset[AgentDepsT],
    mcp_task_config: TaskConfig,
    tool_task_config: TaskConfig,
    tool_task_config_by_name: dict[str, TaskConfig | None],
) -> AbstractToolset[AgentDepsT]:
    """Wrap a toolset to integrate it with Prefect.

    Args:
        toolset: The toolset to wrap.
        mcp_task_config: The Prefect task config to use for MCP server tasks.
        tool_task_config: The default Prefect task config to use for tool calls.
        tool_task_config_by_name: Per-tool task configuration. Keys are tool names, values are TaskConfig or None.
    """
    if isinstance(toolset, FunctionToolset):
        from ._function_toolset import PrefectFunctionToolset

        return PrefectFunctionToolset(
            wrapped=toolset,
            task_config=tool_task_config,
            tool_task_config=tool_task_config_by_name,
        )

    try:
        from pydantic_ai.mcp import MCPServer

        from ._mcp_server import PrefectMCPServer
    except ImportError:
        pass
    else:
        if isinstance(toolset, MCPServer):
            return PrefectMCPServer(
                wrapped=toolset,
                task_config=mcp_task_config,
            )

    return toolset
