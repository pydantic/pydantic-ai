from __future__ import annotations

from abc import ABC
from collections.abc import Callable, Mapping
from typing import Any, Literal, cast

from pydantic_ai import AbstractToolset, FunctionToolset, ToolsetTool, WrapperToolset
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT

from ._types import TaskConfig


def resolve_tool_task_config(
    tool: ToolsetTool[Any] | None,
    tool_name: str,
    tool_task_config: Mapping[str, TaskConfig | None],
) -> TaskConfig | Literal[False]:
    """Resolve per-tool Prefect task config.

    Reads `tool.tool_def.metadata['prefect']` first, then falls back to the explicit
    `tool_task_config` dict keyed by tool name. Returns a `TaskConfig` dict (possibly
    empty), or `False` to skip task wrapping.
    """
    # Metadata set on the tool (via @toolset.tool(metadata={'prefect': ...}), with_metadata, or
    # the `SetToolMetadata` capability) is the primary path.
    if tool is not None and tool.tool_def.metadata is not None:
        metadata_config = tool.tool_def.metadata.get('prefect')
        if metadata_config is False:
            return False
        if metadata_config is not None:
            if not isinstance(metadata_config, dict):
                raise UserError(
                    f"Tool {tool_name!r} has invalid 'prefect' metadata: expected a dict "
                    f'(`TaskConfig`) or `False`, got {type(metadata_config).__name__}.'
                )
            return cast('TaskConfig', metadata_config)
    # Fallback: per-tool dict passed to the deprecated `PrefectAgent`. An explicit `None`
    # disables wrapping; a missing key means "use the base config".
    if tool_name in tool_task_config:
        fallback = tool_task_config[tool_name]
        return False if fallback is None else fallback
    return {}


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
        from pydantic_ai.mcp import MCPToolset

        from ._mcp_toolset import PrefectMCPToolset
    except ImportError:
        pass
    else:
        if isinstance(toolset, MCPToolset):
            return PrefectMCPToolset(
                wrapped=toolset,
                task_config=mcp_task_config,
            )

    return toolset
