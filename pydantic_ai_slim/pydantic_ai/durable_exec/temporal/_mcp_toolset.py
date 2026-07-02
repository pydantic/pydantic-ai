from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from temporalio.workflow import ActivityConfig

from pydantic_ai import ToolsetTool
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.tools import AgentDepsT, ToolDefinition

if TYPE_CHECKING:
    from pydantic_ai.agent.abstract import AbstractAgent

from ._mcp import TemporalMCPToolsetBase
from ._run_context import TemporalRunContext


class TemporalMCPToolset(TemporalMCPToolsetBase[AgentDepsT]):
    """A wrapper for `MCPToolset` that integrates with Temporal, turning `get_tools` and `call_tool` into activities.

    Tool definitions are cached per run (on the run context) to avoid redundant MCP server round-trips,
    respecting the wrapped toolset's `cache_tools` setting.
    """

    def __init__(
        self,
        toolset: MCPToolset[AgentDepsT],
        *,
        activity_name_prefix: str,
        activity_config: ActivityConfig,
        tool_activity_config: dict[str, ActivityConfig | Literal[False]],
        deps_type: type[AgentDepsT],
        run_context_type: type[TemporalRunContext[AgentDepsT]] = TemporalRunContext[AgentDepsT],
        agent: AbstractAgent[AgentDepsT, Any] | None = None,
    ):
        super().__init__(
            toolset,
            activity_name_prefix=activity_name_prefix,
            activity_config=activity_config,
            tool_activity_config=tool_activity_config,
            deps_type=deps_type,
            run_context_type=run_context_type,
            agent=agent,
        )

    @property
    def _toolset(self) -> MCPToolset[AgentDepsT]:
        assert isinstance(self.wrapped, MCPToolset)
        return self.wrapped

    @property
    def _cache_tools(self) -> bool:
        return self._toolset.cache_tools

    def tool_for_tool_def(self, tool_def: ToolDefinition) -> ToolsetTool[AgentDepsT]:
        return self._toolset.tool_for_tool_def(tool_def)
