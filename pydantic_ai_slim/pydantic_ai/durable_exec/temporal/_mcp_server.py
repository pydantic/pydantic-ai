from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from temporalio.workflow import ActivityConfig

from pydantic_ai import ToolsetTool
from pydantic_ai.mcp import MCPServer
from pydantic_ai.tools import AgentDepsT, ToolDefinition

if TYPE_CHECKING:
    from pydantic_ai.agent.abstract import AbstractAgent

from ._mcp import TemporalMCPToolsetBase
from ._run_context import TemporalRunContext


class TemporalMCPServer(TemporalMCPToolsetBase[AgentDepsT]):
    """A wrapper for MCPServer that integrates with Temporal, turning get_tools and call_tool into activities.

    Tool definitions are cached per run (on the run context) to avoid redundant MCP server round-trips,
    respecting the wrapped server's `cache_tools` setting.
    """

    def __init__(
        self,
        server: MCPServer,
        *,
        activity_name_prefix: str,
        activity_config: ActivityConfig,
        tool_activity_config: dict[str, ActivityConfig | Literal[False]],
        deps_type: type[AgentDepsT],
        run_context_type: type[TemporalRunContext[AgentDepsT]] = TemporalRunContext[AgentDepsT],
        agent: AbstractAgent[AgentDepsT, Any] | None = None,
    ):
        super().__init__(
            server,
            activity_name_prefix=activity_name_prefix,
            activity_config=activity_config,
            tool_activity_config=tool_activity_config,
            deps_type=deps_type,
            run_context_type=run_context_type,
            agent=agent,
        )

    @property
    def _server(self) -> MCPServer:
        assert isinstance(self.wrapped, MCPServer)
        return self.wrapped

    @property
    def _cache_tools(self) -> bool:
        return self._server.cache_tools

    def tool_for_tool_def(self, tool_def: ToolDefinition) -> ToolsetTool[AgentDepsT]:
        return self._server.tool_for_tool_def(tool_def)
