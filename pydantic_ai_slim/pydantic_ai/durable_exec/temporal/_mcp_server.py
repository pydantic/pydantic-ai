from __future__ import annotations

from typing import Literal

from temporalio.workflow import ActivityConfig

from pydantic_ai import ToolsetTool
from pydantic_ai.mcp import MCPServer
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition

from ._mcp import TemporalMCPToolset
from ._run_context import TemporalRunContext


class TemporalMCPServer(TemporalMCPToolset[AgentDepsT]):
    """A wrapper for MCPServer that integrates with Temporal, turning get_tools and call_tool into activities.

    Tool definitions are cached across activities to avoid redundant MCP server round-trips,
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
    ):
        super().__init__(
            server,
            activity_name_prefix=activity_name_prefix,
            activity_config=activity_config,
            tool_activity_config=tool_activity_config,
            deps_type=deps_type,
            run_context_type=run_context_type,
        )
        # Cached across activities to avoid redundant MCP connections per activity.
        # Not invalidated by `tools/list_changed` notifications — users who need
        # dynamic tools during a workflow should set `cache_tools=False`.
        self._cached_tool_defs: dict[str, ToolDefinition] | None = None

    @property
    def _server(self) -> MCPServer:
        assert isinstance(self.wrapped, MCPServer)
        return self.wrapped

    def tool_for_tool_def(self, tool_def: ToolDefinition) -> ToolsetTool[AgentDepsT]:
        return self._server.tool_for_tool_def(tool_def)

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        if self._server.cache_tools and self._cached_tool_defs is not None:
            return {name: self.tool_for_tool_def(td) for name, td in self._cached_tool_defs.items()}

        result = await super().get_tools(ctx)
        if self._server.cache_tools:  # pragma: no branch
            self._cached_tool_defs = {name: tool.tool_def for name, tool in result.items()}
        return result
