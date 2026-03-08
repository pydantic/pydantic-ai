from __future__ import annotations

from pydantic_ai import ToolsetTool
from pydantic_ai.mcp import MCPServer
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition

from ._mcp import DBOSMCPToolset
from ._utils import StepConfig


class DBOSMCPServer(DBOSMCPToolset[AgentDepsT]):
    """A wrapper for MCPServer that integrates with DBOS, turning call_tool and get_tools into DBOS steps.

    Tool definitions are cached across steps to avoid redundant MCP server round-trips,
    respecting the wrapped server's `cache_tools` setting.
    """

    def __init__(
        self,
        wrapped: MCPServer,
        *,
        step_name_prefix: str,
        step_config: StepConfig,
    ):
        super().__init__(
            wrapped,
            step_name_prefix=step_name_prefix,
            step_config=step_config,
        )
        # Cached across steps to avoid redundant MCP connections per step.
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
        if self._server.cache_tools:
            self._cached_tool_defs = {name: tool.tool_def for name, tool in result.items()}
        return result
