from __future__ import annotations

from pydantic_ai import ToolsetTool
from pydantic_ai.mcp import MCPServer
from pydantic_ai.tools import AgentDepsT, ToolDefinition

from ._mcp import DBOSMCPToolsetBase
from ._utils import StepConfig


class DBOSMCPServer(DBOSMCPToolsetBase[AgentDepsT]):
    """A wrapper for MCPServer that integrates with DBOS, turning call_tool and get_tools into DBOS steps.

    Tool definitions are cached per run (on the run context) to avoid redundant MCP server round-trips,
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

    @property
    def _server(self) -> MCPServer:
        assert isinstance(self.wrapped, MCPServer)
        return self.wrapped

    @property
    def _cache_tools(self) -> bool:
        return self._server.cache_tools

    def tool_for_tool_def(self, tool_def: ToolDefinition) -> ToolsetTool[AgentDepsT]:
        return self._server.tool_for_tool_def(tool_def)
