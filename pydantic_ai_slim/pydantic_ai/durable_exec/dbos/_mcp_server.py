from __future__ import annotations

from pydantic_ai import ToolsetTool
from pydantic_ai.mcp import MCPServer
from pydantic_ai.tools import AgentDepsT, ToolDefinition

from ._mcp import DBOSMCPToolset
from ._utils import StepConfig


class DBOSMCPServer(DBOSMCPToolset[AgentDepsT]):
    """A wrapper for MCPServer that integrates with DBOS, turning call_tool and get_tools to DBOS steps."""

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

    def tool_for_tool_def(self, tool_def: ToolDefinition) -> ToolsetTool[AgentDepsT]:
        assert isinstance(self.wrapped, MCPServer)
        return self.wrapped.tool_for_tool_def(tool_def)
