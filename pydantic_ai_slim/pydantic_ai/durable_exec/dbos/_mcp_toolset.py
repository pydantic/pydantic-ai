from __future__ import annotations

from pydantic_ai import ToolsetTool
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.tools import AgentDepsT, ToolDefinition

from ._mcp import DBOSMCPToolsetBase
from ._utils import StepConfig


class DBOSMCPToolset(DBOSMCPToolsetBase[AgentDepsT]):
    """A wrapper for `MCPToolset` that integrates with DBOS, turning `call_tool` and `get_tools` into DBOS steps.

    Tool definitions are cached per run (on the run context) to avoid redundant MCP server round-trips,
    respecting the wrapped toolset's `cache_tools` setting.
    """

    def __init__(
        self,
        wrapped: MCPToolset[AgentDepsT],
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
    def _toolset(self) -> MCPToolset[AgentDepsT]:
        assert isinstance(self.wrapped, MCPToolset)
        return self.wrapped

    @property
    def _cache_tools(self) -> bool:
        return self._toolset.cache_tools

    def tool_for_tool_def(self, tool_def: ToolDefinition) -> ToolsetTool[AgentDepsT]:
        return self._toolset.tool_for_tool_def(tool_def)
