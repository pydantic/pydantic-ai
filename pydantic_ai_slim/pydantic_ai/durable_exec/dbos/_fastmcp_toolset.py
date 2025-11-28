from __future__ import annotations

from pydantic_ai import ToolsetTool
from pydantic_ai.tools import AgentDepsT, ToolDefinition
from pydantic_ai.toolsets.fastmcp import FastMCPToolset

from ._mcp import DBOSMCPToolset
from ._utils import StepConfig


class DBOSFastMCPToolset(DBOSMCPToolset[AgentDepsT]):
    """A wrapper for FastMCPToolset that integrates with DBOS, turning call_tool and get_tools to DBOS steps."""

    def __init__(
        self,
        wrapped: FastMCPToolset[AgentDepsT],
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
        assert isinstance(self.wrapped, FastMCPToolset)
        return self.wrapped.tool_for_tool_def(tool_def)
