from __future__ import annotations

from typing import Literal

from temporalio.workflow import ActivityConfig

from pydantic_ai import ToolsetTool
from pydantic_ai.tools import AgentDepsT, ToolDefinition
from pydantic_ai.toolsets.fastmcp import FastMCPToolset

from ._mcp import TemporalMCPToolset
from ._run_context import TemporalRunContext


class TemporalFastMCPToolset(TemporalMCPToolset[AgentDepsT]):
    def __init__(
        self,
        toolset: FastMCPToolset[AgentDepsT],
        *,
        activity_name_prefix: str,
        activity_config: ActivityConfig,
        tool_activity_config: dict[str, ActivityConfig | Literal[False]],
        deps_type: type[AgentDepsT],
        run_context_type: type[TemporalRunContext[AgentDepsT]] = TemporalRunContext[AgentDepsT],
    ):
        super().__init__(
            toolset,
            activity_name_prefix=activity_name_prefix,
            activity_config=activity_config,
            tool_activity_config=tool_activity_config,
            deps_type=deps_type,
            run_context_type=run_context_type,
        )

    def tool_for_tool_def(self, tool_def: ToolDefinition) -> ToolsetTool[AgentDepsT]:
        assert isinstance(self.wrapped, FastMCPToolset)
        return self.wrapped.tool_for_tool_def(tool_def)
