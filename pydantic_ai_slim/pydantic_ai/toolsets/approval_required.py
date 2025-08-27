from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

from .._run_context import AgentDepsT, RunContext
from ..tools import ToolDefinition
from .abstract import ToolsetTool
from .wrapper import WrapperToolset


@dataclass
class ApprovalRequiredToolset(WrapperToolset[AgentDepsT]):
    """TODO: Docstring."""

    approval_required_func: Callable[[RunContext[AgentDepsT], ToolDefinition], bool] = lambda ctx, tool_def: True

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        return {
            name: replace(tool, tool_def=replace(tool.tool_def, kind='unapproved'))
            if not ctx.resuming_after_deferred_tool_calls and self.approval_required_func(ctx, tool.tool_def)
            else tool
            for name, tool in (await super().get_tools(ctx)).items()
        }
