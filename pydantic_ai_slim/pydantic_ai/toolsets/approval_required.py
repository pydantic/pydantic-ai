from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic_ai.exceptions import ApprovalRequired

from .._run_context import AgentDepsT, RunContext
from ..tools import ToolDefinition, ToolSelector, matches_tool_selector
from .abstract import ToolsetTool
from .wrapper import WrapperToolset


@dataclass(init=False)
class ApprovalRequiredToolset(WrapperToolset[AgentDepsT]):
    """A toolset that requires (some) calls to tools it contains to be approved.

    See [toolset docs](../toolsets.md#requiring-tool-approval) for more information.
    """

    tools: ToolSelector[AgentDepsT]
    approval_required_func: Callable[[RunContext[AgentDepsT], ToolDefinition, dict[str, Any]], bool] | None

    def __init__(
        self,
        wrapped: Any,  # AbstractToolset[AgentDepsT] — Any to avoid circular import
        approval_required_func: Callable[[RunContext[AgentDepsT], ToolDefinition, dict[str, Any]], bool] | None = None,
        *,
        tools: ToolSelector[AgentDepsT] = 'all',
    ) -> None:
        self.wrapped = wrapped
        self.tools = tools
        self.approval_required_func = approval_required_func

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        if not ctx.tool_call_approved and await matches_tool_selector(self.tools, ctx, tool.tool_def):
            if self.approval_required_func is None or self.approval_required_func(ctx, tool.tool_def, tool_args):
                raise ApprovalRequired

        return await super().call_tool(name, tool_args, ctx, tool)
